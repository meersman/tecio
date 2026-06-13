"""Split a Tecplot data file into separate grid and solution files.

Solvers that produce full Tecplot files containing both spatial coordinates and solution
variables in a single file can create workflows where the grid geometry is redundantly
stored across every time step.  Separating the coordinate and solution data into
distinct files with the appropriate ``FileType`` metadata allows Tecplot to load a
shared grid once and associate multiple solution files with it, reducing storage
overhead and enabling more efficient transient data management.  ``tecsplit`` performs
this separation automatically, detecting coordinate variables by name or accepting an
explicit list, and supports three operating modes.

Grid file
---------
The grid file (``FileType.GRID``) contains **only** the coordinate variables.  Its zone
list includes only those source zones that carry independent coordinate data — zones
whose coordinate variables are not shared from another zone.  Each grid zone is written
with ``solution_time=0`` and ``strand_id=0`` to mark the file as time-independent.
For FE unstructured zones the node map (connectivity) is included in the grid file.
The dataset title has ``"_grid"`` appended.

Solution file(s)
----------------
Each solution file (``FileType.SOLUTION``) contains the coordinate variables **plus**
the selected solution variables.  Every source zone is written; coordinate variables in
each zone are declared as shared from the corresponding grid zone so that Tecplot 360
resolves geometry from the grid file when both are loaded together.  For FE zones,
connectivity is shared from the same grid zone (``CONNECTIVITYSHAREZONE``).  Solution
time and strand ID are preserved from the source.  The dataset title has ``"_solution"``
appended (or ``"_<varname>"`` in pop mode).

When Tecplot 360 loads a grid file first (zones 1 … M) and then a solution file
(zones M+1 … M+N), VARSHARELIST and CONNECTIVITYSHAREZONE entries in the solution file
reference the **combined-dataset** zone numbering, so grid zone positions (1 … M) are
used as sharing targets.

*Static mesh* — zone 1 has independent coordinate data; all solution zones share
coordinates and FE connectivity from combined zone 1 (the grid zone).

*Multi-block / time-dependent* — zones 1 … M each have independent coordinate data;
solution zone *k* shares from the grid zone that corresponds to its coordinate source.

.. list-table:: Operating modes
   :header-rows: 1
   :widths: 25 75

   * - Mode
     - Output
   * - Default
     - One grid file and one combined solution file containing the coordinate
       variables plus all non-coordinate variables.
   * - ``--pop VAR[,VAR2,...]``
     - One grid file and one solution file per listed variable.  Each solution file
       contains the coordinate variables plus the named variable.
   * - ``--pop-all``
     - One grid file and one solution file per non-coordinate variable.

Output files are written to the same directory as the input unless ``-o`` specifies an
output directory.  Filenames are auto-derived: ``<stem>_grid<ext>`` for the grid file,
``<stem>_solution<ext>`` in default mode, and ``<stem>_<varname><ext>`` for each popped
variable.  Variable names are sanitised for use in filenames; if two variables sanitise
to the same name the command exits without writing any files.


:Usage:

.. code:: bash

    tecsplit [-h] [-o DIR] [-f] [--coords VARS] [--pop VARS | --pop-all] PATH


:Positional Arguments:
    ``PATH``
        Path to the input Tecplot file (``.plt``, ``.szplt``, or ``.dat``) to split.

:Options:
    ``-o DIR``, ``--output DIR``
        Output directory for all generated files. Defaults to the same directory as the
        input file. The directory is created if it does not already exist.

    ``-f``, ``--force``
        Overwrite output files that already exist. Without this flag the command exits
        with an error rather than silently clobbering existing files.

    ``--coords VARS``
        Comma-separated coordinate variable names or one-based indices. Defaults to
        auto-detection of variables named ``x``, ``y``, or ``z`` (case-insensitive,
        exact match). Override this when coordinates carry units or non-standard names,
        e.g.  ``--coords "X [m],Y [m],Z [m]"``.

    ``--pop VARS``
        Comma-separated variable names or one-based indices to write into individual
        solution files.  Coordinate variables are always included in every solution file
        and are silently ignored if listed here.  Mutually exclusive with ``--pop-all``.

    ``--pop-all``
        Write every non-coordinate variable into its own solution file. Mutually
        exclusive with ``--pop``. When many variables are present, note that the source
        file is read once per output file; for large files with many variables ``--pop``
        with an explicit list may be preferable.

:Returns:
    Two or more Tecplot files written to the output directory: one grid file and one or
    more solution files depending on the operating mode.  ``FEPOLYGON`` and
    ``FEPOLYHEDRON`` zones are not supported and are skipped with a warning. Exit code
    is ``0`` on success and non-zero on any error.

Examples:
    Split into grid and combined solution files::

        $ tecsplit flow.szplt

    Write output files to a specific directory::

        $ tecsplit -o /tmp/split flow.szplt

    Pop a single variable into its own solution file::

        $ tecsplit --pop Pressure flow.szplt

    Pop every non-coordinate variable::

        $ tecsplit --pop-all flow.szplt

    Override coordinate detection by name::

        $ tecsplit --coords "X [m],Y [m],Z [m]" flow.szplt

    Call directly from a Python session::

        import tecio.cli.tecsplit.main as tecsplit

        tecsplit(["--pop-all", "-o", "/tmp/split", "flow.szplt"])

See Also:
    * :mod:`tecio.cli.tecmerge`: Merge zones from multiple files into a single output.
    * :mod:`tecio.cli.tecextract`: Extract a subset of zones or variables.
    * :mod:`tecio.cli.tecslice`: Reduce structured zones along IJK axes or by time.

Note:
    When ``--pop-all`` produces N solution files, the source file is read N + 1 times
    (once per output file).  Variable data is read lazily per zone access, so peak
    memory usage is bounded by one zone at a time.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .. import open as tecio_open
from ..libtecio import FileType, ZoneType

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Zone types that cannot be copied by the Write API.
_FE_POLY: frozenset[ZoneType] = frozenset({ZoneType.FEPOLYGON, ZoneType.FEPOLYHEDRON})

#: Variable names treated as spatial coordinates when ``--coords`` is omitted.
_DEFAULT_COORD_NAMES: frozenset[str] = frozenset({"x", "y", "z"})

#: Suffixes appended to the source dataset title for each output file.
_GRID_TITLE_SUFFIX: str = "_grid"
_SOLUTION_TITLE_SUFFIX: str = "_solution"


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="tecsplit",
        description=(
            "Split a full Tecplot file into a grid file and one or more "
            "solution files.  Coordinate variables go to the grid file; "
            "non-coordinate variables go to solution file(s)."
        ),
        epilog=(
            "Operating modes\n"
            "  Default (split)   one grid file + one combined solution file\n"
            "  --pop [VAR, ...]  one grid file + one solution per listed var\n"
            "  --pop-all         one grid file + one solution per non-coord var\n\n"
            "Examples\n"
            "  Split into grid + solution\n"
            "    $ tecsplit flow.szplt\n"
            "  Write outputs to /tmp/split/\n"
            "    $ tecsplit -o /tmp/split flow.szplt\n"
            "  Pop Pressure into its own solution file\n"
            "    $ tecsplit --pop Pressure flow.szplt\n"
            "  Pop every non-coordinate variable\n"
            "    $ tecsplit --pop-all flow.szplt\n"
            "  Specify coordinates by name\n"
            '    $ tecsplit --coords "X [m],Y [m],Z [m]" flow.szplt\n'
        ),
        formatter_class=lambda prog: argparse.RawDescriptionHelpFormatter(
            prog, width=70, max_help_position=24
        ),
    )

    parser.add_argument(
        "filename",
        type=str,
        help="Input Tecplot file to split.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        metavar="DIR",
        help=(
            "Output directory for all generated files.  Defaults to the "
            "same directory as the input file.  The directory is created "
            "if it does not already exist."
        ),
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        default=False,
        help="Overwrite output files that already exist.",
    )
    parser.add_argument(
        "--coords",
        type=str,
        default=None,
        metavar="VARS",
        help=(
            "Comma-separated coordinate variable names or 1-based indices. "
            "Defaults to auto-detection of variables named 'x', 'y', or 'z' "
            "(case-insensitive, exact match)."
        ),
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--pop",
        type=str,
        default=None,
        metavar="VARS",
        help=(
            "Comma-separated variable names or 1-based indices to pop into "
            "individual solution files.  Each variable gets its own output "
            "file named '<stem>_<varname><ext>'.  Coordinate variables are "
            "always included in every solution file and are ignored if listed."
        ),
    )
    mode.add_argument(
        "--pop-all",
        action="store_true",
        default=False,
        dest="pop_all",
        help=(
            "Pop every non-coordinate variable into its own solution file. "
            "Equivalent to listing all non-coordinate variables with --pop."
        ),
    )

    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Variable-spec helpers
# ---------------------------------------------------------------------------


def _parse_var_spec(
    spec: str,
    var_names: list[str],
    label: str = "variable",
) -> list[int]:
    """Parse a comma-separated variable spec into a list of 0-based indices.

    Each token is tried first as a 1-based integer index, then as a
    case-insensitive name match against *var_names*.  Whitespace around
    tokens and separators is stripped before matching.

    Args:
        spec:      Comma-separated names or 1-based integer indices.
        var_names: Ordered dataset variable name list.
        label:     Human-readable label used in error messages.

    Returns:
        List of 0-based variable indices.  May contain duplicates if the
        spec names the same variable multiple times.

    Raises:
        argparse.ArgumentTypeError: If a token is out of range or not found.

    """
    result: list[int] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            idx = int(token)
            if idx < 1 or idx > len(var_names):
                raise argparse.ArgumentTypeError(
                    f"{label.capitalize()} index {idx} out of range "
                    f"[1, {len(var_names)}]."
                )
            result.append(idx - 1)
        except ValueError as exc:
            lower = token.lower()
            matched = [i for i, n in enumerate(var_names) if n.lower() == lower]
            if not matched:
                raise argparse.ArgumentTypeError(
                    f"{label.capitalize()} '{token}' not found.  "
                    f"Available variables: {var_names}"
                ) from exc
            result.extend(matched)
    return result


def _infer_coord_indices(var_names: list[str]) -> list[int]:
    """Return 0-based indices of variables whose names look like coordinates.

    Matches variables whose stripped, lower-cased name is exactly ``'x'``,
    ``'y'``, or ``'z'``.  Use ``--coords`` on the command line to override
    when coordinate names differ from this convention.

    Args:
        var_names: Ordered dataset variable name list.

    Returns:
        0-based indices of auto-detected coordinate variables (may be empty
        if the file uses non-standard coordinate names).

    """
    return [
        i
        for i, name in enumerate(var_names)
        if name.strip().lower() in _DEFAULT_COORD_NAMES
    ]


def _sanitize_varname(name: str) -> str:
    """Convert a variable name to a filesystem-safe string.

    Replaces characters that are not alphanumeric, hyphens, or underscores
    with underscores, then strips leading/trailing underscores.

    Args:
        name: Raw variable name (may include spaces, brackets, slashes, etc.).

    Returns:
        Sanitised string suitable for use as a filename component.  Falls
        back to ``"var"`` if the result would otherwise be empty.

    """
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    return safe.strip("_") or "var"


# ---------------------------------------------------------------------------
# Output path helper
# ---------------------------------------------------------------------------


def _make_output_path(src: Path, suffix: str, output_dir: Path | None) -> Path:
    """Build an output file path with *suffix* appended to the input stem.

    Args:
        src:        Source (input) file path; provides the stem and extension.
        suffix:     String appended to the stem (e.g. ``"_grid"``).
        output_dir: Directory for the output file, or ``None`` to use
                    the same directory as *src*.

    Returns:
        Path of the form ``<dir>/<src.stem><suffix><src.suffix>``.

    """
    directory = output_dir if output_dir is not None else src.parent
    return directory / f"{src.stem}{suffix}{src.suffix}"


# ---------------------------------------------------------------------------
# Coordinate topology analysis
# ---------------------------------------------------------------------------


def _scan_coord_topology(
    reader: Any,
    coord_set: frozenset[int],
) -> tuple[list[int], dict[int, int]]:
    """Identify grid zones and map every source zone to a grid zone number.

    A **grid zone** is a source zone whose coordinate variables all carry
    independent data — they are neither passive nor shared from another zone.
    Every other zone inherits coordinate data (and, for FE zones,
    connectivity) by referencing one of the grid zones.

    The returned *zone_to_grid_num* values use **combined-dataset numbering**.
    When Tecplot 360 loads a grid file (zones 1 … M) and then a solution file
    (zones M+1 … M+N), VARSHARELIST and CONNECTIVITYSHAREZONE entries in the
    solution file reference the combined sequence.  Because grid zones always
    occupy positions 1 … M, the 1-based position of a grid zone in the grid
    file equals its position in the combined dataset.

    Examples::

        # Static mesh — only source zone 0 has independent coord data
        grid_zone_srcs   = [0]
        zone_to_grid_num = {0: 1, 1: 1, 2: 1, ...}

        # 4-block mesh — source zones 0-3 each have independent coord data
        grid_zone_srcs   = [0, 1, 2, 3]
        zone_to_grid_num = {0: 1, 1: 2, 2: 3, 3: 4, 4: 1, 5: 2, ...}

    Args:
        reader:    Open reader instance.
        coord_set: 0-based indices of coordinate variables.

    Returns:
        ``grid_zone_srcs``
            Ordered list of 0-based source zone indices that have independent
            coordinate data; these zones appear in the grid file.
        ``zone_to_grid_num``
            Dict mapping every 0-based source zone index (excluding FEPOLYGON /
            FEPOLYHEDRON zones) to the 1-based grid zone number in the
            combined-dataset sense used as the VARSHARELIST / con_sharing target.

    """
    grid_zone_srcs: list[int] = []

    for i, zone in enumerate(reader.zone):
        if zone.zone_type in _FE_POLY or not coord_set:
            continue
        has_own_coords = all(
            not zone.variable[j].is_passive() and zone.variable[j].shared_zone is None
            for j in coord_set
        )
        if has_own_coords:
            grid_zone_srcs.append(i)

    # 0-based source index -> 1-based grid-file position (= combined position).
    src_to_grid: dict[int, int] = {
        src_i: grid_pos + 1 for grid_pos, src_i in enumerate(grid_zone_srcs)
    }

    first_coord: int | None = min(coord_set) if coord_set else None
    zone_to_grid_num: dict[int, int] = {}

    for i, zone in enumerate(reader.zone):
        if zone.zone_type in _FE_POLY:
            continue
        if i in src_to_grid:
            # This zone is itself a grid zone.
            zone_to_grid_num[i] = src_to_grid[i]
        elif first_coord is not None:
            sv = zone.variable[first_coord].shared_zone  # 1-based or None
            if sv is not None:
                # sv is 1-based in the source; convert to 0-based for lookup.
                zone_to_grid_num[i] = src_to_grid.get(sv - 1, 1)
            else:
                # Coords are passive or unavailable — fall back to grid zone 1.
                zone_to_grid_num[i] = 1
        else:
            zone_to_grid_num[i] = 1

    return grid_zone_srcs, zone_to_grid_num


# ---------------------------------------------------------------------------
# Grid file writer
# ---------------------------------------------------------------------------


def _write_grid_file(
    dst: Path,
    src: Path,
    coord_set: frozenset[int],
    force: bool,
) -> int:
    """Write a grid file containing only coordinate variables.

    Only zones with independent coordinate data (grid zones identified by
    :func:`_scan_coord_topology`) are written.  Each zone is written with
    ``solution_time=0.0`` and ``strand_id=0``.  FE zones include their full
    node map; connectivity is never shared within the grid file because the
    grid file itself is the authoritative connectivity source.

    The variable list contains **only** the coordinate variables in their
    original source order.  Variable-level auxiliary data is remapped to the
    new 1-based grid-file variable indices.  The dataset title has ``"_grid"``
    appended.

    Args:
        dst:       Destination file path.
        src:       Source file path.
        coord_set: 0-based indices of coordinate variables.
        force:     If ``False`` and *dst* exists, raises ``FileExistsError``.

    Returns:
        Number of zones written.

    Raises:
        FileExistsError: If *dst* exists and *force* is ``False``.

    """
    if dst.exists() and not force:
        raise FileExistsError(
            f"Output file already exists: {dst}\nUse --force / -f to overwrite."
        )

    with tecio_open(str(src), "r") as reader:
        var_names: list[str] = reader.variables

        # Grid file variable list: coord vars only, preserving source order.
        coord_sorted: list[int] = sorted(coord_set)
        coord_names: list[str] = [var_names[i] for i in coord_sorted]

        grid_zone_srcs, _ = _scan_coord_topology(reader, coord_set)
        grid_zone_set: frozenset[int] = frozenset(grid_zone_srcs)

        with tecio_open(
            str(dst),
            "w",
            title=reader.title + _GRID_TITLE_SUFFIX,
            variables=coord_names,
            file_type=FileType.GRID,
        ) as writer:
            # Dataset-level auxiliary data.
            if len(reader.auxdata) > 0:
                writer.add_auxdataset_dict(dict(reader.auxdata.items()))

            # Variable-level aux data remapped to grid-file 1-based indices.
            auxvar: dict[int, dict[str, str]] = {}
            for grid_pos, src_pos in enumerate(coord_sorted):
                var_aux = reader.get_var_auxdata(src_pos + 1)
                if len(var_aux) > 0:
                    auxvar[grid_pos + 1] = dict(var_aux.items())
            if auxvar:
                writer.add_auxvar_dict(auxvar)

            n_written = 0
            for i, zone in enumerate(reader.zone):
                if i not in grid_zone_set:
                    continue

                zt = zone.zone_type
                if zt in _FE_POLY:
                    print(
                        f"Warning: zone '{zone.title}' is {zt.name} — "
                        "poly zone copying is not supported, skipping.",
                        file=sys.stderr,
                    )
                    continue

                # Collect coordinate arrays in grid-file variable order.
                data: list[np.ndarray] = []
                locs: list[Any] = []
                skip = False

                for src_pos in coord_sorted:
                    var = zone.variable[src_pos]
                    arr = var.values
                    if arr is None or arr.size == 0:
                        print(
                            f"Warning: zone '{zone.title}' coordinate "
                            f"'{var_names[src_pos]}' has no data — skipping zone.",
                            file=sys.stderr,
                        )
                        skip = True
                        break
                    data.append(arr)
                    locs.append(var.value_location)

                if skip:
                    continue

                zone_aux: dict[str, str] | None = (
                    dict(zone.auxdata.items()) if len(zone.auxdata) > 0 else None
                )

                common_kw: dict[str, Any] = dict(
                    title=zone.title,
                    value_locations=locs,
                    solution_time=0.0,
                    strand_id=0,
                    aux=zone_aux,
                )

                if zt == ZoneType.ORDERED:
                    writer.write_ijk_zone(data=data, **common_kw)
                else:
                    # FE zones: always write connectivity (con_sharing=0).
                    # The grid file is the authoritative connectivity source;
                    # solution file zones share from it.
                    writer.write_fe_zone(
                        zone_type=zt,
                        data=data,
                        node_map=zone.node_map,
                        con_sharing=0,
                        **common_kw,
                    )

                n_written += 1

    return n_written


# ---------------------------------------------------------------------------
# Solution file writer
# ---------------------------------------------------------------------------


def _write_solution_file(
    dst: Path,
    src: Path,
    coord_set: frozenset[int],
    sol_var_set: frozenset[int],
    title_suffix: str,
    force: bool,
) -> int:
    """Write a solution file containing coordinate plus solution variables.

    Every source zone is written.  For each zone:

    * Coordinate variables are always declared as shared from the corresponding
      grid zone (``var_sharing = grid_zone_num``).  No coordinate data is stored
      in the solution file; Tecplot 360 resolves it from the grid file.
    * FE zones additionally share their connectivity from the same grid zone
      (``con_sharing = grid_zone_num``).  The node map is always supplied so
      that writers which require explicit connectivity for their first FE zone
      (SZL, PLT) can satisfy that requirement while still declaring the sharing;
      DAT writers ignore the node map when ``con_sharing > 0``.
    * Solution variables carry their data exactly as in the source; passive
      variables remain passive.

    The variable list is the union of *coord_set* and *sol_var_set* in original
    source order.  Variable-level auxiliary data is remapped to the new 1-based
    solution-file variable indices.  The dataset title has *title_suffix* appended.

    Args:
        dst:          Destination file path.
        src:          Source file path.
        coord_set:    0-based indices of coordinate variables.
        sol_var_set:  0-based indices of solution variables to include.
                      Must not overlap with *coord_set*.
        title_suffix: String appended to the source dataset title
                      (e.g. ``"_solution"`` or ``"_Pressure"``).
        force:        If ``False`` and *dst* exists, raises ``FileExistsError``.

    Returns:
        Number of zones written.

    Raises:
        FileExistsError: If *dst* exists and *force* is ``False``.

    """
    if dst.exists() and not force:
        raise FileExistsError(
            f"Output file already exists: {dst}\nUse --force / -f to overwrite."
        )

    with tecio_open(str(src), "r") as reader:
        var_names: list[str] = reader.variables

        # Solution file variable list: coord vars ∪ sol vars, source order.
        sol_file_src_indices: list[int] = sorted(coord_set | sol_var_set)
        sol_file_var_names: list[str] = [var_names[i] for i in sol_file_src_indices]

        _, zone_to_grid_num = _scan_coord_topology(reader, coord_set)

        with tecio_open(
            str(dst),
            "w",
            title=reader.title + title_suffix,
            variables=sol_file_var_names,
            file_type=FileType.SOLUTION,
        ) as writer:
            # Dataset-level auxiliary data.
            if len(reader.auxdata) > 0:
                writer.add_auxdataset_dict(dict(reader.auxdata.items()))

            # Variable-level aux data remapped to solution-file 1-based indices.
            auxvar: dict[int, dict[str, str]] = {}
            for sol_pos, src_pos in enumerate(sol_file_src_indices):
                var_aux = reader.get_var_auxdata(src_pos + 1)
                if len(var_aux) > 0:
                    auxvar[sol_pos + 1] = dict(var_aux.items())
            if auxvar:
                writer.add_auxvar_dict(auxvar)

            n_written = 0
            for zone_idx, zone in enumerate(reader.zone):
                zt = zone.zone_type
                if zt in _FE_POLY:
                    print(
                        f"Warning: zone '{zone.title}' is {zt.name} — "
                        "poly zone copying is not supported, skipping.",
                        file=sys.stderr,
                    )
                    continue

                # 1-based grid zone number (combined-dataset numbering) that
                # provides coordinate data and FE connectivity for this zone.
                grid_zone_num: int = zone_to_grid_num.get(zone_idx, 1)

                data: list[np.ndarray] = []
                locs: list[Any] = []
                passive: list[bool] = []
                sharing: list[int] = []

                for src_pos in sol_file_src_indices:
                    var = zone.variable[src_pos]

                    if src_pos in coord_set:
                        # Coordinate variable: always shared from the
                        # corresponding grid zone; no data array stored here.
                        passive.append(False)
                        sharing.append(grid_zone_num)
                        locs.append(var.value_location)
                        data.append(np.array([], dtype=np.float32))

                    else:
                        # Solution variable: carry data exactly as in source.
                        is_p = var.is_passive()
                        passive.append(is_p)
                        sharing.append(0)
                        locs.append(var.value_location)

                        if is_p:
                            data.append(np.array([], dtype=np.float32))
                        else:
                            arr = var.values
                            if arr is None or arr.size == 0:
                                # Guard: treat as passive if no data available.
                                passive[-1] = True
                                data.append(np.array([], dtype=np.float32))
                            else:
                                data.append(arr)

                # The Write API expects only arrays for active, non-shared vars.
                writer_data = [
                    arr
                    for arr, is_p, sv in zip(data, passive, sharing, strict=False)
                    if not is_p and sv == 0
                ]
                writer_locs = [
                    loc
                    for loc, is_p, sv in zip(locs, passive, sharing, strict=False)
                    if not is_p and sv == 0
                ]

                zone_aux: dict[str, str] | None = (
                    dict(zone.auxdata.items()) if len(zone.auxdata) > 0 else None
                )

                common_kw: dict[str, Any] = dict(
                    title=zone.title,
                    value_locations=writer_locs,
                    passive_vars=passive,
                    var_sharing=sharing,
                    solution_time=zone.solution_time,
                    strand_id=zone.strand_id,
                    aux=zone_aux,
                )

                if zt == ZoneType.ORDERED:
                    writer.write_ijk_zone(data=writer_data, **common_kw)
                else:
                    # FE zones: share connectivity from the corresponding grid
                    # zone.  node_map is always passed so that SZL and PLT
                    # writers can satisfy their requirement for explicit
                    # connectivity on their first FE zone regardless of whether
                    # con_sharing is set; DAT writers ignore node_map when
                    # con_sharing > 0.
                    writer.write_fe_zone(
                        zone_type=zt,
                        data=writer_data,
                        node_map=zone.node_map,
                        con_sharing=grid_zone_num,
                        **common_kw,
                    )

                n_written += 1

    return n_written


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """Split a full Tecplot file into grid and solution components.

    Returns:
        Exit code — ``0`` on success, ``1`` on error.

    """
    args = _parse_args(argv)

    src = Path(args.filename)
    if not src.exists():
        print(f"Error: input file not found: {src}", file=sys.stderr)
        return 1

    # Resolve / create the output directory.
    if args.output is not None:
        output_dir = Path(args.output)
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            print(f"Error: cannot create output directory: {exc}", file=sys.stderr)
            return 1
        if not output_dir.is_dir():
            print(
                f"Error: --output must be a directory path, got: {output_dir}",
                file=sys.stderr,
            )
            return 1
    else:
        output_dir = None

    # Collect all output paths for pre-flight checks; initialise empty so the
    # except block can reference it even on early failure.
    output_paths: list[Path] = []

    try:
        # ------------------------------------------------------------------
        # 1. Inspect the source file
        # ------------------------------------------------------------------
        with tecio_open(str(src), "r") as reader:
            var_names: list[str] = reader.variables
            num_vars: int = reader.num_vars

        # ------------------------------------------------------------------
        # 2. Resolve coordinate variable indices
        # ------------------------------------------------------------------
        if args.coords is not None:
            try:
                coord_0based = _parse_var_spec(
                    args.coords, var_names, label="coordinate"
                )
            except argparse.ArgumentTypeError as exc:
                print(f"Error: {exc}", file=sys.stderr)
                return 1
            coord_set: frozenset[int] = frozenset(coord_0based)
        else:
            coord_set = frozenset(_infer_coord_indices(var_names))

        if not coord_set:
            print(
                "Warning: no coordinate variables detected.  "
                "The grid file will be empty.  "
                "Use --coords to specify coordinate variables explicitly.",
                file=sys.stderr,
            )

        # ------------------------------------------------------------------
        # 3. Build solution items: (filename_label, sol_var_set) pairs.
        #
        #    sol_var_set holds ONLY the non-coordinate variables active in
        #    each solution file.  Coordinate variables are always added by
        #    _write_solution_file and must not appear in sol_var_set.
        # ------------------------------------------------------------------
        if args.pop_all:
            non_coord = [i for i in range(num_vars) if i not in coord_set]
            if not non_coord:
                print(
                    "Error: all variables are coordinates — nothing to pop.  "
                    "Adjust --coords or use the default split mode.",
                    file=sys.stderr,
                )
                return 1
            sol_items: list[tuple[str, frozenset[int]]] = [
                (_sanitize_varname(var_names[i]), frozenset({i})) for i in non_coord
            ]

        elif args.pop is not None:
            try:
                pop_0based = _parse_var_spec(args.pop, var_names)
            except argparse.ArgumentTypeError as exc:
                print(f"Error: {exc}", file=sys.stderr)
                return 1

            # Deduplicate while preserving first-seen order.
            seen_pop: set[int] = set()
            pop_unique: list[int] = []
            for idx in pop_0based:
                if idx not in seen_pop:
                    seen_pop.add(idx)
                    pop_unique.append(idx)

            # Coordinate variables are always present in every solution file;
            # listing them in --pop has no additional effect.
            coord_overlap = [i for i in pop_unique if i in coord_set]
            if coord_overlap:
                names = [var_names[i] for i in coord_overlap]
                print(
                    f"Warning: coordinate variable(s) {names} listed in --pop "
                    "are always included in every solution file — ignoring.",
                    file=sys.stderr,
                )
                pop_unique = [i for i in pop_unique if i not in coord_set]

            if not pop_unique:
                print(
                    "Error: no non-coordinate variables remain after resolving "
                    "--pop.  Adjust --coords or supply non-coordinate variables.",
                    file=sys.stderr,
                )
                return 1

            sol_items = [
                (_sanitize_varname(var_names[i]), frozenset({i})) for i in pop_unique
            ]

        else:
            # Default split: all non-coordinate variables in one solution file.
            non_coord = [i for i in range(num_vars) if i not in coord_set]
            if not non_coord:
                print(
                    "Error: all variables are coordinates — no solution "
                    "variables to write.  Adjust --coords or check the input.",
                    file=sys.stderr,
                )
                return 1
            sol_items = [("solution", frozenset(non_coord))]

        # ------------------------------------------------------------------
        # 4. Build all output paths and check for collisions
        # ------------------------------------------------------------------
        grid_path = _make_output_path(src, "_grid", output_dir)
        sol_paths: list[Path] = [
            _make_output_path(src, f"_{label}", output_dir) for label, _ in sol_items
        ]
        output_paths = [grid_path, *sol_paths]

        # Detect filename collisions between solution files.
        if len(sol_paths) != len(set(sol_paths)):
            from collections import Counter

            counts = Counter(sol_paths)
            dups = [str(p) for p, n in counts.items() if n > 1]
            print(
                "Error: two or more solution files would have the same path:\n"
                + "\n".join(f"  {d}" for d in dups)
                + "\nRename the conflicting variables or adjust --coords.",
                file=sys.stderr,
            )
            return 1

        # Pre-flight: check for existing outputs before writing anything.
        existing = [p for p in output_paths if p.exists()]
        if existing and not args.force:
            names_str = "\n".join(f"  {p}" for p in existing)
            print(
                f"Error: output file(s) already exist:\n{names_str}\n"
                "Use --force / -f to overwrite.",
                file=sys.stderr,
            )
            return 1

        # ------------------------------------------------------------------
        # 5. Print plan
        # ------------------------------------------------------------------
        coord_names = [var_names[i] for i in sorted(coord_set)]
        print(f"Input   : {src}")
        print(f"  Variables : {num_vars} total")
        print(
            f"  Coords    : "
            f"{coord_names if coord_names else '(none)'} → {grid_path.name}"
        )
        for (_, sol_var_set), sp in zip(sol_items, sol_paths, strict=False):
            # Show the complete variable list the solution file will contain.
            sol_file_indices = sorted(coord_set | sol_var_set)
            sol_file_names = [var_names[i] for i in sol_file_indices]
            print(f"  Solution  : {sol_file_names} → {sp.name}")

        # ------------------------------------------------------------------
        # 6. Write the grid file
        # ------------------------------------------------------------------
        print(f"\nWriting grid     : {grid_path}")
        n_g = _write_grid_file(
            dst=grid_path,
            src=src,
            coord_set=coord_set,
            force=args.force,
        )
        print(f"  {n_g} zone(s) written.")

        # ------------------------------------------------------------------
        # 7. Write each solution file
        # ------------------------------------------------------------------
        for (label, sol_var_set), sp in zip(sol_items, sol_paths, strict=False):
            print(f"Writing solution : {sp}")
            # "_solution" for default split; "_<varname>" for pop modes.
            title_suffix = (
                _SOLUTION_TITLE_SUFFIX if label == "solution" else f"_{label}"
            )
            n_s = _write_solution_file(
                dst=sp,
                src=src,
                coord_set=coord_set,
                sol_var_set=sol_var_set,
                title_suffix=title_suffix,
                force=args.force,
            )
            print(f"  {n_s} zone(s) written.")

    except FileExistsError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        # Remove any partially written files so the directory is left clean.
        for p in output_paths:
            p.unlink(missing_ok=True)
        return 1

    n_out = len(output_paths)
    print(f"\nDone. {n_out} file(s) written.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
