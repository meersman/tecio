"""Split a Tecplot data file into separate grid and solution files.

Solvers that produce full Tecplot files containing both spatial coordinates and solution
variables in a single file can create workflows where the grid geometry is redundantly
stored across every time step. Separating the coordinate and solution data into distinct
files with the appropriate ``FileType`` metadata allows Tecplot to load a shared grid
once and associate multiple solution files with it, reducing storage overhead and
enabling more efficient transient data management. ``tecsplit`` performs this separation
automatically, detecting coordinate variables by name or accepting an explicit list, and
supports three operating modes depending on how finely the solution variables should be
distributed across output files.

.. list-table:: Operating modes
   :header-rows: 1
   :widths: 25 75

   * - Mode
     - Output
   * - Default
     - One grid file and one combined solution file containing all non-coordinate
       variables.
   * - ``--pop VAR[,VAR2,...]``
     - One grid file and one solution file per listed variable. Each named variable is
       the sole active variable in its output file; all others are passive.
   * - ``--pop-all``
     - One grid file and one solution file per non-coordinate variable.  Equivalent to
       listing every non-coordinate variable with ``--pop``.

Output files are written to the same directory as the input unless ``-o`` specifies an
output directory.  Filenames are always auto-derived: ``<stem>_grid<ext>`` for the grid
file, ``<stem>_solution<ext>`` in default mode, and ``<stem>_<varname><ext>`` for each
popped variable.  Variable names are sanitised for use in filenames; if two variables
sanitise to the same name the command exits without writing any files.

:Positional Arguments:
    ``filename``
        Path to the input Tecplot file (``.plt``, ``.szplt``, or ``.dat``) to split.

:Options:
    ``-o, --output DIR``
        Output directory for all generated files. Defaults to the same directory as the
        input file. The directory is created if it does not already exist.

    ``-f, --force``
        Overwrite output files that already exist. Without this flag the command exits
        with an error rather than silently clobbering existing files.

    ``--coords VARS``
        Comma-separated coordinate variable names or one-based indices. Defaults to
        auto-detection of variables named ``x``, ``y``, or ``z`` (case-insensitive,
        exact match). Override this when coordinates carry units or non-standard names,
        e.g.  ``--coords "X [m],Y [m],Z [m]"``.

    ``--pop VARS``
        Comma-separated variable names or one-based indices to write into individual
        solution files. Mutually exclusive with ``--pop-all``.

    ``--pop-all``
        Write every non-coordinate variable into its own solution file. Mutually
        exclusive with ``--pop``. When many variables are present, note that the source
        file is read once per output file; for large files with many variables ``--pop``
        with an explicit list may be preferable.

:Returns:
    Two or more Tecplot files written to the output directory: one grid file and one or
    more solution files depending on the operating mode.  All zone types are copied in
    full, including auxiliary data, solution time, and strand IDs. ``FEPOLYGON`` and
    ``FEPOLYHEDRON`` zones are not supported and are skipped with a warning. Exit code
    is ``0`` on success and non-zero if the input file cannot be read, a variable cannot
    be resolved, a filename collision is detected, or an output file already exists and
    ``--force`` is not set.

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

        import tecio.cli.tecsplit.main as tecplit
        tecsplit(["--pop-all", "-o", "/tmp/split", "flow.szplt"])

See Also:
    * :mod:`tecio.cli.tecmerge`: Merge zones from multiple files into a single output —
      the inverse of splitting.
    * :mod:`tecio.cli.tecextract`: Extract a subset of zones or variables into a single
      output file.
    * :mod:`tecio.cli.tecslice`: Reduce structured zones along IJK axes or filter by
      solution time.

Note:
    When ``--pop-all`` produces N solution files, the source file is read N + 1 times
    (once per output file).  Variable data is read lazily per zone access, so peak
    memory usage is bounded by one zone at a time. For very large files with many
    variables, consider splitting into a modest batch with ``--pop`` instead.
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
        "-o", "--output",
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
        "-f", "--force",
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
            "file named '<stem>_<varname><ext>'."
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
        except ValueError:
            lower = token.lower()
            matched = [i for i, n in enumerate(var_names) if n.lower() == lower]
            if not matched:
                raise argparse.ArgumentTypeError(
                    f"{label.capitalize()} '{token}' not found.  "
                    f"Available variables: {var_names}"
                )
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
# Zone-copy engine
# ---------------------------------------------------------------------------


def _write_zones_subset(
    reader: Any,
    writer: Any,
    active_0based: frozenset[int],
    num_vars: int,
) -> int:
    """Copy all supported zones from *reader* to *writer* with variable filtering.

    Variables whose 0-based index appears in *active_0based* are written with
    their original passive / sharing status from the source file.  All other
    variables are forced passive (their data is discarded).

    Zone-level auxiliary data, solution time, strand ID, and FE connectivity
    are always copied verbatim.  Variable sharing cross-references use the
    same zone numbering as the source because every zone is written in source
    order (none are skipped).

    FEPOLYGON and FEPOLYHEDRON zones are not supported by the Write API and
    are skipped with a message to stderr.

    Args:
        reader:       Open reader instance (szl.Read, plt.Read, or dat.Read).
        writer:       Open writer instance whose variable list matches the
                      source dataset.
        active_0based: 0-based indices of variables to keep active.  All
                       other variables are written as passive.
        num_vars:     Total number of variables in the dataset.

    Returns:
        Number of zones successfully written.

    """
    n_written = 0

    for zone in reader.zone:
        zt = zone.zone_type

        if zt in _FE_POLY:
            print(
                f"Warning: zone '{zone.title}' is {zt.name} — "
                "poly zone copying is not supported, skipping.",
                file=sys.stderr,
            )
            continue

        data: list[np.ndarray] = []
        locs: list[Any] = []
        passive: list[bool] = []
        sharing: list[int] = []

        for j in range(num_vars):
            var = zone.variable[j]

            if j in active_0based:
                # Preserve original passive / sharing status.
                is_p = var.is_passive()
                sv = var.shared_zone  # None or positive source zone index
                share_int = sv if sv is not None else 0
                passive.append(is_p)
                sharing.append(share_int)
                locs.append(var.value_location)

                if is_p or share_int != 0:
                    # Passive or shared: no data array needed.
                    data.append(np.array([], dtype=np.float32))
                else:
                    arr = var.values
                    if arr is None or arr.size == 0:
                        # Guard against readers that return empty arrays for
                        # variables that are structurally present but have no
                        # data (e.g. some DAT edge cases).
                        passive[-1] = True
                        data.append(np.array([], dtype=np.float32))
                    else:
                        data.append(arr)
            else:
                # Force this variable passive — discard its data.
                passive.append(True)
                sharing.append(0)
                locs.append(var.value_location)
                data.append(np.array([], dtype=np.float32))

        # The Write API expects only arrays for active, non-shared variables.
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
            writer.write_fe_zone(
                zone_type=zt,
                data=writer_data,
                node_map=zone.node_map,
                **common_kw,
            )

        n_written += 1

    return n_written


# ---------------------------------------------------------------------------
# Single-file writers (open → copy → close)
# ---------------------------------------------------------------------------


def _write_one_file(
    dst: Path,
    src: Path,
    active_0based: frozenset[int],
    file_type: FileType,
    force: bool,
) -> int:
    """Open *src*, copy zones with *active_0based* active, write to *dst*.

    Dataset-level and variable-level auxiliary data are forwarded for all
    variables (not just those in *active_0based*), so that the output file
    retains the full metadata context of the source.

    Args:
        dst:           Destination file path.
        src:           Source file path.
        active_0based: 0-based indices of variables to write actively.
        file_type:     ``FileType.GRID`` or ``FileType.SOLUTION``.
        force:         If ``False`` and *dst* exists, raises ``FileExistsError``.

    Returns:
        Number of zones written.

    Raises:
        FileExistsError: If *dst* already exists and *force* is ``False``.

    """
    if dst.exists() and not force:
        raise FileExistsError(
            f"Output file already exists: {dst}\n"
            "Use --force / -f to overwrite."
        )

    with tecio_open(str(src), "r") as reader:
        num_vars: int = reader.num_vars
        var_names: list[str] = reader.variables

        with tecio_open(
            str(dst),
            "w",
            title=reader.title,
            variables=var_names,
            file_type=file_type,
        ) as writer:
            # Forward dataset-level auxiliary data.
            if len(reader.auxdata) > 0:
                writer.add_auxdataset_dict(dict(reader.auxdata.items()))

            # Forward variable-level auxiliary data for every variable so
            # that metadata (units, descriptions, etc.) is not lost even if
            # the variable is passive in this output file.
            auxvar: dict[int, dict[str, str]] = {}
            for i in range(num_vars):
                var_aux = reader.get_var_auxdata(i + 1)
                if len(var_aux) > 0:
                    auxvar[i + 1] = dict(var_aux.items())
            if auxvar:
                writer.add_auxvar_dict(auxvar)

            return _write_zones_subset(reader, writer, active_0based, num_vars)


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
        # ------------------------------------------------------------------ #
        # 1. Inspect the source file.                                         #
        # ------------------------------------------------------------------ #
        with tecio_open(str(src), "r") as reader:
            var_names: list[str] = reader.variables
            num_vars: int = reader.num_vars

        # ------------------------------------------------------------------ #
        # 2. Resolve coordinate variable indices.                             #
        # ------------------------------------------------------------------ #
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
                "The grid file will be written with all variables passive.  "
                "Use --coords to specify coordinate variables explicitly.",
                file=sys.stderr,
            )

        # ------------------------------------------------------------------ #
        # 3. Build the list of (label, solution_frozenset) pairs.             #
        #    label  → used in the output filename as <stem>_<label><ext>      #
        #    sol_set → 0-based variable indices active in that solution file   #
        # ------------------------------------------------------------------ #
        if args.pop_all:
            # One solution file per non-coordinate variable.
            non_coord = [i for i in range(num_vars) if i not in coord_set]
            if not non_coord:
                print(
                    "Error: all variables are coordinates — nothing to pop.  "
                    "Adjust --coords or use the default split mode.",
                    file=sys.stderr,
                )
                return 1
            sol_items: list[tuple[str, frozenset[int]]] = [
                (_sanitize_varname(var_names[i]), frozenset({i}))
                for i in non_coord
            ]

        elif args.pop is not None:
            # One solution file per explicitly listed variable.
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

            # Warn if coordinate variables were explicitly popped — they will
            # appear active in both the grid file and their own solution file.
            coord_overlap = [i for i in pop_unique if i in coord_set]
            if coord_overlap:
                names = [var_names[i] for i in coord_overlap]
                print(
                    f"Warning: coordinate variable(s) {names} included in "
                    "--pop.  They will appear active in both the grid file "
                    "and their own solution file.",
                    file=sys.stderr,
                )

            sol_items = [
                (_sanitize_varname(var_names[i]), frozenset({i}))
                for i in pop_unique
            ]

        else:
            # Default split mode: all non-coordinate variables in one file.
            non_coord = [i for i in range(num_vars) if i not in coord_set]
            if not non_coord:
                print(
                    "Error: all variables are coordinates — no solution "
                    "variables to write.  Adjust --coords or check the input.",
                    file=sys.stderr,
                )
                return 1
            sol_items = [("solution", frozenset(non_coord))]

        # ------------------------------------------------------------------ #
        # 4. Build all output paths and check for collisions.                 #
        # ------------------------------------------------------------------ #
        grid_path = _make_output_path(src, "_grid", output_dir)
        sol_paths: list[Path] = [
            _make_output_path(src, f"_{label}", output_dir)
            for label, _ in sol_items
        ]
        output_paths = [grid_path, *sol_paths]

        # Detect filename collisions between solution files (can happen when
        # two variable names sanitise to the same string).
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

        # ------------------------------------------------------------------ #
        # 5. Print plan.                                                      #
        # ------------------------------------------------------------------ #
        coord_names = [var_names[i] for i in sorted(coord_set)]
        print(f"Input   : {src}")
        print(f"  Variables : {num_vars} total")
        print(
            f"  Coords    : "
            f"{coord_names if coord_names else '(none)'} → {grid_path.name}"
        )
        for (label, sol_set), sp in zip(sol_items, sol_paths, strict=False):
            active_names = [var_names[i] for i in sorted(sol_set)]
            print(f"  Solution  : {active_names} → {sp.name}")

        # ------------------------------------------------------------------ #
        # 6. Write the grid file.                                             #
        # ------------------------------------------------------------------ #
        print(f"\nWriting grid     : {grid_path}")
        n_g = _write_one_file(
            dst=grid_path,
            src=src,
            active_0based=coord_set,
            file_type=FileType.GRID,
            force=args.force,
        )
        print(f"  {n_g} zone(s) written.")

        # ------------------------------------------------------------------ #
        # 7. Write each solution file.                                        #
        # ------------------------------------------------------------------ #
        for (label, sol_set), sp in zip(sol_items, sol_paths, strict=False):
            print(f"Writing solution : {sp}")
            n_s = _write_one_file(
                dst=sp,
                src=src,
                active_0based=sol_set,
                file_type=FileType.SOLUTION,
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
