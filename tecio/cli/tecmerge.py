r"""Merge zones from multiple Tecplot data files into a single output file.

Post-processing workflows commonly produce results distributed across multiple files.
``tecmerge`` collects all zones from an arbitrary number of input files into a single
output, reconciling variable lists across sources by taking their union and writing any
variable absent from a given source as passive.  Input files may be specified explicitly
or via a quoted glob pattern and may be any mix of supported formats.  When merging
time-step sequences, solution times and strand IDs can be assigned automatically from a
start time and either a fixed interval or an end time.

:Usage:

.. code:: bash

    tecmerge [-h] --output PATH [--force] [--title STRING] [--assign-time-strands]
             [-start VALUE] [-delta VALUE | -end VALUE] [-strand ID] FILE [FILE ...]

:Positional Arguments:
    ``FILE [FILE ...]``
        One or more input Tecplot files (``.plt``, ``.szplt``, or ``.dat``). Glob
        patterns are expanded by the tool — quote the pattern to prevent premature shell
        expansion (e.g.  ``"step_*.szplt"``). Files are merged in the order given or
        matched.

:Options:
    ``-o PATH``, ``--output PATH``
        Output file path. Required. The extension controls the output format:
        ``.szplt``, ``.plt``, or ``.dat``.

    ``-f``, ``--force``
        Overwrite the output file if it already exists. Without this flag the command
        exits with an error rather than silently clobbering an existing file.

    ``--title STRING``
        Dataset title to write to the output file. Defaults to the title of the first
        input file.

    ``--assign-time-strands``
        Assign evenly-spaced solution times and a strand ID to all zones, treating each
        input file as one time step. Requires ``-start`` and either ``-delta`` or
        ``-end``.

    ``-start VALUE``
        Solution time of the first input file. Used with ``--assign-time-strands``.

    ``-delta VALUE``
        Constant time increment between successive input files.  Mutually exclusive with
        ``-end``.

    ``-end VALUE``
        Solution time of the last input file. The time step is computed as ``(end -
        start) / (N - 1)`` where ``N`` is the number of input files. Mutually exclusive
        with ``-delta``.

    ``-strand INT``
        Strand ID to assign to all zones when using ``--assign-time-strands``. Defaults
        to ``1``.

:Returns:
    A new Tecplot file written to the output path containing all zones from every input
    file. Variables absent from a source file are written as passive. Exit code is ``0``
    on success and non-zero if any input file cannot be read, the output file already
    exists and ``--force`` is not set, or conflicting time-strand options are supplied.

Examples:
    Merge two files explicitly::

        $ tecmerge part1.szplt part2.szplt -o combined.szplt

    Merge a sequence matched by a glob pattern::

        $ tecmerg "results_*.szplt" -o combined.szplt

    Merge a time series and assign solution time metadata::

        $ tecmerge --assign-time-strands -start 0.0 -delta 0.1 \\
                   "step_*.szplt" -o transient.szplt

    Call directly from a Python session::

        import tecio.cli.tecmerge.main as tecmerge

        tecmerge(["part1.szplt", "part2.szplt", "--output", "combined.szplt"])

See Also:
    * :mod:`tecio.cli.tecextract`: Extract a zone/variable subset from a single file —
      the inverse of merging.
    * :mod:`tecio.cli.tecsplit`: Split a file into separate grid and solution files.
    * :mod:`tecio.cli.tecslice`: Extract planar slices from volumetric zone data.

"""
from __future__ import annotations

import argparse
import glob
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .. import open as tecio_open
from ..libtecio import ZoneType


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="tecmerge",
        description=(
            "Merge zones from multiple Tecplot files into a single output file.  "
            "Variables not present in a source file are written as passive."
        ),
        epilog=(
            "Example usage:\n"
            "  Merge explicit files\n"
            "    $ tecmerge -o combined.szplt part1.szplt part2.szplt\n"
            "  Merge via glob\n"
            '    $ tecmerge -o combined.szplt "results_*.szplt"\n'
            "  Assign time/strand metadata\n"
            "    $ tecmerge --assign-time-strands -start 0.0 -delta 0.1 \\\n"
            '               -o transient.szplt "step_*.szplt"\n'
        ),
        formatter_class=lambda prog: argparse.RawDescriptionHelpFormatter(
            prog, width=70, max_help_position=24
        ),
    )

    # Input files
    parser.add_argument(
        "files",
        type=str,
        nargs="+",
        metavar="FILE",
        help=(
            "Input Tecplot files.  Glob patterns are expanded (quote the "
            "pattern to prevent shell expansion).  Files are merged in the "
            "order given / matched."
        ),
    )

    # Output
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        metavar="PATH",
        help=(
            "Output file path.  The extension controls the output format "
            "(.szplt, .plt, .dat)."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Overwrite the output file if it already exists.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        metavar="STRING",
        help=(
            "Dataset title for the output file.  Defaults to the title of "
            "the first input file."
        ),
    )

    # Time strand assignment
    ts = parser.add_argument_group("time/strand assignment (--assign-time-strands)")
    ts.add_argument(
        "--assign-time-strands",
        action="store_true",
        default=False,
        dest="assign_ts",
        help=(
            "Assign evenly-spaced solution times and a strand ID to all "
            "zones.  Each input file corresponds to one time step.  "
            "Requires -start and one of -delta or -end."
        ),
    )
    ts.add_argument(
        "-start",
        type=float,
        default=None,
        metavar="VALUE",
        help="Solution time of the first input file.",
    )
    # -delta and -end are mutually exclusive -- either specifies the spacing,
    # the other specifies the endpoint.  Providing both over-constrains the
    # problem and is therefore disallowed at the parser level.
    step_group = ts.add_mutually_exclusive_group()
    step_group.add_argument(
        "-delta",
        type=float,
        default=None,
        metavar="VALUE",
        help="Constant time step between successive input files.",
    )
    step_group.add_argument(
        "-end",
        type=float,
        default=None,
        metavar="VALUE",
        help=(
            "Solution time of the last input file.  The step is computed as "
            "(end - start) / (N - 1) where N is the number of input files."
        ),
    )
    ts.add_argument(
        "-strand",
        type=int,
        default=1,
        metavar="ID",
        help="Strand ID to assign to all zones.  Default: 1.",
    )

    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _expand_inputs(patterns: list[str]) -> list[Path]:
    """Expand a list of file paths / glob patterns to a sorted list of Paths.

    Args:
        patterns: Strings that may be literal file paths or glob patterns.

    Returns:
        Deduplicated, ordered list of :class:`Path` objects.

    Raises:
        SystemExit: If a pattern matches no files.

    """
    seen: set[Path] = set()
    result: list[Path] = []

    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if not matches:
            # Try treating as a literal path even if glob found nothing.
            p = Path(pattern)
            if p.exists():
                matches = [str(p)]
            else:
                raise FileNotFoundError(f"no files matched pattern: {pattern!r}")
        for m in matches:
            p = Path(m).resolve()
            if p not in seen:
                seen.add(p)
                result.append(p)

    return result


def _build_var_union(
    readers: list[Any],
) -> tuple[list[str], list[list[int | None]]]:
    """Compute the union variable list and per-reader index maps.

    Args:
        readers: List of open ``Read`` instances.

    Returns:
        A 2-tuple of:
        - ``union_vars``: Ordered list of all unique variable names
          (preserving first-seen order).
        - ``index_maps``: For each reader, a list of length
          ``len(union_vars)`` where each entry is either the 0-based
          local variable index for that reader, or ``None`` if the
          reader does not have that variable.

    """
    union_vars: list[str] = []
    union_set: dict[str, int] = {}  # name -> position in union_vars

    for reader in readers:
        for name in reader.variables:
            if name not in union_set:
                union_set[name] = len(union_vars)
                union_vars.append(name)

    index_maps: list[list[int | None]] = []
    for reader in readers:
        local_names = reader.variables
        local_map: dict[str, int] = {n: i for i, n in enumerate(local_names)}
        row: list[int | None] = [local_map.get(uname) for uname in union_vars]
        index_maps.append(row)

    return union_vars, index_maps


def _write_zone(
    writer: Any,
    zone: Any,
    union_vars: list[str],
    local_index_map: list[int | None],
    solution_time: float | None,
    strand_id: int | None,
) -> None:
    """Write one zone to *writer* using the reconciled variable list.

    Variables present in *zone* are copied verbatim.  Variables absent from
    *zone* (``None`` entries in *local_index_map*) are written as passive.

    Args:
        writer:           Open ``Write`` instance.
        zone:             Source ``ReadZone``.
        union_vars:       Full union variable name list.
        local_index_map:  Map from union index -> local 0-based var index
                          (``None`` = not present in this file).
        solution_time:    Override solution time, or ``None`` to keep original.
        strand_id:        Override strand ID, or ``None`` to keep original.

    """
    zt = zone.zone_type

    active_data: list[np.ndarray] = []
    active_locs: list[Any] = []
    passive_vars: list[bool] = []
    var_sharing: list[int] = []

    for local_idx in local_index_map:
        if local_idx is None:
            # Variable not in this file -- mark passive.
            passive_vars.append(True)
            var_sharing.append(0)
            active_locs.append(None)
            active_data.append(np.array([], dtype=np.float32))
            continue

        var = zone.variable[local_idx]
        passive_vars.append(var.is_passive())
        sv = var.shared_zone
        # Drop cross-file sharing -- zones from different files cannot share.
        var_sharing.append(0)
        active_locs.append(var.value_location)

        if var.is_passive() or sv is not None:
            active_data.append(np.array([], dtype=np.float32))
            continue

        arr = var.values
        if arr is None or arr.size == 0:
            passive_vars[-1] = True
            active_data.append(np.array([], dtype=np.float32))
        else:
            active_data.append(arr)

    # Filter to active, non-shared variables for the writer.
    writer_data = [
        arr
        for arr, is_p, sv in zip(active_data, passive_vars, var_sharing, strict=False)
        if not is_p and sv == 0
    ]
    # Replace None locations (from passive-by-absence) with a sentinel;
    # they will never reach the writer but the filter must align.
    writer_locs = [
        loc if loc is not None else active_locs[0]
        for loc, is_p, sv in zip(active_locs, passive_vars, var_sharing, strict=False)
        if not is_p and sv == 0
    ]

    zone_aux: dict[str, str] | None = None
    if len(zone.auxdata) > 0:
        zone_aux = dict(zone.auxdata.items())

    s_time = solution_time if solution_time is not None else zone.solution_time
    s_id = strand_id if strand_id is not None else zone.strand_id

    common_kw: dict[str, Any] = dict(
        title=zone.title,
        value_locations=writer_locs,
        passive_vars=passive_vars,
        var_sharing=var_sharing,
        solution_time=s_time,
        strand_id=s_id,
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """Merge zones from multiple Tecplot files into one.

    Returns:
        Exit code -- ``0`` on success, ``1`` on error.

    """
    args = _parse_args(argv)

    # Validate time/strand options
    if args.assign_ts:
        if args.start is None:
            print("Error: --assign-time-strands requires -start.", file=sys.stderr)
            return 1
        if args.delta is None and args.end is None:
            print(
                "Error: --assign-time-strands requires either -delta or -end.",
                file=sys.stderr,
            )
            return 1
    # Expand input globs
    try:
        input_paths = _expand_inputs(args.files)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    if not input_paths:
        print("Error: no input files found.", file=sys.stderr)
        return 1

    dst = Path(args.output)
    if dst.exists() and not args.force:
        print(
            f"Error: output file already exists: {dst}\nUse --force to overwrite.",
            file=sys.stderr,
        )
        return 1

    n_files = len(input_paths)
    print(f"Merging {n_files} file(s)  ->  {dst}")
    for p in input_paths:
        print(f"  {p}")

    # Build time schedule
    times: list[float] | None = None
    if args.assign_ts:
        if args.delta is not None:
            times = [args.start + i * args.delta for i in range(n_files)]
        else:
            if n_files == 1:
                times = [args.start]
            else:
                step = (args.end - args.start) / (n_files - 1)
                times = [args.start + i * step for i in range(n_files)]
        print(
            f"\nTime assignment: strand={args.strand}, "
            f"times={[f'{t:.6g}' for t in times]}"
        )

    try:
        # Open all readers
        readers: list[Any] = [tecio_open(str(p), "r") for p in input_paths]

        # Build union variable list
        union_vars, index_maps = _build_var_union(readers)
        n_union = len(union_vars)

        print(f"\nUnion variable list ({n_union}): {union_vars}")

        # Report any variables that will be passive in some files.
        for fi, (reader, imap) in enumerate(zip(readers, index_maps, strict=False)):
            missing = [union_vars[ui] for ui, li in enumerate(imap) if li is None]
            if missing:
                print(
                    f"  {input_paths[fi].name}: variables set passive "
                    f"(not in file): {missing}"
                )

        # Resolve output title
        out_title = args.title if args.title is not None else readers[0].title

        # Open writer and stream all zones
        with tecio_open(
            str(dst),
            "w",
            title=out_title,
            variables=union_vars,
            file_type=readers[0].file_type,
        ) as writer:
            # Dataset aux data from first file only.
            if len(readers[0].auxdata) > 0:
                writer.add_auxdataset_dict(dict(readers[0].auxdata.items()))

            # Variable aux data: first file wins for variables it has.
            auxvar: dict[int, dict[str, str]] = {}
            for ui, local_idx in enumerate(index_maps[0]):
                if local_idx is None:
                    continue
                var_aux = readers[0].get_var_auxdata(local_idx + 1)
                if len(var_aux) > 0:
                    auxvar[ui + 1] = dict(var_aux.items())
            if auxvar:
                writer.add_auxvar_dict(auxvar)

            total_zones = 0
            for fi, (reader, imap) in enumerate(zip(readers, index_maps, strict=False)):
                sol_time = times[fi] if times is not None else None
                s_id = args.strand if times is not None else None

                for zone in reader.zone:
                    zt = zone.zone_type
                    if zt in (ZoneType.FEPOLYGON, ZoneType.FEPOLYHEDRON):
                        print(
                            f"Warning: zone '{zone.title}' in "
                            f"{input_paths[fi].name} is {zt.name} "
                            "-- skipping.",
                            file=sys.stderr,
                        )
                        continue

                    _write_zone(
                        writer=writer,
                        zone=zone,
                        union_vars=union_vars,
                        local_index_map=imap,
                        solution_time=sol_time,
                        strand_id=s_id,
                    )
                    total_zones += 1

        # Close all readers
        for reader in readers:
            reader.__exit__(None, None, None)

    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        dst.unlink(missing_ok=True)
        return 1

    print(f"\nDone. {total_zones} zone(s) written to: {dst}")
    return 0


if __name__ == "__main__":
    main()
