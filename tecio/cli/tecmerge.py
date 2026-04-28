#!/usr/bin/env python3
"""Command line interface to merge zones from multiple Tecplot files into one.

Input files may be any mix of supported formats (.szplt, .plt, .dat).  The
output format is controlled by the ``-o`` extension.

Variable reconciliation
-----------------------
All input files must share at least one common variable name.  The output
variable list is the union of all variable names found across all files,
preserving the order in which names are first encountered.  For any zone
where a variable from the union list was not present in the source file, that
variable is written as **passive**.

Time / strand assignment (``--assign-time-strands``)
-----------------------------------------------------
When ``--assign-time-strands`` is given, all zones written to the output are
assigned a strand ID and evenly-spaced solution times.  Each *input file*
maps to one time step; zones within a file all receive that file's time value.

Required sub-options:

    ``-start VALUE``    Solution time of the first file.
    ``-strand ID``      Strand ID to assign to all zones (default: 1).

Exactly one of the following to define the step interval:

    ``-delta VALUE``    Constant time step between files.
    ``-end VALUE``      End time; step is computed as (end - start) / (N - 1).

Example usage::

    # Merge three explicit files into a single SZL file
    $ tecmerge -o combined.szplt part1.szplt part2.szplt part3.szplt

    # Merge using a glob pattern
    $ tecmerge -o combined.szplt "results_*.szplt"

    # Mix formats
    $ tecmerge -o combined.szplt run1.plt run2.dat run3.szplt

    # Merge a time series and assign strand/time metadata
    $ tecmerge --assign-time-strands -start 0.0 -delta 0.1 -strand 1 \\
               -o transient.szplt "step_*.szplt"

    # Using -end instead of -delta
    $ tecmerge --assign-time-strands -start 0.0 -end 1.0 -strand 1 \\
               -o transient.szplt "step_*.szplt"
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
            "    $ tecmerge -o combined.szplt \"results_*.szplt\"\n"
            "  Assign time/strand metadata\n"
            "    $ tecmerge --assign-time-strands -start 0.0 -delta 0.1 \\\n"
            "               -o transient.szplt \"step_*.szplt\"\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
    ts = parser.add_argument_group(
        "time/strand assignment (--assign-time-strands)"
    )
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
                print(
                    f"Error: no files matched pattern: {pattern!r}",
                    file=sys.stderr,
                )
                sys.exit(1)
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
        row: list[int | None] = [
            local_map.get(uname) for uname in union_vars
        ]
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
        for arr, is_p, sv in zip(active_data, passive_vars, var_sharing)
        if not is_p and sv == 0
    ]
    # Replace None locations (from passive-by-absence) with a sentinel;
    # they will never reach the writer but the filter must align.
    writer_locs = [
        loc if loc is not None else active_locs[0]
        for loc, is_p, sv in zip(active_locs, passive_vars, var_sharing)
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
            print(
                "Error: --assign-time-strands requires -start.", file=sys.stderr
            )
            return 1
        if args.delta is None and args.end is None:
            print(
                "Error: --assign-time-strands requires either -delta or -end.",
                file=sys.stderr,
            )
            return 1
    # Expand input globs
    input_paths = _expand_inputs(args.files)
    if not input_paths:
        print("Error: no input files found.", file=sys.stderr)
        return 1

    dst = Path(args.output)
    if dst.exists() and not args.force:
        print(
            f"Error: output file already exists: {dst}\n"
            "Use --force to overwrite.",
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
        readers: list[Any] = [
            tecio_open(str(p), "r") for p in input_paths
        ]

        # Build union variable list
        union_vars, index_maps = _build_var_union(readers)
        n_union = len(union_vars)

        print(f"\nUnion variable list ({n_union}): {union_vars}")

        # Report any variables that will be passive in some files.
        for fi, (reader, imap) in enumerate(zip(readers, index_maps)):
            missing = [
                union_vars[ui]
                for ui, li in enumerate(imap)
                if li is None
            ]
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
            for fi, (reader, imap) in enumerate(zip(readers, index_maps)):
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
