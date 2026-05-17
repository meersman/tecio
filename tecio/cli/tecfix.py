"""Command line interface to fix Tecplot files that contain ``NaN`` or ``Inf`` values.

For each zone, any variable whose array contains ``NaN`` or ``Inf`` is marked
**passive** in the output file.  A zone-level auxiliary data record is also
written that documents exactly which variables were affected and why, using
the naming convention::

    Variable<N>  →  "NaN" | "Inf" | "NaN and Inf"

where ``<N>`` is the 1-based variable index.

The fixed file is written alongside the original with the suffix ``_fixed``
appended to the stem (e.g. ``flow.szplt`` → ``flow_fixed.szplt``), unless an
explicit output path is given with ``--output``.

Example usage::

    # Fix a blown-up SZL file
    $ tecfix flow.szplt

    # Write to an explicit path
    $ tecfix --output clean.szplt flow.szplt

    # Overwrite an existing output file
    $ tecfix --force flow.szplt

Note:
    - Only floating-point variables (``FLOAT`` and ``DOUBLE``) are inspected;
      integer variables cannot represent NaN / Inf and are always copied as-is.
    - Variables that are already passive or shared in the source file are
      forwarded unchanged (they remain passive / shared).
    - If a zone has no bad variables its aux data is not modified beyond what
      was already present in the source file.
    - The output format matches the input format (extension is preserved).
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from .. import open as tecio_open
from ..libtecio import DataType, ZoneType

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: DataTypes that can hold NaN / Inf.
_FLOAT_TYPES: frozenset[DataType] = frozenset({DataType.FLOAT, DataType.DOUBLE})

#: FE zone types that the Write API cannot copy.
_FE_POLY: frozenset[ZoneType] = frozenset({ZoneType.FEPOLYGON, ZoneType.FEPOLYHEDRON})


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="tecfix",
        description=(
            "Rewrite a Tecplot file with NaN / Inf variable arrays set to passive. "
            "Affected variables are documented in zone-level auxiliary data."
        ),
        epilog=(
            "Examples\n"
            "  tecfix flow.szplt                  # writes flow_fixed.szplt\n"
            "  tecfix --output clean.szplt flow.szplt\n"
            "  tecfix --force flow.szplt          # overwrite existing _fixed file\n"
            "  tecfix --dry-run flow.szplt        # report bad variables, no output\n"
        ),
        formatter_class=lambda prog: argparse.RawDescriptionHelpFormatter(
            prog, width=70, max_help_position=24
        ),
    )
    parser.add_argument(
        "filename",
        type=str,
        help="Input Tecplot file to inspect and fix.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Explicit output file path.  Defaults to <stem>_fixed<ext> "
            "in the same directory as the input file."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Overwrite the output file if it already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        dest="dry_run",
        help=(
            "Scan for bad values and print a report without writing any output file."
        ),
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Inspection helpers
# ---------------------------------------------------------------------------


def _classify_array(arr: npt.NDArray) -> str | None:
    """Return a short description of any NaN / Inf in *arr*, or ``None``.

    Args:
        arr: A flat or shaped NumPy array of a floating-point dtype.

    Returns:
        ``"NaN"``, ``"Inf"``, ``"NaN and Inf"``, or ``None`` if the array
        is clean or empty.

    """
    # Some readers (e.g. DAT) may return an empty or None array for passive
    # or otherwise unavailable variables.  Treat these as clean -- there is
    # nothing to inspect and numpy operations like isnan would error or
    # return misleading results on a zero-size array.
    if arr is None or arr.size == 0:
        return None

    has_nan = bool(np.any(np.isnan(arr)))
    has_inf = bool(np.any(np.isinf(arr)))

    if has_nan and has_inf:
        return "NaN and Inf"
    if has_nan:
        return "NaN"
    if has_inf:
        return "Inf"
    return None


# ---------------------------------------------------------------------------
# Per-zone processing
# ---------------------------------------------------------------------------


def _process_zone(
    zone: Any,
    num_vars: int,
    var_names: list[str],
) -> tuple[
    list[np.ndarray],
    list[Any],
    list[bool],
    list[int],
    dict[str, str],
    dict[str, str],
]:
    """Inspect one zone and build the data/metadata needed by the writer.

    For each variable:
    - If the variable is already passive or shared, it is forwarded exactly
      as-is.  Sharing references are 1-based zone indices that are identical
      in the output file because tecfix writes every zone in order without
      skipping any.
    - If the variable is a floating-point type and contains NaN or Inf it is
      marked passive and recorded in *fix_auxdata*.
    - Otherwise it is included in the active data list verbatim.

    Args:
        zone:      A ``ReadZone`` instance from the source file.
        num_vars:  Total number of variables in the dataset.
        var_names: Ordered list of variable names (0-based).

    Returns:
        A 6-tuple of:
        ``(active_data, active_locs, passive_vars, var_sharing,
           existing_aux, fix_auxdata)``

        ``active_data``    – arrays for variables that are active and not shared.
        ``active_locs``    – value locations aligned with *active_data*.
        ``passive_vars``   – bool list (len == num_vars), ``True`` = passive.
        ``var_sharing``    – int list (len == num_vars), 0 = no sharing,
                             positive = 1-based source zone (unchanged).
        ``existing_aux``   – dict copied from the zone's original aux data.
        ``fix_auxdata``    – dict describing variables set passive by this tool.

    """
    active_data: list[np.ndarray] = []
    active_locs: list[Any] = []
    passive_vars: list[bool] = []
    var_sharing: list[int] = []
    fix_auxdata: dict[str, str] = {}

    for j, var in enumerate(zone.variable):
        already_passive = var.is_passive()
        sv = var.shared_zone  # None or 0-based source zone index

        # Pass sharing through verbatim.  tecfix writes all zones in the
        # same order as the source, so source zone N is always output zone N.
        share_int = sv if sv is not None else 0

        passive_vars.append(already_passive)
        var_sharing.append(share_int)

        if already_passive or share_int != 0:
            # Passive or shared -- nothing to inspect or store.
            active_data.append(np.array([], dtype=np.float32))
            active_locs.append(var.value_location)
            continue

        # Load the array.
        arr: np.ndarray = var.values
        active_locs.append(var.value_location)

        # Guard against readers that return None or an empty array for
        # variables that are unavailable (e.g. DAT reader passive vars).
        if arr is None or arr.size == 0:
            passive_vars[-1] = True
            active_data.append(np.array([], dtype=np.float32))
            continue

        # Only floating-point arrays can hold NaN / Inf.
        bad_label: str | None = None
        if var.data_type in _FLOAT_TYPES:
            bad_label = _classify_array(arr)

        if bad_label is not None:
            # Mark as passive; record the reason.
            passive_vars[-1] = True
            key = f"Variable{j + 1}"
            fix_auxdata[key] = bad_label
            active_data.append(np.array([], dtype=np.float32))
        else:
            active_data.append(arr)

    # Filter to only the arrays / locations the writer needs (active, non-shared).
    writer_data = [
        arr
        for arr, is_p, sv in zip(active_data, passive_vars, var_sharing, strict=False)
        if not is_p and sv == 0
    ]
    writer_locs = [
        loc
        for loc, is_p, sv in zip(active_locs, passive_vars, var_sharing, strict=False)
        if not is_p and sv == 0
    ]

    # Collect existing zone aux data.
    existing_aux: dict[str, str] = {}
    if len(zone.auxdata) > 0:
        existing_aux = dict(zone.auxdata.items())

    return (
        writer_data,
        writer_locs,
        passive_vars,
        var_sharing,
        existing_aux,
        fix_auxdata,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """Inspect and fix a Tecplot file.

    Returns:
        Exit code — ``0`` on success (including clean files), ``1`` on error.

    """
    args = _parse_args(argv)

    src = Path(args.filename)
    if not src.exists():
        print(f"Error: input file not found: {src}", file=sys.stderr)
        return 1

    # Determine output path (irrelevant for --dry-run but computed early so we
    # can catch conflicts before doing any work).
    if args.output is not None:
        dst = Path(args.output)
    else:
        dst = src.with_stem(src.stem + "_fixed")

    if not args.dry_run and dst.exists() and not args.force:
        print(
            f"Error: output file already exists: {dst}\nUse --force to overwrite.",
            file=sys.stderr,
        )
        return 1

    # ------------------------------------------------------------------
    # Scan / fix
    # ------------------------------------------------------------------

    total_fixed = 0  # count of (zone, variable) pairs made passive

    try:
        with tecio_open(str(src), "r") as reader:
            num_vars: int = reader.num_vars
            var_names: list[str] = reader.variables

            if args.dry_run:
                # Scan only — report and exit without writing.
                print(f"Scanning: {src}")
                print(f"  Variables : {num_vars}")
                print(f"  Zones     : {reader.num_zones}")

                any_bad = False
                for i, zone in enumerate(reader.zone):
                    zone_bad: dict[str, str] = {}
                    for j, var in enumerate(zone.variable):
                        if var.is_passive() or var.shared_zone is not None:
                            continue
                        if var.data_type not in _FLOAT_TYPES:
                            continue
                        label = _classify_array(var.values)
                        if label is not None:
                            zone_bad[f"Variable{j + 1}"] = label

                    if zone_bad:
                        any_bad = True
                        total_fixed += len(zone_bad)
                        print(
                            f"\n  Zone {i + 1}: '{zone.title}' — "
                            f"{len(zone_bad)} bad variable(s)"
                        )
                        for key, label in zone_bad.items():
                            idx = int(key.replace("Variable", "")) - 1
                            print(f"    {key} ({var_names[idx]}): {label}")

                if not any_bad:
                    print("\nNo NaN or Inf values found. File is clean.")
                else:
                    print(
                        f"\nTotal: {total_fixed} variable(s) across "
                        f"{reader.num_zones} zone(s) contain NaN or Inf."
                    )
                return 0

            # ------------------------------------------------------------------
            # Write fixed output.
            # ------------------------------------------------------------------
            print(f"Fixing: {src}  →  {dst}")

            with tecio_open(
                str(dst),
                "w",
                title=reader.title,
                variables=var_names,
                file_type=reader.file_type,
            ) as writer:
                # Forward dataset-level aux data.
                if len(reader.auxdata) > 0:
                    writer.add_auxdataset_dict(dict(reader.auxdata.items()))

                # Forward variable-level aux data.
                auxvar: dict[int, dict[str, str]] = {}
                for i in range(num_vars):
                    var_aux = reader.get_var_auxdata(i + 1)
                    if len(var_aux) > 0:
                        auxvar[i + 1] = dict(var_aux.items())
                if auxvar:
                    writer.add_auxvar_dict(auxvar)

                for i, zone in enumerate(reader.zone):
                    zt = zone.zone_type

                    if zt in _FE_POLY:
                        print(
                            f"Warning: zone {i + 1} ('{zone.title}') is {zt.name} "
                            "and cannot be copied — skipping.",
                            file=sys.stderr,
                        )
                        continue

                    (
                        writer_data,
                        writer_locs,
                        passive_vars,
                        var_sharing,
                        existing_aux,
                        fix_auxdata,
                    ) = _process_zone(zone, num_vars, var_names)

                    if fix_auxdata:
                        n = len(fix_auxdata)
                        total_fixed += n
                        print(
                            f"  Zone {i + 1}: '{zone.title}' — "
                            f"setting {n} variable(s) passive:"
                        )
                        for key, label in fix_auxdata.items():
                            idx = int(key.replace("Variable", "")) - 1
                            print(f"    {key} ({var_names[idx]}): {label}")

                    # Merge existing aux data with any fix records.
                    merged_aux: dict[str, str] | None = {
                        **existing_aux,
                        **fix_auxdata,
                    } or None

                    # Sanitize solution_time: if it is NaN the zone metadata
                    # is corrupt.  Reset both solution_time and strand_id to 0
                    # so Tecplot treats the zone as stationary and the writer
                    # does not propagate the bad value downstream.
                    sol_time = zone.solution_time
                    strand = zone.strand_id
                    if math.isnan(sol_time):
                        sol_time = 0.0
                        strand = 0
                        fix_auxdata["SolutionTime"] = "NaN"
                        total_fixed += 1
                        print(
                            f"  Zone {i + 1}: '{zone.title}' — "
                            "solution_time is NaN, reset to 0.0 (strand_id -> 0)"
                        )
                        # Rebuild merged_aux to pick up the SolutionTime entry.
                        merged_aux = {**existing_aux, **fix_auxdata} or None

                    common_kw: dict[str, Any] = dict(
                        title=zone.title,
                        value_locations=writer_locs,
                        passive_vars=passive_vars,
                        var_sharing=var_sharing,
                        solution_time=sol_time,
                        strand_id=strand,
                        aux=merged_aux,
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

    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        if not args.dry_run:
            dst.unlink(missing_ok=True)
        return 1

    if total_fixed == 0:
        print("No NaN or Inf values found. Output file is a clean copy.")
    else:
        print(
            f"\nDone. {total_fixed} variable array(s) set passive. "
            f"Output written to: {dst}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
