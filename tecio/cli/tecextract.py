#!/usr/bin/env python3
"""Command line interface to extract a subset of zones and/or variables
from a Tecplot file.

Zones and variables are specified by 1-based index.  Multiple values are
given as comma-separated lists (no spaces).  The output format is inferred
from the output file extension, so passing ``-o result.dat`` produces an
ASCII DAT file regardless of the input format.

Example usage::

    # Extract zones 1 and 3, all variables
    $ tecextract -zones 1,3 flow.szplt

    # Extract variables 1, 2, 5, all zones
    $ tecextract -variables 1,2,5 flow.szplt

    # Extract zone 2, variables 1-3, write to PLT
    $ tecextract -zones 2 -variables 1,2,3 -o subset.plt flow.szplt

    # Convert format while extracting (SZL -> DAT)
    $ tecextract -o flow_subset.dat flow.szplt
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .. import open as tecio_open
from ..libtecio import ZoneType

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_index_list(value: str) -> list[int]:
    """Parse a comma-separated string of 1-based integers.

    Args:
        value: String like ``"1,3,5"`` or ``"2"``.

    Returns:
        List of integers.

    Raises:
        argparse.ArgumentTypeError: On invalid input.

    """
    try:
        return [int(v.strip()) for v in value.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Expected comma-separated integers, got: {value!r}"
        )


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="tecextract",
        description=(
            "Extract a subset of zones and/or variables from a Tecplot file. "
            "Output format is determined by the -o extension."
        ),
        epilog=(
            "Example usage:\n"
            "  Extract zones 1 and 3\n"
            "    $ tecextract -zones 1,3 <file>\n"
            "  Extract variables 1, 2, 5\n"
            "    $ tecextract -variables 1,2,5 <file>\n"
            "  Extract and convert format\n"
            "    $ tecextract -zones 1,2 -o subset.dat <file>\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "filename",
        type=str,
        help="Input Tecplot file.",
    )
    parser.add_argument(
        "-zones",
        type=_parse_index_list,
        default=None,
        metavar="LIST",
        help=(
            "Comma-separated list of 1-based zone indices to extract "
            "(e.g. -zones 1,3,5).  Default is all zones."
        ),
    )
    parser.add_argument(
        "-variables",
        type=_parse_index_list,
        default=None,
        metavar="LIST",
        help=(
            "Comma-separated list of 1-based variable indices to extract "
            "(e.g. -variables 1,2,5).  Default is all variables."
        ),
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Output file path.  The extension controls the output format "
            "(.szplt, .plt, .dat).  Defaults to <stem>_extract<ext> in "
            "the same directory as the input."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Overwrite the output file if it already exists.",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """Extract zones and/or variables from a Tecplot file.

    Returns:
        Exit code -- ``0`` on success, ``1`` on error.

    """
    args = _parse_args(argv)

    src = Path(args.filename)
    if not src.exists():
        print(f"Error: input file not found: {src}", file=sys.stderr)
        return 1

    if args.output is not None:
        dst = Path(args.output)
    else:
        dst = src.with_stem(src.stem + "_extract")

    if dst.exists() and not args.force:
        print(
            f"Error: output file already exists: {dst}\nUse --force to overwrite.",
            file=sys.stderr,
        )
        return 1

    try:
        with tecio_open(str(src), "r") as reader:
            num_vars: int = reader.num_vars
            all_var_names: list[str] = reader.variables

            # Resolve and validate zone filter.
            if args.zones is not None:
                for z in args.zones:
                    if z < 1 or z > reader.num_zones:
                        print(
                            f"Error: zone index {z} out of range "
                            f"[1, {reader.num_zones}].",
                            file=sys.stderr,
                        )
                        return 1
                zone_set: set[int] = set(args.zones)
            else:
                zone_set = set(range(1, reader.num_zones + 1))

            # Resolve and validate variable filter.
            if args.variables is not None:
                for v in args.variables:
                    if v < 1 or v > num_vars:
                        print(
                            f"Error: variable index {v} out of range [1, {num_vars}].",
                            file=sys.stderr,
                        )
                        return 1
                var_set: set[int] = set(args.variables)
                # Ordered list preserving original index order.
                out_var_indices: list[int] = [
                    v for v in range(1, num_vars + 1) if v in var_set
                ]
            else:
                out_var_indices = list(range(1, num_vars + 1))

            out_var_names: list[str] = [all_var_names[v - 1] for v in out_var_indices]

            print(f"Extracting: {src}  ->  {dst}")
            print(f"  Zones     : {sorted(zone_set)} of {reader.num_zones}")
            print(f"  Variables : {out_var_indices} of {num_vars} ({out_var_names})")

            with tecio_open(
                str(dst),
                "w",
                title=reader.title,
                variables=out_var_names,
                file_type=reader.file_type,
            ) as writer:
                # Forward dataset-level aux data.
                if len(reader.auxdata) > 0:
                    writer.add_auxdataset_dict(dict(reader.auxdata.items()))

                # Forward variable-level aux data for kept variables.
                auxvar: dict[int, dict[str, str]] = {}
                for new_idx, orig_idx in enumerate(out_var_indices, start=1):
                    var_aux = reader.get_var_auxdata(orig_idx)
                    if len(var_aux) > 0:
                        auxvar[new_idx] = dict(var_aux.items())
                if auxvar:
                    writer.add_auxvar_dict(auxvar)

                for i, zone in enumerate(reader.zone):
                    zone_num = i + 1
                    if zone_num not in zone_set:
                        continue

                    zt = zone.zone_type
                    if zt in (ZoneType.FEPOLYGON, ZoneType.FEPOLYHEDRON):
                        print(
                            f"Warning: zone {zone_num} ('{zone.title}') is "
                            f"{zt.name} and cannot be written -- skipping.",
                            file=sys.stderr,
                        )
                        continue

                    # Build per-variable metadata for the requested subset.
                    active_data: list[np.ndarray] = []
                    active_locs: list[Any] = []
                    passive_vars: list[bool] = []
                    var_sharing: list[int] = []

                    for orig_idx in out_var_indices:
                        var = zone.variable[orig_idx - 1]
                        passive_vars.append(var.is_passive())
                        sv = var.shared_zone
                        # Sharing refers to zones by their index in the
                        # *output* file, which may differ from the source.
                        # We cannot safely remap shares to a reduced zone
                        # set, so drop sharing — write as independent data.
                        var_sharing.append(0)
                        active_locs.append(var.value_location)

                        if var.is_passive() or sv is not None:
                            active_data.append(np.array([], dtype=np.float32))
                        else:
                            arr = var.values
                            if arr is None or arr.size == 0:
                                passive_vars[-1] = True
                                active_data.append(np.array([], dtype=np.float32))
                            else:
                                active_data.append(arr)

                    writer_data = [
                        arr
                        for arr, is_p in zip(active_data, passive_vars, strict=False)
                        if not is_p
                    ]
                    writer_locs = [
                        loc
                        for loc, is_p in zip(active_locs, passive_vars, strict=False)
                        if not is_p
                    ]

                    zone_aux: dict[str, str] | None = None
                    if len(zone.auxdata) > 0:
                        zone_aux = dict(zone.auxdata.items())

                    common_kw: dict[str, Any] = dict(
                        title=zone.title,
                        value_locations=writer_locs,
                        passive_vars=passive_vars,
                        var_sharing=var_sharing,
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

    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        dst.unlink(missing_ok=True)
        return 1

    print(f"Done. Output written to: {dst}")
    return 0


if __name__ == "__main__":
    main()
