#!/usr/bin/env python3
"""Command line interface to scale and/or offset variables in a Tecplot file.

Variables can be identified by 1-based index or by name.  The transformation
applied to each selected value is::

    new_value = old_value * scale + offset

Both ``-scale`` and ``-offset`` default to 1.0 and 0.0 respectively, so
either can be omitted.  Use ``-zone`` to restrict the operation to a single
zone; without it all zones are processed.

The output file is written to ``<stem>_scaled<ext>`` unless ``-o`` is given.
The output format is controlled by the ``-o`` extension, allowing format
conversion at the same time.

Example usage::

    # Convert pressure (var 4) from Pa to kPa across all zones
    $ tecscale -variable 4 -scale 1e-3 flow.szplt

    # Same using variable name
    $ tecscale -variable Pressure -scale 1e-3 flow.szplt

    # Shift temperature by -273.15 (K -> degC), zone 2 only
    $ tecscale -variable Temperature -offset -273.15 -zone 2 flow.szplt

    # Scale and offset, write to DAT
    $ tecscale -variable 3 -scale 0.3048 -o flow_ft.dat flow.szplt
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .. import open as tecio_open
from ..libtecio import DataType, ZoneType


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="tecscale",
        description=(
            "Scale and/or offset one or more variables in a Tecplot file.  "
            "Transformation: new = old * scale + offset."
        ),
        epilog=(
            "Examples:\n"
            "  Scale variable 4 by 1e-3 (Pa -> kPa)\n"
            "    $ tecscale -variable 4 -scale 1e-3 <file>\n"
            "  Same using variable name\n"
            "    $ tecscale -variable Pressure -scale 1e-3 <file>\n"
            "  Offset temperature in zone 2 only\n"
            "    $ tecscale -variable Temperature -offset -273.15 -zone 2 <file>\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "filename",
        type=str,
        help="Input Tecplot file.",
    )
    parser.add_argument(
        "-variable",
        type=str,
        required=True,
        metavar="INDEX_OR_NAME",
        help=(
            "Variable to scale: either a 1-based integer index or the "
            "variable name (case-insensitive)."
        ),
    )
    parser.add_argument(
        "-scale",
        type=float,
        default=1.0,
        metavar="FLOAT",
        help="Multiplicative scale factor.  Default: 1.0.",
    )
    parser.add_argument(
        "-offset",
        type=float,
        default=0.0,
        metavar="FLOAT",
        help="Additive offset applied after scaling.  Default: 0.0.",
    )
    parser.add_argument(
        "-zone",
        type=int,
        default=None,
        metavar="INDEX",
        help=(
            "1-based zone index to apply the transformation to.  "
            "Default is all zones."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Output file path.  The extension controls the output format.  "
            "Defaults to <stem>_scaled<ext> in the same directory."
        ),
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        default=False,
        help="Overwrite the output file if it already exists.",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_variable(spec: str, var_names: list[str]) -> int:
    """Return a 0-based variable index from a name or 1-based integer string.

    Args:
        spec:      User-supplied string (e.g. ``"3"`` or ``"Pressure"``).
        var_names: Ordered list of variable names from the reader.

    Returns:
        0-based variable index.

    Raises:
        SystemExit: If the spec cannot be resolved.

    """
    # Try integer first.
    try:
        idx = int(spec)
        if idx < 1 or idx > len(var_names):
            print(
                f"Error: variable index {idx} out of range [1, {len(var_names)}].",
                file=sys.stderr,
            )
            sys.exit(1)
        return idx - 1
    except ValueError:
        pass

    # Try case-insensitive name match.
    spec_lower = spec.lower()
    for i, name in enumerate(var_names):
        if name.lower() == spec_lower:
            return i

    print(
        f"Error: variable '{spec}' not found.  "
        f"Available: {var_names}",
        file=sys.stderr,
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """Scale and/or offset a variable in a Tecplot file.

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
        dst = src.with_stem(src.stem + "_scaled")

    if dst.exists() and not args.force:
        print(
            f"Error: output file already exists: {dst}\n"
            "Use --force to overwrite.",
            file=sys.stderr,
        )
        return 1

    try:
        with tecio_open(str(src), "r") as reader:
            var_names: list[str] = reader.variables
            num_vars: int = reader.num_vars

            # Resolve target variable.
            var_idx0: int = _resolve_variable(args.variable, var_names)

            # Validate zone if specified.
            if args.zone is not None:
                if args.zone < 1 or args.zone > reader.num_zones:
                    print(
                        f"Error: zone index {args.zone} out of range "
                        f"[1, {reader.num_zones}].",
                        file=sys.stderr,
                    )
                    return 1

            print(
                f"Scaling '{var_names[var_idx0]}' (var {var_idx0 + 1}): "
                f"new = old * {args.scale} + {args.offset}"
            )
            if args.zone is not None:
                print(f"  Zone {args.zone} only.")
            else:
                print(f"  All {reader.num_zones} zones.")

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

                # Forward all variable-level aux data.
                auxvar: dict[int, dict[str, str]] = {}
                for i in range(num_vars):
                    var_aux = reader.get_var_auxdata(i + 1)
                    if len(var_aux) > 0:
                        auxvar[i + 1] = dict(var_aux.items())
                if auxvar:
                    writer.add_auxvar_dict(auxvar)

                for i, zone in enumerate(reader.zone):
                    zone_num = i + 1
                    zt = zone.zone_type

                    if zt in (ZoneType.FEPOLYGON, ZoneType.FEPOLYHEDRON):
                        print(
                            f"Warning: zone {zone_num} ('{zone.title}') is "
                            f"{zt.name} and cannot be written -- skipping.",
                            file=sys.stderr,
                        )
                        continue

                    apply_scale = (
                        args.zone is None or zone_num == args.zone
                    )

                    active_data: list[np.ndarray] = []
                    active_locs: list[Any] = []
                    passive_vars: list[bool] = []
                    var_sharing: list[int] = []

                    for j, var in enumerate(zone.variable):
                        passive_vars.append(var.is_passive())
                        sv = var.shared_zone
                        var_sharing.append((sv + 1) if sv is not None else 0)
                        active_locs.append(var.value_location)

                        if var.is_passive() or sv is not None:
                            active_data.append(
                                np.array([], dtype=np.float32)
                            )
                            continue

                        arr = var.values
                        if arr is None or arr.size == 0:
                            passive_vars[-1] = True
                            active_data.append(
                                np.array([], dtype=np.float32)
                            )
                            continue

                        if apply_scale and j == var_idx0:
                            # Apply transformation, promoting to float64 to
                            # preserve precision across the operation.
                            arr = arr.astype(np.float64)
                            arr = arr * args.scale + args.offset

                        active_data.append(arr)

                    writer_data = [
                        arr
                        for arr, is_p, sv in zip(
                            active_data, passive_vars, var_sharing
                        )
                        if not is_p and sv == 0
                    ]
                    writer_locs = [
                        loc
                        for loc, is_p, sv in zip(
                            active_locs, passive_vars, var_sharing
                        )
                        if not is_p and sv == 0
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
