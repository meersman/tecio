"""Command line interface to print per-variable statistics for a Tecplot file.

For each zone and each variable, prints the minimum, maximum, mean, and
standard deviation of the data array.  Passive and shared variables are
noted but skipped.

Example usage::

    # Print stats for all zones and variables
    $ tecstat flow.szplt

    # Stats for zone 2 only
    $ tecstat -zone 2 flow.szplt

    # Stats for variable 3 only, all zones
    $ tecstat -variable 3 flow.szplt

    # Stats for zone 1, variable 2
    $ tecstat -zone 1 -variable 2 flow.szplt
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

import numpy as np

from .. import open as tecio_open
from ..libtecio import ZoneType


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="tecstat",
        description="Print per-variable statistics for a Tecplot file.",
        epilog=(
            "Example usage:\n"
            "  Print stats for all zones and variables\n"
            "    $ tecstat <file>\n"
            "  Print stats for zone 2 only\n"
            "    $ tecstat -zone 2 <file>\n"
            "  Print stats for variable 3 only\n"
            "    $ tecstat -variable 3 <file>\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "filename",
        type=str,
        help="Input Tecplot file.",
    )
    parser.add_argument(
        "-zone",
        type=int,
        default=None,
        metavar="INDEX",
        help="1-based zone index to report.  Default is all zones.",
    )
    parser.add_argument(
        "-variable",
        type=int,
        default=None,
        metavar="INDEX",
        help="1-based variable index to report.  Default is all variables.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Print per-variable statistics for a Tecplot file.

    Returns:
        Exit code -- ``0`` on success, ``1`` on error.

    """
    args = _parse_args(argv)

    try:
        with tecio_open(args.filename, "r") as tec:
            print(f"\nFile    : {args.filename}")
            print(f"Title   : {tec.title}")
            print(f"Vars    : {tec.num_vars}")
            print(f"Zones   : {tec.num_zones}")

            # Column header
            print()
            print(
                f"{'Zone':>5}  {'Var':>4}  {'Name':<20}  "
                f"{'Min':>14}  {'Max':>14}  {'Mean':>14}  {'Std':>14}  "
                f"{'Location':<12}  Note"
            )
            print("-" * 110)

            for i, zone in enumerate(tec.zone):
                zone_num = i + 1
                if args.zone is not None and zone_num != args.zone:
                    continue

                for j in range(tec.num_vars):
                    var_num = j + 1
                    if args.variable is not None and var_num != args.variable:
                        continue

                    var = zone.variable[j]
                    loc = var.value_location.name if var.value_location is not None else "?"
                    name = var.name[:20]

                    if var.is_passive():
                        print(
                            f"{zone_num:>5}  {var_num:>4}  {name:<20}  "
                            f"{'':>14}  {'':>14}  {'':>14}  {'':>14}  "
                            f"{loc:<12}  passive"
                        )
                        continue

                    if var.shared_zone is not None:
                        print(
                            f"{zone_num:>5}  {var_num:>4}  {name:<20}  "
                            f"{'':>14}  {'':>14}  {'':>14}  {'':>14}  "
                            f"{loc:<12}  shared (zone {var.shared_zone + 1})"
                        )
                        continue

                    arr = var.values
                    if arr is None or arr.size == 0:
                        print(
                            f"{zone_num:>5}  {var_num:>4}  {name:<20}  "
                            f"{'':>14}  {'':>14}  {'':>14}  {'':>14}  "
                            f"{loc:<12}  no data"
                        )
                        continue

                    # Cast to float64 for stable statistics regardless of
                    # the on-disk data type (INT16, BYTE, etc.).
                    farr = arr.astype(np.float64)
                    vmin = float(np.min(farr))
                    vmax = float(np.max(farr))
                    vmean = float(np.mean(farr))
                    vstd = float(np.std(farr))

                    print(
                        f"{zone_num:>5}  {var_num:>4}  {name:<20}  "
                        f"{vmin:>14.6g}  {vmax:>14.6g}  "
                        f"{vmean:>14.6g}  {vstd:>14.6g}  "
                        f"{loc:<12}"
                    )

    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    main()
