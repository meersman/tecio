r"""Print per-variable statistics for a Tecplot data file.

Verifying that simulation output is physically reasonable, falls within expected bounds,
or that a variable has not collapsed to a constant is a routine step in CFD
post-processing. Extracting these figures from a binary file would otherwise require
opening it in Tecplot or writing a dedicated script. ``tecstats`` reports the minimum,
maximum, mean, and standard deviation for every variable in every zone directly to the
terminal, with optional CSV output for further analysis or archival. Passive and shared
variables are noted but skipped. Used alongside ``tecfix``, this tool provides a
lightweight diagnostic layer for validating file contents before and after any
corrective operation.

:Usage:

.. code:: bash

    tecstats [-h] [-zone INDEX] [-variable INDEX] [-csv] [-f] PATH

:Positional Arguments:
    ``PATH``
        Path to the input Tecplot file (``.plt``, ``.szplt``, or ``.dat``) to analyse.

:Options:
    ``-zone INDEX``
        Restrict output to the zone at the given one-based index. If omitted, all zones
        are reported.

    ``-variable INDEX``
        Restrict output to the variable at the given one-based index.  If omitted, all
        variables are reported.

    ``-csv``
        Write statistics to a CSV file in addition to the terminal output. The filename
        is derived automatically from the input file stem with a ``_stats`` suffix,
        preceded by optional ``_zone_N`` and ``_var_N`` segments when the corresponding
        filters are active. For example:

        .. list-table::
           :header-rows: 1
           :widths: 50 50

           * - Command
             - Output filename
           * - ``tecstats -csv flow.szplt``
             - ``flow_stats.csv``
           * - ``tecstats -csv -zone 2 flow.szplt``
             - ``flow_zone_2_stats.csv``
           * - ``tecstats -csv -variable 3 flow.szplt``
             - ``flow_var_3_stats.csv``
           * - ``tecstats -csv -zone 2 -variable 3 flow.szplt``
             - ``flow_zone_2_var_3_stats.csv``

    ``-f``, ``--force``
        Overwrite the output CSV file if it already exists. Without this flag the
        command exits with an error rather than silently clobbering an existing file.

:Returns:
    Statistics are written to standard output. If ``-csv`` is set, a CSV file is also
    written to the same directory as the input file with an automatically derived
    name. Exit code is ``0`` on success and non-zero if the input file cannot be read,
    an invalid index is supplied, or the CSV file already exists and ``--force`` is not
    set.

Examples:
    Print statistics for all zones and variables::

        $ tecstats flow.szplt

    Restrict to zone 2 only::

        $ tecstats -zone 2 flow.szplt

    Restrict to variable 3 across all zones::

        $ tecstats -variable 3 flow.szplt

    Write results to a CSV file::

        $ tecstats -csv flow.szplt

    Call directly from a Python session::

        import tecio.cli.tecstats.main as tecstats

        tecstats(["-zone", "2", "-variable", "3", "flow.szplt"])

See Also:
    * :mod:`tecio.cli.tecdump`: Inspect the full contents and metadata of a file,
      including auxiliary data and raw variable arrays.
    * :mod:`tecio.cli.tecfix`: Rewrite a file with invalid variable arrays set to
       passive once bad values have been identified via statistics.

"""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .. import open as tecio_open

# ---------------------------------------------------------------------------
# Column layout constants
# ---------------------------------------------------------------------------

#: Console display width for zone title and variable name columns.
_TITLE_WIDTH: int = 20
_NAME_WIDTH: int = 20

#: Total separator line width.
#: Zone(5) + Title(20) + Var(4) + Name(20) + Min(14) + Max(14) + Mean(14)
#: + Std(14) + Location(12) + column gaps + "Note" label ≈ 140
_SEP_WIDTH: int = 140


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="tecstats",
        description="Print per-variable statistics for a Tecplot file.",
        epilog=(
            "Example usage:\n"
            "  Print stats for all zones and variables\n"
            "    $ tecstats <file>\n"
            "  Print stats for zone 2 only\n"
            "    $ tecstats -zone 2 <file>\n"
            "  Print stats for variable 3 only\n"
            "    $ tecstats -variable 3 <file>\n"
            "  Write results to a CSV file (auto-named from input stem)\n"
            "    $ tecstats -csv <file>                 # <stem>_stats.csv\n"
            "  CSV with zone/variable filter suffixes\n"
            "    $ tecstats -csv -zone 2 -variable 3 <file>  # <stem>_zone_2_var_3_stats.csv\n"  # noqa: E501
        ),
        formatter_class=lambda prog: argparse.RawDescriptionHelpFormatter(
            prog, width=70, max_help_position=24
        ),
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
    parser.add_argument(
        "-csv",
        action="store_true",
        default=False,
        dest="write_csv",
        help=(
            "Write statistics to a CSV file.  The filename is derived "
            "automatically from the input file stem with a _stats suffix "
            "always appended, preceded by optional _zone_N and _var_N "
            "segments when -zone or -variable are active "
            "(e.g. flow_zone_2_var_3_stats.csv)."
        ),
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        default=False,
        help="Overwrite the output CSV file if it already exists.",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# CSV filename helper
# ---------------------------------------------------------------------------


def _build_csv_path(
    input_path: str,
    zone: int | None,
    variable: int | None,
) -> Path:
    """Return the auto-generated CSV output path.

    The stem of *input_path* is used as the base name.  ``_zone_N`` and
    ``_var_N`` segments are inserted (in that order) when the corresponding
    filter arguments are not ``None``, and ``_stats`` is always appended as
    the final suffix before the ``.csv`` extension.

    Args:
        input_path: Path to the input Tecplot file.
        zone:       1-based zone filter index, or ``None``.
        variable:   1-based variable filter index, or ``None``.

    Returns:
        :class:`~pathlib.Path` with a ``.csv`` suffix in the same directory
        as the input file.

    Examples:
        >>> _build_csv_path("results/flow.szplt", None, None)
        PosixPath('results/flow_stats.csv')
        >>> _build_csv_path("results/flow.szplt", 2, None)
        PosixPath('results/flow_zone_2_stats.csv')
        >>> _build_csv_path("results/flow.szplt", 2, 3)
        PosixPath('results/flow_zone_2_var_3_stats.csv')

    """
    src = Path(input_path)
    stem = src.stem
    if zone is not None:
        stem = f"{stem}_zone_{zone}"
    if variable is not None:
        stem = f"{stem}_var_{variable}"
    stem = f"{stem}_stats"
    return src.with_name(stem + ".csv")


# ---------------------------------------------------------------------------
# Console formatting helpers
# ---------------------------------------------------------------------------


def _print_header() -> None:
    """Print the column header and separator line to stdout."""
    print(
        f"\n{'Zone':>5}  {'Zone Title':<{_TITLE_WIDTH}}  {'Var':>4}  "
        f"{'Name':<{_NAME_WIDTH}}  "
        f"{'Min':>14}  {'Max':>14}  {'Mean':>14}  {'Std':>14}  "
        f"{'Location':<12}  Note"
    )
    print("-" * _SEP_WIDTH)


def _format_console_row(
    zone_num: int,
    zone_title: str,
    var_num: int,
    var_name: str,
    vmin: float | str,
    vmax: float | str,
    vmean: float | str,
    vstd: float | str,
    loc: str,
    note: str = "",
) -> str:
    """Return a formatted console statistics row string."""
    z_title = zone_title[:_TITLE_WIDTH]
    v_name = var_name[:_NAME_WIDTH]

    if isinstance(vmin, float):
        nums = f"{vmin:>14.6g}  {vmax:>14.6g}  {vmean:>14.6g}  {vstd:>14.6g}"
    else:
        nums = f"{'':>14}  {'':>14}  {'':>14}  {'':>14}"

    return (
        f"{zone_num:>5}  {z_title:<{_TITLE_WIDTH}}  {var_num:>4}  "
        f"{v_name:<{_NAME_WIDTH}}  {nums}  {loc:<12}  {note}"
    )


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

_CSV_FIELDNAMES: list[str] = [
    "zone_num",
    "zone_title",
    "var_num",
    "var_name",
    "min",
    "max",
    "mean",
    "std",
    "location",
    "note",
]


def _csv_row(
    zone_num: int,
    zone_title: str,
    var_num: int,
    var_name: str,
    vmin: float | str,
    vmax: float | str,
    vmean: float | str,
    vstd: float | str,
    loc: str,
    note: str = "",
) -> dict[str, str | int | float]:
    """Build one CSV row as a plain dict keyed on :data:`_CSV_FIELDNAMES`."""
    return {
        "zone_num": zone_num,
        "zone_title": zone_title,
        "var_num": var_num,
        "var_name": var_name,
        "min": vmin,
        "max": vmax,
        "mean": vmean,
        "std": vstd,
        "location": loc,
        "note": note,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """Print per-variable statistics for a Tecplot file.

    Returns:
        Exit code -- ``0`` on success, ``1`` on error.

    """
    args = _parse_args(argv)

    # Resolve and validate the CSV output path before doing any work so that
    # a naming conflict fails fast rather than after a potentially long read.
    csv_path: Path | None = None
    if args.write_csv:
        csv_path = _build_csv_path(args.filename, args.zone, args.variable)
        if csv_path.exists() and not args.force:
            print(
                f"Error: output file already exists: {csv_path}\n"
                "Use --force / -f to overwrite.",
                file=sys.stderr,
            )
            return 1

    # Accumulate CSV rows during iteration; written at the end so that a
    # mid-run error never leaves a partial file on disk.
    csv_rows: list[dict] = []

    try:
        with tecio_open(args.filename, "r") as tec:
            print(f"\nFile    : {args.filename}")
            print(f"Title   : {tec.title}")
            print(f"Vars    : {tec.num_vars}")
            print(f"Zones   : {tec.num_zones}")

            _print_header()

            for i, zone in enumerate(tec.zone):
                zone_num = i + 1
                if args.zone is not None and zone_num != args.zone:
                    continue

                zone_title: str = zone.title or ""

                for j in range(tec.num_vars):
                    var_num = j + 1
                    if args.variable is not None and var_num != args.variable:
                        continue

                    var = zone.variable[j]
                    loc = (
                        var.value_location.name
                        if var.value_location is not None
                        else "?"
                    )
                    name: str = var.name

                    if var.is_passive():
                        note = "passive"
                        print(
                            _format_console_row(
                                zone_num,
                                zone_title,
                                var_num,
                                name,
                                "",
                                "",
                                "",
                                "",
                                loc,
                                note,
                            )
                        )
                        if csv_path is not None:
                            csv_rows.append(
                                _csv_row(
                                    zone_num,
                                    zone_title,
                                    var_num,
                                    name,
                                    "",
                                    "",
                                    "",
                                    "",
                                    loc,
                                    note,
                                )
                            )
                        continue

                    if var.shared_zone is not None:
                        note = f"shared (zone {var.shared_zone + 1})"
                        print(
                            _format_console_row(
                                zone_num,
                                zone_title,
                                var_num,
                                name,
                                "",
                                "",
                                "",
                                "",
                                loc,
                                note,
                            )
                        )
                        if csv_path is not None:
                            csv_rows.append(
                                _csv_row(
                                    zone_num,
                                    zone_title,
                                    var_num,
                                    name,
                                    "",
                                    "",
                                    "",
                                    "",
                                    loc,
                                    note,
                                )
                            )
                        continue

                    arr = var.values
                    if arr is None or arr.size == 0:
                        note = "no data"
                        print(
                            _format_console_row(
                                zone_num,
                                zone_title,
                                var_num,
                                name,
                                "",
                                "",
                                "",
                                "",
                                loc,
                                note,
                            )
                        )
                        if csv_path is not None:
                            csv_rows.append(
                                _csv_row(
                                    zone_num,
                                    zone_title,
                                    var_num,
                                    name,
                                    "",
                                    "",
                                    "",
                                    "",
                                    loc,
                                    note,
                                )
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
                        _format_console_row(
                            zone_num,
                            zone_title,
                            var_num,
                            name,
                            vmin,
                            vmax,
                            vmean,
                            vstd,
                            loc,
                        )
                    )
                    if csv_path is not None:
                        csv_rows.append(
                            _csv_row(
                                zone_num,
                                zone_title,
                                var_num,
                                name,
                                vmin,
                                vmax,
                                vmean,
                                vstd,
                                loc,
                            )
                        )

    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    # Write CSV atomically only after all rows have been collected without
    # error, keeping the output directory free of partial files.
    if csv_path is not None:
        try:
            with csv_path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDNAMES)
                writer.writeheader()
                writer.writerows(csv_rows)
            print(f"\nCSV written to: {csv_path}")
        except OSError as exc:
            print(f"Error writing CSV: {exc}", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    main()
