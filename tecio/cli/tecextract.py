r"""Extract a subset of zones and/or variables from a Tecplot data file.

Large Tecplot files produced by CFD solvers frequently contain many zones and variables,
only a fraction of which are relevant to a given analysis. Loading the full file into
Tecplot or writing a dedicated extraction script is impractical when the goal is simply
to isolate a surface zone or a handful of flow variables for downstream processing.
``tecextract`` addresses this by writing a new Tecplot file containing only the
requested subset, with the output format determined by the file extension of the output
path. This makes it straightforward to reduce file size, change format, or prepare a
subset for use with another tool in a single command.

:Usage:

.. code:: bash

    tecextract [-h] [-zones LIST] [-variables LIST] [-o PATH] [-f] PATH

:Positional Arguments:
    ``PATH``
        Path to the input Tecplot binary file (``.plt`` or ``.szplt``).

:Options:
    ``-zones LIST``
        Comma-separated list of one-based zone indices to extract (e.g. ``-zones
        1,3,5``). If omitted, all zones are written to the output.

    ``-variables LIST``
        Comma-separated list of one-based variable indices to extract (e.g. ``-variables
        1,2,5``). If omitted, all variables are written to the output.

    ``-o PATH``, ``--output PATH``
        Output file path. The extension controls the output format: ``.szplt``,
        ``.plt``, or ``.dat``. Defaults to ``<stem>_extract<ext>`` in the same directory
        as the input file.

    ``-f``, ``--force``
        Overwrite the output file if it already exists. Without this flag the command
        exits with an error rather than silently clobbering an existing file.

:Returns:
    A new Tecplot file written to the output path containing only the requested zones
    and variables. Exit code is ``0`` on success and non-zero if the input file cannot
    be read, an invalid index is supplied, or the output file already exists and
    ``--force`` is not set.

Examples:
    Extract zones 1 and 3::

        $ tecextract -zones 1,3 solution.szplt

    Extract variables 1, 2, and 5::

        $ tecextract -variables 1,2,5 solution.szplt

    Extract a zone subset and convert to ASCII in one step::

        $ tecextract -zones 1,2 -o subset.dat solution.szplt

    Call directly from a Tecplot macro or Python session, passing arguments as a list of
    strings::

        import tecio.cli.tecextract.main as tecextract

        tecextract([
            "-zones",
            "1,2",
            "-variables",
            "1,2,5",
            "-o",
            "subset.szplt",
            "solution.szplt",
        ])

See Also:
    * :mod:`tecio.cli.tecsplit` - Split a file into separate grid and solution files
      rather than extracting a subset into a single output.
    * :mod:`tecio.cli.tecslice` - Slice a Tecplot file containing structured data along
      IJK indices and/or solution time.
    * :mod:`tecio.cli.tecmerge` - Merge zones from multiple files into a single output —
      the inverse operation to ``tecextract``.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .. import TecplotFEZoneReader, TecplotOrderedZoneReader, ZoneType
from .. import open as tecio_open

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


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
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Expected comma-separated integers, got: {value!r}"
        ) from exc


# --------------------------------------------------------------------------------------
# Argument parsing
# --------------------------------------------------------------------------------------


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="tecextract",
        description=(
            # -|--------------------|---------------------------------------------|
            "Extract a subset of zones and/or variables from a Tecplot file.\n"
            "Output format is determined by the -o extension."
        ),
        epilog=(
            # -|--------------------|---------------------------------------------|
            "Example usage:\n"
            "  Extract zones 1 and 3\n"
            "    $ tecextract -zones 1,3 <file>\n"
            "  Extract variables 1, 2, 5\n"
            "    $ tecextract -variables 1,2,5 <file>\n"
            "  Extract and convert format\n"
            "    $ tecextract -zones 1,2 -o subset.dat <file>\n"
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
        "-zones",
        type=_parse_index_list,
        default=None,
        metavar="LIST",
        help=(
            "Comma-separated list of 1-based zone indices to extract "
            "(e.g. -zones 1,3,5). Default is all zones."
        ),
    )
    parser.add_argument(
        "-variables",
        type=_parse_index_list,
        default=None,
        metavar="LIST",
        help=(
            "Comma-separated list of 1-based variable indices to extract "
            "(e.g. -variables 1,2,5). Default is all variables."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Output file path. The extension controls the output format "
            "(.szplt, .plt, .dat). Defaults to <stem>_extract<ext> in "
            "the same directory as the input."
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

                # Maps a source zone's 1-based index to its 1-based index in the output
                # file, populated as zones are actually written:
                # - A variable/connectivity shared from a zone that's also in this map
                #   can have its sharing preserved (just pointing at the new, compacted
                #   index) instead of being materialized as independent data
                # - Only a share whose source zone was excluded from the extraction
                #   genuinely has nowhere to point and must fall back to real data.
                zone_index_map: dict[int, int] = {}

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
                        is_passive = var.is_passive()
                        passive_vars.append(is_passive)
                        active_locs.append(var.value_location)

                        sv = var.shared_zone
                        remapped = zone_index_map.get(sv) if sv is not None else None

                        if is_passive:
                            var_sharing.append(0)
                            active_data.append(np.array([], dtype=np.float32))
                        elif remapped is not None:
                            # Source zone was also extracted -> preserve the data
                            # sharing relationship
                            var_sharing.append(remapped)
                            active_data.append(np.array([], dtype=np.float32))
                        else:
                            # Variables not shared at all, or shared from a zone that is
                            # not output, in which case branch sharing -> write the
                            # actual values as independent data
                            var_sharing.append(0)
                            arr = var.values
                            if arr is None or arr.size == 0:
                                passive_vars[-1] = True
                                active_data.append(np.array([], dtype=np.float32))
                            else:
                                active_data.append(arr)

                    writer_data = [
                        arr
                        for arr, is_p, sv in zip(
                            active_data, passive_vars, var_sharing, strict=False
                        )
                        if not is_p and sv == 0
                    ]
                    writer_locs = [
                        loc
                        for loc, is_p, sv in zip(
                            active_locs, passive_vars, var_sharing, strict=False
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

                    if isinstance(zone, TecplotOrderedZoneReader):
                        writer.write_ijk_zone(data=writer_data, **common_kw)
                    elif isinstance(zone, TecplotFEZoneReader):
                        con_src = zone.shared_connectivity
                        con_remapped = (
                            zone_index_map.get(con_src) if con_src is not None else None
                        )
                        writer.write_fe_zone(
                            zone_type=zt,
                            data=writer_data,
                            node_map=None if con_remapped else zone.node_map,
                            con_sharing=con_remapped,
                            **common_kw,
                        )
                    else:
                        raise NotImplementedError(
                            f"Zone '{zone.title}' is neither an ordered nor a "
                            "classic FE zone; extraction is not supported for it."
                        )

                    # Record where this source zone landed in the output, so a later
                    # zone sharing from it can point at the real (compacted) index
                    # instead of falling back to independent data.
                    zone_index_map[zone_num] = writer.current_zone

    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        dst.unlink(missing_ok=True)
        return 1

    print(f"Done. Output written to: {dst}")
    return 0


if __name__ == "__main__":
    main()
