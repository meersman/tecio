"""Command line interface to convert between Tecplot file formats.

Supports conversions between:
    - SZL  (``.szplt``)
    - PLT  (``.plt``)
    - DAT  (``.dat``)

The output format is selected with a format flag.  The output file is written
to the same directory as the input file with the appropriate extension, unless
an explicit path is given with ``-o``.

Example usage::

    # SZL → DAT
    $ teconvert -dat flow.szplt

    # PLT → SZL
    $ teconvert -szplt flow.plt

    # DAT → PLT, explicit output path
    $ teconvert -plt -o /tmp/flow.plt flow.dat

    # Force overwrite of an existing output file
    $ teconvert --force -dat flow.szplt
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
# Constants
# ---------------------------------------------------------------------------

#: Canonical extension for each format flag (flag name -> extension).
_FLAG_TO_EXT: dict[str, str] = {
    "szplt": ".szplt",
    "plt": ".plt",
    "dat": ".dat",
}

#: FE zone types supported by the Write API.
_FE_SIMPLE: frozenset[ZoneType] = frozenset({
    ZoneType.FELINESEG,
    ZoneType.FETRIANGLE,
    ZoneType.FEQUADRILATERAL,
    ZoneType.FETETRAHEDRON,
    ZoneType.FEBRICK,
})


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="teconvert",
        description="Convert a Tecplot file to a different format.",
        epilog=(
            "Output format flags (exactly one required)\n"
            "  -szplt  ->  Tecplot SZL binary (.szplt)\n"
            "  -plt    ->  Tecplot PLT binary (.plt)\n"
            "  -dat    ->  Tecplot ASCII DAT (.dat)\n\n"
            "Examples\n"
            "  teconvert -dat flow.szplt                  # SZL -> ASCII DAT\n"
            "  teconvert -szplt flow.plt                  # PLT -> SZL\n"
            "  teconvert -plt -o /tmp/out.plt flow.dat    # explicit output path\n"
            "  teconvert --force -dat flow.szplt          # overwrite existing\n"
        ),
        formatter_class=lambda prog: argparse.RawDescriptionHelpFormatter(
            prog, width=70, max_help_position=24
        ),
    )
    parser.add_argument(
        "filename",
        type=str,
        help="Input Tecplot file to convert.",
    )

    # Mutually exclusive format flags -- exactly one must be given.
    fmt_group = parser.add_mutually_exclusive_group(required=True)
    fmt_group.add_argument(
        "-szplt",
        action="store_const",
        const="szplt",
        dest="format",
        help="Convert to Tecplot SZL binary format (.szplt).",
    )
    fmt_group.add_argument(
        "-plt",
        action="store_const",
        const="plt",
        dest="format",
        help="Convert to Tecplot PLT binary format (.plt).",
    )
    fmt_group.add_argument(
        "-dat",
        action="store_const",
        const="dat",
        dest="format",
        help="Convert to Tecplot ASCII DAT format (.dat).",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Overwrite the output file if it already exists.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Explicit output file path.  Defaults to the input file stem with "
            "the new extension in the same directory as the input."
        ),
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Zone copying helper
# ---------------------------------------------------------------------------


def _copy_zones(reader: Any, writer: Any) -> None:
    """Stream all zones from *reader* into the open *writer*.

    Each zone is reproduced at its original data type and value location.
    Connectivity (node maps) is copied verbatim for FE zones.  Zone-level
    auxiliary data is forwarded as well.

    Args:
        reader: An open ``Read`` instance (szl, plt, or dat).
        writer: An open ``Write`` instance for the target format.

    Raises:
        NotImplementedError: If a FEPOLYGON or FEPOLYHEDRON zone is
            encountered, because the ``Write`` API does not yet support them.

    """
    for zone in reader.zone:
        zt = zone.zone_type

        if zt in (ZoneType.FEPOLYGON, ZoneType.FEPOLYHEDRON):
            raise NotImplementedError(
                f"Zone '{zone.title}' has unsupported type {zt.name}. "
                "FEPOLYGON and FEPOLYHEDRON zones cannot be converted."
            )

        # Collect per-variable arrays and metadata up front.
        data: list[np.ndarray] = []
        value_locations: list[Any] = []
        passive_vars: list[bool] = []
        var_sharing: list[int] = []

        for var in zone.variable:
            passive_vars.append(var.is_passive())
            sv = var.shared_zone  # None or 0-based zone index
            # Write API expects 0 = no sharing, positive = 1-based zone source.
            var_sharing.append(sv if sv is not None else 0)
            value_locations.append(var.value_location)

            if var.is_passive() or sv is not None:
                data.append(np.array([], dtype=np.float32))
            else:
                data.append(var.values)

        # Only pass arrays for active, non-shared variables.
        active_data = [
            arr
            for arr, is_p, sv in zip(data, passive_vars, var_sharing, strict=False)
            if not is_p and sv == 0
        ]
        active_locs = [
            loc
            for loc, is_p, sv in zip(
                value_locations, passive_vars, var_sharing, strict=False
            )
            if not is_p and sv == 0
        ]

        # Collect existing zone-level aux data.
        zone_aux: dict[str, str] | None = None
        if len(zone.auxdata) > 0:
            zone_aux = dict(zone.auxdata.items())

        common_kw: dict[str, Any] = dict(
            title=zone.title,
            value_locations=active_locs,
            passive_vars=passive_vars,
            var_sharing=var_sharing,
            solution_time=zone.solution_time,
            strand_id=zone.strand_id,
            aux=zone_aux,
        )

        if zt == ZoneType.ORDERED:
            writer.write_ijk_zone(data=active_data, **common_kw)
        else:
            writer.write_fe_zone(
                zone_type=zt,
                data=active_data,
                node_map=zone.node_map,
                **common_kw,
            )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """Convert a Tecplot file to the requested format.

    Returns:
        Exit code -- ``0`` on success, ``1`` on error.

    """
    args = _parse_args(argv)

    src = Path(args.filename)
    if not src.exists():
        print(f"Error: input file not found: {src}", file=sys.stderr)
        return 1

    src_ext = src.suffix.lower()
    dst_ext = _FLAG_TO_EXT[args.format]

    if src_ext == dst_ext:
        print(
            f"Warning: source and destination formats are identical ({dst_ext}). "
            "Nothing to do.",
            file=sys.stderr,
        )
        return 0

    # Determine output path.
    dst = Path(args.output) if args.output is not None else src.with_suffix(dst_ext)

    if dst.exists() and not args.force:
        print(
            f"Error: output file already exists: {dst}\nUse --force to overwrite.",
            file=sys.stderr,
        )
        return 1

    print(f"Converting: {src}  ->  {dst}")

    try:
        with tecio_open(str(src), "r") as reader:
            # Propagate dataset-level metadata to the writer.
            title: str = reader.title
            variables: list[str] = reader.variables
            file_type = reader.file_type

            with tecio_open(
                str(dst),
                "w",
                title=title,
                variables=variables,
                file_type=file_type,
            ) as writer:
                # Forward dataset-level aux data.
                if len(reader.auxdata) > 0:
                    writer.add_auxdataset_dict(dict(reader.auxdata.items()))

                # Forward variable-level aux data.
                auxvar: dict[int, dict[str, str]] = {}
                for i in range(reader.num_vars):
                    var_aux = reader.get_var_auxdata(i + 1)
                    if len(var_aux) > 0:
                        auxvar[i + 1] = dict(var_aux.items())
                if auxvar:
                    writer.add_auxvar_dict(auxvar)

                _copy_zones(reader, writer)

    except NotImplementedError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        # Remove incomplete output so the user is not left with a partial file.
        dst.unlink(missing_ok=True)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"Error during conversion: {exc}", file=sys.stderr)
        dst.unlink(missing_ok=True)
        return 1

    print(f"Done. Output written to: {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
