r"""Convert a Tecplot data file to a MATLAB ``.mat`` file.

MATLAB is widely used for engineering analysis and post-processing, but it cannot read
Tecplot binary files directly, and the Tecplot ASCII format is awkward to parse on the
MATLAB side. Moving data across normally requires a Tecplot license or a hand-written
reader.  ``tec2mat`` bridges this gap by reading any supported Tecplot format
(``.szplt``, ``.plt``, or ``.dat``) and writing a single MATLAB ``.mat`` file via
:func:`scipy.io.savemat`. Each input file maps to exactly one output file
(``flow.szplt`` -> ``flow.mat``): every zone becomes a named struct, every variable is
preserved at its native precision, and connectivity and metadata are retained so the
dataset can be reconstructed in MATLAB without a Tecplot installation.

:Usage:

.. code:: bash

    tec2mat [-h] [-o PATH] [-f] [-c] [--oned-as {column,row}] PATH

:Positional Arguments:
    ``PATH``
        Path to the input Tecplot file (``.plt``, ``.szplt``, or ``.dat``) to convert.

:Options:
    ``-o PATH``, ``--output PATH``
        Output file path. Defaults to the input file stem with a ``.mat`` extension in
        the same directory as the input file.

    ``-f``, ``--force``
        Overwrite the output file if it already exists. Without this flag the command
        exits with an error rather than silently clobbering an existing file.

    ``-c``, ``--compress``
        Compress the variable arrays inside the ``.mat`` file (passes
        ``do_compression=True`` to :func:`scipy.io.savemat`). Reduces file size at the
        cost of some write/read time.

    ``--oned-as {column,row}``
        Orientation for one-dimensional arrays (finite-element nodal/cell vectors and
        the per-variable metadata arrays) in the ``.mat`` file.  ``column`` (the
        default) writes ``N x 1`` column vectors; ``row`` writes ``1 x N``. Has no
        effect on the two- and three-dimensional arrays of ordered zones.

:Returns:
    A single MATLAB ``.mat`` file written to the output path. Exit code is ``0`` on
    success and non-zero if the input file cannot be read or the output file already
    exists and ``--force`` is not set.

Output structure:
    The ``.mat`` file contains one ``info`` struct describing the dataset and one
    ``zone_<n>`` struct per zone (1-based, matching MATLAB's and Tecplot's indexing)::

        info                      struct
          .title                  char
          .file_type              char    'FULL' | 'GRID' | 'SOLUTION'
          .num_zones              double
          .num_vars               double
          .var_names              cell    {'x', 'y', 'p'}   (real variable names)

        zone_1                    struct
          .title                  char
          .zone_type              char    'ORDERED' | 'FETRIANGLE' | ...
          .solution_time          double
          .strand_id              double
          .I, .J, .K              double  (ordered zones)
          .num_nodes              double  (finite-element zones)
          .num_elements           double  (finite-element zones)
          .var_1 ... .var_N       array   one field per dataset variable
          .var_status             cell    'active' | 'passive' | 'shared'
          .var_locations          cell    'NODAL' | 'CELL_CENTERED' | ''
          .var_dtypes             cell    'FLOAT' | 'DOUBLE' | 'INT32' | ...
          .var_shared_from        array   1-based source zone, or 0 if not shared
          .node_map               array   (num_elements x nodes_per_cell), FE only,
                                          omitted if connectivity is shared
          .node_map_shared_from   double  1-based source zone, FE only, present only
                                          if this zone shares its connectivity

    Variable arrays are stored at their on-disk NumPy dtype, so single/double/integer
    precision is preserved. The real variable names are kept only in ``info.var_names``
    because they are frequently not valid MATLAB field names (e.g. ``"X [ft]"``); the
    per-zone data is addressed by 1-based index (``var_1`` ...) instead.

    Passive and shared variables carry no data: their ``var_<k>`` field is an empty
    matrix ``[]``. A shared variable is therefore never duplicated on disk -- the data
    lives in its source zone and ``var_shared_from`` records where, so the MATLAB user
    can dereference it (see the examples below). Shared FE connectivity follows the
    same convention: a zone sharing its node map has no ``node_map`` field at all, only
    ``node_map_shared_from``.

Examples:
    Convert an SZL file to ``flow.mat``::

        $ tec2mat flow.szplt

    Convert a PLT file with compression enabled::

        $ tec2mat -c flow.plt

    Convert to an explicit path, writing 1-D arrays as row vectors::

        $ tec2mat -o /tmp/out.mat --oned-as row flow.dat

    Call directly from a Python session::

        import tecio.cli.tec2mat.main as tec2mat

        tec2mat(["-c", "-o", "flow.mat", "flow.szplt"])

    Load the result in MATLAB and read a variable directly::

        d = load('flow.mat');
        x = d.zone_1.var_1;        % first variable of the first zone
        p = d.zone_1.var_3;        % third variable

    Variables shared from another zone are stored once. Resolve them with a small
    helper that follows ``var_shared_from``::

        function v = tecvar(d, zoneIdx, varIdx)
            z = d.(sprintf('zone_%d', zoneIdx));
            src = z.var_shared_from(varIdx);
            if src == 0
                v = z.(sprintf('var_%d', varIdx));
            else
                v = d.(sprintf('zone_%d', src)).(sprintf('var_%d', varIdx));
            end
        end

    Shared FE connectivity resolves the same way, via ``node_map_shared_from``::

        function m = tecnodemap(d, zoneIdx)
            z = d.(sprintf('zone_%d', zoneIdx));
            if isfield(z, 'node_map')
                m = z.node_map;
            else
                m = d.(sprintf('zone_%d', z.node_map_shared_from)).node_map;
            end
        end

See Also:
    * :mod:`tecio.cli.teconvert` - Convert between Tecplot file formats (``.szplt``,
      ``.plt``, ``.dat``) without leaving the Tecplot ecosystem.
    * :mod:`tecio.cli.tecdump` - Inspect the full contents and metadata of a file before
      converting it.

Note:
    ``tec2mat`` requires SciPy (:mod:`scipy.io`). Install it with ``pip install
    scipy`` if it is not already available.

Note:
    :func:`scipy.io.savemat` writes the MAT version 5 format and assembles the whole
    file in memory before writing. Very large datasets are therefore limited by
    available memory and by the ~4 GB-per-variable ceiling of the MAT v5 format.

Note:
    FEPOLYGON and FEPOLYHEDRON zones are written with their variable data, but their
    face-based connectivity cannot be read and the ``node_map`` field is omitted. A
    warning is printed for each such zone.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .. import (
    TecplotFEZoneReader,
    TecplotOrderedZoneReader,
    TecplotReader,
    TecplotZoneReader,
)
from .. import open as tecio_open
from ..libtecio import ZoneType

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------

#: Zone types whose connectivity is face-based and cannot be read; their
#: node map is omitted from the output.
_FE_POLY: frozenset[ZoneType] = frozenset({
    ZoneType.FEPOLYGON,
    ZoneType.FEPOLYHEDRON,
})


# --------------------------------------------------------------------------------------
# Argument parsing
# --------------------------------------------------------------------------------------


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="tec2mat",
        description=(
            # -|-------------------|---------------------------------------------|
            "Convert a Tecplot file to a MATLAB .mat file. Each input file maps\n"
            "to one output file, with every zone stored as a named struct."
        ),
        epilog=(
            "Example usage:\n"
            "  Convert SZL to MAT\n"
            "    $ tec2mat flow.szplt\n"
            "  Convert PLT with compression\n"
            "    $ tec2mat -c flow.plt\n"
            "  Explicit output path with row vectors\n"
            "    $ tec2mat -o out.mat --oned-as row flow.dat\n"
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
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Output file path. Defaults to the input file stem with a "
            ".mat extension in the same directory as the input."
        ),
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        default=False,
        help="Overwrite the output file if it already exists.",
    )
    parser.add_argument(
        "-c",
        "--compress",
        action="store_true",
        default=False,
        dest="compress",
        help="Compress the variable arrays inside the .mat file.",
    )
    parser.add_argument(
        "--oned-as",
        choices=("column", "row"),
        default="column",
        dest="oned_as",
        metavar="{column,row}",
        help=(
            "Orientation for 1-D arrays (FE vectors and per-variable "
            "metadata).  'column' (default) writes Nx1 vectors; 'row' "
            "writes 1xN. Ordered-zone arrays are unaffected."
        ),
    )
    return parser.parse_args(argv)


# --------------------------------------------------------------------------------------
# Conversion helpers
# --------------------------------------------------------------------------------------


def _build_info_dict(reader: TecplotReader) -> dict[str, Any]:
    """Build the dataset-level ``info`` struct for the output file.

    The returned dict is stored by :func:`scipy.io.savemat` as a MATLAB struct named
    ``info``, holding the dataset title, file type, zone/variable counts, and the
    ordered list of real variable names as a MATLAB cell array. The real names are
    kept here as metadata because they are frequently not valid MATLAB field names
    (e.g. ``"X [ft]"``); the per-zone arrays are addressed by 1-based index instead.

    Args:
        reader: An open reader instance (SZL, PLT, or DAT).

    Returns:
        Mapping of field name to value for the ``info`` struct.

    """
    return {
        "title": reader.title,
        "file_type": reader.file_type.name,
        "num_zones": int(reader.num_zones),
        "num_vars": int(reader.num_vars),
        # An object array of strings becomes a MATLAB cell array, preserving
        # names that are not valid MATLAB identifiers.
        "var_names": np.array(reader.variables, dtype=object),
    }


def _zone_to_dict(zone: TecplotZoneReader, num_vars: int) -> dict[str, Any]:
    """Build one zone's struct for the output file.

    The returned dict becomes a MATLAB struct (named ``zone_<n>`` by the caller). It
    contains the zone metadata, one ``var_<k>`` field per dataset variable (1-based),
    and, for simple finite-element zones, the ``node_map`` connectivity array.

    Each ``var_<k>`` holds the variable's data array at its native NumPy dtype, so the
    on-disk precision is preserved on the MATLAB side. Passive and shared variables
    carry no data: their ``var_<k>`` field is an empty matrix ``[]`` and the
    per-variable metadata records how to interpret it:

    * ``var_status[k]``      -- ``'active'``, ``'passive'``, or ``'shared'``.
    * ``var_locations[k]``   -- ``'NODAL'``, ``'CELL_CENTERED'``, or ``''``.
    * ``var_dtypes[k]``      -- on-disk :class:`~tecio.libtecio.DataType` name.
    * ``var_shared_from[k]`` -- 1-based source zone for a shared variable, else ``0``.

    A shared variable is therefore not duplicated on disk; the data lives in the source
    zone (``zone_<var_shared_from>.var_<k>``) and the MATLAB user dereferences it from
    there (see the module docstring for a helper).

    Args:
        zone:     A zone reader instance from any supported format.
        num_vars: Total number of variables in the dataset.

    Returns:
        Mapping of field name to value for this zone's struct.

    """
    d: dict[str, Any] = {
        "title": zone.title,
        "zone_type": zone.zone_type.name,
        "solution_time": float(zone.solution_time),
        "strand_id": int(zone.strand_id),
    }

    if isinstance(zone, TecplotOrderedZoneReader):
        ni, nj, nk = zone.dimensions
        d["I"] = int(ni)
        d["J"] = int(nj)
        d["K"] = int(nk)
    elif isinstance(zone, TecplotFEZoneReader):
        d["num_nodes"] = int(zone.num_nodes)
        d["num_elements"] = int(zone.num_elements)

    # Empty placeholder for passive/shared variables -> MATLAB [].
    empty = np.array([])

    status: list[str] = []
    locations: list[str] = []
    dtypes: list[str] = []
    shared_from = np.zeros(num_vars, dtype=np.int32)

    for j in range(num_vars):
        var = zone.variable[j]

        loc = var.value_location
        locations.append(loc.name if loc is not None else "")
        dtypes.append(var.data_type.name)

        if var.is_passive():
            d[f"var_{j + 1}"] = empty
            status.append("passive")
            continue

        src = var.shared_zone
        if src is not None:
            # Stored once in the source zone; record the 1-based source so the
            # MATLAB user can dereference zone_<src>.var_<k>.
            d[f"var_{j + 1}"] = empty
            shared_from[j] = int(src)
            status.append("shared")
            continue

        arr = var.values
        if arr is None or arr.size == 0:
            # No data available (e.g. an empty array from the DAT reader) --
            # treat as passive so the slot is well defined.
            d[f"var_{j + 1}"] = empty
            status.append("passive")
            continue

        # Active variable: store the array verbatim at its native dtype.
        d[f"var_{j + 1}"] = arr
        status.append("active")

    # Object arrays of strings become MATLAB cell arrays.
    d["var_status"] = np.array(status, dtype=object)
    d["var_locations"] = np.array(locations, dtype=object)
    d["var_dtypes"] = np.array(dtypes, dtype=object)
    d["var_shared_from"] = shared_from

    # Connectivity for simple FE zones only. Ordered zones have no node map, and poly
    # zones expose none through the readers.
    #
    # A zone that shares connectivity is handled the same way as a shared variable
    # above: rather than duplicating the (potentially large) node map into every zone's
    # struct, store the 1-based source zone number so the MATLAB user can dereference
    # zone_<src>.node_map themselves.
    if isinstance(zone, TecplotFEZoneReader) and zone.zone_type not in _FE_POLY:
        con_src = zone.shared_connectivity
        if con_src is not None:
            d["node_map_shared_from"] = np.int32(con_src)
        else:
            node_map = zone.node_map
            if node_map is not None:
                d["node_map"] = node_map

    return d


# --------------------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """Convert a Tecplot file to a MATLAB ``.mat`` file.

    Returns:
        Exit code -- ``0`` on success, ``1`` on error.

    """
    args = _parse_args(argv)

    # SciPy is an optional dependency, imported lazily so that merely importing this
    # module (e.g. to call main() from a script) does not require SciPy unless the tool
    # is actually run.
    try:
        from scipy import io as scipy_io
    except ImportError:
        print(
            "Error: tec2mat requires SciPy, which is not installed.  "
            "Install it with: pip install scipy",
            file=sys.stderr,
        )
        return 1

    src = Path(args.filename)
    if not src.exists():
        print(f"Error: input file not found: {src}", file=sys.stderr)
        return 1

    dst = Path(args.output) if args.output is not None else src.with_suffix(".mat")

    if dst.exists() and not args.force:
        print(
            f"Error: output file already exists: {dst}\nUse --force to overwrite.",
            file=sys.stderr,
        )
        return 1

    print(f"Converting: {src}  ->  {dst}")

    try:
        with tecio_open(str(src), "r") as reader:
            num_vars: int = reader.num_vars

            print(f"  Title     : {reader.title}")
            print(f"  Variables : {num_vars} {reader.variables}")
            print(f"  Zones     : {reader.num_zones}")

            mdict: dict[str, Any] = {"info": _build_info_dict(reader)}

            for i, zone in enumerate(reader.zone, start=1):
                if zone.zone_type in _FE_POLY:
                    print(
                        f"Warning: zone {i} ('{zone.title}') is "
                        f"{zone.zone_type.name}; its connectivity cannot be read "
                        "and the node map will be omitted.",
                        file=sys.stderr,
                    )
                mdict[f"zone_{i}"] = _zone_to_dict(zone, num_vars)

            scipy_io.savemat(
                str(dst),
                mdict,
                do_compression=args.compress,
                oned_as=args.oned_as,
                long_field_names=True,
            )

    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        dst.unlink(missing_ok=True)
        return 1

    print(f"Done. Output written to: {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
