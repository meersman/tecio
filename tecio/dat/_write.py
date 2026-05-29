r"""Tecplot ASCII DAT file writer API.

Supported ``DATAPACKING`` modes for writing:
    * ``BLOCK`` — one contiguous value block per variable (default).
    * ``POINT`` — one row of all active nodal variable values per node,
      followed by a separate row-per-cell section for cell-centred variables.
      Pass ``datapacking="POINT"`` to :meth:`~Write.write_ijk_zone` or
      :meth:`~Write.write_fe_zone`.
"""

from __future__ import annotations

import contextlib

# Standard library
import io
import os
from collections.abc import Sequence
from typing import Any

# Third-party
import numpy as np
import numpy.typing as npt

from ..libtecio import (
    DataPacking,
    DataType,
    FaceNeighborMode,
    FileType,
    ValueLocation,
    ZoneType,
)

# ---------------------------------------------------------------------------
# Shared module-level constants
# ---------------------------------------------------------------------------

#: FE zone types fully supported for reading and writing.
_FE_SIMPLE: frozenset[ZoneType] = frozenset({
    ZoneType.FELINESEG,
    ZoneType.FETRIANGLE,
    ZoneType.FEQUADRILATERAL,
    ZoneType.FETETRAHEDRON,
    ZoneType.FEBRICK,
})

#: Zone types whose connectivity is face-based (not yet supported).
_FE_POLY: frozenset[ZoneType] = frozenset({
    ZoneType.FEPOLYGON,
    ZoneType.FEPOLYHEDRON,
})

#: Nodes per element for each supported simple FE type.
_NODES_PER_ELEM: dict[ZoneType, int] = {
    ZoneType.FELINESEG: 2,
    ZoneType.FETRIANGLE: 3,
    ZoneType.FEQUADRILATERAL: 4,
    ZoneType.FETETRAHEDRON: 4,
    ZoneType.FEBRICK: 8,
}

#: ASCII keyword → ZoneType (lower-cased at parse time).
_STR_TO_ZONETYPE: dict[str, ZoneType] = {
    "ordered": ZoneType.ORDERED,
    "felineseg": ZoneType.FELINESEG,
    "fetriangle": ZoneType.FETRIANGLE,
    "fequadrilateral": ZoneType.FEQUADRILATERAL,
    "fetetrahedron": ZoneType.FETETRAHEDRON,
    "febrick": ZoneType.FEBRICK,
    "fepolygon": ZoneType.FEPOLYGON,
    "fepolyhedron": ZoneType.FEPOLYHEDRON,
}

#: ASCII keyword → FileType.
_STR_TO_FILETYPE: dict[str, FileType] = {
    "full": FileType.FULL,
    "grid": FileType.GRID,
    "solution": FileType.SOLUTION,
}

#: ZoneType → ASCII keyword (for writing).
_ZONETYPE_STR: dict[ZoneType, str] = {
    ZoneType.ORDERED: "Ordered",
    ZoneType.FELINESEG: "FELineSeg",
    ZoneType.FETRIANGLE: "FETriangle",
    ZoneType.FEQUADRILATERAL: "FEQuadrilateral",
    ZoneType.FETETRAHEDRON: "FETetrahedron",
    ZoneType.FEBRICK: "FEBrick",
    ZoneType.FEPOLYGON: "FEPolygon",
    ZoneType.FEPOLYHEDRON: "FEPolyhedron",
}

#: FileType → ASCII keyword (FULL omitted from output).
_FILETYPE_STR: dict[FileType, str] = {
    FileType.GRID: "GRID",
    FileType.SOLUTION: "SOLUTION",
}

#: DataType → NumPy dtype string (used by Write for casting).
_DT_TO_DTYPE: dict[DataType, str] = {
    DataType.FLOAT: "f4",
    DataType.DOUBLE: "f8",
    DataType.INT32: "i4",
    DataType.INT16: "i2",
    DataType.BYTE: "u1",
}

#: Values per line for Write data blocks.
_VALUES_PER_LINE: int = 5


# ===========================================================================
# Shared internal helpers
# ===========================================================================


def _quote(s: str) -> str:
    """Wrap *s* in double-quotes, escaping embedded double-quotes."""
    return '"' + str(s).replace('"', '\\"') + '"'


def _unquote(s: str) -> str:
    r"""Remove surrounding double-quotes and unescape internal ``\\"``."""
    s = s.strip()
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        return s[1:-1].replace('\\"', '"')
    return s


def _strip_comment(line: str) -> str:
    """Remove a Tecplot ``#`` comment and trailing whitespace."""
    idx = line.find("#")
    return line[:idx].rstrip() if idx >= 0 else line.rstrip()


def _infer_data_type(arr: npt.NDArray) -> DataType:
    """Return the most appropriate :class:`DataType` for *arr*'s dtype."""
    dtype = arr.dtype
    if dtype.kind == "f":
        return DataType.DOUBLE if dtype.itemsize >= 8 else DataType.FLOAT
    if dtype.kind in ("i", "u"):
        if dtype.itemsize >= 4:
            return DataType.INT32
        if dtype.itemsize == 2:
            return DataType.INT16
        return DataType.BYTE
    return DataType.FLOAT


# ===========================================================================
# Write helpers
# ===========================================================================


def _make_float_fmt(sig_digits: int) -> str:
    """Return a ``format()``-compatible scientific-notation format string."""
    return f".{max(sig_digits - 1, 0)}e"


def _stage_float_array(buf: io.StringIO, arr: npt.NDArray, fmt: str) -> None:
    """Write a 1-D float array to *buf* in scientific notation."""
    flat = np.asarray(arr).ravel()
    vpl = _VALUES_PER_LINE
    for start in range(0, flat.size, vpl):
        chunk = flat[start : start + vpl]
        buf.write("\t".join(format(float(v), fmt) for v in chunk) + "\n")


def _stage_point_rows(
    buf: io.StringIO,
    cols: list[npt.NDArray],
    fmt: str,
) -> None:
    """Write *cols* as ``DATAPACKING=POINT`` rows into *buf*.

    Each element of *cols* is the flat value array for one variable.  One
    tab-separated row is written per point (node or cell): all variable values
    for that point appear on the same line.  All column arrays must have the
    same length.  Does nothing when *cols* is empty.

    Example:
        >>> _stage_point_rows(buf, [x_flat, y_flat, p_flat], ".8e")
    """
    if not cols:
        return
    # Stack into a (n_points, n_vars) matrix then write row by row.
    matrix = np.column_stack(cols) if len(cols) > 1 else cols[0].reshape(-1, 1)
    for row in matrix:
        buf.write("\t".join(format(float(v), fmt) for v in row) + "\n")


def _stage_connectivity_row(buf: io.StringIO, row: npt.NDArray) -> None:
    """Write one element's node indices to *buf* as space-separated ints."""
    buf.write(" ".join(str(int(n)) for n in row) + "\n")


# ===========================================================================
# Write class
# ===========================================================================


class Write:
    r"""Context-manager writer for Tecplot 360 ASCII (``.dat``) files.

    The public interface is identical to :class:`szl.Write` and
    :class:`plt.Write`.

    Parameters:
        path:
            Destination file path.
        title:
            Dataset title.  Defaults to ``"untitled"``.
        variables:
            Variable name list.  ``None`` defers file creation until the first
            zone-writing call (lazy open).
        file_type:
            :class:`FileType` enum.  Defaults to :attr:`FileType.FULL`.
        sig_digits:
            Significant digits for scientific-notation float output.  Default
            is ``9``; use ``17`` for full ``float64`` round-trip fidelity.

    Attributes:
        auxdataset : dict[str, str]
            Dataset-level auxiliary data buffer.
        auxvar : dict[int | str, dict[str, str]]
            Variable-level auxiliary data buffer.
        current_zone : int
            Count of successfully written zones.
    """

    path: str
    """Output file path."""

    title: str
    """Dataset title string."""

    variables: list[str] | None
    """Variable name list, or ``None`` if the file has not been opened yet."""

    file_type: FileType
    """File type (FULL, GRID, or SOLUTION)."""

    current_zone: int
    """Count of successfully written zones."""

    SIG_DIGITS: int = 9

    def __init__(
        self,
        path: str,
        title: str = "untitled",
        variables: list[str] | None = None,
        file_type: FileType = FileType.FULL,
        sig_digits: int | None = None,
    ) -> None:
        """Store configuration; open the file immediately if *variables* given."""
        self.path: str = str(path)
        self.title: str = title
        self.variables: list[str] | None = variables
        self.file_type: FileType = file_type
        self.current_zone: int = 0
        self.auxdataset: dict[str, str] = {}
        self.auxvar: dict[int | str, dict[str, str]] = {}

        self._fp: io.TextIOWrapper | None = None
        self._opened: bool = False
        self._sig_digits: int = (
            sig_digits if sig_digits is not None else self.SIG_DIGITS
        )
        self._float_fmt: str = _make_float_fmt(self._sig_digits)

        if self.variables is not None:
            self._open(self.variables)

    # -- context manager ------------------------------------------------------

    def __enter__(self) -> Write:
        """Support ``with`` statement."""
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Close the file on context-manager exit."""
        try:
            self.close()
        except Exception:
            if exc_type is None:
                raise

    # -- lifecycle ------------------------------------------------------------

    def _open(self, var_names: list[str]) -> None:
        """Open the output file and write the dataset header.

        Raises:
            ValueError: If ``var_names`` is empty.
        """
        if not var_names:
            raise ValueError("Write requires at least one variable name.")
        self.variables = var_names
        self._fp = open(self.path, "w", encoding="utf-8", newline="\n")  # noqa: SIM115
        self._write_file_header()
        self._opened = True

    def close(self) -> None:
        """Flush and close the output file (safe to call more than once)."""
        if self._fp is not None:
            self._fp.flush()
            self._fp.close()
            self._fp = None
            self._opened = False

    def flush_aux(self) -> None:
        """Write buffered dataset- and variable-level aux data to the file.

        Called automatically before the first zone.

        Raises:
            IOError: If the file has not been opened yet.
        """
        if self._fp is None:
            raise OSError("flush_aux() called before the file was opened.")

        for name, value in self.auxdataset.items():
            self._fp.write(f"DATASETAUXDATA {name}={_quote(value)}\n")

        for key, subdict in self.auxvar.items():
            if isinstance(key, int):
                var_idx = key - 1
                if var_idx not in range(len(self.variables)):
                    raise IndexError(
                        f"Variable index {key} out of bounds [1, {len(self.variables)}]"
                    )
            elif isinstance(key, str):
                try:
                    var_idx = self.variables.index(key)
                except ValueError as exc:
                    raise KeyError(
                        f"Variable aux data key {key!r} not found in "
                        f"variable list ({self.variables})"
                    ) from exc
            else:
                raise TypeError(f"Aux data key must be str or 1-based int, got {key!r}")

            one_based = var_idx + 1
            for name, value in subdict.items():
                self._fp.write(f"VARAUXDATA {one_based} {name}={_quote(value)}\n")

        self.auxdataset.clear()
        self.auxvar.clear()

    def add_auxdataset_dict(self, auxdict: dict[str, Any]) -> None:
        """Buffer dataset-level auxiliary data from input dictionary."""
        self.auxdataset.update(auxdict)

    def add_auxvar_dict(self, auxdict: dict[int, dict[str, Any]]) -> None:
        """Buffer variable-level auxiliary data from input dictionary."""
        self.auxvar.update(auxdict)

    # -- zone writers ---------------------------------------------------------

    def write_ijk_zone(
        self,
        data: Sequence[npt.NDArray],
        title: str | None = None,
        variables: list[str] | None = None,
        value_locations: Sequence[ValueLocation] | None = None,
        passive_vars: Sequence[bool | int] | None = None,
        var_sharing: Sequence[int] | None = None,
        solution_time: float = 0.0,
        strand_id: int = 0,
        aux: dict[str, Any] | None = None,
        datapacking: DataPacking | str = DataPacking.BLOCK,
    ) -> None:
        """Write a complete IJK-ordered zone.

        Zone dimensions are inferred from the shape of the first NODAL array.

        Parameters:
            datapacking:
                :class:`~tecio.libtecio.DataPacking` member or equivalent
                string (``"BLOCK"`` or ``"POINT"``).  POINT format writes one
                row of all nodal variable values per node, which many
                third-party tools read as plain CSV rows.  A separate
                row-per-cell section follows for any cell-centred variables.

        Raises:
            ValueError: On variable-count or array-shape mismatch, or if
                *datapacking* is not a recognised value.
        """
        if isinstance(datapacking, str):
            try:
                datapacking = DataPacking[datapacking.upper()]
            except KeyError:
                raise ValueError(
                    f"datapacking={datapacking!r} is not supported; "
                    "use DataPacking.BLOCK, DataPacking.POINT, or their "
                    "string equivalents."
                ) from None
        if title is None:
            title = f"IJK_Zone_{self.current_zone + 1}"
        if variables is None:
            variables = [f"V{i}" for i in range(1, len(data) + 1)]

        if not self._opened:
            self._open(variables)
            self.flush_aux()

        variable_types = [_infer_data_type(np.asarray(arr)) for arr in data]

        if value_locations is None:
            value_locations = [ValueLocation.NODAL] * len(data)
        if passive_vars is None:
            passive_vars = [False] * len(self.variables)
        if var_sharing is None:
            var_sharing = [0] * len(self.variables)

        # Validate count
        if len(data) != len(self.variables):
            expected = sum(
                1
                for p, s in zip(passive_vars, var_sharing, strict=True)
                if not p and not s
            )
            if len(data) != expected:
                self._handle_zone_error()
                raise ValueError(
                    f"Expected {expected} data arrays for active variables, "
                    f"got {len(data)}."
                )

        # Infer dimensions
        nodal = [
            i for i, loc in enumerate(value_locations) if loc == ValueLocation.NODAL
        ]
        cell = [
            i
            for i, loc in enumerate(value_locations)
            if loc == ValueLocation.CELL_CENTERED
        ]

        if nodal:
            ref = np.asarray(data[nodal[0]])
            ndims = ref.ndim
            if ndims not in (1, 2, 3):
                self._handle_zone_error()
                raise ValueError(f"Arrays must be 1D, 2D, or 3D; got {ndims}D.")
            nodal_shape = ref.shape + (1,) * (3 - ndims)
            cell_shape = tuple(max(d - 1, 1) for d in nodal_shape)
            imax, jmax, kmax = nodal_shape
        elif cell:
            ref = np.asarray(data[cell[0]])
            ndims = ref.ndim
            if ndims not in (1, 2, 3):
                self._handle_zone_error()
                raise ValueError(f"Arrays must be 1D, 2D, or 3D; got {ndims}D.")
            cell_shape = ref.shape + (1,) * (3 - ndims)
            nodal_shape = tuple(max(d + 1, 1) for d in cell_shape)
            imax, jmax, kmax = nodal_shape
        else:
            self._handle_zone_error()
            raise ValueError("Could not determine zone dimensions.")

        # Validate shapes
        for i, (arr, loc) in enumerate(zip(data, value_locations, strict=True)):
            arr_np = np.asarray(arr)
            if arr_np.ndim != ndims:
                self._handle_zone_error()
                raise ValueError(f"Array {i} is {arr_np.ndim}D, expected {ndims}D.")
            shape = arr_np.shape + (1,) * (3 - arr_np.ndim)
            if loc == ValueLocation.NODAL and shape != nodal_shape:
                self._handle_zone_error()
                raise ValueError(
                    f"Array {i} is NODAL but has shape {shape}, expected {nodal_shape}."
                )
            if loc == ValueLocation.CELL_CENTERED and shape != cell_shape:
                self._handle_zone_error()
                raise ValueError(
                    f"Array {i} is CELL_CENTERED but has shape {shape}, "
                    f"expected {cell_shape}."
                )

        active_var_idx = [
            vi
            for vi, (p, s) in enumerate(
                zip(passive_vars, var_sharing, strict=True), start=1
            )
            if not p and not s
        ]
        vl_global = [ValueLocation.NODAL] * len(self.variables)
        for li, vi in enumerate(active_var_idx):
            vl_global[vi - 1] = value_locations[li]

        buf = io.StringIO()
        self._stage_zone_header(
            buf,
            title,
            ZoneType.ORDERED,
            imax=imax,
            jmax=jmax,
            kmax=kmax,
            solution_time=solution_time,
            strand_id=strand_id,
            passive_vars=passive_vars,
            var_sharing=var_sharing,
            value_locations_global=vl_global,
            aux=aux,
            datapacking=datapacking,
        )

        if datapacking == DataPacking.POINT:
            # Nodal section: one row per node, all nodal vars per row.
            nodal_cols = [
                np.asarray(arr, dtype=_DT_TO_DTYPE[dt]).ravel(order="F")
                for arr, dt, loc in zip(
                    data, variable_types, value_locations, strict=True
                )
                if loc == ValueLocation.NODAL
            ]
            _stage_point_rows(buf, nodal_cols, self._float_fmt)
            # Cell-centred section: one row per cell, all CC vars per row.
            cc_cols = [
                np.asarray(arr, dtype=_DT_TO_DTYPE[dt]).ravel(order="F")
                for arr, dt, loc in zip(
                    data, variable_types, value_locations, strict=True
                )
                if loc == ValueLocation.CELL_CENTERED
            ]
            _stage_point_rows(buf, cc_cols, self._float_fmt)
        else:
            for arr, dt in zip(data, variable_types, strict=False):
                cast = np.asarray(arr, dtype=_DT_TO_DTYPE[dt]).ravel(order="F")
                _stage_float_array(buf, cast, self._float_fmt)

        self._fp.write(buf.getvalue())
        self.current_zone += 1

    def write_fe_zone(
        self,
        zone_type: ZoneType,
        data: Sequence[npt.NDArray],
        node_map: npt.ArrayLike | None = None,
        title: str | None = None,
        variables: list[str] | None = None,
        value_locations: Sequence[ValueLocation] | None = None,
        passive_vars: Sequence[bool | int] | None = None,
        var_sharing: Sequence[int] | None = None,
        con_sharing: int = 0,
        face_neighbors: npt.ArrayLike | None = None,
        face_nbr_mode: FaceNeighborMode = FaceNeighborMode.LOCAL_ONE_TO_ONE,
        solution_time: float = 0.0,
        strand_id: int = 0,
        aux: dict[str, Any] | None = None,
        datapacking: DataPacking | str = DataPacking.BLOCK,
    ) -> None:
        """Write a complete finite-element zone.

        ``FEPOLYGON`` and ``FEPOLYHEDRON`` raise :exc:`NotImplementedError`.

        Parameters:
            datapacking:
                :class:`~tecio.libtecio.DataPacking` member or equivalent
                string (``"BLOCK"`` or ``"POINT"``).  POINT format writes one
                row of all nodal variable values per node, followed by a
                separate row-per-cell section for cell-centred variables.

        Raises:
            NotImplementedError: For unsupported zone types.
            ValueError: On variable-count or array-length mismatch, or if
                *datapacking* is not a recognised value.
        """
        if isinstance(datapacking, str):
            try:
                datapacking = DataPacking[datapacking.upper()]
            except KeyError:
                raise ValueError(
                    f"datapacking={datapacking!r} is not supported; "
                    "use DataPacking.BLOCK, DataPacking.POINT, or their "
                    "string equivalents."
                ) from None
        if zone_type in _FE_POLY:
            raise NotImplementedError(
                f"Zone type {zone_type.name!r} is not supported by write_fe_zone."
            )
        if zone_type not in _FE_SIMPLE:
            raise NotImplementedError(
                f"Zone type {zone_type!r} is not supported by write_fe_zone."
            )

        if title is None:
            title = f"FE_Zone_{self.current_zone + 1}"
        if variables is None:
            variables = [f"V{i}" for i in range(1, len(data) + 1)]

        if not self._opened:
            self._open(variables)
            self.flush_aux()

        variable_types = [_infer_data_type(np.asarray(arr)) for arr in data]

        if value_locations is None:
            value_locations = [ValueLocation.NODAL] * len(data)
        if passive_vars is None:
            passive_vars = [False] * len(self.variables)
        if var_sharing is None:
            var_sharing = [0] * len(self.variables)

        # Validate count
        if len(data) != len(self.variables):
            expected = sum(
                1
                for p, s in zip(passive_vars, var_sharing, strict=True)
                if not p and not s
            )
            if expected == 0:
                self._handle_zone_error()
                raise ValueError("No active variables to write.")
            if len(data) == 0 or len(data) != expected:
                self._handle_zone_error()
                raise ValueError(
                    f"Expected {expected} data arrays for active variables, "
                    f"got {len(data)}."
                )

        node_map_arr = np.asarray(node_map)
        num_cells = int(node_map_arr.shape[0])
        num_nodes = int(node_map_arr.max())

        for i, (arr, loc) in enumerate(zip(data, value_locations, strict=True)):
            arr_np = np.asarray(arr)
            if loc == ValueLocation.NODAL and arr_np.size != num_nodes:
                self._handle_zone_error()
                raise ValueError(
                    f"Array {i} is NODAL but has {arr_np.size} values; "
                    f"expected {num_nodes}."
                )
            if loc == ValueLocation.CELL_CENTERED and arr_np.size != num_cells:
                self._handle_zone_error()
                raise ValueError(
                    f"Array {i} is CELL_CENTERED but has {arr_np.size} values; "
                    f"expected {num_cells}."
                )

        active_var_idx = [
            vi
            for vi, (p, s) in enumerate(
                zip(passive_vars, var_sharing, strict=True), start=1
            )
            if not p and not s
        ]
        vl_global = [ValueLocation.NODAL] * len(self.variables)
        for li, vi in enumerate(active_var_idx):
            vl_global[vi - 1] = value_locations[li]

        buf = io.StringIO()
        self._stage_zone_header(
            buf,
            title,
            zone_type,
            num_nodes=num_nodes,
            num_elements=num_cells,
            solution_time=solution_time,
            strand_id=strand_id,
            passive_vars=passive_vars,
            var_sharing=var_sharing,
            value_locations_global=vl_global,
            con_sharing=con_sharing,
            aux=aux,
            datapacking=datapacking,
        )

        if datapacking == DataPacking.POINT:
            # Nodal section: one row per node, all nodal vars per row.
            nodal_cols = [
                np.asarray(arr, dtype=_DT_TO_DTYPE[dt]).ravel()
                for arr, dt, loc in zip(
                    data, variable_types, value_locations, strict=True
                )
                if loc == ValueLocation.NODAL
            ]
            _stage_point_rows(buf, nodal_cols, self._float_fmt)
            # Cell-centred section: one row per cell, all CC vars per row.
            cc_cols = [
                np.asarray(arr, dtype=_DT_TO_DTYPE[dt]).ravel()
                for arr, dt, loc in zip(
                    data, variable_types, value_locations, strict=True
                )
                if loc == ValueLocation.CELL_CENTERED
            ]
            _stage_point_rows(buf, cc_cols, self._float_fmt)
        else:
            for arr, dt in zip(data, variable_types, strict=False):
                cast = np.asarray(arr, dtype=_DT_TO_DTYPE[dt]).ravel()
                _stage_float_array(buf, cast, self._float_fmt)

        if not con_sharing:
            conn = np.asarray(node_map, dtype=np.intp).reshape(
                num_cells, _NODES_PER_ELEM[zone_type]
            )
            for row in conn:
                _stage_connectivity_row(buf, row)

        self._fp.write(buf.getvalue())
        self.current_zone += 1

    # -- private helpers ------------------------------------------------------

    def _handle_zone_error(self) -> None:
        """Delete the output file if no zone has been committed yet."""
        if self.current_zone == 0:
            if self._fp is not None:
                self._fp.close()
                self._fp = None
                self._opened = False
            with contextlib.suppress(OSError):
                os.remove(self.path)

    def _write_file_header(self) -> None:
        """Write TITLE, optional FILETYPE, and VARIABLES lines."""
        fp = self._fp
        fp.write(f"TITLE     = {_quote(self.title)}\n")
        if self.file_type in _FILETYPE_STR:
            fp.write(f"FILETYPE  = {_FILETYPE_STR[self.file_type]}\n")
        var_lines = "\n".join(_quote(v) for v in self.variables)
        fp.write(f"VARIABLES = {var_lines}\n")

    def _stage_zone_header(
        self,
        buf: io.StringIO,
        title: str,
        zone_type: ZoneType,
        imax: int = 0,
        jmax: int = 0,
        kmax: int = 0,
        num_nodes: int = 0,
        num_elements: int = 0,
        solution_time: float = 0.0,
        strand_id: int = 0,
        passive_vars: Sequence[bool | int] | None = None,
        var_sharing: Sequence[int] | None = None,
        value_locations_global: list[ValueLocation] | None = None,
        con_sharing: int = 0,
        aux: dict[str, Any] | None = None,
        datapacking: DataPacking = DataPacking.BLOCK,
    ) -> None:
        """Write a ``ZONE`` header block into the staging buffer *buf*."""
        zt_str = _ZONETYPE_STR[zone_type]

        buf.write(f"ZONE T={_quote(title)}\n")
        buf.write(f" STRANDID={strand_id}, SOLUTIONTIME={float(solution_time)}\n")

        if zone_type == ZoneType.ORDERED:
            buf.write(f" I={imax}, J={jmax}, K={kmax}, ZONETYPE={zt_str}\n")
        else:
            buf.write(
                f" Nodes={num_nodes}, Elements={num_elements}, ZONETYPE={zt_str}\n"
            )

        buf.write(f" DATAPACKING={datapacking.name}\n")

        if value_locations_global:
            cc = [
                i + 1
                for i, loc in enumerate(value_locations_global)
                if loc == ValueLocation.CELL_CENTERED
            ]
            if cc:
                buf.write(
                    f" VARLOCATION=([{','.join(str(i) for i in cc)}]=CELLCENTERED)\n"
                )

        if passive_vars:
            pidx = [str(i + 1) for i, f in enumerate(passive_vars) if f]
            if pidx:
                buf.write(f" PASSIVEVARLIST=[{','.join(pidx)}]\n")

        if var_sharing and any(var_sharing):
            entries = [f"[{i + 1}]={z}" for i, z in enumerate(var_sharing) if z]
            if entries:
                buf.write(f" VARSHARELIST=({','.join(entries)})\n")

        if con_sharing:
            buf.write(f" CONNECTIVITYSHAREZONE={con_sharing}\n")

        if aux:
            for name, value in aux.items():
                buf.write(f" AUXDATA {name}={_quote(value)}\n")
