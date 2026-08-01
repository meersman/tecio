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

from .._meta import WriterMeta, ZoneMeta
from ..libtecio import (
    DataPacking,
    DataType,
    FaceNeighborMode,
    FileType,
    ValueLocation,
    ZoneType,
)

# -------------------------------------------------------------------------------------
# Module-level constants
# -------------------------------------------------------------------------------------

# FE zone types fully supported for reading and writing
_FE_SIMPLE: frozenset[ZoneType] = frozenset({
    ZoneType.FELINESEG,
    ZoneType.FETRIANGLE,
    ZoneType.FEQUADRILATERAL,
    ZoneType.FETETRAHEDRON,
    ZoneType.FEBRICK,
})

# Zone types whose connectivity is face-based
_FE_POLY: frozenset[ZoneType] = frozenset({
    ZoneType.FEPOLYGON,
    ZoneType.FEPOLYHEDRON,
})

# Nodes per element for each supported simple FE type
_NODES_PER_ELEM: dict[ZoneType, int] = {
    ZoneType.FELINESEG: 2,
    ZoneType.FETRIANGLE: 3,
    ZoneType.FEQUADRILATERAL: 4,
    ZoneType.FETETRAHEDRON: 4,
    ZoneType.FEBRICK: 8,
}

# ASCII keyword to ZoneType
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

# ASCII keyword to FileType
_STR_TO_FILETYPE: dict[str, FileType] = {
    "full": FileType.FULL,
    "grid": FileType.GRID,
    "solution": FileType.SOLUTION,
}

# ZoneType  ASCII keyword
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

# FileType to ASCII keyword
_FILETYPE_STR: dict[FileType, str] = {
    FileType.GRID: "GRID",
    FileType.SOLUTION: "SOLUTION",
}

# DataType to NumPy dtype string
_DT_TO_DTYPE: dict[DataType, str] = {
    DataType.FLOAT: "f4",
    DataType.DOUBLE: "f8",
    DataType.INT32: "i4",
    DataType.INT16: "i2",
    DataType.BYTE: "u1",
}

# DataType to ASCII DT= keyword
_DATATYPE_STR: dict[DataType, str] = {
    DataType.FLOAT: "SINGLE",
    DataType.DOUBLE: "DOUBLE",
    DataType.INT32: "LONGINT",
    DataType.INT16: "SHORTINT",
    DataType.BYTE: "BYTE",
}

# ASCII DT= keyword (and common aliases) to DataType, case-insensitive lookup
_STR_TO_DATATYPE: dict[str, DataType] = {
    "single": DataType.FLOAT,
    "float": DataType.FLOAT,
    "double": DataType.DOUBLE,
    "longint": DataType.INT32,
    "shortint": DataType.INT16,
    "byte": DataType.BYTE,
}

# Significant digits that guarantee a lossless round-trip for each floating precision
# (matches the IEEE 754 theoretical bounds)
_SIG_DIGITS_FOR_PRECISION: dict[DataType, int] = {
    DataType.FLOAT: 9,
    DataType.DOUBLE: 17,
}

# Values per line for Write data blocks
_VALUES_PER_LINE: int = 5

#: Indentation applied to every zone-header line after the first
_INDENT: str = "  "

# Indentation applied to data and connectivity lines
_DATA_INDENT: str = _INDENT + "  "

#: Separator written between adjacent values on a data line
_VALUE_SEP: str = "  "


# =====================================================================================
# Helpers
# =====================================================================================


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


def _normalize_precision(precision: DataType | str) -> DataType:
    """Return the :class:`DataType` for *precision*, accepting a string alias.

    Accepts the :class:`DataType` enum directly, or a case-insensitive string.

    Raises:
        ValueError: If *precision* isn't FLOAT/DOUBLE (or a recognized
                    string alias for one of them).
    """
    if isinstance(precision, str):
        try:
            precision = _STR_TO_DATATYPE[precision.strip().lower()]
        except KeyError:
            raise ValueError(
                f"precision={precision!r} is not recognized; use 'single' or "
                "'double' (or DataType.FLOAT / DataType.DOUBLE)."
            ) from None
    if precision not in (DataType.FLOAT, DataType.DOUBLE):
        raise ValueError(
            f"precision={precision!r} is not supported; precision only "
            "applies to floating-point variables -- use DataType.FLOAT or "
            "DataType.DOUBLE."
        )
    return precision


def _resolve_written_type(inferred: DataType, precision: DataType) -> DataType:
    """Return the :class:`DataType` actually written for one variable.

    *precision* overrides *inferred* only when *inferred* is itself a floating-point
    type (FLOAT or DOUBLE). Integer-inferred variables (INT32/INT16/BYTE) always keep
    their own inferred type, unaffected by *precision*. A variable holding a meaningful
    integer (a CPU number, an index, a count) should never be silently coerced by a
    setting that's conceptually about floating-point precision.
    """
    if inferred in (DataType.FLOAT, DataType.DOUBLE):
        return precision
    return inferred


def _make_float_fmt(sig_digits: int) -> str:
    """Return a ``format()``-compatible scientific-notation format string."""
    return f" .{max(sig_digits - 1, 0)}e"


def _stage_float_array(buf: io.StringIO, arr: npt.NDArray, fmt: str) -> None:
    """Write a 1-D float array to *buf* in scientific notation.

    Values are written ``_VALUES_PER_LINE`` per line, each line indented by
    ``_DATA_INDENT`` and the values separated by ``_VALUE_SEP`` so that columns align.
    """
    flat = np.asarray(arr).ravel()
    vpl = _VALUES_PER_LINE
    for start in range(0, flat.size, vpl):
        chunk = flat[start : start + vpl]
        buf.write(
            _DATA_INDENT + _VALUE_SEP.join(format(float(v), fmt) for v in chunk) + "\n"
        )


def _stage_point_rows(
    buf: io.StringIO,
    cols: list[npt.NDArray],
    fmt: str,
) -> None:
    """Write *cols* as ``DATAPACKING=POINT`` rows into *buf*.

    Each element of *cols* is the flat value array for one variable.  One tab-separated
    row is written per point (node or cell): all variable values for that point appear
    on the same line.  All column arrays must have the same length.  Does nothing when
    *cols* is empty.

    Example:
        >>> _stage_point_rows(buf, [x_flat, y_flat, p_flat], ".8e")
    """
    if not cols:
        return
    # Stack into a (n_points, n_vars) matrix then write row by row.
    matrix = np.column_stack(cols) if len(cols) > 1 else cols[0].reshape(-1, 1)
    for row in matrix:
        buf.write(
            _DATA_INDENT + _VALUE_SEP.join(format(float(v), fmt) for v in row) + "\n"
        )


def _stage_connectivity_row(buf: io.StringIO, row: npt.NDArray) -> None:
    """Write one element's node indices to *buf* as space-separated ints."""
    buf.write(_DATA_INDENT + " ".join(str(int(n)) for n in row) + "\n")


# =====================================================================================
# Write class
# =====================================================================================


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
        precision:
            Uniform floating point precision for the whole file.  :attr:`DataType.FLOAT`
            (default; ``"single"``) or :attr:`DataType.DOUBLE` (``"double"``). Also
            accepts those strings directly (``"single"``/``"double"``, or ``"float"``
            for the enum's own name). Applies only to floating-point variables: a
            ``float64`` array is downcast to ``SINGLE`` under the default, but integer
            variables (LONGINT/ SHORTINT/BYTE) always keep their own inferred type in
            the zone's ``DT=`` declaration regardless of *precision*. Also sets the
            significant digit count used for every variable uniformly (9 for FLOAT, 17
            for DOUBLE).

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

    precision: DataType
    """Uniform floating-point precision for the file."""

    current_zone: int
    """Count of successfully written zones."""

    def __init__(
        self,
        path: str,
        title: str = "untitled",
        variables: list[str] | None = None,
        file_type: FileType = FileType.FULL,
        *,
        precision: DataType | str = DataType.FLOAT,
    ) -> None:
        """Store configuration; open the file immediately if *variables* given.

        Raises:
            ValueError: If *precision* is not :attr:`DataType.FLOAT` /
                        :attr:`DataType.DOUBLE` (or a recognized string alias for one of
                        them).
        """
        self.path: str = str(path)
        self.title: str = title
        self.variables: list[str] | None = variables
        self.file_type: FileType = file_type
        self.precision: DataType = _normalize_precision(precision)
        self.current_zone: int = 0
        self.auxdataset: dict[str, str] = {}
        self.auxvar: dict[int, dict[str, str]] = {}

        self._fp: io.TextIOWrapper | None = None
        self._opened: bool = False
        self._float_fmt: str = _make_float_fmt(
            _SIG_DIGITS_FOR_PRECISION[self.precision]
        )

        # Running record of everything committed to the file so far (header, aux counts,
        # per-zone dimensions/sharing). Used to validate var_sharing / con_sharing on
        # subsequent zones against an earlier zone.
        self._meta = WriterMeta(
            path=self.path,
            title=self.title,
            file_type=self.file_type,
            file_format="dat",
        )

        if self.variables is not None:
            self._open(self.variables)

    # -- Context manager --------------------------------------------------------------

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

    # -- Validation checks and errror handling ----------------------------------------

    def _check_handle(self) -> io.TextIOWrapper:
        """Return the file handle, raising if the file has been closed."""
        if self._fp is None:
            raise ValueError(f"I/O operation on closed file: '{self.path}'")
        return self._fp

    def _check_variables(self) -> list[str]:
        """Return the variable list, raising if the file has not been opened yet."""
        if self.variables is None:
            raise RuntimeError(
                "Attempted to access variable name list before they were set. "
                "Ensure variables are set on initialization or zone write."
            )
        return self.variables

    @property
    def meta(self) -> WriterMeta:
        """Read-only record of everything written to this file so far."""
        return self._meta

    # -- File lifecycle ---------------------------------------------------------------

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
        self._meta.set_variables(self.variables)

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
            self._check_handle().write(f"DATASETAUXDATA {name}={_quote(value)}\n")

        for key, subdict in self.auxvar.items():
            if isinstance(key, int):
                var_idx = key - 1
                if var_idx not in range(len(self._check_variables())):
                    raise IndexError(
                        f"Variable index {key} out of bounds "
                        f"[1, {len(self._check_variables())}]"
                    )
            elif isinstance(key, str):
                try:
                    var_idx = self._check_variables().index(key)
                except ValueError as exc:
                    raise KeyError(
                        f"Variable aux data key {key!r} not found in "
                        f"variable list ({self._check_variables()})"
                    ) from exc
            else:
                raise TypeError(f"Aux data key must be str or 1-based int, got {key!r}")

            one_based = var_idx + 1
            for name, value in subdict.items():
                self._check_handle().write(
                    f"VARAUXDATA {one_based} {name}={_quote(value)}\n"
                )

        self._meta.note_dataset_aux(len(self.auxdataset))
        self._meta.note_var_aux(sum(len(subdict) for subdict in self.auxvar.values()))
        self.auxdataset.clear()
        self.auxvar.clear()

    def add_auxdataset_dict(self, auxdict: dict[str, Any]) -> None:
        """Buffer dataset-level auxiliary data from input dictionary."""
        self.auxdataset.update(auxdict)

    def add_auxvar_dict(self, auxdict: dict[int, dict[str, Any]]) -> None:
        """Buffer variable-level auxiliary data from input dictionary."""
        self.auxvar.update(auxdict)

    # -- Structured zone writer --------------------------------------------------------

    def write_ijk_zone(
        self,
        data: Sequence[npt.ArrayLike],
        *,
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

        Args:
            data:            One NumPy array per dataset variable.  Array shape is used
                             to infer ``imax``, ``jmax``, and ``kmax``; Fortran
                             (column-major) order is assumed.  Pass ``None`` to write a
                             zone header only.
            title:           Zone title.  Defaults to ``"IJK_Zone_{current_zone + 1}"``.
            variables:       Variable name list.  Required on the first call when the
                             file has not been opened yet (lazy-open path); ignored once
                             the file is already initialised. Default to ``[V1, V2, V3,
                             ...]`` if not provided in open or zone call.
            value_locations: Per-variable :class:`~libtecio.ValueLocation`.  Defaults to
                             all ``NODAL``.
            passive_vars:    Per-variable passive flags.  Defaults to all active
                             (``False``).
            var_sharing:     Per-variable share-from zone index (1-based).  Defaults to
                             no sharing (all zeros).
            solution_time:   Solution time for transient data.  Use ``0.0`` for static
                             zones. Default to ``0.0`` if not defined.
            strand_id:       Strand ID for transient data.  Use ``0`` for static
                             zones. Default to ``0`` (static) if not defined.
            aux:             Zone-level auxiliary data as ``{name: value}`` string
                             pairs.
            datapacking:     Must be :attr:`~tecio.libtecio.DataPacking.BLOCK` (the
                             default).  :attr:`~tecio.libtecio.DataPacking.POINT` is an
                             ASCII-only layout and is not supported by the SZL binary
                             format. Defined only for parity with ASCII writer.

        Raises:
            NotImplementedError: If *datapacking* is
                                 :attr:`~tecio.libtecio.DataPacking.POINT`.
            ValueError:          If I/O operation attempted on closed or None file
                                 handle.
            RuntimeError:        If attempting to write variable aux before variables
                                 are defined.
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

        variable_types = [
            _resolve_written_type(_infer_data_type(np.asarray(arr)), self.precision)
            for arr in data
        ]

        if value_locations is None:
            value_locations = [ValueLocation.NODAL] * len(data)
        if passive_vars is None:
            passive_vars = [False] * len(self._check_variables())
        if var_sharing is None:
            var_sharing = [0] * len(self._check_variables())

        # Validate count
        if len(data) != len(self._check_variables()):
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

        # Determine which dataset variables are supplied locally (not passive or shared)
        # and translate to 0-based local index
        active_var_idx = [
            vi
            for vi, (p, s) in enumerate(
                zip(passive_vars, var_sharing, strict=True), start=1
            )
            if not p and not s
        ]
        active_local_idx = {vi: i for i, vi in enumerate(active_var_idx)}

        # Determine validation reference data array shape. NODAL local or shared
        # variable arrays gives shape dimensions directly. CELL_CENTERED arrays can be
        # ambiguous if there is a degenerate axis (2D cells vs 3D with only 1 cell along
        # an axis appear the same). Therefore CELL_CENTERED is only used as fallback
        # method if no NODAL variables are available.
        nodal_shape: tuple[int, ...] | None = None
        ndims: int | None = None  # set only when nodal_shape came from a local array
        cell_fallback: tuple[int, ...] | None = None
        cell_fallback_ndims: int | None = None

        for var_idx in range(1, len(self._check_variables()) + 1):
            if passive_vars[var_idx - 1]:
                continue
            src = var_sharing[var_idx - 1]
            if src:
                if nodal_shape is None:
                    src_zone = self._meta.zone(src)
                    if src_zone is None or src_zone.dimensions is None:
                        self._handle_zone_error()
                        raise ValueError(
                            f"Variable {var_idx} shares from zone {src}, "
                            "which has not been written yet, or is not an "
                            "ORDERED zone."
                        )
                    nodal_shape = src_zone.dimensions
                continue

            local_arr = np.asarray(data[active_local_idx[var_idx]])
            loc = value_locations[active_local_idx[var_idx]]
            arr_ndims = local_arr.ndim
            if arr_ndims not in (1, 2, 3):
                self._handle_zone_error()
                raise ValueError(
                    f"Arrays must be 1D, 2D, or 3D; got {arr_ndims}-D array.  "
                    "For time-dependent data, write each time step as a separate zone."
                )
            shape = local_arr.shape + (1,) * (3 - arr_ndims)
            if loc == ValueLocation.NODAL:
                if nodal_shape is None:
                    nodal_shape, ndims = shape, arr_ndims
            elif cell_fallback is None:
                cell_fallback, cell_fallback_ndims = shape, arr_ndims

        if nodal_shape is None:
            if cell_fallback is not None:
                nodal_shape = tuple(n + 1 for n in cell_fallback)
                ndims = cell_fallback_ndims
            else:
                self._handle_zone_error()
                raise ValueError("Could not determine zone dimensions.")

        cell_shape = tuple(max(n - 1, 1) for n in nodal_shape)
        imax, jmax, kmax = nodal_shape

        # Validate every non-passive dataset variable array (including shared vars)
        # against reference
        for var_idx in range(1, len(self._check_variables()) + 1):
            if passive_vars[var_idx - 1]:
                continue
            src = var_sharing[var_idx - 1]
            if src:
                src_zone = self._meta.zone(src)
                if src_zone is None or src_zone.dimensions is None:
                    self._handle_zone_error()
                    raise ValueError(
                        f"Variable {var_idx} shares from zone {src}, which "
                        "has not been written yet, or is not an ORDERED "
                        "zone."
                    )
                if src_zone.dimensions != nodal_shape:
                    self._handle_zone_error()
                    raise ValueError(
                        f"Variable {var_idx} shares from zone {src} with "
                        f"dimensions {src_zone.dimensions}, which does not "
                        f"match this zone's dimensions {nodal_shape}."
                    )
                continue

            i = active_local_idx[var_idx]
            local_arr, loc = np.asarray(data[i]), value_locations[i]
            if ndims is not None and local_arr.ndim != ndims:
                self._handle_zone_error()
                raise ValueError(f"Array {i} is {local_arr.ndim}D, expected {ndims}D.")
            shape = local_arr.shape + (1,) * (3 - local_arr.ndim)
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

        # Define global var type and value locations using active variable index
        variable_types_global = [DataType.DOUBLE] * len(self._check_variables())
        value_locations_global = [ValueLocation.NODAL] * len(self._check_variables())

        # Replace the active indices with the real types/locations
        for local_idx, var_idx in enumerate(active_var_idx):
            variable_types_global[var_idx - 1] = variable_types[local_idx]
            value_locations_global[var_idx - 1] = value_locations[local_idx]

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
            value_locations_global=value_locations_global,
            variable_types_global=variable_types_global,
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

        self._check_handle().write(buf.getvalue())
        self.current_zone += 1

        # Finally set zone metadata after successfully completing TecIO calls
        self._meta.record_zone(
            ZoneMeta(
                index=self.current_zone,
                title=title,
                zone_type=ZoneType.ORDERED,
                solution_time=solution_time,
                strand_id=strand_id,
                num_aux_items=len(aux) if aux else 0,
                dimensions=(imax, jmax, kmax),
                value_locations=tuple(value_locations_global),
                passive_vars=tuple(bool(p) for p in passive_vars),
                shared_vars=tuple(int(s) for s in var_sharing),
                data_types=tuple(variable_types_global),
            )
        )

    # -- Unstructured zone writer ------------------------------------------------------

    def write_fe_zone(
        self,
        data: Sequence[npt.ArrayLike],
        zone_type: ZoneType,
        *,
        node_map: npt.ArrayLike | None = None,
        title: str | None = None,
        variables: list[str] | None = None,
        value_locations: Sequence[ValueLocation] | None = None,
        passive_vars: Sequence[bool | int] | None = None,
        var_sharing: Sequence[int] | None = None,
        con_sharing: int | None = None,
        face_neighbors: npt.ArrayLike | None = None,
        face_nbr_mode: FaceNeighborMode = FaceNeighborMode.LOCAL_ONE_TO_ONE,
        solution_time: float = 0.0,
        strand_id: int = 0,
        aux: dict[str, Any] | None = None,
        datapacking: DataPacking | str = DataPacking.BLOCK,
    ) -> None:
        """Write a complete finite-element zone.

        Node and cell counts are inferred from *node_map*, or if *node_map* is omitted
        from the zone referenced by *con_sharing*.

        Warning:
            ``FEPOLYGON`` and ``FEPOLYHEDRON`` raise :exc:`NotImplementedError`.

        Args:
            data:            Sequence of 1-D arrays, one per dataset variable.  NODAL
                             arrays must have length ``num_nodes``; CELL_CENTERED arrays
                             must have length ``num_cells``.  ``num_nodes`` and
                             ``num_cells`` are inferred from ``node_map`` (or from the
                             ``con_sharing`` source zone when ``node_map`` is omitted).
            zone_type:       FE zone type from the ZoneType enum.  Must be one of the
                             types in ``_FE_SIMPLE``.
            node_map:        Integer array of shape ``(num_cells, nodes_per_cell)``
                             containing 1-based node indices.  32- or 64-bit write is
                             chosen automatically based on the maximum index value.
                             Required unless ``con_sharing`` is set, in which case the
                             connectivity -- and the node/cell counts derived from it --
                             are inherited from the source zone instead.
            title:           Zone title string.  Defaults to ``"FE_Zone_{current_zone +
                             1}"`` if not provided.
            variables:       Variable name list.  Required only when the file has not
                             been opened yet (lazy-open path).  Ignored on subsequent
                             zones once the file is already initialised. Default to
                             ``[V1, V2, V3, ...]`` if not provided in open or zone call.
            value_locations: Per-variable ValueLocation.  Defaults to all NODAL.
            passive_vars:    Per-variable passive flags.  Defaults to all active
                             (False).
            var_sharing:     Per-variable share from zone index.  Defaults to no
                             sharing.  Cross-checked against ``node_map`` /
                             ``con_sharing`` for a consistent node/cell count.
            con_sharing:     Optional zone index that the connectivity is shared from.
                             ``None`` or ``0`` indicates no sharing (this zone owns its
                             connectivity). The first zone in a dataset must own its
                             connectivity. Connectivity cannot be shared when face
                             neighbor mode is set to global. Connectivity cannot be
                             shared between cell-based and face-based finite element
                             zones.
            face_neighbors:  Optional face-neighbor connectivity array.
                             ``num_face_cons`` in the zone header is set to
                             ``len(face_neighbors)`` automatically when this is
                             supplied.
            face_nbr_mode:   Face-neighbor mode, used only when ``face_neighbors`` is
                             provided. Defaults to LOCAL_ONE_TO_ONE.
            solution_time:   Solution time for transient data (0.0 = static).
            strand_id:       Strand ID for transient data (0 = static).
            aux:             Zone-level auxiliary data as ``{name: value}`` strings.
            datapacking:     Must be :attr:`~tecio.libtecio.DataPacking.BLOCK` (the
                             default).  :attr:`~tecio.libtecio.DataPacking.POINT` is an
                             ASCII-only layout and is not supported by the SZL binary
                             format. Defined only for parity with ASCII writer.

        Raises:
            NotImplementedError: For FEPOLYGON, FEPOLYHEDRON, or if *datapacking*
                                 is :attr:`~tecio.libtecio.DataPacking.POINT`.
            ValueError:          On variable count or array length mismatch; if
                                 ``node_map`` is omitted without ``con_sharing``; or if
                                 ``var_sharing``/``con_sharing`` reference a zone with
                                 no recorded node/cell count, or one whose count
                                 disagrees with this zone's.
            ValueError:          If I/O operation attempted on closed or None file
                                 handle.
            RuntimeError:        If attempting to write variable aux before variables
                                 are defined.

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

        variable_types = [
            _resolve_written_type(_infer_data_type(np.asarray(arr)), self.precision)
            for arr in data
        ]

        # Default passive / sharing arrays
        if value_locations is None:
            value_locations = [ValueLocation.NODAL] * len(data)
        if passive_vars is None:
            passive_vars = [False] * len(self._check_variables())
        if var_sharing is None:
            var_sharing = [0] * len(self._check_variables())
        if con_sharing is None:
            con_sharing = 0

        # Validate count
        if len(data) != len(self._check_variables()):
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

        # Derive num_nodes and num_cells from node_map (read meta if shared)
        if node_map is not None:
            node_map_arr = np.asarray(node_map)
            num_cells = int(node_map_arr.shape[0])
            num_nodes = int(node_map_arr.max())
        elif con_sharing:
            src_zone = self._meta.zone(con_sharing)
            if (
                src_zone is None
                or src_zone.num_nodes is None
                or src_zone.num_elements is None
            ):
                self._handle_zone_error()
                raise ValueError(
                    f"con_sharing={con_sharing} references a zone that has "
                    "not been written yet, or is not a finite-element zone."
                )
            num_nodes = src_zone.num_nodes
            num_cells = src_zone.num_elements
        else:
            self._handle_zone_error()
            raise ValueError(
                "node_map must be provided unless connectivity is shared "
                "from another zone via con_sharing."
            )

        # Shared variable data shape validation
        for var_idx, src in enumerate(var_sharing, start=1):
            if not src:
                continue
            src_zone = self._meta.zone(src)
            if src_zone is None:
                self._handle_zone_error()
                raise ValueError(
                    f"Variable {var_idx} shares from zone {src}, which has "
                    "not been written yet."
                )
            # Check shared variable value location to determine validation reference
            src_loc = (
                src_zone.value_locations[var_idx - 1]
                if var_idx - 1 < len(src_zone.value_locations)
                else ValueLocation.NODAL
            )
            if src_loc == ValueLocation.CELL_CENTERED:
                if src_zone.num_elements != num_cells:
                    self._handle_zone_error()
                    raise ValueError(
                        f"Variable {var_idx} shares from zone {src} with "
                        f"{src_zone.num_elements} cells, which does not "
                        f"match this zone's cell count of {num_cells}."
                    )
            elif src_zone.num_nodes != num_nodes:
                self._handle_zone_error()
                raise ValueError(
                    f"Variable {var_idx} shares from zone {src} with "
                    f"{src_zone.num_nodes} nodes, which does not match "
                    f"this zone's node count of {num_nodes}."
                )

        # Local variable data shape validation
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

        # Determine 1-based index of dataset variables to write (not passive or shared)
        active_var_idx = [
            vi
            for vi, (p, s) in enumerate(
                zip(passive_vars, var_sharing, strict=True), start=1
            )
            if not p and not s
        ]

        # Define global var type and value locations using active variable index
        variable_types_global = [DataType.DOUBLE] * len(self._check_variables())
        value_locations_global = [ValueLocation.NODAL] * len(self._check_variables())

        # Replace the active indices with the real types/locations
        for local_idx, var_idx in enumerate(active_var_idx):
            variable_types_global[var_idx - 1] = variable_types[local_idx]
            value_locations_global[var_idx - 1] = value_locations[local_idx]

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
            value_locations_global=value_locations_global,
            variable_types_global=variable_types_global,
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

        # Write connectivity (if not shared)
        if not con_sharing:
            assert node_map is not None
            conn = np.asarray(node_map, dtype=np.intp).reshape(
                num_cells, _NODES_PER_ELEM[zone_type]
            )
            for row in conn:
                _stage_connectivity_row(buf, row)

        self._check_handle().write(buf.getvalue())
        self.current_zone += 1

        # Finally set zone metadata after successfully completing TecIO calls
        self._meta.record_zone(
            ZoneMeta(
                index=self.current_zone,
                title=title,
                zone_type=zone_type,
                solution_time=solution_time,
                strand_id=strand_id,
                num_aux_items=len(aux) if aux else 0,
                num_nodes=num_nodes,
                num_elements=num_cells,
                value_locations=tuple(value_locations_global),
                passive_vars=tuple(bool(p) for p in passive_vars),
                shared_vars=tuple(int(s) for s in var_sharing),
                data_types=tuple(variable_types_global),
            )
        )

    # -- Private helpers --------------------------------------------------------------

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
        fp = self._check_handle()
        fp.write(f"TITLE     = {_quote(self.title)}\n")
        if self.file_type in _FILETYPE_STR:
            fp.write(f"FILETYPE  = {_FILETYPE_STR[self.file_type]}\n")
        var_lines = "\n".join(_quote(v) for v in self._check_variables())
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
        variable_types_global: list[DataType] | None = None,
        con_sharing: int = 0,
        aux: dict[str, Any] | None = None,
        datapacking: DataPacking = DataPacking.BLOCK,
    ) -> None:
        """Write a ``ZONE`` header block into the staging buffer *buf*."""
        zt_str = _ZONETYPE_STR[zone_type]

        # First line is never indented; every subsequent header line is indented.
        buf.write(f"ZONE T={_quote(title)}\n")
        buf.write(
            f"{_INDENT}STRANDID={strand_id}, SOLUTIONTIME={float(solution_time)}\n"
        )

        if zone_type == ZoneType.ORDERED:
            buf.write(f"{_INDENT}I={imax}, J={jmax}, K={kmax}, ZONETYPE={zt_str}\n")
        else:
            buf.write(
                f"{_INDENT}Nodes={num_nodes}, Elements={num_elements}, "
                f"ZONETYPE={zt_str}\n"
            )

        buf.write(f"{_INDENT}DATAPACKING={datapacking.name}\n")

        if variable_types_global:
            dt_str = " ".join(_DATATYPE_STR[dt] for dt in variable_types_global)
            buf.write(f"{_INDENT}DT=({dt_str})\n")

        if value_locations_global:
            cc = [
                i + 1
                for i, loc in enumerate(value_locations_global)
                if loc == ValueLocation.CELL_CENTERED
            ]
            if cc:
                buf.write(
                    f"{_INDENT}VARLOCATION=([{','.join(str(i) for i in cc)}]"
                    "=CELLCENTERED)\n"
                )

        if passive_vars:
            pidx = [str(i + 1) for i, f in enumerate(passive_vars) if f]
            if pidx:
                buf.write(f"{_INDENT}PASSIVEVARLIST=[{','.join(pidx)}]\n")

        if var_sharing and any(var_sharing):
            entries = [f"[{i + 1}]={z}" for i, z in enumerate(var_sharing) if z]
            if entries:
                buf.write(f"{_INDENT}VARSHARELIST=({','.join(entries)})\n")

        if con_sharing:
            buf.write(f"{_INDENT}CONNECTIVITYSHAREZONE={con_sharing}\n")

        if aux:
            for name, value in aux.items():
                buf.write(f"{_INDENT}AUXDATA {name}={_quote(value)}\n")
