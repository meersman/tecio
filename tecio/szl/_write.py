"""Write Tecplot SZL (``.szplt``) files via the TecIO C library.

Supports lazy-open (deferred until first zone write), buffered auxiliary
data, and automatic dtype inference from NumPy arrays.

Notes:
    - flush_aux():
        - write dataset aux if self.aux_dataset is not empty
        - write variable aux if self.aux_var is not empty
        - after writing set both to empty with self.aux_dataset.clear() and
          self.aux_var.clear()
    - Zone writers:
        - write_ijk_zone(): write zone header and optionally data for input ORDERED data
            - set default value location as NODAL
            - write default data as DOUBLE
            - support data provided as a list[npt.NDArrayLike, ...]
            - if var list already defined for whole dataset, do not require, but if not
              defined, throw error
    - write_fe_zone():
"""

from __future__ import annotations

import ctypes
from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from .. import libtecio
from .._meta import WriterMeta, ZoneMeta
from ..libtecio import (
    DataPacking,
    DataType,
    FaceNeighborMode,
    FileType,
    ValueLocation,
    ZoneType,
)

# --------------------------------------------------------------------------------------
# Module-level constants
# --------------------------------------------------------------------------------------

# FE zone types that use tec_zone_create_fe
_FE_SIMPLE: frozenset[ZoneType] = frozenset({
    ZoneType.FELINESEG,
    ZoneType.FETRIANGLE,
    ZoneType.FEQUADRILATERAL,
    ZoneType.FETETRAHEDRON,
    ZoneType.FEBRICK,
})


# ======================================================================================
# Helpers
# ======================================================================================


def _infer_data_type(dt: DataType | np.dtype) -> DataType:
    """Return the closest C-supported DataType for a DataType or NumPy dtype."""
    # Mapping of NumPy type categories to closest DataType
    closest_dtype_map = {
        np.dtype(np.float64): DataType.DOUBLE,
        np.dtype(np.float32): DataType.FLOAT,
        np.dtype(np.float16): DataType.FLOAT,  # promote half → float
        np.dtype(np.int64): DataType.INT32,  # promote 64-bit → INT32
        np.dtype(np.int32): DataType.INT32,
        np.dtype(np.int16): DataType.INT16,
        np.dtype(np.int8): DataType.BYTE,  # small ints → BYTE
        np.dtype(np.uint8): DataType.BYTE,
    }
    if isinstance(dt, DataType):
        return dt

    dt_np = np.dtype(dt)

    for key in closest_dtype_map:
        if dt_np == key:
            return closest_dtype_map[key]

    if np.issubdtype(dt_np, np.floating):
        return DataType.FLOAT if dt_np.itemsize <= 4 else DataType.DOUBLE

    if np.issubdtype(dt_np, np.signedinteger):
        return DataType.INT16 if dt_np.itemsize <= 2 else DataType.INT32

    if np.issubdtype(dt_np, np.unsignedinteger):
        return DataType.BYTE

    raise ValueError(f"Unsupported dtype: {dt_np}")


# ASCII/enum-name aliases for the two precision options
_STR_TO_PRECISION: dict[str, DataType] = {
    "single": DataType.FLOAT,
    "float": DataType.FLOAT,
    "double": DataType.DOUBLE,
}


def _normalize_precision(precision: DataType | str | None) -> DataType | None:
    """Return the :class:`DataType` for *precision*, or ``None`` unchanged.

    Default (``None``) means no override, infer each variable's type from its own array
    automatically. Otherwise accepts the enum directly, or a case-insensitive string.

    Raises:
        ValueError: If *precision* is neither ``None`` nor FLOAT/DOUBLE (or a recognized
                    string alias for one of them).
    """
    if precision is None:
        return None
    if isinstance(precision, str):
        try:
            precision = _STR_TO_PRECISION[precision.strip().lower()]
        except KeyError:
            raise ValueError(
                f"precision={precision!r} is not recognized; use 'single' or "
                "'double' (or DataType.FLOAT / DataType.DOUBLE), or None to "
                "infer each variable's type automatically."
            ) from None
    if precision not in (DataType.FLOAT, DataType.DOUBLE):
        raise ValueError(
            f"precision={precision!r} is not supported; precision only "
            "applies to floating-point variables -- use DataType.FLOAT, "
            "DataType.DOUBLE, or None."
        )
    return precision


def _resolve_written_type(inferred: DataType, precision: DataType | None) -> DataType:
    """Return the :class:`DataType` actually written for one variable.

    *precision* overrides *inferred* only when *inferred* is itself a floating-point
    type (FLOAT or DOUBLE). Integer-inferred variables (INT32/INT16/BYTE) always keep
    their own inferred type, unaffected by *precision*. A variable holding a meaningful
    integer (a CPU number, an index, a count) should never be silently coerced by a
    setting that's conceptually about floating-point precision.
    """
    if precision is None:
        return inferred
    if inferred in (DataType.FLOAT, DataType.DOUBLE):
        return precision
    return inferred


# ======================================================================================
# Local functions
# ======================================================================================


def _write_data(
    handle: ctypes.c_void_p,
    zone_num: int,
    var_num: int,
    data: npt.ArrayLike,
    dt: np.dtype | DataType | None = None,
) -> None:
    """Write a single variable's data array to an SZL file.

    Infers the data type from the array dtype and dispatches to the
    appropriate C write function. Arrays are ravelled in Fortran order.

    Args:
        handle (ctypes.c_void_p): C library file handle.
        zone_num (int): 1-based zone index.
        var_num (int): 1-based variable index.
        data (npt.ArrayLike): Array of values to write.
        dt (np.dtype | DataType | None): Optional explicit data type override.

    Output defaults:
        1. Inferrs data_typeype for nummpy arrays
        2. Defaults to double precision for array-like (list, tuple, etc)
        3. Optionally casts to input DataType or numpy dtype.
        4. Assumes data is in the correct shape and order (column major / Fortran order)
    """
    # Mappings between C-supported data types and numpy dtypes
    dtype_to_datatype: dict[np.dtype, DataType] = {
        np.dtype(np.float64): DataType.DOUBLE,
        np.dtype(np.float32): DataType.FLOAT,
        np.dtype(np.int32): DataType.INT32,
        np.dtype(np.int16): DataType.INT16,
        np.dtype(np.uint8): DataType.BYTE,
    }
    datatype_to_dtype: dict[DataType, np.dtype] = {
        DataType.DOUBLE: np.dtype(np.float64),
        DataType.FLOAT: np.dtype(np.float32),
        DataType.INT32: np.dtype(np.int32),
        DataType.INT16: np.dtype(np.int16),
        DataType.BYTE: np.dtype(np.uint8),
    }

    if dt is not None:
        data_type = _infer_data_type(dt)
        arr = np.ascontiguousarray(data, dtype=datatype_to_dtype[data_type]).ravel(
            order="F"
        )
    else:
        arr = np.ascontiguousarray(data).ravel(order="F")
        data_type = dtype_to_datatype[arr.dtype]

    if data_type == DataType.DOUBLE:
        libtecio.tec_zone_var_write_double_values(handle, zone_num, var_num, arr)
    elif data_type == DataType.FLOAT:
        libtecio.tec_zone_var_write_float_values(handle, zone_num, var_num, arr)
    elif data_type == DataType.INT32:
        libtecio.tec_zone_var_write_int32_values(handle, zone_num, var_num, arr)
    elif data_type == DataType.INT16:
        libtecio.tec_zone_var_write_int16_values(handle, zone_num, var_num, arr)
    elif data_type == DataType.BYTE:
        libtecio.tec_zone_var_write_uint8_values(handle, zone_num, var_num, arr)
    else:
        raise ValueError(f"Unsupported DataType: {data_type!r}")


def _write_connectivity(
    handle: ctypes.c_void_p,
    zone_num: int,
    node_map: npt.ArrayLike,
    face_neighbors: npt.ArrayLike | None = None,
) -> None:
    """Write FE zone connectivity: node map and optional face-neighbor connections.

    Integer width (32 or 64 bit) is chosen automatically from the
    maximum index value in each array.

    Args:
        handle (ctypes.c_void_p): C library file handle.
        zone_num (int): 1-based zone index.
        node_map (npt.ArrayLike): Connectivity array, 1-based node indices.
        face_neighbors (npt.ArrayLike | None): Optional face-neighbor
            connection array.

    Note:
        Both arrays are written using the minimum integer width capable of
        representing the maximum index value present in each array.  No copy
        is made for C-contiguous input arrays — ravel(order="C") returns a flat
        view of the node_map or face_neighbors without making a copy

    Note:
        Node and face-neighbor integer widths are chosen independently based
        on the maximum value in each respective array.
    """
    # Node map
    node_map_flat = np.ascontiguousarray(node_map).ravel(order="C")
    if node_map_flat.max() > np.iinfo(np.int32).max:
        libtecio.tec_zone_node_map_write64(handle, zone_num, node_map_flat)
    else:
        libtecio.tec_zone_node_map_write32(handle, zone_num, node_map_flat)

    # Face neighbors (optional)
    if face_neighbors is not None:
        face_nbr_flat = np.ascontiguousarray(face_neighbors).ravel(order="C")
        if face_nbr_flat.max() > np.iinfo(np.int32).max:
            libtecio.tec_zone_face_nbr_write_connections64(
                handle, zone_num, face_nbr_flat
            )
        else:
            libtecio.tec_zone_face_nbr_write_connections32(
                handle, zone_num, face_nbr_flat
            )


def _write_zone_aux_data(
    handle: ctypes.c_void_p, aux: dict[int, dict[str, Any]]
) -> None:
    """Write zone-level auxiliary data.

    Args:
        handle (ctypes.c_void_p): C library file handle.
        aux (dict[int, dict[str, Any]]): Mapping of
            ``{zone_index: {name: value}}``.

    Note:
        Aux data should be structured as {zone_idx: {name, value}}
    """
    for zone_idx, subdict in aux.items():
        for name, value in subdict.items():
            libtecio.tec_zone_add_aux_data(handle, zone_idx, str(name), str(value))


def _write_variable_aux_data(
    handle: ctypes.c_void_p, aux: dict[int, dict[str, Any]]
) -> None:
    """Write variable-level auxiliary data.

    Args:
        handle (ctypes.c_void_p): C library file handle.
        aux (dict[int, dict[str, Any]]): Mapping of
            ``{var_index: {name: value}}``.

    Hint:
        Aux data should be structured as ``{var_idx: {name, value}}``
    """
    for var_idx, subdict in aux.items():
        for name, value in subdict.items():
            libtecio.tec_var_add_aux_data(handle, var_idx, str(name), str(value))


def _write_dataset_aux_data(handle: ctypes.c_void_p, aux: dict[str, Any]) -> None:
    """Write dataset-level auxiliary data.

    Args:
        handle (ctypes.c_void_p): C library file handle.
        aux (dict[str, Any]): Mapping of ``{name: value}``.

    Hint:
        Aux data should be structured as ``{var_idx: {name, value}}``
    """
    for name, value in aux.items():
        libtecio.tec_data_set_add_aux_data(handle, str(name), str(value))


def _write_aux_data(handle: ctypes.c_void_p, aux: dict[str, dict[Any, Any]]) -> None:
    """Write a combined auxiliary data dictionary to the file.

    Args:
        handle (ctypes.c_void_p): C library file handle.
        aux (dict[str, dict[Any]]): Dict with keys ``"AUXDATA"``,
            ``"AUXVAR"``, ``"AUXZONE"``, each containing the appropriate
            nested structure.

    Example:
        {
            "AUXDATASET":
                {name1: value1}
                {name2: value2}
            "AUXVAR": {
                        1:
                           {name1: value1}
                        2:
                           {name1: value1}
                    },
            "AUXZONE": {
                        1:
                            {name1: value1}
                            {name2: value2}
                    }
        }
    """
    for auxtype, auxdict in aux.items():
        if auxtype.lower() == "auxdata":
            for name, value in auxdict.items():
                libtecio.tec_data_set_add_aux_data(handle, str(name), str(value))
        elif auxtype.lower() == "auxvar":
            for var, subdict in auxdict.items():
                for name, value in subdict.items():
                    libtecio.tec_var_add_aux_data(handle, var, name, value)
        elif auxtype.lower() == "auxzone":
            for zone, subdict in auxdict.items():
                for name, value in subdict.items():
                    libtecio.tec_zone_add_aux_data(handle, zone, name, value)


# ======================================================================================
# Write class
# ======================================================================================


class Write:
    """Write Tecplot SZL (``.szplt``) files with a lazy-open file handle.

    Supports lazy-open: if *variables* is ``None`` at construction, the
    file is created on the first zone write. Auxiliary data is buffered
    and flushed automatically before the first zone.

    The tecio library requires a list of variables when the output file is
    opened. However if writing data on the fly, it may be beneficial to store file
    outputs until the first zone is passed to the writer. Then file header will be
    garanteed to be consistent with the first zone variables.

    For the SZL API, file contents can be written out of order after creating zones.

    Args:
        path:         Output file path.
        title:        Dataset title.
        variables:    Variable name list. ``None`` defers file creation.
        file_type:    File type enum (FULL, GRID, or SOLUTION).
        precision:    Optional whole file floating point precision override
                      (:attr:`DataType.FLOAT`/``"single"`` or
                      :attr:`DataType.DOUBLE`/``"double"``). Defaults to ``None``: each
                      variable's type is inferred automatically from its own array's
                      dtype.

    Attributes:
        path:         Output file path.
        title:        Dataset title string.
        variables:    Variable name list, or ``None`` if the file has not been opened
                      yet.
        file_type:    File type (FULL, GRID, or SOLUTION).
        precision:    Whole-file floating-point override, or ``None`` for automatic
                      per-variable inference.
        current_zone: The index of the most recently written zone.  Before any zones
                      have been written, ``current_zone`` is ``0``.  During a call to a
                      zone writing method, ``current_zone`` still refers to the
                      previously written zone.  ``current_zone`` is incremented only
                      after a zone writing method successfully completes.
        auxdataset:   Buffered dataset-level auxiliary data, flushed before the first
                      zone.
        auxvar:       Buffered variable-level auxiliary data, flushed before the first
                      zone.
    """

    def __init__(
        self,
        path: str,
        title: str = "untitled",
        variables: list[str] | None = None,
        file_type: FileType = FileType.FULL,
        *,
        precision: DataType | str | None = None,
    ) -> None:
        """Store minimum necessary info until first zone is ready to write.

        Raises:
            ValueError: If *precision* is neither ``None`` nor
                        :attr:`DataType.FLOAT`/:attr:`DataType.DOUBLE` (or a recognized
                        string alias for one of them).
        """
        self.path = path
        self.title = title
        self.variables = variables
        self.file_type = file_type
        self.precision: DataType | None = _normalize_precision(precision)
        self.current_zone = 0
        self.current_var = 0

        # Add created data to the buffer and flush once a file handle is created.
        # Dataset-level aux data buffer (flushed on first zone)
        self.auxdataset: dict[str, str] = {}
        # Variable-level aux data buffer: {var_name: {key: value}}
        self.auxvar: dict[int, dict[str, str]] = {}

        # Running record of everything committed to the file so far (header, aux counts,
        # per-zone dimensions/sharing). Used to validate var_sharing / con_sharing on
        # subsequent zones against an earlier zone.
        self._meta = WriterMeta(
            path=self.path,
            title=self.title,
            file_type=self.file_type,
            file_format="szplt",
        )

        # Initialize if all needed info provided, else set to null
        if self.variables is not None:
            self.handle: ctypes.c_void_p | None = libtecio.tec_file_writer_open(
                filename=self.path,
                variables=self.variables,
                title=self.title,
                file_type=self.file_type,
                use_szl=1,
            )
            self._meta.set_variables(self.variables)
        else:
            # Variables needed
            self.handle = None

    # -- Context manager ---------------------------------------------------------------

    def __enter__(self) -> Write:
        """Context manager to automatically open, close, and flush file."""
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Exit Write class context manager regardless of exceptions.

        Only raise an exception if closing the file fails, not if an exception is raised
        in the with block.
        """
        try:
            self.close()
        except Exception:
            if exc_type is None:
                raise

    # -- Validation checks and errror handling -----------------------------------------

    def _check_handle(self) -> ctypes.c_void_p:
        """Return file handle catching errors if the writer has already been closed.

        This ensures the that each libtecio call will execute or return an appropriate
        ValueError.
        """
        if self.handle is None:
            raise RuntimeError(f"I/O operation on closed file: '{self.path}'")
        else:
            return self.handle

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

    # -- File lifecycle ----------------------------------------------------------------

    def _open(self, var_names: list[str]) -> None:
        """Open the file handle.  Called exactly once on the first zone."""
        self.variables = var_names
        self.handle = libtecio.tec_file_writer_open(
            filename=self.path,
            variables=self.variables,
            title=self.title,
            file_type=self.file_type,
            use_szl=1,
        )
        self._meta.set_variables(self.variables)

    def close(self) -> None:
        """Finalise and flush the file (safe to call more than once)."""
        if self.handle is not None:
            self.flush_aux()
            libtecio.tec_file_writer_close(self.handle)
            self.handle = None

    def add_auxdataset_dict(self, auxdict: dict[str, Any]) -> None:
        """Create buffered auxdataset items from input dictionary."""
        self.auxdataset.update(auxdict)

    def add_auxvar_dict(self, auxdict: dict[int, dict[str, Any]]) -> None:
        """Create buffered auxvar items from input dictionary."""
        self.auxvar.update(auxdict)

    def flush_aux(self) -> None:
        """Write buffered dataset and variable aux data to the file.

        Called automatically before the first zone is created.  You only
        need to call this directly if you want to be explicit.

        The C library requires dataset and variable aux data to be written
        after the file is intialized. Therefore in cases of lazy loading
        (buffered file info until first zone is defined), aux data is also
        buffered then flushed on first zone creation.
        """
        # Write dataset-level aux data
        for name, value in self.auxdataset.items():
            libtecio.tec_data_set_add_aux_data(
                self._check_handle(),
                str(name),
                str(value),
            )

        # Variable-level aux data.  Keys may be 1-based int indices or names.
        for key, subdict in self.auxvar.items():
            if isinstance(key, int):
                var_idx = key - 1  # Convert to 0-based
                if var_idx not in range(len(self._check_variables())):
                    raise IndexError(
                        f"Variable index {var_idx} out of bounds "
                        f"[1, {len(self._check_variables())}]"
                    )
            elif isinstance(key, str):
                try:
                    var_idx = self.variables.index(key)
                except ValueError as exc:
                    raise KeyError(
                        f"Variable aux data key '{key}' not found in "
                        f"variable list ({self.variables})"
                    ) from exc
            else:
                raise TypeError(
                    f"Aux data key must be a variable name (str) or 1-based "
                    f"index (int), got {key!r}"
                )

            for name, value in subdict.items():
                libtecio.tec_var_add_aux_data(
                    self._check_handle(), var_idx + 1, str(name), str(value)
                )

        # Record counts, then clear buffers — each item is written exactly once.
        self._meta.note_dataset_aux(len(self.auxdataset))
        self._meta.note_var_aux(sum(len(subdict) for subdict in self.auxvar.values()))
        self.auxdataset.clear()
        self.auxvar.clear()

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

        Dimensions are inferred from the first array's shape. Arrays may be 1-D, 2-D, or
        3-D; missing trailing dimensions default to 1.

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

        Note:
            If the file is already open, ``data`` and ``variables`` may be omitted to
            write a zone header only.  If the file has not been opened yet,
            ``variables`` must be provided on this call.

        Note:
            Data arrays are written as DOUBLE precision by default.  To write other
            types, cast the NumPy arrays before calling (e.g.
            ``arr.astype(np.float32)``).

        Note:
            Separate grid files (where all variables are cell-centred) are not handled
            automatically; use the low-level API for that case.
        """
        # Validate inputs
        if isinstance(datapacking, str):
            try:
                datapacking = DataPacking[datapacking.upper()]
            except KeyError:
                raise ValueError(
                    f"datapacking={datapacking!r} is not a recognised value; "
                    "use DataPacking.BLOCK or the string 'BLOCK'."
                ) from None
        if datapacking != DataPacking.BLOCK:
            raise NotImplementedError(
                "DATAPACKING=POINT is an ASCII-only layout and is not supported "
                "by the SZL binary format.  Use DataPacking.BLOCK (the default) "
                "or write to a .dat file instead."
            )

        # Convert input data to NumPy arrays
        arrays: list[npt.NDArray] = [np.asarray(arr) for arr in data]

        # Set default title if none provided
        if title is None:
            title = f"IJK_Zone_{self.current_zone + 1}"

        # Set default variable names if none provided. Only relevant if file lazily
        # loaded and first zone
        if variables is None:
            variables = [f"V{i}" for i in range(1, len(arrays) + 1)]

        # Open and initialize the file if lazily loaded
        if self.handle is None:
            self._open(variables)
            self.flush_aux()

        # Get variable types (local variable -> length = number of supplied data arrays)
        variable_types = [
            _resolve_written_type(_infer_data_type(arr.dtype), self.precision)
            for arr in arrays
        ]

        # Set default value loacations
        if value_locations is None:
            # Local variable -> length = number of supplied data arrays
            value_locations = [ValueLocation.NODAL] * len(arrays)

        # Default passive / sharing arrays — length equals dataset variable count
        if passive_vars is None:
            passive_vars = [False] * len(self._check_variables())
        if var_sharing is None:
            var_sharing = [0] * len(self._check_variables())

        # Validate active variable count
        if len(arrays) != len(self._check_variables()):
            # Calcuate the number of expected variables based on passive and shared
            # variables
            expected_vars = sum(
                1
                for is_passive, share_zone in zip(
                    passive_vars, var_sharing, strict=True
                )
                if not is_passive and not share_zone
            )
            if expected_vars == 0:
                raise ValueError(
                    "No active variables to write — all variables are either "
                    "passive or shared."
                )
            elif len(arrays) == 0:
                raise ValueError(
                    f"No data arrays provided. Expected {expected_vars} active "
                    "variable arrays based on passive_vars and var_sharing."
                )
            elif len(arrays) != expected_vars:
                raise ValueError(
                    f"Expected {expected_vars} data arrays, got {len(arrays)}"
                )

        # Determine which dataset variables are supplied locally (not passive or shared)
        # and translate to 0-based local index
        active_var_idx = [
            var_idx
            for var_idx, (is_passive, sharing_zone_idx) in enumerate(
                zip(passive_vars, var_sharing, strict=True),
                start=1,
            )
            if (not is_passive) and (not sharing_zone_idx)
        ]
        active_local_idx = {var_idx: i for i, var_idx in enumerate(active_var_idx)}

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
                        raise ValueError(
                            f"Variable {var_idx} shares from zone {src}, "
                            "which has not been written yet, or is not an "
                            "ORDERED zone."
                        )
                    nodal_shape = src_zone.dimensions
                continue

            arr = arrays[active_local_idx[var_idx]]
            loc = value_locations[active_local_idx[var_idx]]
            arr_ndims = arr.ndim
            if arr_ndims not in (1, 2, 3):
                raise ValueError(
                    f"Arrays must be 1D, 2D, or 3D; got {arr_ndims}-D array.  "
                    "For time-dependent data, write each time step as a separate zone."
                )
            shape = arr.shape + (1,) * (3 - arr_ndims)
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
                    raise ValueError(
                        f"Variable {var_idx} shares from zone {src}, which "
                        "has not been written yet, or is not an ORDERED "
                        "zone."
                    )
                if src_zone.dimensions != nodal_shape:
                    raise ValueError(
                        f"Variable {var_idx} shares from zone {src} with "
                        f"dimensions {src_zone.dimensions}, which does not "
                        f"match this zone's dimensions {nodal_shape}."
                    )
                continue

            i = active_local_idx[var_idx]
            arr, loc = arrays[i], value_locations[i]
            if ndims is not None and arr.ndim != ndims:
                raise ValueError(f"Array {i} is {arr.ndim}D, expected {ndims}D")
            shape = arr.shape + (1,) * (3 - arr.ndim)
            if (loc == ValueLocation.NODAL) and (shape != nodal_shape):
                raise ValueError(
                    f"Array {i} is NODAL but has shape {shape}, expected {nodal_shape}"
                )
            if (loc == ValueLocation.CELL_CENTERED) and (shape != cell_shape):
                raise ValueError(
                    f"Array {i} is CELL_CENTERED but has shape {shape}, "
                    f"expected {cell_shape}"
                )

        # Define global var type and value locations using active variable index
        variable_types_global = [DataType.DOUBLE] * len(self._check_variables())
        value_locations_global = [ValueLocation.NODAL] * len(self._check_variables())

        # Replace the active indices with the real types/locations
        for local_idx, var_idx in enumerate(active_var_idx):
            variable_types_global[var_idx - 1] = variable_types[local_idx]
            value_locations_global[var_idx - 1] = value_locations[local_idx]

        # Write zone header
        self.current_zone = libtecio.tec_zone_create_ijk(
            self._check_handle(),
            title,
            imax,
            jmax,
            kmax,
            var_types=variable_types_global,
            value_locations=value_locations_global,
            var_sharing=var_sharing,
            pas_vars=passive_vars,
        )

        # Unsteady options
        if strand_id != 0 or solution_time != 0.0:
            libtecio.tec_zone_set_unsteady_options(
                handle=self._check_handle(),
                zone=self.current_zone,
                strand=strand_id,
                solution_time=solution_time,
            )

        # Write aux data
        if aux is not None:
            _write_zone_aux_data(self._check_handle(), {self.current_zone: aux})

        # Write active data only
        for var_idx, arr, dtype in zip(
            active_var_idx, arrays, variable_types, strict=True
        ):
            self.current_var = var_idx
            _write_data(
                self._check_handle(),
                zone_num=self.current_zone,
                var_num=var_idx,
                data=arr,
                dt=dtype,
            )

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
        from the zone referenced by *con_sharing*. The 32- or 64-bit write path is
        chosen automatically from the max index.

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
                             sharing.  A shared variable's type and value location are
                             inherited from its source zone and are cross-checked
                             against ``node_map`` / ``con_sharing`` for a consistent
                             node/cell count.
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
            ValueError:          On variable count or array length mismatch, if
                                 ``node_map`` is omitted without ``con_sharing``, or if
                                 ``var_sharing``/``con_sharing`` reference a zone with
                                 no recorded node/cell count, or one whose count
                                 disagrees with this zone's.
            ValueError:          If I/O operation attempted on closed or None file
                                 handle.
            RuntimeError:        If attempting to write variable aux before variables
                                 are defined.

        Note:
            FE variable arrays are 1-D and node-ordered. ``write_data`` handles dtype
            inference and F-order ravel internally, but 1-D arrays are unaffected by
            memory order.
        """
        # Validate input
        if isinstance(datapacking, str):
            try:
                datapacking = DataPacking[datapacking.upper()]
            except KeyError:
                raise ValueError(
                    f"datapacking={datapacking!r} is not a recognised value; "
                    "use DataPacking.BLOCK or the string 'BLOCK'."
                ) from None
        if datapacking != DataPacking.BLOCK:
            raise NotImplementedError(
                "DATAPACKING=POINT is an ASCII-only layout and is not supported "
                "by the SZL binary format.  Use DataPacking.BLOCK (the default) "
                "or write to a .dat file instead."
            )
        if zone_type not in _FE_SIMPLE:
            raise NotImplementedError(
                f"Zone type {zone_type.name!r} is not supported by szl file formats. "
                "Polygon and polyhedral zones require the low-level API."
            )

        # Convert input data to NumPy arrays
        arrays: list[npt.NDArray] = [np.asarray(arr) for arr in data]

        # Set default title if none provided
        if title is None:
            title = f"FE_Zone_{self.current_zone + 1}"

        # Default variable names (only used on lazy-open first-zone call)
        if variables is None:
            variables = [f"V{i}" for i in range(1, len(data) + 1)]

        # Open and initialise the file if lazily loaded
        if self.handle is None:
            self._open(variables)
            self.flush_aux()

        # Infer per-variable data types from array dtypes
        variable_types = [
            _resolve_written_type(_infer_data_type(arr.dtype), self.precision)
            for arr in arrays
        ]

        # Set default value locations
        if value_locations is None:
            value_locations = [ValueLocation.NODAL] * len(arrays)

        # Default passive / sharing arrays
        if passive_vars is None:
            passive_vars = [False] * len(self._check_variables())
        if var_sharing is None:
            var_sharing = [0] * len(self._check_variables())
        if con_sharing is None:
            con_sharing = 0

        # Check data for consistent number of variables
        if len(arrays) != len(self._check_variables()):
            # Calcuate the number of expected variables based on passive and shared
            # variables
            expected_vars = sum(
                1
                for is_passive, sharing_zone_idx in zip(
                    passive_vars, var_sharing, strict=True
                )
                if not is_passive and not sharing_zone_idx
            )
            if expected_vars == 0:
                raise ValueError(
                    "No active variables to write. All variables are either passive or "
                    "shared."
                )
            elif len(arrays) == 0:
                raise ValueError(
                    "No data arrays provided for active variables. Expected "
                    f"{expected_vars} active variables based on passive_vars and "
                    "var_sharing settings."
                )
            elif len(arrays) != expected_vars:
                raise ValueError(
                    f"Expected {expected_vars} data arrays, got {len(arrays)}"
                )

        # Derive num_nodes and num_cells from node_map (read meta if shared)
        if node_map is not None:
            node_map_arr = np.asarray(node_map)
            num_cells = node_map_arr.shape[0]
            num_nodes = int(node_map_arr.max())
        elif con_sharing:
            src_zone = self._meta.zone(con_sharing)
            if (
                src_zone is None
                or src_zone.num_nodes is None
                or src_zone.num_elements is None
            ):
                raise ValueError(
                    f"con_sharing={con_sharing} references a zone that has "
                    "not been written yet, or is not a finite-element zone."
                )
            num_nodes = src_zone.num_nodes
            num_cells = src_zone.num_elements
        else:
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
                    raise ValueError(
                        f"Variable {var_idx} shares from zone {src} with "
                        f"{src_zone.num_elements} cells, which does not "
                        f"match this zone's cell count of {num_cells}."
                    )
            elif src_zone.num_nodes != num_nodes:
                raise ValueError(
                    f"Variable {var_idx} shares from zone {src} with "
                    f"{src_zone.num_nodes} nodes, which does not match "
                    f"this zone's node count of {num_nodes}."
                )

        # Local variable data shape validation
        for i, (arr, loc) in enumerate(zip(arrays, value_locations, strict=True)):
            if (loc == ValueLocation.NODAL) and (arr.size != num_nodes):
                raise ValueError(
                    f"Array {i} is NODAL but has {arr.size} values, "
                    f"expected {num_nodes}"
                )
            elif (loc == ValueLocation.CELL_CENTERED) and (arr.size != num_cells):
                raise ValueError(
                    f"Array {i} is CELL_CENTERED but has {arr.size} values, "
                    f"expected {num_cells}"
                )

        # Determine face-neighbor count for the zone header
        face_neighbors_arr: npt.NDArray | None = (
            np.asarray(face_neighbors) if face_neighbors is not None else None
        )
        num_face_cons = len(face_neighbors_arr) if face_neighbors_arr is not None else 0

        # Determine 1-based index of dataset variables to write (not passive or shared)
        active_var_idx = [
            var_idx
            for var_idx, (is_passive, share_zone) in enumerate(
                zip(passive_vars, var_sharing, strict=True),
                start=1,
            )
            if (not is_passive) and (not share_zone)
        ]

        # Define global var type and value locations using active variable index
        variable_types_global = [DataType.DOUBLE] * len(self._check_variables())
        value_locations_global = [ValueLocation.NODAL] * len(self._check_variables())

        # Replace the active indices with the real types/locations
        for local_idx, var_idx in enumerate(active_var_idx):
            variable_types_global[var_idx - 1] = variable_types[local_idx]
            value_locations_global[var_idx - 1] = value_locations[local_idx]

        # Write zone header
        self.current_zone = libtecio.tec_zone_create_fe(
            self._check_handle(),
            title,
            zone_type,
            num_nodes,
            num_cells,
            var_types=variable_types_global,
            value_locations=value_locations_global,
            pas_vars=passive_vars,
            var_sharing=var_sharing,
            con_sharing=con_sharing,
            num_face_cons=num_face_cons,
            face_nbr_mode=face_nbr_mode,
        )

        # Unsteady options
        if strand_id != 0 or solution_time != 0.0:
            libtecio.tec_zone_set_unsteady_options(
                handle=self._check_handle(),
                zone=self.current_zone,
                strand=strand_id,
                solution_time=solution_time,
            )

        # Write zone-level aux data
        if aux is not None:
            _write_zone_aux_data(self._check_handle(), {self.current_zone: aux})

        # Write active data only
        for var_idx, arr, dtype in zip(
            active_var_idx, arrays, variable_types, strict=True
        ):
            self.current_var = var_idx
            _write_data(
                self._check_handle(),
                zone_num=self.current_zone,
                var_num=var_idx,
                data=arr,
                dt=dtype,
            )

        # Write connectivity (if not shared)
        if not con_sharing:
            assert node_map is not None
            _write_connectivity(
                self._check_handle(),
                self.current_zone,
                node_map,
                face_neighbors_arr,
            )

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
