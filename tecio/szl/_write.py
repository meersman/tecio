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
from ..libtecio import (
    DataPacking,
    DataType,
    FaceNeighborMode,
    FileType,
    ValueLocation,
    ZoneType,
)

# FE zone types that use tec_zone_create_fe
_FE_SIMPLE: frozenset[ZoneType] = frozenset({
    ZoneType.FELINESEG,
    ZoneType.FETRIANGLE,
    ZoneType.FEQUADRILATERAL,
    ZoneType.FETETRAHEDRON,
    ZoneType.FEBRICK,
})


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
        path (str): Output file path.
        title (str): Dataset title.
        variables (list[str] | None): Variable name list. ``None`` defers
            file creation.
        file_type (FileType): File type enum (FULL, GRID, or SOLUTION).

    Attributes:
        current_zone: 1-based index of the most recently created zone.
        auxdataset: Buffered dataset-level auxiliary data.
        auxvar: Buffered variable-level auxiliary data.
    """

    # Class-level annotations so autodoc can discover instance attributes.
    path: str
    """Output file path."""

    title: str
    """Dataset title string."""

    variables: list[str] | None
    """Variable name list, or ``None`` if the file has not been opened yet."""

    file_type: FileType
    """File type (FULL, GRID, or SOLUTION)."""

    current_zone: int
    """1-based index of the most recently written zone (initialized to 0)."""

    auxdataset: dict[str, Any]
    """Buffered dataset-level auxiliary data, flushed before the first zone."""

    auxvar: dict[int, dict[str, Any]]
    """Buffered variable-level auxiliary data, flushed before the first zone."""

    def __init__(
        self,
        path: str,
        title: str = "untitled",
        variables: list[str] | None = None,
        file_type: FileType = FileType.FULL,
    ) -> None:
        """Store minimum necessary info until first zone is ready to write."""
        self.path = path
        self.title = title
        self.variables = variables
        self.file_type = file_type
        self.current_zone = 0
        self.current_var = 0

        # Add created data to the buffer and flush once a file handle is created.
        # Dataset-level aux data buffer (flushed on first zone)
        self.auxdataset: dict[str, str] = {}
        # Variable-level aux data buffer: {var_name: {key: value}}
        self.auxvar: dict[int, dict[str, str]] = {}

        # Initialize if all needed info provided, else set to null
        if self.variables is not None:
            self.handle: ctypes.c_void_p | None = libtecio.tec_file_writer_open(
                filename=self.path,
                variables=self.variables,
                title=self.title,
                file_type=self.file_type,
                use_szl=1,
            )
        else:
            # Variables needed
            self.handle = None

    # Context manager
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
            libtecio.tec_data_set_add_aux_data(self.handle, str(name), str(value))

        # Variable-level aux data.  Keys may be 1-based int indices or names.
        for key, subdict in self.auxvar.items():
            if isinstance(key, int):
                var_idx = key - 1  # Convert to 0-based
                if var_idx not in range(len(self.variables)):
                    raise IndexError(
                        f"Variable index {var_idx} out of bounds "
                        f"[1, {len(self.variables)}]"
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
                    self.handle, var_idx + 1, str(name), str(value)
                )

        # Clear buffers — each item is written exactly once.
        self.auxdataset.clear()
        self.auxvar.clear()

    # ------------------------------------------------------------------
    # Zone writers
    # ------------------------------------------------------------------

    def write_ijk_zone(
        self,
        data: Sequence[npt.NDArray] | None,
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

        Dimensions are inferred from the first array's shape. Arrays may
        be 1-D, 2-D, or 3-D; missing trailing dimensions default to 1.

        Args:
            data: One NumPy array per dataset variable.  Array shape is used
                to infer ``imax``, ``jmax``, and ``kmax``; Fortran (column-major)
                order is assumed.  Pass ``None`` to write a zone header only.
            title: Zone title.  Defaults to ``"IJK_Zone_{current_zone + 1}"``.
            variables: Variable name list.  Required on the first call when the
                file has not been opened yet (lazy-open path); ignored once the
                file is already initialised.
            value_locations: Per-variable :class:`~libtecio.ValueLocation`.
                Defaults to all ``NODAL``.
            passive_vars: Per-variable passive flags.  Defaults to all active
                (``False``).
            var_sharing: Per-variable share-from zone index (1-based).
                Defaults to no sharing (all zeros).
            solution_time: Solution time for transient data.  Use ``0.0`` for
                static zones.
            strand_id: Strand ID for transient data.  Use ``0`` for static zones.
            aux: Zone-level auxiliary data as ``{name: value}`` string pairs.
            datapacking: Must be :attr:`~tecio.libtecio.DataPacking.BLOCK` (the
                default).  :attr:`~tecio.libtecio.DataPacking.POINT` is an
                ASCII-only layout and is not supported by the SZL binary format.

        Note:
            If the file is already open, ``data`` and ``variables`` may be
            omitted to write a zone header only.  If the file has not been
            opened yet, ``variables`` must be provided on this call.

        Note:
            Data arrays are written as DOUBLE precision by default.  To write
            other types, cast the NumPy arrays before calling (e.g.
            ``arr.astype(np.float32)``).

        Note:
            Separate grid files (where all variables are cell-centred) are not
            handled automatically; use the low-level API for that case.

        Raises:
            NotImplementedError: If *datapacking* is
                :attr:`~tecio.libtecio.DataPacking.POINT`.
        """
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
        # Set default title if none provided
        if title is None:
            title = f"IJK_Zone_{self.current_zone + 1}"

        # Set default variable names if none provided. Only relevant if file lazily
        # loaded and first zone
        if variables is None:
            variables = [f"V{i}" for i in range(1, len(data) + 1)]

        # Open and initialize the file if lazily loaded
        if self.handle is None:
            self._open(variables)
            self.flush_aux()

        # Get variable types (local variable -> length = number of supplied data arrays)
        variable_types = [_infer_data_type(arr.dtype) for arr in data]

        # Set default value loacations
        if value_locations is None:
            # Local variable -> length = number of supplied data arrays
            value_locations = [ValueLocation.NODAL] * len(data)

        # Default passive / sharing arrays — length equals dataset variable count
        if passive_vars is None:
            passive_vars = [False] * len(self.variables)
        if var_sharing is None:
            var_sharing = [0] * len(self.variables)

        # Validate active variable count
        if len(data) != len(self.variables):
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
            elif len(data) == 0:
                raise ValueError(
                    f"No data arrays provided. Expected {expected_vars} active "
                    "variable arrays based on passive_vars and var_sharing."
                )
            elif len(data) != expected_vars:
                raise ValueError(
                    f"Expected {expected_vars} data arrays, got {len(data)}"
                )

        # Infer zone dimensions from first nodal array or first cell array if no nodal
        nodal_indices = [
            i for i, loc in enumerate(value_locations) if loc == ValueLocation.NODAL
        ]
        cell_indices = [
            i
            for i, loc in enumerate(value_locations)
            if loc == ValueLocation.CELL_CENTERED
        ]
        if len(nodal_indices) >= 1:
            ndims = data[nodal_indices[0]].ndim
            if ndims not in (1, 2, 3):
                raise ValueError(
                    f"Arrays must be 1D, 2D, or 3D; got {ndims}-D array.  "
                    "For time-dependent data, write each time step as a separate zone."
                )

            nodal_shape = data[nodal_indices[0]].shape + (1,) * (3 - ndims)
            cell_shape = tuple(max(i - 1, 1) for i in nodal_shape)
            imax, jmax, kmax = nodal_shape
        elif len(cell_indices) >= 1:
            ndims = data[cell_indices[0]].ndim
            if ndims not in (1, 2, 3):
                raise ValueError(
                    f"Arrays must be 1D, 2D, or 3D; got {ndims}-D array.  "
                    "For time-dependent data, write each time step as a separate zone."
                )

            cell_shape = data[cell_indices[0]].shape + (1,) * (3 - ndims)
            nodal_shape = tuple(max(i + 1, 1) for i in cell_shape)
            imax, jmax, kmax = nodal_shape
        else:
            raise ValueError(
                "Could not determine nodal and cell-centered indices. "
                f"Got Nodal: {nodal_indices}, Cell-centered: {cell_indices}"
            )

        # Data shape validation
        for i, (arr, loc) in enumerate(zip(data, value_locations, strict=True)):
            # Check dimension of array
            if arr.ndim != ndims:
                raise ValueError(f"Array {i} is {arr.ndim}D, expected {ndims}D")
            shape = arr.shape + (1,) * (3 - arr.ndim)
            if (loc == ValueLocation.NODAL) and (shape != nodal_shape):
                raise ValueError(
                    f"Array {i} is NODAL but has shape {shape}, expected {nodal_shape}"
                )
            elif (loc == ValueLocation.CELL_CENTERED) and (shape != cell_shape):
                raise ValueError(
                    f"Array {i} is CELL_CENTERED but has shape {shape}, "
                    f"expected {cell_shape}"
                )

        # Determine which variables to write based on (not passive and not shared,
        # 1-based)
        active_var_idx = [
            var_idx
            for var_idx, (is_passive, sharing_zone_idx) in enumerate(
                zip(passive_vars, var_sharing, strict=True),
                start=1,
            )
            if (not is_passive) and (not sharing_zone_idx)
        ]

        # Define global var type and value locations using active variable index
        variable_types_global = [DataType.DOUBLE] * len(self.variables)
        value_locations_global = [ValueLocation.NODAL] * len(self.variables)
        # Replace the active indices with the real types/locations
        for local_idx, var_idx in enumerate(active_var_idx):
            variable_types_global[var_idx - 1] = variable_types[local_idx]
            value_locations_global[var_idx - 1] = value_locations[local_idx]

        # Write zone header
        self.current_zone = libtecio.tec_zone_create_ijk(
            self.handle,
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
                handle=self.handle,
                zone=self.current_zone,
                strand=strand_id,
                solution_time=solution_time,
            )

        # Write aux data
        if aux is not None:
            write_zone_aux_data(self.handle, {self.current_zone: aux})

        # Write active data only
        for var_idx, arr, dtype in zip(
            active_var_idx, data, variable_types, strict=True
        ):
            self.current_var = var_idx
            write_data(
                self.handle,
                zone_num=self.current_zone,
                var_num=var_idx,
                data=arr,
                dt=dtype,
            )

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

        Node and cell counts are inferred from *node_map*. The 32- or
        64-bit write path is chosen automatically from the max index.

        Args:
            zone_type: FE zone type from the ZoneType enum.  Must be one of the types in
                ``_FE_SIMPLE``.
            data: Sequence of 1-D arrays, one per dataset variable.  NODAL arrays must
                have length ``num_nodes``; CELL_CENTERED arrays must have length
                ``num_cells``.  ``num_nodes`` and ``num_cells`` are inferred from
                ``node_map``.
            node_map: Integer array of shape ``(num_cells, nodes_per_cell)`` containing
                1-based node indices.  32- or 64-bit write is chosen automatically based
                on the maximum index value.
            title: Zone title string.  Defaults to ``"FE_Zone_{current_zone + 1}"`` if
                not provided.
            variables: Variable name list.  Required only when the file has not been
                opened yet (lazy-open path).  Ignored on subsequent zones once the file
                is already initialised.
            value_locations: Per-variable ValueLocation.  Defaults to all NODAL.
            passive_vars: Per-variable passive flags.  Defaults to all active (False).
            var_sharing: Per-variable share from zone index.  Defaults to no sharing.
            con_sharing: optional zone index that the connectivity is shared from Pass 0
                to indicate no connectivity. You must pass 0 for the first zone in a
                dataset. Connectivity cannot be shared when face neighbor mode is set to
                global. Connectivity cannot be shared between cell-based and face-based
                finite element zones.
            face_neighbors: Optional face-neighbor connectivity array.
                ``num_face_cons`` in the zone header is set to ``len(face_neighbors)``
                automatically when this is supplied.
            face_nbr_mode: Face-neighbor mode, used only when ``face_neighbors`` is
                provided.  Defaults to LOCAL_ONE_TO_ONE.
            solution_time: Solution time for transient data (0.0 = static).
            strand_id: Strand ID for transient data (0 = static).
            aux: Zone-level auxiliary data as ``{name: value}`` strings.
            datapacking: Must be :attr:`~tecio.libtecio.DataPacking.BLOCK` (the
                default).  :attr:`~tecio.libtecio.DataPacking.POINT` is an
                ASCII-only layout and is not supported by the SZL binary format.

        Raises:
            NotImplementedError: For FEPOLYGON, FEPOLYHEDRON, or if *datapacking*
                is :attr:`~tecio.libtecio.DataPacking.POINT`.
            ValueError: On variable count or array length mismatch.

        Note:
            FE variable arrays are 1-D and node-ordered — no axis-ordering
            considerations apply (unlike IJK zones).  ``write_data`` handles
            dtype inference and F-order ravel internally; 1-D arrays are
            unaffected by memory order.
        """
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
                f"Zone type {zone_type.name!r} is not supported by write_fe_zone. "
                "Polygon and polyhedral zones require the low-level API."
            )

        # Set default title if none provided
        if title is None:
            title = f"FE_Zone_{self.current_zone + 1}"

        # Set default variable names if none provided — only relevant on
        # the lazy-open first-zone call
        if variables is None:
            variables = [f"V{i}" for i in range(1, len(data) + 1)]

        # Open and initialise the file if lazily loaded
        if self.handle is None:
            self._open(variables)
            self.flush_aux()

        # Infer per-variable data types from array dtypes
        variable_types = [_infer_data_type(arr.dtype) for arr in data]

        # Set default value locations
        if value_locations is None:
            value_locations = [ValueLocation.NODAL] * len(data)

        # Default passive / sharing arrays
        if passive_vars is None:
            passive_vars = [False] * len(self.variables)
        if var_sharing is None:
            var_sharing = [0] * len(self.variables)

        # Check data for consistent number of variables
        if len(data) != len(self.variables):
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
            elif len(data) == 0:
                raise ValueError(
                    "No data arrays provided for active variables. Expected "
                    f"{expected_vars} active variables based on passive_vars and "
                    "var_sharing settings."
                )
            elif len(data) != expected_vars:
                raise ValueError(
                    f"Expected {expected_vars} data arrays, got {len(data)}"
                )

        # Derive num_nodes and num_cells from node_map.
        # node_map is (num_cells, nodes_per_cell) with 1-based indices, so
        # the maximum value equals the total number of nodes.
        node_map_arr = np.asarray(node_map)
        num_cells = node_map_arr.shape[0]
        num_nodes = int(node_map_arr.max())

        # Validate per-variable array lengths against node / cell counts
        for i, (arr, loc) in enumerate(zip(data, value_locations, strict=True)):
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
        num_face_cons = len(face_neighbors) if face_neighbors is not None else 0

        # Determine which variables to write based on (not passive and not shared,
        # 1-based)
        active_var_idx = [
            var_idx
            for var_idx, (is_passive, share_zone) in enumerate(
                zip(passive_vars, var_sharing, strict=True),
                start=1,
            )
            if (not is_passive) and (not share_zone)
        ]

        # Define global var type and value locations using active variable index
        variable_types_global = [DataType.DOUBLE] * len(self.variables)
        value_locations_global = [ValueLocation.NODAL] * len(self.variables)

        # Replace the active indices with the real types/locations
        for local_idx, var_idx in enumerate(active_var_idx):
            variable_types_global[var_idx - 1] = variable_types[local_idx]
            value_locations_global[var_idx - 1] = value_locations[local_idx]

        # Write zone header
        self.current_zone = libtecio.tec_zone_create_fe(
            self.handle,
            title,
            zone_type,
            num_nodes,
            num_cells,
            var_types=variable_types_global,
            value_locations=value_locations_global,
            pas_vars=passive_vars,
            var_sharing=var_sharing,
            num_face_cons=num_face_cons,
            face_nbr_mode=face_nbr_mode,
        )

        # Unsteady options
        if strand_id != 0 or solution_time != 0.0:
            libtecio.tec_zone_set_unsteady_options(
                handle=self.handle,
                zone=self.current_zone,
                strand=strand_id,
                solution_time=solution_time,
            )

        # Write zone-level aux data
        if aux is not None:
            write_zone_aux_data(self.handle, {self.current_zone: aux})

        # Write active data only
        for var_idx, arr, dtype in zip(
            active_var_idx, data, variable_types, strict=True
        ):
            self.current_var = var_idx
            write_data(
                self.handle,
                zone_num=self.current_zone,
                var_num=var_idx,
                data=arr,
                dt=dtype,
            )

        # Write connectivity
        if (not con_sharing) or (self.current_zone == 1):
            # If first zone, must supply connectivity
            write_connectivity(self.handle, self.current_zone, node_map, face_neighbors)


def write_data(
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


def write_connectivity(
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


def write_zone_aux_data(
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


def write_variable_aux_data(
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


def write_dataset_aux_data(handle: ctypes.c_void_p, aux: dict[str, Any]) -> None:
    """Write dataset-level auxiliary data.

    Args:
        handle (ctypes.c_void_p): C library file handle.
        aux (dict[str, Any]): Mapping of ``{name: value}``.

    Hint:
        Aux data should be structured as ``{var_idx: {name, value}}``
    """
    for name, value in aux.items():
        libtecio.tec_data_set_add_aux_data(handle, str(name), str(value))


def write_aux_data(handle: ctypes.c_void_p, aux: dict[str, dict[Any]]) -> None:
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
