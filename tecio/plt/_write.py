"""Higher level API for writing PLT binary files using the classic TecIO API."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from .. import libtecio
from ..libtecio import (
    DataType,
    FaceNeighborMode,
    FileFormat,
    FileType,
    ValueLocation,
    ZoneType,
)

# ---------------------------------------------------------------------------
# Zone-type sets (mirrors szl.py)
# ---------------------------------------------------------------------------

#: FE zone types supported by :meth:`Write.write_fe_zone`.
_FE_SIMPLE: frozenset[ZoneType] = frozenset({
    ZoneType.FELINESEG,
    ZoneType.FETRIANGLE,
    ZoneType.FEQUADRILATERAL,
    ZoneType.FETETRAHEDRON,
    ZoneType.FEBRICK,
})


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _infer_data_type(dt: DataType | np.dtype) -> DataType:
    """Return a C-supported :class:`DataType` for a :class:`DataType` or NumPy dtype.

    Maps NumPy dtypes to the closest supported Tecplot data type.  For PLT format only
    FLOAT and DOUBLE are supported; all integer types are upcast to DOUBLE.  precision
    is promoted to ``FLOAT``; 64-bit integers are promoted to ``INT32``; ``int8`` and
    ``uint8`` map to ``BYTE``.
    """
    closest: dict[np.dtype, DataType] = {
        np.dtype(np.float64): DataType.DOUBLE,
        np.dtype(np.float32): DataType.FLOAT,
        np.dtype(np.float16): DataType.FLOAT,
        np.dtype(np.int64): DataType.INT32,
        np.dtype(np.int32): DataType.INT32,
        np.dtype(np.int16): DataType.INT16,
        np.dtype(np.int8): DataType.BYTE,
        np.dtype(np.uint8): DataType.BYTE,
    }

    if isinstance(dt, DataType):
        return dt

    dt_np = np.dtype(dt)

    for key, val in closest.items():
        if dt_np == key:
            return val

    if np.issubdtype(dt_np, np.floating):
        return DataType.FLOAT if dt_np.itemsize <= 4 else DataType.DOUBLE

    if np.issubdtype(dt_np, np.signedinteger):
        return DataType.INT16 if dt_np.itemsize <= 2 else DataType.INT32

    if np.issubdtype(dt_np, np.unsignedinteger):
        return DataType.BYTE

    raise ValueError(f"Unsupported dtype: {dt_np}")


# ---------------------------------------------------------------------------
# Write class
# ---------------------------------------------------------------------------


class Write:
    """Write Tecplot PLT (``.plt``) files using the classic TecIO API.

    The classic TecIO API maintains a single implicit global file context;
    only one PLT file may be open at a time.  This class wraps that
    procedural API behind the same interface as :class:`szl.Write`.

    Dataset- and variable-level auxiliary data can be set at any time before
    the first zone is written.  They are buffered and flushed automatically
    when the first zone header is created.  Zone-level auxiliary data is
    passed directly to each zone-writing method.

    Like :class:`szl.Write`, the file can be opened *eagerly* (when
    ``variables`` is supplied to ``__init__``) or *lazily* (deferred until
    the first :meth:`write_ijk_zone` or :meth:`write_fe_zone` call, at
    which point ``variables`` must be provided to that call).

    .. warning::
        Unlike the SZL writer, zone data must be written strictly in order:
        header → variable data → connectivity.  This is enforced internally
        by the write methods.

    Args:
        path:      Output file path (should end in ``.plt``).
        title:     Dataset title string.
        variables: Variable name list.  If provided the file is opened
                   immediately.  If ``None``, the file is opened on the first
                   zone write.
        file_type: :class:`~libtecio.FileType` enum.  Defaults to
                   :attr:`~libtecio.FileType.FULL`.

    Example — eager open::

        with plt.Write("out.plt", variables=["X", "Y", "P"]) as w:
            w.write_ijk_zone(data=[x, y, p], title="Zone 1")

    Example — lazy open::

        with plt.Write("out.plt") as w:
            w.write_ijk_zone(
                data=[x, y, p],
                variables=["X", "Y", "P"],
                title="Zone 1",
            )

    """

    def __init__(
        self,
        path: str,
        title: str = "untitled",
        variables: list[str] | None = None,
        file_type: FileType = FileType.FULL,
    ) -> None:
        """Store configuration; open the file immediately if *variables* given."""
        self.path = path
        self.title = title
        self.variables: list[str] | None = variables
        self.file_type = file_type
        self.current_zone: int = 0

        # Buffered aux data — flushed to disk before the first zone is created.
        # Dataset-level: {name: value}
        self.auxdataset: dict[str, str] = {}
        # Variable-level: {var_name_or_1based_index: {name: value}}
        self.auxvar: dict[int, dict[str, str]] = {}

        # Track whether the file has been opened so we know whether to call
        # tecini142 inside open().
        self._opened: bool = False

        if self.variables is not None:
            self._open(self.variables)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> Write:
        """Support ``with`` statement — returns *self*."""
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Close the file on context-manager exit.

        The file is closed regardless of whether an exception was raised in
        the ``with`` block.  If closing itself raises, that secondary
        exception is only re-raised when the ``with`` block completed without
        error; otherwise the original exception takes precedence.
        """
        try:
            self.close()
        except Exception:
            if exc_type is None:
                raise

    # ------------------------------------------------------------------
    # File lifecycle
    # ------------------------------------------------------------------

    def _open(self, var_names: list[str]) -> None:
        """Call ``tecini142`` and record the variable list.

        This is called at most once per instance — either from ``__init__``
        (eager open) or from the first zone-writing method (lazy open).

        Args:
            var_names: Ordered list of variable name strings.

        """
        self.variables = var_names
        libtecio.tecini142(
            filename=self.path,
            variables=self.variables,
            title=self.title,
            file_format=FileFormat.PLT,
            file_type=self.file_type,
        )
        self._opened = True

    def close(self) -> None:
        """Finalize and close the PLT file (safe to call more than once).

        Calls ``tecend142`` only if the file was opened.
        """
        if self._opened:
            libtecio.tecend142()
            self._opened = False

    def add_auxdataset_dict(self, auxdict: dict[str, Any]) -> None:
        """Create buffered auxdataset items from input dictionary."""
        self.auxdataset.update(auxdict)

    def add_auxvar_dict(self, auxdict: dict[int, dict[str, Any]]) -> None:
        """Create buffered auxvar items from input dictionary."""
        self.auxvar.update(auxdict)

    def flush_aux(self) -> None:
        """Write buffered dataset- and variable-level aux data to the file.

        Must be called *after* ``tecini142`` and *before* the first
        ``teczne142``.  The zone-writing methods call this automatically;
        users need not call it directly unless flushing explicitly.

        In the classic API:

        * Dataset aux data is written via ``tecauxstr142``.
        * Variable aux data is written via ``tecvauxstr142``.

        Both must appear before the first zone header.
        """
        # Dataset-level aux data
        for name, value in self.auxdataset.items():
            libtecio.tecauxstr142(str(name), str(value))

        # Variable-level aux data.  Keys may be 1-based int indices or names.
        for key, subdict in self.auxvar.items():
            if isinstance(key, int):
                var_idx = key - 1  # Convert to 0-based
                if var_idx < 1 not in range(len(self.variables)):
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
                libtecio.tecvauxstr142(var_idx + 1, str(name), str(value))

        # Clear buffers — each item is written exactly once.
        self.auxdataset.clear()
        self.auxvar.clear()

    # ------------------------------------------------------------------
    # Zone writers
    # ------------------------------------------------------------------

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
    ) -> None:
        """Write a complete IJK-ordered zone.

        The zone header (``teczne142``), variable data (``tecdat142``), and
        optional zone aux data (``teczauxstr142``) are all written in strict
        sequence, as required by the classic TecIO API.

        The zone dimensions ``imax × jmax × kmax`` are inferred from the
        shape of the first nodal data array.  1-D arrays produce a
        ``(N, 1, 1)`` zone; 2-D arrays produce an ``(I, J, 1)`` zone.

        Args:
            data:            Sequence of NumPy arrays, one per active variable.
                             NODAL arrays must all share the same shape.
                             CELL_CENTERED arrays must have shape
                             ``(imax-1, jmax-1, kmax-1)`` (minimum 1 in each
                             dimension).
            title:           Zone title.  Defaults to
                             ``"IJK_Zone_{current_zone + 1}"``.
            variables:       Variable names — required when the file has not
                             been opened yet (lazy-open path).  Ignored on
                             subsequent zones once the file is initialised.
            value_locations: Per-variable :class:`ValueLocation`.  Defaults
                             to all :attr:`~ValueLocation.NODAL`.
            passive_vars:    Per-variable passive flags (one per dataset
                             variable, not just active variables).  Defaults
                             to all active (``False``).
            var_sharing:     Per-variable source-zone sharing indices (one per
                             dataset variable).  ``0`` means no sharing.
                             Defaults to no sharing.
            solution_time:   Solution time for transient data (``0.0``
                             indicates steady-state).
            strand_id:       Strand ID for transient data (``0`` indicates
                             steady-state).
            aux:             Zone-level auxiliary data as ``{name: value}``
                             strings.  Written immediately after the zone
                             header and before variable data.

        Raises:
            ValueError: If the number of supplied data arrays does not match
                        the number of active (non-passive, non-shared)
                        variables, or if array shapes are inconsistent.

        """
        # Default title
        if title is None:
            title = f"IJK_Zone_{self.current_zone + 1}"

        # Default variable names (only used on lazy-open first-zone call)
        if variables is None:
            variables = [f"V{i}" for i in range(1, len(data) + 1)]

        # Lazy open and flush buffered aux data before the first zone
        if not self._opened:
            self._open(variables)
            self.flush_aux()

        # Per-active-variable data types inferred from array dtypes
        variable_types = [_infer_data_type(arr.dtype) for arr in data]

        # Default value locations — all nodal
        if value_locations is None:
            value_locations = [ValueLocation.NODAL] * len(data)

        # Default passive / sharing arrays — length equals dataset variable count
        if passive_vars is None:
            passive_vars = [False] * len(self.variables)
        if var_sharing is None:
            var_sharing = [0] * len(self.variables)

        # Validate active variable count
        if len(data) != len(self.variables):
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
            if len(data) == 0:
                raise ValueError(
                    f"No data arrays provided. Expected {expected_vars} active "
                    "variable arrays based on passive_vars and var_sharing."
                )
            if len(data) != expected_vars:
                raise ValueError(
                    f"Expected {expected_vars} data arrays for active variables, "
                    f"got {len(data)}."
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

        # Validate individual array shapes
        for i, (arr, loc) in enumerate(zip(data, value_locations, strict=True)):
            if arr.ndim != ndims:
                raise ValueError(f"Array {i} is {arr.ndim}D, expected {ndims}D.")
            shape = arr.shape + (1,) * (3 - arr.ndim)
            if loc == ValueLocation.NODAL and shape != nodal_shape:
                raise ValueError(
                    f"Array {i} is NODAL but has shape {shape}, expected {nodal_shape}."
                )
            if loc == ValueLocation.CELL_CENTERED and shape != cell_shape:
                raise ValueError(
                    f"Array {i} is CELL_CENTERED but has shape {shape}, "
                    f"expected {cell_shape}."
                )

        # Determine active variable indices (1-based, dataset-level)
        active_var_idx = [
            var_idx
            for var_idx, (is_passive, share_zone) in enumerate(
                zip(passive_vars, var_sharing, strict=True),
                start=1,
            )
            if not is_passive and not share_zone
        ]

        # Build global (dataset-length) value-location and type lists —
        # passive / shared positions use placeholder values (they are not
        # written but the arrays must be the right length for teczne142).
        value_locations_global = [ValueLocation.NODAL] * len(self.variables)
        for local_idx, var_idx in enumerate(active_var_idx):
            value_locations_global[var_idx - 1] = value_locations[local_idx]

        # Write zone header
        libtecio.teczne142(
            zone_title=title,
            zone_type=ZoneType.ORDERED,
            imax=imax,
            jmax=jmax,
            kmax=kmax,
            value_locations=value_locations_global,
            pas_vars=passive_vars,
            var_sharing=var_sharing if any(var_sharing) else None,
            con_sharing=0,
            strand=strand_id,
            solution_time=solution_time,
        )
        self.current_zone += 1

        # Zone-level aux data must be written immediately after teczne142 and
        # before the first tecdat142 call.
        if aux is not None:
            for name, value in aux.items():
                libtecio.teczauxstr142(str(name), str(value))

        # Write active variable data in dataset-variable order
        for var_idx, arr, dtype in zip(
            active_var_idx, data, variable_types, strict=True
        ):
            self.current_var = var_idx
            write_data(arr, dtype)

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
    ) -> None:
        """Write a complete finite-element zone.

        The full sequence required by the classic TecIO API is issued
        internally:

        1. ``teczne142`` — zone header.
        2. ``teczauxstr142`` — zone-level aux data (if any), before data.
        3. ``tecdat142`` — one call per active variable.
        4. ``tecnode142`` — node map.
        5. ``tecface142`` — face-neighbour connections (if provided).

        ``FEPOLYGON`` and ``FEPOLYHEDRON`` zone types are not supported
        (they require ``tecpolyface142`` / ``tecpolybconn142``); use the
        low-level libtecio API directly for those types.

        Args:
            zone_type:       FE zone type.  Must be one of :data:`_FE_SIMPLE`.
            data:            Sequence of 1-D NumPy arrays, one per active
                             variable.  NODAL arrays must have length
                             ``num_nodes``; CELL_CENTERED arrays must have
                             length ``num_cells``.  Both counts are inferred
                             from *node_map*.
            node_map:        Integer array of shape
                             ``(num_cells, nodes_per_cell)`` with 1-based node
                             indices.
            title:           Zone title.  Defaults to
                             ``"FE_Zone_{current_zone + 1}"``.
            variables:       Variable names — required on the lazy-open first
                             zone call.
            value_locations: Per-variable :class:`ValueLocation`.  Defaults
                             to all :attr:`~ValueLocation.NODAL`.
            passive_vars:    Per-variable passive flags (dataset-length).
                             Defaults to all active.
            var_sharing:     Per-variable sharing zone indices (dataset-
                             length).  Defaults to no sharing.
            con_sharing:     Source-zone index for connectivity sharing.
                             Pass ``0`` for no sharing (mandatory for the
                             first zone).
            face_neighbors:  Optional face-neighbour connection array.  Its
                             length sets ``num_face_connections`` in the zone
                             header automatically.
            face_nbr_mode:   Face-neighbour mode; used only when
                             *face_neighbors* is provided.
            solution_time:   Solution time for transient data.
            strand_id:       Strand ID for transient data.
            aux:             Zone-level auxiliary data as ``{name: value}``.

        Raises:
            NotImplementedError: If *zone_type* is not in :data:`_FE_SIMPLE`.
            ValueError:          On variable count or array length mismatch.

        """
        if zone_type not in _FE_SIMPLE:
            raise NotImplementedError(
                f"Zone type {zone_type.name!r} is not supported by "
                "write_fe_zone.  FEPOLYGON and FEPOLYHEDRON zones require "
                "the low-level libtecio API."
            )

        # Default title
        if title is None:
            title = f"FE_Zone_{self.current_zone + 1}"

        # Default variable names (only used on lazy-open first-zone call)
        if variables is None:
            variables = [f"V{i}" for i in range(1, len(data) + 1)]

        # Lazy open and flush buffered aux data before the first zone
        if not self._opened:
            self._open(variables)
            self.flush_aux()

        # Per-active-variable data types
        variable_types = [_infer_data_type(arr.dtype) for arr in data]

        # Default value locations — all nodal
        if value_locations is None:
            value_locations = [ValueLocation.NODAL] * len(data)

        # Default passive / sharing arrays
        if passive_vars is None:
            passive_vars = [False] * len(self.variables)
        if var_sharing is None:
            var_sharing = [0] * len(self.variables)

        # Validate active variable count
        if len(data) != len(self.variables):
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
            if len(data) == 0:
                raise ValueError(
                    f"No data arrays provided. Expected {expected_vars} active "
                    "variable arrays based on passive_vars and var_sharing."
                )
            if len(data) != expected_vars:
                raise ValueError(
                    f"Expected {expected_vars} data arrays for active variables, "
                    f"got {len(data)}."
                )

        # Derive num_nodes and num_cells from node_map.
        # node_map shape is (num_cells, nodes_per_cell); max value = num_nodes.
        node_map_arr = np.asarray(node_map)
        num_cells = node_map_arr.shape[0]
        num_nodes = int(node_map_arr.max())

        # Validate per-variable array lengths
        for i, (arr, loc) in enumerate(zip(data, value_locations, strict=True)):
            if loc == ValueLocation.NODAL and arr.size != num_nodes:
                raise ValueError(
                    f"Array {i} is NODAL but has {arr.size} values; "
                    f"expected {num_nodes}."
                )
            if loc == ValueLocation.CELL_CENTERED and arr.size != num_cells:
                raise ValueError(
                    f"Array {i} is CELL_CENTERED but has {arr.size} values; "
                    f"expected {num_cells}."
                )

        # Face-neighbour count for zone header
        num_face_cons = len(face_neighbors) if face_neighbors is not None else 0

        # Determine active variable indices (1-based, dataset-level)
        active_var_idx = [
            var_idx
            for var_idx, (is_passive, share_zone) in enumerate(
                zip(passive_vars, var_sharing, strict=True),
                start=1,
            )
            if not is_passive and not share_zone
        ]

        # Build global value-location list for teczne142
        value_locations_global = [ValueLocation.NODAL] * len(self.variables)
        for local_idx, var_idx in enumerate(active_var_idx):
            value_locations_global[var_idx - 1] = value_locations[local_idx]

        # Write zone header.
        # imax = num_nodes, jmax = num_cells, kmax = 0 for simple FE zones.
        libtecio.teczne142(
            zone_title=title,
            zone_type=zone_type,
            imax=num_nodes,
            jmax=num_cells,
            kmax=0,
            value_locations=value_locations_global,
            pas_vars=passive_vars,
            var_sharing=var_sharing if any(var_sharing) else None,
            con_sharing=con_sharing,
            strand=strand_id,
            solution_time=solution_time,
            num_face_connections=num_face_cons,
            face_nbr_mode=face_nbr_mode
            if num_face_cons > 0
            else FaceNeighborMode.LOCAL_ONE_TO_ONE,
        )
        self.current_zone += 1

        # Zone-level aux data must be immediately after teczne142, before
        # any tecdat142 calls.
        if aux is not None:
            for name, value in aux.items():
                libtecio.teczauxstr142(str(name), str(value))

        # Write active variable data in dataset-variable order
        for var_idx, arr, dtype in zip(
            active_var_idx, data, variable_types, strict=True
        ):
            self.current_var = var_idx
            write_data(arr, dtype)

        # Write connectivity (must come after all tecdat142 calls)
        if (not con_sharing) or (self.current_zone == 1):
            # If first zone, must supply connectivity
            write_connectivity(node_map, face_neighbors)


def write_data(arr: npt.ArrayLike, dt: np.dtype | DataType | None = None) -> None:
    """Write a single variable's data using ``tecdat142``.

    ``tecdat142`` only accepts ``float32`` or ``float64``; integer types are
    upcast to ``float64``.  The array is ravelled in Fortran (column-major)
    order before writing, which matches Tecplot's expected BLOCK layout.

    Args:
        arr:       Array-like of values to write.
        dt:        Intended :class:`DataType` for the variable.

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

    data_type = _infer_data_type(dt) if dt is not None else dtype_to_datatype[arr.dtype]

    # tecdat142 already returns a contiguous array so no need to cast before calling
    if data_type in (DataType.FLOAT,):
        libtecio.tecdat142(arr.ravel(order="F"), is_double=False)
    else:
        # DOUBLE, INT32, INT16, BYTE — all written as float64 in PLT format.
        libtecio.tecdat142(arr.ravel(order="F"), is_double=True)


def write_connectivity(
    node_map: npt.ArrayLike,
    face_neighbors: npt.ArrayLike | None = None,
) -> None:
    """Write FE connectivity: node map and optional face-neighbour data.

    Args:
        node_map:       Integer array of shape ``(num_cells, nodes_per_cell)``
                        with 1-based node indices.
        face_neighbors: Optional face-neighbour connection array.

    """
    nodes_flat = np.ascontiguousarray(node_map, dtype=np.int32).ravel(order="C")
    libtecio.tecnode142(nodes_flat)

    if face_neighbors is not None:
        face_flat = np.ascontiguousarray(face_neighbors, dtype=np.int32).ravel(
            order="C"
        )
        libtecio.tecface142(face_flat)
