"""Shared, format-independent base class for Tecplot writers.

Every format's writer (SZL, PLT, DAT) exposes the same lifecycle: construct with a path
and optional variable list (eager or lazy open), buffer auxiliary data with
:meth:`~TecplotWriter.add_auxdataset_dict`/ :meth:`~TecplotWriter.add_auxvar_dict`,
write zones with ``write_ijk_zone``/ ``write_fe_zone``, and close (directly or via
context manager). This module defines the parts of that lifecycle that are identical
across formats once, so the three writers can no longer drift apart the way the readers
had.

Notes:
    * Unlike the reader hierarchy, writer instances are not made immutable.  A writer is
      inherently a mutable, sequentially-appended-to object (``current_zone`` grows, aux
      buffers fill and drain, a file handle opens and closes), there's no "already-read,
      now frozen" state to protect the way there was for reader Zone/Variable objects.
    * :meth:`~TecplotWriter.flush_aux` centralizes the aux-data buffer draining and key
      resolution (a 1-based int or a variable name), which was previously duplicated
      nearly verbatim in all three formats. Each format only implements the two small
      hooks that actually differ, writing one dataset-level item and one variable-level
      item.
    * ``_open``, ``close``, ``write_ijk_zone``, and ``write_fe_zone`` stay abstract: how
      a file is opened/closed and how a zone is actually written are genuinely
      format-specific (a live C handle for SZL, a global implicit context for PLT's
      classic API, a plain text file for DAT). Their signatures here are the common
      shape for documentation; Python doesn't enforce exact signature matching on
      ``abstractmethod``, so a format may add its own extra keyword-only parameters
      (SZL's ``flush``, for subzone flushing mid-write) without conflict.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from ._constants import (
    DataPacking,
    DataType,
    FaceNeighborMode,
    FileType,
    ValueLocation,
    ZoneType,
)
from ._meta import WriterMeta

_STR_TO_PRECISION: dict[str, DataType] = {
    "single": DataType.FLOAT,
    "float": DataType.FLOAT,
    "double": DataType.DOUBLE,
}

# Faces per cell
_FACES_PER_ELEM: dict[ZoneType, int] = {
    ZoneType.FELINESEG: 2,
    ZoneType.FETRIANGLE: 3,
    ZoneType.FEQUADRILATERAL: 4,
    ZoneType.FETETRAHEDRON: 4,
    ZoneType.FEBRICK: 6,
}


def count_face_connections(flat: npt.NDArray, mode: FaceNeighborMode) -> int:
    """Count connections in a flat face-neighbor array for the given mode.

    Shared across formats. Always derived from *flat* itself, never asked for
    separately: unlike reading (where a dedicated count avoids loading a potentially
    large array just to know its size), on write the caller already has the full array
    in memory, so deriving costs nothing extra, and it makes a mismatched explicit
    count, silently producing a corrupted file, structurally impossible.

    ``LOCAL_ONE_TO_ONE``/``GLOBAL_ONE_TO_ONE`` connections are a fixed 3 or 4 values
    each, so the count is *len(flat)* divided by that. The "many" modes are ragged: each
    record starts with a 4-value header whose own 4th value (``nz``) gives the count of
    additional neighbor references that follow it, so counting means walking record by
    record until the array is exhausted, the same walk
    :func:`~tecio.reshape_face_connections` does to parse them, just counting steps
    instead of collecting values.

    Raises:
        ValueError: If *flat*'s length doesn't cleanly divide for the one-to-one modes,
            or ends mid-record for the "many" modes, either way, this is a reliable
            signal of malformed input.
    """
    n = len(flat)
    if mode in (FaceNeighborMode.LOCAL_ONE_TO_ONE, FaceNeighborMode.GLOBAL_ONE_TO_ONE):
        values_per = 3 if mode is FaceNeighborMode.LOCAL_ONE_TO_ONE else 4
        if n % values_per != 0:
            raise ValueError(
                f"face_neighbors has {n} values, not a multiple of "
                f"{values_per} as required for {mode.name}."
            )
        return n // values_per

    is_global = mode is FaceNeighborMode.GLOBAL_ONE_TO_MANY
    i = 0
    count = 0
    while i < n:
        if i + 4 > n:
            raise ValueError(
                f"face_neighbors is truncated: a new record starts at "
                f"index {i} but fewer than 4 values remain ({n - i})."
            )
        nz = int(flat[i + 3])
        record_len = 4 + (2 * nz if is_global else nz)
        if i + record_len > n:
            raise ValueError(
                f"face_neighbors is truncated: the record starting at "
                f"index {i} needs {record_len} values but only "
                f"{n - i} remain."
            )
        i += record_len
        count += 1
    return count


def validate_face_neighbor_sharing(
    con_sharing: int | None, face_neighbor_mode: FaceNeighborMode | None
) -> None:
    """Raise if a global face-neighbor mode is combined with con_sharing.

    Shared across formats. Per the classic API's own documented constraint:
    connectivity, and any face-neighbor data associated with it, cannot be shared
    between zones when the face-neighbor mode is global, only local modes support
    sharing (implicitly, along with connectivity itself, via con_sharing, there's no
    separate face-neighbor-sharing mechanism).
    """
    global_modes = (
        FaceNeighborMode.GLOBAL_ONE_TO_ONE,
        FaceNeighborMode.GLOBAL_ONE_TO_MANY,
    )
    if con_sharing and face_neighbor_mode in global_modes:
        raise ValueError(
            f"face_neighbor_mode={face_neighbor_mode.name} cannot be "
            "combined with con_sharing: connectivity (and any face-"
            "neighbor data with it) cannot be shared between zones when "
            "the face-neighbor mode is global."
        )


def validate_face_neighbors(
    flat: npt.NDArray,
    mode: FaceNeighborMode,
    zone_type: ZoneType,
    num_cells: int,
) -> None:
    """Validate face-neighbor connections against this zone's own shape.

    Shared across formats. Structural compatibility only: every local cell reference is
    a valid 1-based cell index for this zone, and every face index is valid for this
    zone's cell type. Doesn't independently verify the *geometric* adjacency itself
    (that two cells stated as sharing a face genuinely share the node set that face
    implies), that would mean recomputing adjacency from node_map ourselves, duplicating
    the very auto-detection Tecplot already does; this only rules out references that
    are structurally impossible regardless of geometry.

    Remote references (the zone/cell pairs in the global modes) aren't checked, they
    refer to a different zone's own cell count, which isn't known here.

    Raises:
        ValueError: If any local cell reference is out of range [1, num_cells], or any
            face index is out of range [1, faces_per_cell] for *zone_type*.
    """
    faces_per_cell = _FACES_PER_ELEM.get(zone_type)
    if faces_per_cell is None:
        return  # poly/mixed zones: no fixed face count, not this function's job

    def check_cell(cz: int, pos: int) -> None:
        if not 1 <= cz <= num_cells:
            raise ValueError(
                f"face_neighbors[{pos}]: cell {cz} is out of range "
                f"[1, {num_cells}] for this zone."
            )

    def check_face(fz: int, pos: int) -> None:
        if not 1 <= fz <= faces_per_cell:
            raise ValueError(
                f"face_neighbors[{pos}]: face {fz} is out of range "
                f"[1, {faces_per_cell}] for {zone_type.name} "
                f"({faces_per_cell} faces per cell)."
            )

    if mode is FaceNeighborMode.LOCAL_ONE_TO_ONE:
        for i in range(0, len(flat), 3):
            check_cell(int(flat[i]), i)
            check_face(int(flat[i + 1]), i + 1)
            check_cell(int(flat[i + 2]), i + 2)
    elif mode is FaceNeighborMode.GLOBAL_ONE_TO_ONE:
        for i in range(0, len(flat), 4):
            check_cell(int(flat[i]), i)
            check_face(int(flat[i + 1]), i + 1)
    else:
        is_global = mode is FaceNeighborMode.GLOBAL_ONE_TO_MANY
        i = 0
        n = len(flat)
        while i < n:
            cz, fz, nz = int(flat[i]), int(flat[i + 1]), int(flat[i + 3])
            check_cell(cz, i)
            check_face(fz, i + 1)
            if is_global:
                record_len = 4 + 2 * nz
            else:
                record_len = 4 + nz
                for j in range(nz):
                    check_cell(int(flat[i + 4 + j]), i + 4 + j)
            i += record_len


@dataclass
class PreparedOrderedZone:
    """Validated, normalized inputs for writing one ordered (IJK) zone.

    Everything a writer needs to proceed straight to its own format-specific zone
    creation dispatch call, already validated and shape-consistent.
    """

    value_locations: list[ValueLocation]
    passive_vars: list[bool]
    var_sharing: list[int]
    active_var_idx: list[int]  # 1-based dataset indices being written now
    imax: int
    jmax: int
    kmax: int
    value_locations_global: list[ValueLocation]
    variable_types_global: list[DataType]


def prepare_ordered_zone(
    arrays: Sequence[npt.NDArray],
    variable_types: Sequence[DataType],
    *,
    value_locations: Sequence[ValueLocation] | None,
    passive_vars: Sequence[bool | int] | None,
    var_sharing: Sequence[int] | None,
    dataset_variables: Sequence[str],
    meta: WriterMeta,
    on_error: Callable[[], None] | None = None,
) -> PreparedOrderedZone:
    """Validate and normalize inputs for writing an ordered (IJK) zone.

    Shared across formats: derives imax/jmax/kmax from the supplied arrays' shapes,
    resolves value_locations/passive_vars/var_sharing defaults, validates the
    active-variable count and every array's shape (local and shared) against a single
    reference shape, and builds the dataset-level (not just active-variable)
    value_locations/ variable_types arrays every format's zone-creation call needs.

    Args:
        arrays:            NumPy arrays already converted from the caller's raw data,
                           one per active (non-passive, non-shared) variable.
        variable_types:    Per-active-variable data type, one per array in
                           *arrays*. Computed by the caller: type-inference strategy is
                           currently
                           format-specific (SZL/DAT infer from each array's own dtype,
                           PLT always uses its
                           configured precision), so it isn't derived here.
        value_locations:   Per-active-variable value location, or None for
                           all-NODAL.
        passive_vars:      Per-dataset-variable passive flags, or None for all
                           active.
        var_sharing:       Per-dataset-variable source-zone sharing indices, or
                           None for no sharing.
        dataset_variables: The full dataset variable name list, already
                           established (after the writer's own lazy-open handling).
        meta:              The writer's own zone metadata, for resolving shared
                           variables' source zones.
        on_error:          Optional callback invoked once, before any ValueError is
                           raised, for a format needing extra cleanup on failure (DAT's
                           partial-file deletion; SZL/PLT need none, the C library
                           handles it, so they pass None).

    Returns:
        A :class:`PreparedOrderedZone` with everything else needed to proceed to the
        format-specific zone-creation call.

    Raises:
        ValueError: On any inconsistency: wrong active-variable count, shape mismatch
            (local or shared), or a shared variable referencing a zone that doesn't
            exist yet or isn't ORDERED.

    """
    try:
        if value_locations is None:
            value_locations = [ValueLocation.NODAL] * len(arrays)
        if passive_vars is None:
            passive_vars = [False] * len(dataset_variables)
        if var_sharing is None:
            var_sharing = [0] * len(dataset_variables)

        if len(arrays) != len(dataset_variables):
            expected_vars = sum(
                1
                for is_passive, share_zone in zip(
                    passive_vars, var_sharing, strict=True
                )
                if not is_passive and not share_zone
            )
            if expected_vars == 0:
                raise ValueError(
                    "No active variables to write. All variables are "
                    "either passive or shared."
                )
            if len(arrays) == 0:
                raise ValueError(
                    f"No data arrays provided. Expected {expected_vars} "
                    "active variable arrays based on passive_vars and "
                    "var_sharing."
                )
            if len(arrays) != expected_vars:
                raise ValueError(
                    f"Expected {expected_vars} data arrays, got {len(arrays)}."
                )

        active_var_idx = [
            var_idx
            for var_idx, (is_passive, sharing_zone_idx) in enumerate(
                zip(passive_vars, var_sharing, strict=True), start=1
            )
            if (not is_passive) and (not sharing_zone_idx)
        ]
        active_local_idx = {var_idx: i for i, var_idx in enumerate(active_var_idx)}

        # Determine validation reference shape. NODAL local or shared variable arrays
        # give shape dimensions directly; CELL_CENTERED is only used as a fallback
        # (ambiguous for a degenerate axis).
        nodal_shape: tuple[int, ...] | None = None
        ndims: int | None = None
        cell_fallback: tuple[int, ...] | None = None
        cell_fallback_ndims: int | None = None

        for var_idx in range(1, len(dataset_variables) + 1):
            if passive_vars[var_idx - 1]:
                continue
            src = var_sharing[var_idx - 1]
            if src:
                if nodal_shape is None:
                    src_zone = meta.zone(src)
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
                    f"Arrays must be 1D, 2D, or 3D; got {arr_ndims}-D "
                    "array. For time-dependent data, write each time step "
                    "as a separate zone."
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

        # Validate every non-passive dataset variable (local and shared) against the
        # reference shape.
        for var_idx in range(1, len(dataset_variables) + 1):
            if passive_vars[var_idx - 1]:
                continue
            src = var_sharing[var_idx - 1]
            if src:
                src_zone = meta.zone(src)
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
                raise ValueError(f"Array {i} is {arr.ndim}D, expected {ndims}D.")
            shape = arr.shape + (1,) * (3 - arr.ndim)
            if (loc == ValueLocation.NODAL) and (shape != nodal_shape):
                raise ValueError(
                    f"Array {i} is NODAL but has shape {shape}, expected {nodal_shape}."
                )
            if (loc == ValueLocation.CELL_CENTERED) and (shape != cell_shape):
                raise ValueError(
                    f"Array {i} is CELL_CENTERED but has shape {shape}, "
                    f"expected {cell_shape}."
                )

        variable_types_global = [DataType.DOUBLE] * len(dataset_variables)
        value_locations_global = [ValueLocation.NODAL] * len(dataset_variables)
        for local_idx, var_idx in enumerate(active_var_idx):
            variable_types_global[var_idx - 1] = variable_types[local_idx]
            value_locations_global[var_idx - 1] = value_locations[local_idx]

    except ValueError:
        if on_error is not None:
            on_error()
        raise

    return PreparedOrderedZone(
        value_locations=list(value_locations),
        passive_vars=[bool(p) for p in passive_vars],
        var_sharing=[int(s) for s in var_sharing],
        active_var_idx=active_var_idx,
        imax=imax,
        jmax=jmax,
        kmax=kmax,
        value_locations_global=value_locations_global,
        variable_types_global=variable_types_global,
    )


@dataclass
class PreparedFEZone:
    """Validated, normalized inputs for writing one finite-element zone.

    Everything a writer needs to proceed straight to its own format-specific
    zone-creation dispatch call, already validated and shape-consistent.
    """

    value_locations: list[ValueLocation]
    passive_vars: list[bool]
    var_sharing: list[int]
    con_sharing: int
    active_var_idx: list[int]  # 1-based dataset indices being written now
    num_nodes: int
    num_cells: int
    value_locations_global: list[ValueLocation]
    variable_types_global: list[DataType]
    face_neighbors_arr: npt.NDArray | None
    face_neighbor_mode: FaceNeighborMode | None
    num_face_connections: int


def prepare_fe_zone(
    arrays: Sequence[npt.NDArray],
    variable_types: Sequence[DataType],
    zone_type: ZoneType,
    *,
    node_map: npt.ArrayLike | None,
    value_locations: Sequence[ValueLocation] | None,
    passive_vars: Sequence[bool | int] | None,
    var_sharing: Sequence[int] | None,
    con_sharing: int | None,
    face_neighbors: npt.ArrayLike | None,
    face_neighbor_mode: FaceNeighborMode | None,
    dataset_variables: Sequence[str],
    meta: WriterMeta,
    on_error: Callable[[], None] | None = None,
) -> PreparedFEZone:
    """Validate and normalize inputs for writing a finite-element zone.

    Shared across formats: derives num_nodes/num_cells from node_map (or the con_sharing
    source zone), resolves value_locations/passive_vars/ var_sharing defaults, validates
    the active-variable count and every array's size (local and shared) against those
    counts, validates and derives face-neighbor state (mode, count, structural
    compatibility), and builds the dataset-level value_locations/variable_types arrays
    every format's zone-creation call needs.

    Args:
        arrays: NumPy arrays already converted from the caller's raw data, one per
            active (non-passive, non-shared) variable.
        variable_types: Per-active-variable data type, one per array in
            *arrays*. Computed by the caller (see
            :func:`prepare_ordered_zone` for why this isn't derived here).
        zone_type: This zone's :class:`~tecio.ZoneType`.
        node_map: Connectivity array, or None if *con_sharing* is set.
        value_locations: Per-active-variable value location, or None for all-NODAL.
        passive_vars: Per-dataset-variable passive flags, or None for all active.
        var_sharing: Per-dataset-variable source-zone sharing indices, or None for no
            sharing.
        con_sharing: Source zone index to share connectivity from, or None (equivalent
            to 0) for no sharing.
        face_neighbors: Optional flat face-neighbor connectivity array.  Left in its
            natural dtype (never forced to a particular integer width here): a writer
            needing a specific dtype for its own dispatch call (PLT's int32-only classic
            API, for instance) casts the returned array itself.
        face_neighbor_mode: Required if *face_neighbors* is given, invalid if it isn't;
            see :func:`validate_face_neighbor_sharing`.
        dataset_variables: The full dataset variable name list, already established
            (after the writer's own lazy-open handling).
        meta: The writer's own zone metadata, for resolving shared connectivity/variable
            source zones.
        on_error: Optional callback invoked once, before any ValueError is raised, for a
            format needing extra cleanup on failure (DAT's partial-file deletion;
            SZL/PLT need none, the C library handles it, so they pass None).

    Returns:
        A :class:`PreparedFEZone` with everything else needed to proceed to the
        format-specific zone-creation call.

    Raises:
        ValueError: On any inconsistency: wrong active-variable count, array-size
            mismatch (local or shared), con_sharing/var_sharing referencing a zone that
            doesn't exist yet or isn't FE, or any face-neighbor validation failure.
    """
    try:
        if con_sharing is None:
            con_sharing = 0

        if node_map is not None:
            node_map_arr = np.asarray(node_map)
            num_cells = node_map_arr.shape[0]
            num_nodes = int(node_map_arr.max())
        elif con_sharing:
            src_zone = meta.zone(con_sharing)
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

        if value_locations is None:
            value_locations = [ValueLocation.NODAL] * len(arrays)
        if passive_vars is None:
            passive_vars = [False] * len(dataset_variables)
        if var_sharing is None:
            var_sharing = [0] * len(dataset_variables)

        if len(arrays) != len(dataset_variables):
            expected_vars = sum(
                1
                for is_passive, sharing_zone_idx in zip(
                    passive_vars, var_sharing, strict=True
                )
                if not is_passive and not sharing_zone_idx
            )
            if expected_vars == 0:
                raise ValueError(
                    "No active variables to write. All variables are "
                    "either passive or shared."
                )
            if len(arrays) == 0:
                raise ValueError(
                    "No data arrays provided for active variables. "
                    f"Expected {expected_vars} active variables based on "
                    "passive_vars and var_sharing settings."
                )
            if len(arrays) != expected_vars:
                raise ValueError(
                    f"Expected {expected_vars} data arrays, got {len(arrays)}."
                )

        active_var_idx = [
            var_idx
            for var_idx, (is_passive, sharing_zone_idx) in enumerate(
                zip(passive_vars, var_sharing, strict=True), start=1
            )
            if (not is_passive) and (not sharing_zone_idx)
        ]

        # Shared variable data shape validation
        for var_idx, src in enumerate(var_sharing, start=1):
            if not src:
                continue
            src_zone = meta.zone(src)
            if src_zone is None:
                raise ValueError(
                    f"Variable {var_idx} shares from zone {src}, which has "
                    "not been written yet."
                )
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
                    f"expected {num_nodes}."
                )
            elif (loc == ValueLocation.CELL_CENTERED) and (arr.size != num_cells):
                raise ValueError(
                    f"Array {i} is CELL_CENTERED but has {arr.size} "
                    f"values, expected {num_cells}."
                )

        variable_types_global = [DataType.DOUBLE] * len(dataset_variables)
        value_locations_global = [ValueLocation.NODAL] * len(dataset_variables)
        for local_idx, var_idx in enumerate(active_var_idx):
            variable_types_global[var_idx - 1] = variable_types[local_idx]
            value_locations_global[var_idx - 1] = value_locations[local_idx]

        # Face-neighbor connections
        if face_neighbors is not None and face_neighbor_mode is None:
            face_neighbor_mode = FaceNeighborMode.LOCAL_ONE_TO_ONE
        elif face_neighbors is None and face_neighbor_mode is not None:
            raise ValueError(
                "face_neighbor_mode was given without face_neighbors; "
                "that combination has nothing to apply the mode to."
            )
        validate_face_neighbor_sharing(con_sharing, face_neighbor_mode)
        face_neighbors_arr: npt.NDArray | None = None
        num_face_connections = 0
        if face_neighbors is not None and not con_sharing:
            # Face-neighbor data is inherited from the same source zone
            assert face_neighbor_mode is not None  # narrowed above
            face_neighbors_arr = np.ascontiguousarray(face_neighbors).ravel(order="C")
            num_face_connections = count_face_connections(
                face_neighbors_arr, face_neighbor_mode
            )
            validate_face_neighbors(
                face_neighbors_arr, face_neighbor_mode, zone_type, num_cells
            )

    except ValueError:
        if on_error is not None:
            on_error()
        raise

    return PreparedFEZone(
        value_locations=list(value_locations),
        passive_vars=[bool(p) for p in passive_vars],
        var_sharing=[int(s) for s in var_sharing],
        con_sharing=con_sharing,
        active_var_idx=active_var_idx,
        num_nodes=num_nodes,
        num_cells=num_cells,
        value_locations_global=value_locations_global,
        variable_types_global=variable_types_global,
        face_neighbors_arr=face_neighbors_arr,
        face_neighbor_mode=(
            face_neighbor_mode if face_neighbors_arr is not None else None
        ),
        num_face_connections=num_face_connections,
    )


def normalize_precision(
    precision: DataType | str | None, *, allow_none: bool
) -> DataType | None:
    """Return the :class:`~tecio.libtecio.DataType` for *precision*.

    Args:
        precision: ``None``, the enum directly, or a case-insensitive string
            (``"single"``/``"float"``/``"double"``).
        allow_none: Whether ``None`` is a valid result. SZL: yes, per-variable data type
            is inferred automatically from each array, with no file-wide override to
            normalize. PLT and DAT: no, ``precision`` must resolve to a concrete
            FLOAT/DOUBLE, but for different reasons and with different scope:

            * PLT's classic API has one ``VIsDouble`` flag for the entire file (set once
              at ``tecini142``); every variable, including integer-valued data, is
              declared and stored at that single type. There is no per-variable type in
              PLT's zone header at all.
            * DAT's ``precision`` only decides two things: the ASCII significant-digit
              count used to format floating values, and which of FLOAT/DOUBLE is
              declared for variables whose own array is itself float-typed. A variable
              inferred as an integer type keeps its own INT32/INT16/BYTE in the zone's
              ``DT=`` declaration regardless of ``precision`` (see
              :func:`~tecio._dat_write._resolve_written_type`), since that declaration
              is what Tecplot uses to allocate memory on read, not the printed digit
              count. A value can be written as ``1.000000000e0`` and still be declared
              (and read back) as an integer.

    Raises:
        ValueError: If *precision* is ``None`` and *allow_none* is False, or if it's
            neither ``None`` nor FLOAT/DOUBLE (or a recognized string alias for one of
            them).
    """
    if precision is None:
        if allow_none:
            return None
        raise ValueError(
            "precision=None is not supported by this format; use "
            "DataType.FLOAT/DataType.DOUBLE (or 'single'/'double')."
        )
    if isinstance(precision, str):
        try:
            precision = _STR_TO_PRECISION[precision.strip().lower()]
        except KeyError:
            raise ValueError(
                f"precision={precision!r} is not recognized; use 'single' or "
                "'double' (or DataType.FLOAT / DataType.DOUBLE)."
            ) from None
    if precision not in (DataType.FLOAT, DataType.DOUBLE):
        raise ValueError(
            f"precision={precision!r} is not supported; precision only "
            "applies to floating-point variables -- use DataType.FLOAT, "
            "DataType.DOUBLE, or None."
        )
    return precision


class TecplotWriter(ABC):
    """Shared interface and lifecycle for all Tecplot file writers.

    Concrete subclasses (:class:`~tecio.TecplotSzlWriter`, ...) differ in how a file is
    actually opened, closed, and written to, but share the same aux-data buffering,
    variable-list handling, and context-manager lifecycle, so application code can be
    written against this base without caring which format is being produced.

    Args:
        path: Output file path.
        title: Dataset title.
        variables: Variable name list. ``None`` defers file creation until the first
            zone-writing call (lazy open).
        file_type: File type enum (FULL, GRID, or SOLUTION).

    Attributes:
        path: Output file path.
        title: Dataset title string.
        variables: Variable name list, or ``None`` if the file has not been opened yet.
        file_type: File type (FULL, GRID, or SOLUTION).
        current_zone: Index of the most recently written zone. ``0`` before any zone has
            been written; incremented only after a zone-writing method successfully
            completes.
        auxdataset: Buffered dataset-level auxiliary data, flushed before the first
            zone.
        auxvar: Buffered variable-level auxiliary data, flushed before the first zone.
    """

    def __init__(
        self,
        path: str,
        title: str,
        variables: list[str] | None,
        file_type: FileType,
    ) -> None:
        self.path: str = str(path)
        self.title: str = title
        self.variables: list[str] | None = variables
        self.file_type: FileType = file_type
        self.current_zone: int = 0
        self.auxdataset: dict[str, str] = {}
        self.auxvar: dict[int, dict[str, str]] = {}
        self._meta = WriterMeta(
            path=self.path,
            title=self.title,
            file_type=self.file_type,
            file_format=self._file_format,
        )
        if self.variables is not None:
            self._open(self.variables)

    @property
    @abstractmethod
    def _file_format(self) -> str:
        """Format tag for :class:`~tecio._meta.WriterMeta`, e.g. ``'szplt'``."""

    # -- Context manager ---------------------------------------------------------------

    def __enter__(self) -> TecplotWriter:
        """Support ``with`` statement, returns *self*."""
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        """Close the file on context-manager exit.

        The file is closed regardless of whether an exception was raised in the ``with``
        block. If closing itself raises, that secondary exception is only re-raised when
        the ``with`` block completed without error; otherwise the original exception
        takes precedence.
        """
        try:
            self.close()
        except Exception:
            if exc_type is None:
                raise

    # -- Validation --------------------------------------------------------------------

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

    # -- File lifecycle: format-specific -----------------------------------------------

    @abstractmethod
    def _open(self, var_names: list[str]) -> None:
        """Open the file/write the header. Called at most once per instance."""

    @abstractmethod
    def close(self) -> None:
        """Finalize and close the file. Safe to call more than once."""

    # -- Aux data: buffering and key resolution are shared, the actual write isn't -----

    def add_auxdataset_dict(self, auxdict: dict[str, Any]) -> None:
        """Buffer dataset-level auxiliary data from a dictionary."""
        self.auxdataset.update(auxdict)

    def add_auxvar_dict(self, auxdict: dict[int, dict[str, Any]]) -> None:
        """Buffer variable-level auxiliary data from a dictionary."""
        self.auxvar.update(auxdict)

    def _resolve_var_index(self, key: int | str) -> int:
        """Return the 0-based variable index for an aux-data key.

        Args:
            key: A 1-based variable index, or an exact variable name.

        Raises:
            IndexError: If a 1-based index key is out of range.
            KeyError: If a name key doesn't match any variable.
            TypeError: If *key* is neither ``int`` nor ``str``.
        """
        if isinstance(key, bool):
            raise TypeError(
                f"Aux data key must be a variable name (str) or 1-based "
                f"index (int), got {key!r}"
            )
        if isinstance(key, int):
            var_idx = key - 1
            if var_idx not in range(len(self._check_variables())):
                raise IndexError(
                    f"Variable index {key} out of bounds "
                    f"[1, {len(self._check_variables())}]"
                )
            return var_idx
        if isinstance(key, str):
            try:
                return self._check_variables().index(key)
            except ValueError as exc:
                raise KeyError(
                    f"Variable aux data key {key!r} not found in variable "
                    f"list ({self.variables})"
                ) from exc
        raise TypeError(
            f"Aux data key must be a variable name (str) or 1-based index "
            f"(int), got {key!r}"
        )

    def flush_aux(self) -> None:
        """Write buffered dataset- and variable-level aux data to the file.

        Called automatically before the first zone is written. Only needed directly if
        you want to flush explicitly, e.g. before checking :attr:`meta`.
        """
        for name, value in self.auxdataset.items():
            self._write_dataset_aux_item(str(name), str(value))

        for key, subdict in self.auxvar.items():
            var_idx = self._resolve_var_index(key)
            for name, value in subdict.items():
                self._write_var_aux_item(var_idx + 1, str(name), str(value))

        # Record counts, then clear buffers -- each item is written exactly once.
        self._meta.note_dataset_aux(len(self.auxdataset))
        self._meta.note_var_aux(sum(len(subdict) for subdict in self.auxvar.values()))
        self.auxdataset.clear()
        self.auxvar.clear()

    @abstractmethod
    def _write_dataset_aux_item(self, name: str, value: str) -> None:
        """Write one dataset-level aux item. Called only from :meth:`flush_aux`."""

    @abstractmethod
    def _write_var_aux_item(
        self, one_based_var_index: int, name: str, value: str
    ) -> None:
        """Write one variable-level aux item. Called only from :meth:`flush_aux`."""

    # -- Zone writers: fully format-specific ------------------------------------------

    @abstractmethod
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
        """Write a complete IJK-ordered zone. See the concrete subclass for details."""

    @abstractmethod
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
        face_neighbor_mode: FaceNeighborMode | None = None,
        solution_time: float = 0.0,
        strand_id: int = 0,
        aux: dict[str, Any] | None = None,
    ) -> None:
        """Write a complete finite-element zone."""
