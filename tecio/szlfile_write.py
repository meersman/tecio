"""Simplified Write class for szlfile.py."""

from __future__ import annotations

import ctypes
from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from . import libtecio
from .libtecio import (
    DataType,
    FaceNeighborMode,
    FileType,
    TecioError,
    ValueLocation,
    ZoneType,
)


# Dtype helpers (local to avoid circular imports)
_NUMPY_TO_DATATYPE: dict[np.dtype, DataType] = {
    np.dtype(np.float64): DataType.DOUBLE,
    np.dtype(np.float32): DataType.FLOAT,
    np.dtype(np.int32):   DataType.INT32,
    np.dtype(np.int16):   DataType.INT16,
    np.dtype(np.uint8):   DataType.BYTE,
}


def _infer_data_type(arr: npt.NDArray) -> DataType:
    """Return the ``DataType`` that best matches *arr*'s numpy dtype."""
    dt = _NUMPY_TO_DATATYPE.get(arr.dtype)
    if dt is not None:
        return dt
    k, s = arr.dtype.kind, arr.dtype.itemsize
    if k == "f":
        return DataType.DOUBLE if s == 8 else DataType.FLOAT
    if k in ("i", "u"):
        if s >= 4:
            return DataType.INT32
        return DataType.INT16 if s == 2 else DataType.BYTE
    return DataType.DOUBLE


def _cast(arr: npt.NDArray, dt: DataType) -> npt.NDArray:
    """Return *arr* cast to the numpy dtype matching *dt*."""
    _map = {
        DataType.DOUBLE: np.float64,
        DataType.FLOAT:  np.float32,
        DataType.INT32:  np.int32,
        DataType.INT16:  np.int16,
        DataType.BYTE:   np.uint8,
    }
    return np.ascontiguousarray(arr, dtype=_map[dt])


# FE zone types that use tec_zone_create_fe
_FE_SIMPLE: frozenset[ZoneType] = frozenset({
    ZoneType.FELINESEG,
    ZoneType.FETRIANGLE,
    ZoneType.FEQUADRILATERAL,
    ZoneType.FETETRAHEDRON,
    ZoneType.FEBRICK,
})



class Write:
    """Write Tecplot SZL (``.szplt``) files with a lazy-open file handle."""

    def __init__(
        self,
        path: str,
        title: str = "untitled",
        file_type: FileType = FileType.FULL,
    ) -> None:
        self._path = path
        self.title = title
        self.file_type = file_type

        # Dataset-level aux data buffer (flushed on first zone)
        self.auxdata: dict[str, str] = {}
        # Variable-level aux data buffer: {var_name: {key: value}}
        self.var_auxdata: dict[str, dict[str, str]] = {}

        # Internals — None until the file is lazily opened
        self._handle: ctypes.c_void_p | None = None
        self._var_names: list[str] | None = None  # locked on first zone


    # Context manager
    def __enter__(self) -> Write:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def add_zone(
        self,
        title: str,
        zone_type: ZoneType,
        dimensions: tuple[int, int, int],
        variables: dict[str, npt.ArrayLike],
        node_map: npt.ArrayLike | None = None,
        value_locations: Sequence[ValueLocation] | None = None,
        solution_time: float = 0.0,
        strand_id: int = 0,
        auxdata: dict[str, str] | None = None,
    ) -> None:
        """Write one zone to the file."""
        var_names = list(variables.keys())
        arrays = [np.asarray(v) for v in variables.values()]

        # Lazy open on first zone
        if self._handle is None:
            self._open(var_names)

        # Validate variable names against first zone
        if var_names != self._var_names:
            raise ValueError(
                f"Variable names {var_names!r} do not match the names "
                f"locked by the first zone {self._var_names!r}. All zones "
                "must supply the same variables in the same order."
            )

        n_vars = len(var_names)
        I, J, K = dimensions

        # Infer per-variable types and locations
        var_types = [_infer_data_type(a) for a in arrays]

        if value_locations is None:
            value_locations = [ValueLocation.NODAL] * n_vars

        # Create zone header
        if zone_type == ZoneType.ORDERED:
            zone_num = libtecio.tec_zone_create_ijk(
                handle=self._handle,
                zone_title=title,
                I=I,
                J=J,
                K=K,
                var_types=var_types,
                value_locations=value_locations,
            )
        elif zone_type in _FE_SIMPLE:
            zone_num = libtecio.tec_zone_create_fe(
                handle=self._handle,
                zone_title=title,
                zone_type=zone_type,
                num_nodes=I,
                num_elements=J,
                var_types=var_types,
                value_locations=value_locations,
            )
        else:
            raise NotImplementedError(
                f"Zone type {zone_type.name!r} is not yet supported by Write. "
                "Use TecData.write_szl for polygon/polyhedral zones."
            )

        # Unsteady options
        if strand_id != 0 or solution_time != 0.0:
            libtecio.tec_zone_set_unsteady_options(
                handle=self._handle,
                zone=zone_num,
                strand=strand_id,
                solution_time=solution_time,
            )

        # Variable data
        for var_idx, (arr, dt) in enumerate(zip(arrays, var_types)):
            _write_var(self._handle, zone_num, var_idx + 1, arr.ravel(), dt)

        # Connectivity
        if zone_type in _FE_SIMPLE and node_map is not None:
            flat = np.asarray(node_map).ravel()
            if flat.dtype == np.int64 or int(flat.max()) > np.iinfo(np.int32).max:
                arr64 = np.ascontiguousarray(flat, dtype=np.int64)
                libtecio.tec_zone_node_map_write64(self._handle, zone_num, arr64)
            else:
                arr32 = np.ascontiguousarray(flat, dtype=np.int32)
                libtecio.tec_zone_node_map_write32(self._handle, zone_num, arr32)

        # Zone aux data
        if auxdata:
            for name, value in auxdata.items():
                libtecio.tec_zone_add_aux_data(self._handle, zone_num, name, value)

    def close(self) -> None:
        """Finalise and flush the file.  Safe to call more than once."""
        if self._handle is not None:
            libtecio.tec_file_writer_close(self._handle)
            self._handle = None

    def _open(self, var_names: list[str]) -> None:
        """Open the file handle.  Called exactly once on the first zone."""
        self._var_names = var_names
        self._handle = libtecio.tec_file_writer_open(
            fname=self._path,
            title=self.title,
            variables=",".join(var_names),
            file_type=self.file_type,
            use_szl=1,
        )
        # Flush buffered dataset-level aux data
        for name, value in self.auxdata.items():
            libtecio.tec_data_set_add_aux_data(self._handle, name, value)
        # Flush buffered variable-level aux data
        for var_name, items in self.var_auxdata.items():
            if var_name not in self._var_names:
                raise ValueError(
                    f"var_auxdata key {var_name!r} is not in variable list "
                    f"{self._var_names!r}"
                )
            var_1based = self._var_names.index(var_name) + 1
            for name, value in items.items():
                libtecio.tec_var_add_aux_data(
                    self._handle, var_1based, name, value
                )


def _write_var(
    handle: ctypes.c_void_p,
    zone_num: int,
    var_num: int,
    data: npt.NDArray,
    dt: DataType,
) -> None:
    """Write one variable's data to *zone_num* using the correct typed call."""
    arr = _cast(data, dt)
    if dt == DataType.DOUBLE:
        libtecio.tec_zone_var_write_double_values(handle, zone_num, var_num, arr)
    elif dt == DataType.FLOAT:
        libtecio.tec_zone_var_write_float_values(handle, zone_num, var_num, arr)
    elif dt == DataType.INT32:
        libtecio.tec_zone_var_write_int32_values(handle, zone_num, var_num, arr)
    elif dt == DataType.INT16:
        libtecio.tec_zone_var_write_int16_values(handle, zone_num, var_num, arr)
    elif dt == DataType.BYTE:
        libtecio.tec_zone_var_write_uint8_values(handle, zone_num, var_num, arr)
    else:
        raise ValueError(f"Unsupported DataType: {dt!r}")
