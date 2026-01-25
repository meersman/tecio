from __future__ import annotations

import ctypes
from typing import Iterable, Optional, Sequence

import numpy as np
import numpy.typing as npt

import szlio
from szlio import DataType, FileType, ValueLocation


def open_file(
    file_name: str,
    dataset_title: str,
    var_names: Iterable[str],
    file_type: FileType = FileType.FULL,
    grid_file_handle: Optional[ctypes.c_void_p] = None,
) -> ctypes.c_void_p:
    """High-level: open an SZL writer file and return a file handle.

    file_type must be a szlio.FileType enum.
    """
    if not isinstance(file_type, FileType):
        raise TypeError("file_type must be a szlio.FileType enum")

    var_names_csv = ",".join(var_names)
    # pass the enum directly; szlio.tec_file_writer_open expects FileType
    return szlio.tec_file_writer_open(
        file_name,
        dataset_title,
        var_names_csv,
        file_type,
        use_szl=1,
        grid_file_handle=grid_file_handle,
    )


def close_file(file_handle: ctypes.c_void_p) -> None:
    """High-level: close writer handle."""
    szlio.tec_file_writer_close(file_handle)


def create_ordered_zone(
    file_handle: ctypes.c_void_p,
    zone_name: str,
    shape: Sequence[int],
    var_sharing: Optional[Sequence[int]] = None,
    var_data_types: Optional[Sequence[DataType]] = None,
    value_locations: Optional[Sequence[ValueLocation]] = None,
) -> int:
    """
    Create an ordered zone. `shape` is (I,J,K).
    Returns zone index (int).

    var_data_types must be a sequence of szlio.DataType enums (if provided).
    value_locations must be a sequence of szlio.ValueLocation enums (if provided).
    """
    I, J, K = shape
    return szlio.tec_zone_create_ijk(
        file_handle,
        zone_name,
        int(I),
        int(J),
        int(K),
        var_types=var_data_types,
        var_sharing=var_sharing,
        value_locations=value_locations,
    )


def zone_set_solution_time(
    file_handle: ctypes.c_void_p, zone: int, strand: int = 0, solution_time: float = 0.0
) -> None:
    """Set unsteady options (strand id and solution time) for a zone."""
    szlio.tec_zone_set_unsteady_options(
        file_handle, zone, strand=strand, solution_time=solution_time
    )


def zone_write_double_values(
    file_handle: ctypes.c_void_p, zone: int, var: int, values: npt.ArrayLike
) -> None:
    # accept numpy arrays or lists; enforce float64
    arr = np.ascontiguousarray(values, dtype=np.float64)
    szlio.tec_zone_var_write_double_values(file_handle, zone, var, arr)


def zone_write_float_values(
    file_handle: ctypes.c_void_p, zone: int, var: int, values: npt.ArrayLike
) -> None:
    arr = np.ascontiguousarray(values, dtype=np.float32)
    szlio.tec_zone_var_write_float_values(file_handle, zone, var, arr)


def zone_write_int32_values(
    file_handle: ctypes.c_void_p, zone: int, var: int, values: npt.ArrayLike
) -> None:
    arr = np.ascontiguousarray(values, dtype=np.int32)
    szlio.tec_zone_var_write_int32_values(file_handle, zone, var, arr)


def zone_write_int16_values(
    file_handle: ctypes.c_void_p, zone: int, var: int, values: npt.ArrayLike
) -> None:
    arr = np.ascontiguousarray(values, dtype=np.int16)
    szlio.tec_zone_var_write_int16_values(file_handle, zone, var, arr)


def zone_write_uint8_values(
    file_handle: ctypes.c_void_p, zone: int, var: int, values: npt.ArrayLike
) -> None:
    arr = np.ascontiguousarray(values, dtype=np.uint8)
    szlio.tec_zone_var_write_uint8_values(file_handle, zone, var, arr)
