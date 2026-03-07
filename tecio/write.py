from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt

from . import libtecio
from .libtecio import DataType, FileType, ValueLocation, ZoneType


class Write:
    """
    Write provides a high level API to write data to Tecplot szplt
    formatted binary files.
    """

    def __init__(
        self,
        path: str,
        dataset_title: str = "Untitled",
        var_names: Iterable[str] = [],
        file_type: FileType = FileType.FULL,
        grid_file_handle: Optional[ctypes.c_void_p] = None,
    ):
        if not isinstance(file_type, FileType):
            raise TypeError("file_type must be a libtecio.FileType enum")

        self._var_string = ",".join(var_names)
        self._handle = libtecio.tec_file_writer_open(
            path,
            dataset_title,
            self._var_string,
            file_type,
            use_szl=1,
            grid_file_handle=grid_file_handle,
        )

    def __enter__(self) -> Write:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def close(self) -> None:
        if self._handle is not None:
            libtecio.tec_file_writer_close(self._handle)
            self._handle = None


class WriteZone:
    """
    WriteZone provides a high level API with tecio functions to write
    szplt binary formatted zone data.
    """

    def zone_write_solution_time(
        file_handle: ctypes.c_void_p,
        zone: int,
        strand: int = 0,
        solution_time: float = 0.0,
    ) -> None:
        """Set unsteady options (strand id and solution time) for a zone."""
        libtecio.tec_zone_set_unsteady_options(
            file_handle, zone, strand=strand, solution_time=solution_time
        )

    def write_zone_ordered(
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

        var_data_types must be a sequence of libtecio.DataType enums (if provided).
        value_locations must be a sequence of libtecio.ValueLocation enums (if provided).
        """
        I, J, K = shape
        return libtecio.tec_zone_create_ijk(
            file_handle,
            zone_name,
            int(I),
            int(J),
            int(K),
            var_types=var_data_types,
            var_sharing=var_sharing,
            value_locations=value_locations,
        )

    def _zone_write_double_values(
        file_handle: ctypes.c_void_p, zone: int, var: int, values: npt.ArrayLike
    ) -> None:
        # accept numpy arrays or lists; enforce float64
        arr = np.ascontiguousarray(values, dtype=np.float64)
        libtecio.tec_zone_var_write_double_values(file_handle, zone, var, arr)

    def _zone_write_float_values(
        file_handle: ctypes.c_void_p, zone: int, var: int, values: npt.ArrayLike
    ) -> None:
        arr = np.ascontiguousarray(values, dtype=np.float32)
        libtecio.tec_zone_var_write_float_values(file_handle, zone, var, arr)

    def _zone_write_int32_values(
        file_handle: ctypes.c_void_p, zone: int, var: int, values: npt.ArrayLike
    ) -> None:
        arr = np.ascontiguousarray(values, dtype=np.int32)
        libtecio.tec_zone_var_write_int32_values(file_handle, zone, var, arr)

    def _zone_write_int16_values(
        file_handle: ctypes.c_void_p, zone: int, var: int, values: npt.ArrayLike
    ) -> None:
        arr = np.ascontiguousarray(values, dtype=np.int16)
        libtecio.tec_zone_var_write_int16_values(file_handle, zone, var, arr)

    def _zone_write_uint8_values(
        file_handle: ctypes.c_void_p, zone: int, var: int, values: npt.ArrayLike
    ) -> None:
        arr = np.ascontiguousarray(values, dtype=np.uint8)
        libtecio.tec_zone_var_write_uint8_values(file_handle, zone, var, arr)
