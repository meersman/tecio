from __future__ import annotations

import ctypes
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Union
import numpy.typing as npt

import szlio
from szlio import FileType, ZoneType, DataType, ValueLocation


class SzlFile:
    """
    SZLFile provides an interface to read information and data from Tecplot
    SZPLT files.
    """

    def __init__(self, file_name):
        self.handle = szlio.tec_file_reader_open(file_name)
        self.zones = [
            Zone(self.handle, i + 1, self.num_vars) for i in range(self.num_zones)
        ]
        # todo: make dataset var variable list

    @property
    def type(self) -> FileType:
        return szlio.tec_file_get_type(self.handle)

    @property
    def title(self) -> str:
        return szlio.tec_data_set_get_title(self.handle)

    @property
    def num_vars(self) -> int:
        return szlio.tec_data_set_get_num_vars(self.handle)

    @property
    def num_zones(self) -> int:
        return szlio.tec_data_set_get_num_zones(self.handle)

    @property
    def num_auxdata_items(self) -> int:
        return szlio.tec_data_set_aux_data_get_num_items(self.handle)

    @property
    def num_zones(self) -> int:
        return szlio.tec_data_set_get_num_zones(self.handle)

    @property
    def num_auxdata_items(self) -> int:
        return szlio.tec_data_set_aux_data_get_num_items(self.handle)


@dataclass
class Zone:
    handle: ctypes.c_void_p
    zone_index: int
    num_vars: int

    @property
    def variables(self) -> Variable:
        return [
            Variable(self.handle, self.zone_index, i + 1) for i in range(self.num_vars)
        ]

    def __post_init__(self) -> Tuple[int, int, int]:
        self.I, self.J, self.K = szlio.tec_zone_get_ijk(self.handle, self.zone_index)

    @property
    def title(self) -> str:
        return szlio.tec_zone_get_title(self.handle, self.zone_index)

    @property
    def type(self) -> ZoneType:
        return ZoneType(szlio.tec_zone_get_type(self.handle, self.zone_index))

    def is_enabled(self) -> bool:
        return szlio.tec_zone_is_enabled(self.handle, self.zone_index)

    @property
    def num_points(self) -> int:
        if self.type == ZoneType.ORDERED:
            return self.I * self.J * self.K
        else:
            return self.I

    @property
    def num_elements(self) -> int:
        if self.type == ZoneType.ORDERED:
            return self.I * self.J * self.K
        else:
            return self.J

    @property
    def dimensions(self) -> Tuple[int, int, int]:
        return (self.I, self.J, self.K)

    @property
    def nodes_per_cell(self) -> int:
        """Returns how many nodes per cell based on FE type."""
        if self.type == ZoneType.FELINESEG:
            return 2
        elif self.type == ZoneType.FETRIANGLE:
            return 3
        elif self.type == ZoneType.FEQUADRILATERAL:
            return 4
        elif self.type == ZoneType.FETETRAHEDRON:
            return 4
        elif self.type == ZoneType.FEBRICK:
            return 8
        else:
            raise Exception("ZoneType does not have a consistent number of nodes")

    @property
    def solution_time(self) -> int:
        return szlio.tec_zone_get_solution_time(self.handle, self.zone_index)

    @property
    def strand_id(self) -> int:
        return szlio.tec_zone_get_strand_id(self.handle, self.zone_index)

    @property
    def node_map(self) -> npt.NDArray[np.int64]:
        is64bit = szlio.is_64bit(self.handle, self.zone_index)

        if is64bit:
            return szlio.tec_zone_node_map_get_64(
                self.handle, self.zone_index, self.num_elements, self.nodes_per_cell
            )
        else:
            return szlio.tec_zone_node_map_get(
                self.handle, self.zone_index, self.num_elements, self.nodes_per_cell
            ).astype(np.int64)


@dataclass
class Variable:
    handle: ctypes.c_void_p
    zone_index: int
    var_index: int

    @property
    def name(self) -> str:
        return szlio.tec_var_get_name(self.handle, self.var_index)

    def is_enabled(self) -> bool:
        return szlio.tec_var_is_enabled(self.handle, self.var_index)

    @property
    def type(self) -> DataType:
        return szlio.tec_zone_var_get_type(self.handle, self.zone_index, self.var_index)

    @property
    def value_location(self) -> ValueLocation:
        return szlio.tec_zone_var_get_value_location(
            self.handle, self.zone_index, self.var_index
        )

    def is_passive(self) -> bool:
        return szlio.tec_zone_var_is_passive(
            self.handle, self.zone_index, self.var_index
        )

    @property
    def shared_zone(self) -> int | None:
        """Outputs shared zone index (0 if none)"""
        return szlio.tec_zone_var_get_shared_zone(
            self.handle, self.zone_index, self.var_index
        )

    @property
    def num_values(self) -> int:
        return szlio.tec_zone_var_get_num_values(
            self.handle, self.zone_index, self.var_index
        )

    @property
    def values(self, value_range: Tuple[Optional[int], Optional[int]] = (None, None)) -> Union[
            NDArray[np.float32],
            NDArray[np.float64],
            NDArray[np.int32],
            NDArray[np.int16],
            NDArray[np.uint8]
    ]:
        data_type = self.type

        if value_range == (None, None):
            start_index = 1
            num_values = self.num_values
        else:
            start_index = value_range[0]
            num_values = value_range[1] - value_range[0]

            if (start_index > self.num_values() or start_index < 0) or (
                num_values < 0 or num_values > self.num_values()
            ):
                raise Exception(f"Variable value range incorrect: {value_range}")

        if data_type == DataType.FLOAT:
            return szlio.tec_zone_var_get_float_values(
                self.handle, self.zone_index, self.var_index, start_index, num_values
            )

        elif data_type == DataType.DOUBLE:
            return szlio.tec_zone_var_get_double_values(
                self.handle, self.zone_index, self.var_index, start_index, num_values
            )

        elif data_type == DataType.INT32:
            return szlio.tec_zone_var_get_int32_values(
                self.handle, self.zone_index, self.var_index, start_index, num_values
            )

        elif data_type == DataType.INT16:
            return szlio.tec_zone_var_get_int16_values(
                self.handle, self.zone_index, self.var_index, start_index, num_values
            )

        elif data_type == DataType.BYTE:
            return szlio.tec_zone_var_get_uint8_values(
                self.handle, self.zone_index, self.var_index, start_index, num_values
            )
