from __future__ import annotations

import ctypes
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Iterator, Tuple, Optional, Union
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
        self._auxdata: Optional[AuxData] = None
        self._var_auxdata: Optional[List[AuxData]] = None
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
    def auxdata(self) -> AuxData:
        """Get dataset-level auxiliary data."""
        if self._auxdata is None:
            self._auxdata = AuxData(self.handle, "dataset")
        return self._auxdata

    @property
    def var_auxdata(self) -> List[AuxData]:
        """
        Get list of variable-level auxiliary data.
        
        Returns:
            List of AuxData objects, one per variable (1-indexed to match Tecplot)
        """
        if self._var_auxdata is None:
            # Create list with None at index 0 for 1-based indexing
            self._var_auxdata = [None]
            for i in range(self.num_vars):
                self._var_auxdata.append(AuxData(self.handle, "var", i + 1))
        return self._var_auxdata

    def get_var_auxdata(self, var_index: int) -> AuxData:
        """
        Get auxiliary data for a specific variable.

        Args:
            var_index: Variable index (1-based)

        Returns:
            AuxData object for the specified variable
        """
        if var_index < 1 or var_index > self.num_vars:
            raise IndexError(
                f"Variable index {var_index} out of range [1, {self.num_vars}]"
            )
        return self.var_auxdata[var_index]


@dataclass
class Zone:
    _handle: ctypes.c_void_p
    zone_index: int
    num_vars: int
    _auxdata: Optional[AuxData] = None
    _variables: Optional[List[Variable]] = None
    _node_map: Optional[npt.NDArray[np.int64]] = None
    # Note: could cache all properties if they are shown as bottlenecks in profiling. Or could leave everything as methods and save data in a more flexible data structure. 

    def __post_init__(self) -> Tuple[int, int, int]:
        self.I, self.J, self.K = szlio.tec_zone_get_ijk(self._handle, self.zone_index)

    @property
    def variables(self) -> List[Variable]:
        # Check cached private variables -> don't run C functions each time this is called if already defined
        if self._variables is None:
            self._variables = [
                Variable(self._handle, self.zone_index, i + 1)
                for i in range(self.num_vars)
            ]
        return self._variables
    
    @property
    def title(self) -> str:
        return szlio.tec_zone_get_title(self._handle, self.zone_index)

    @property
    def type(self) -> ZoneType:
        return ZoneType(szlio.tec_zone_get_type(self._handle, self.zone_index))

    def is_enabled(self) -> bool:
        return szlio.tec_zone_is_enabled(self._handle, self.zone_index)

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
            raise ValueError("ZoneType does not have a consistent number of nodes")

    @property
    def solution_time(self) -> float:
        return szlio.tec_zone_get_solution_time(self._handle, self.zone_index)

    @property
    def strand_id(self) -> int:
        return szlio.tec_zone_get_strand_id(self._handle, self.zone_index)

    @property
    def node_map(self) -> npt.NDArray[np.int64]:
        if self._node_map is None:
            is64bit = szlio.is_64bit(self._handle, self.zone_index)
            
            if is64bit:
                self._node_map = szlio.tec_zone_node_map_get_64(
                    self._handle, self.zone_index, self.num_elements, self.nodes_per_cell
                )
            else:
                self._node_map = szlio.tec_zone_node_map_get(
                    self._handle, self.zone_index, self.num_elements, self.nodes_per_cell
                ).astype(np.int64)
        return self._node_map

    @property
    def auxdata(self) -> AuxData:
        """Get zone-level auxiliary data."""
        if self._auxdata is None:
            self._auxdata = AuxData(self._handle, "zone", self.zone_index)
        return self._auxdata


@dataclass
class Variable:
    _handle: ctypes.c_void_p
    zone_index: int
    var_index: int

    @property
    def name(self) -> str:
        return szlio.tec_var_get_name(self._handle, self.var_index)

    def is_enabled(self) -> bool:
        return szlio.tec_var_is_enabled(self._handle, self.var_index)

    @property
    def type(self) -> DataType:
        return szlio.tec_zone_var_get_type(self._handle, self.zone_index, self.var_index)

    @property
    def value_location(self) -> ValueLocation:
        return szlio.tec_zone_var_get_value_location(
            self._handle, self.zone_index, self.var_index
        )

    def is_passive(self) -> bool:
        return szlio.tec_zone_var_is_passive(
            self._handle, self.zone_index, self.var_index
        )

    @property
    def shared_zone(self) -> Optional[int]:
        """Outputs shared zone index (0 if none)"""
        return szlio.tec_zone_var_get_shared_zone(
            self._handle, self.zone_index, self.var_index
        )

    @property
    def num_values(self) -> int:
        return szlio.tec_zone_var_get_num_values(
            self._handle, self.zone_index, self.var_index
        )

    @property
    def values(
            self
    ) -> Union[
        npt.NDArray[np.float32],
        npt.NDArray[np.float64],
        npt.NDArray[np.int32],
        npt.NDArray[np.int16],
        npt.NDArray[np.uint8]
    ]:
        """Get all values for this variable."""
        return self.get_values()

    def get_values(
            self, value_range: Tuple[Optional[int], Optional[int]] = (None, None)
    ) -> Union[
        npt.NDArray[np.float32],
        npt.NDArray[np.float64],
        npt.NDArray[np.int32],
        npt.NDArray[np.int16],
        npt.NDArray[np.uint8],
    ]:
        """
        Get variable values with optional range specification.

        Args:
            value_range: Tuple of (start_index, end_index). If (None, None),
                         retrieves all values.

        Returns:
            NumPy array of values with appropriate dtype
        """
        data_type = self.type

        if value_range == (None, None):
            start_index = 1
            num_values = self.num_values
        else:
            start_index = value_range[0]
            end_index = value_range[1]
            
            if start_index is None or end_index is None:
                raise ValueError("Both start and end indices must be specified")
            
            num_values = end_index - start_index

            if start_index > self.num_values or start_index < 1:
                raise ValueError(
                    f"Start index {start_index} out of range [1, {self.num_values}]"
                )
            if num_values < 0 or end_index > self.num_values:
                raise ValueError(
                    f"Invalid value range: ({start_index}, {end_index})"
                )

        if data_type == DataType.FLOAT:
            return szlio.tec_zone_var_get_float_values(
                self._handle, self.zone_index, self.var_index, start_index, num_values
            )

        elif data_type == DataType.DOUBLE:
            return szlio.tec_zone_var_get_double_values(
                self._handle, self.zone_index, self.var_index, start_index, num_values
            )

        elif data_type == DataType.INT32:
            return szlio.tec_zone_var_get_int32_values(
                self._handle, self.zone_index, self.var_index, start_index, num_values
            )

        elif data_type == DataType.INT16:
            return szlio.tec_zone_var_get_int16_values(
                self._handle, self.zone_index, self.var_index, start_index, num_values
            )

        elif data_type == DataType.BYTE:
            return szlio.tec_zone_var_get_uint8_values(
                self._handle, self.zone_index, self.var_index, start_index, num_values
            )

        raise ValueError(f"Unknown data type: {data_type}")

    
class AuxData:
    """
    AuxData provides a dictionary-like interface for accessing Tecplot
    auxiliary data with automatic type conversion.
    
    Values are stored as strings in the SZL file but can be retrieved
    as integers or floats using the as_int() and as_float() methods.
    """

    def __init__(
        self,
        handle: ctypes.c_void_p,
        aux_type: str,
        index: Optional[int] = None,
    ):
        """
        Initialize AuxData wrapper.

        Args:
            handle: File handle C pointer
            aux_type: Type of auxiliary data ('dataset', 'var', or 'zone')
            index: Variable or zone index (1-based), not needed for dataset
        """
        self._handle = handle
        self._aux_type = aux_type
        self._index = index
        self._data: Optional[Dict[str, str]] = None

    def _load_data(self) -> None:
        """Load auxiliary data from file into internal dictionary."""
        if self._data is not None:
            return

        self._data = {}

        if self._aux_type == "dataset":
            num_items = szlio.tec_data_set_aux_data_get_num_items(self._handle)
            for i in range(num_items):
                name, value = szlio.tec_data_set_aux_data_get_item(
                    self._handle, i + 1
                )
                self._data[name] = value

        elif self._aux_type == "var":
            if self._index is None:
                raise ValueError("Variable index required for variable aux data")
            num_items = szlio.tec_var_aux_data_get_num_items(
                self._handle, self._index
            )
            for i in range(num_items):
                name, value = szlio.tec_var_aux_data_get_item(
                    self._handle, self._index, i + 1
                )
                self._data[name] = value

        elif self._aux_type == "zone":
            if self._index is None:
                raise ValueError("Zone index required for zone aux data")
            num_items = szlio.tec_zone_aux_data_get_num_items(
                self._handle, self._index
            )
            for i in range(num_items):
                name, value = szlio.tec_zone_aux_data_get_item(
                    self._handle, self._index, i + 1
                )
                self._data[name] = value

        else:
            raise ValueError(f"Invalid aux_type: {self._aux_type}")

    @property
    def data(self) -> Dict[str, str]:
        """Return the underlying dictionary of auxiliary data."""
        self._load_data()
        return self._data

    def __len__(self) -> int:
        """Return number of auxiliary data items."""
        return len(self.data)

    def __getitem__(self, key: str) -> str:
        """Get auxiliary data value by name."""
        return self.data[key]

    def __contains__(self, key: str) -> bool:
        """Check if auxiliary data name exists."""
        return key in self.data

    def __iter__(self) -> Iterator[str]:
        """Iterate over auxiliary data names."""
        return iter(self.data)

    def get(self, key: str, default: Any = None) -> str:
        """Get auxiliary data value with optional default."""
        return self.data.get(key, default)

    def keys(self) -> Iterator[str]:
        """Return iterator over auxiliary data names."""
        return self.data.keys()

    def values(self) -> Iterator[str]:
        """Return iterator over auxiliary data values."""
        return self.data.values()

    def items(self) -> Iterator[tuple[str, str]]:
        """Return iterator over (name, value) pairs."""
        return self.data.items()

    def as_int(self, key: str, default: Optional[int] = None) -> Optional[int]:
        """
        Get auxiliary data value as integer.

        Args:
            key: Auxiliary data name
            default: Default value if key not found or conversion fails

        Returns:
            Integer value or default
        """
        try:
            return int(self[key])
        except (KeyError, ValueError):
            return default

    def as_float(self, key: str, default: Optional[float] = None) -> Optional[float]:
        """
        Get auxiliary data value as float.

        Args:
            key: Auxiliary data name
            default: Default value if key not found or conversion fails

        Returns:
            Float value or default
        """
        try:
            return float(self[key])
        except (KeyError, ValueError):
            return default

    def as_bool(self, key: str, default: Optional[bool] = None) -> Optional[bool]:
        """
        Get auxiliary data value as boolean.

        Recognizes common boolean string representations:
        - True: 'true', 't', 'yes', 'y', '1' (case-insensitive)
        - False: 'false', 'f', 'no', 'n', '0' (case-insensitive)

        Args:
            key: Auxiliary data name
            default: Default value if key not found or conversion fails

        Returns:
            Boolean value or default
        """
        try:
            value = self[key].lower().strip()
            if value in ('true', 't', 'yes', 'y', '1'):
                return True
            elif value in ('false', 'f', 'no', 'n', '0'):
                return False
            else:
                return default
        except (KeyError, AttributeError):
            return default

    def __repr__(self) -> str:
        """Return string representation of AuxData."""
        return f"AuxData({self.data})"

    def __str__(self) -> str:
        """Return string representation of AuxData."""
        return str(self.data)
