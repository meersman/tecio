"""Read Tecplot SZL (``.szplt``) files via the TecIO C library.

Zone metadata is queried on demand from the C library; variable data is
read lazily when :attr:`ReadVariable.values` is accessed.

Todo:
    Fix ReadAuxData to separate out dataset, variable, and zone aux data functions
"""

from __future__ import annotations

import ctypes
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from .. import libtecio
from ..libtecio import DataType, FileType, ValueLocation, ZoneType


class Read:
    """Reader for Tecplot ``.szplt`` files.

    Metadata is queried lazily from the C library. Variable data arrays
    are read on first access via :attr:`ReadVariable.values`.

    Args:
        file_name: Path to the ``.szplt`` file.
    """

    def __init__(self, file_name):
        """Initialize with a C-pointer file handle, metadata, and a list of zones."""
        if not Path(file_name).exists():
            raise FileNotFoundError(f"No such file or directory: '{file_name}'")
        self.handle = libtecio.tec_file_reader_open(file_name)
        self.zone = [
            ReadZone(self.handle, i + 1, self.num_vars) for i in range(self.num_zones)
        ]
        self._auxdata: ReadAuxData | None = None
        self._var_auxdata: list[ReadAuxData] | None = None

    def __enter__(self) -> Read:
        """Context manager for Read class."""
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Exit Read class context manager regardless of exceptions.

        Only raise an exception if closing the file fails, not if an exception is raised
        in the with block.
        """
        try:
            self.close()
        except Exception:
            if exc_type is None:
                raise

    @property
    def file_type(self) -> FileType:
        """File type enum (FULL, GRID, or SOLUTION)."""
        return libtecio.tec_file_get_type(self.handle)

    @property
    def title(self) -> str:
        """Dataset title string."""
        return libtecio.tec_data_set_get_title(self.handle)

    @property
    def num_vars(self) -> int:
        """Number of variables in the dataset."""
        return libtecio.tec_data_set_get_num_vars(self.handle)

    @property
    def variables(self) -> list[str]:
        """Ordered list of variable name strings."""
        # Read list of var
        return [self.zone[0].variable[i].name for i in range(self.num_vars)]

    @property
    def num_zones(self) -> int:
        """Number of zones in the file."""
        return libtecio.tec_data_set_get_num_zones(self.handle)

    @property
    def num_auxdata_items(self) -> int:
        """Number of dataset-level auxiliary data items."""
        return libtecio.tec_data_set_aux_data_get_num_items(self.handle)

    @property
    def auxdata(self) -> ReadAuxData:
        """Per-variable auxiliary data (1-indexed to match Tecplot)."""
        if self._auxdata is None:
            self._auxdata = ReadAuxData(self.handle, "dataset")
        return self._auxdata

    @property
    def var_auxdata(self) -> list[ReadAuxData]:
        """Per-variable auxiliary data (1-indexed to match Tecplot)."""
        if self._var_auxdata is None:
            # Create list with None at index 0 for 1-based indexing
            self._var_auxdata = [None]
            for i in range(self.num_vars):
                self._var_auxdata.append(ReadAuxData(self.handle, "var", i + 1))
        return self._var_auxdata

    def get_var_auxdata(self, var_index: int) -> ReadAuxData:
        """Get list of variable-level auxiliary data.

        Returns:
            List of AuxData objects, one per variable (1-indexed to match Tecplot)

        Raises:
            IndexError: If *var_index* is out of range.
        """
        if var_index < 1 or var_index > self.num_vars:
            raise IndexError(
                f"Variable index {var_index} out of range [1, {self.num_vars}]"
            )
        return self.var_auxdata[var_index]

    def get_zone_auxdata(self, zone_index: int) -> ReadAuxData:
        """Return auxiliary data for zone *zone_index* (1-based).

        Raises:
            IndexError: If *zone_index* is out of range.
        """
        if zone_index < 1 or zone_index > self.num_zones:
            raise IndexError(
                f"Variable index {zone_index} out of range [1, {self.num_zones}]"
            )
        return ReadAuxData(self.handle, "zone", zone_index)

    def close(self) -> None:
        """Close the file reader handle."""
        if self.handle is not None:
            libtecio.tec_file_reader_close(self.handle)
            self.handle = None


@dataclass
class ReadZone:
    """Zone reader for SZL files.

    Provides access to zone metadata and per-variable data arrays.

    Args:
        _handle (ctypes.c_void_p): C library file handle.
        zone_index (int): 1-based zone index.
        num_vars (int): Number of variables in the dataset.
    """

    _handle: ctypes.c_void_p
    zone_index: int
    num_vars: int
    _auxdata: ReadAuxData | None = None
    _variable: list[ReadVariable] | None = None
    # Note: For simplicity in calling, lists of objects are initially set to none, then
    #       cached once called.

    def __post_init__(self) -> tuple[int, int, int]:
        """Set data dimensions as attributes."""
        self.I, self.J, self.K = libtecio.tec_zone_get_ijk(
            self._handle, self.zone_index
        )

    @property
    def variable(self) -> list[ReadVariable]:
        """List of :class:`ReadVariable` objects (0-indexed)."""
        # Check cached private variables -> don't run C functions each time this is
        # called if already defined
        if self._variable is None:
            self._variable = [
                ReadVariable(self._handle, self.zone_index, i + 1)
                for i in range(self.num_vars)
            ]
        return self._variable

    def __getattr__(self, name: str) -> Any:
        """Access variable data by name (case-insensitive).

        Example:
            zone.pressure -> returns the NumPy array for variable "Pressure"

        """
        # Only called if normal attributes do not exist
        for var in self.variables:
            if var.name.lower() == name.lower():  # case-insensitive match
                return var.values
        # If no match, raise normal AttributeError
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    @property
    def title(self) -> str:
        """Zone title string."""
        return libtecio.tec_zone_get_title(self._handle, self.zone_index)

    @property
    def zone_type(self) -> ZoneType:
        """Current zone :class:`ZoneType` enum."""
        return ZoneType(libtecio.tec_zone_get_type(self._handle, self.zone_index))

    def is_enabled(self) -> bool:
        """Return True if the zone is enabled."""
        return libtecio.tec_zone_is_enabled(self._handle, self.zone_index)

    @property
    def num_nodes(self) -> int:
        """Number of nodes in current zone."""
        if self.zone_type == ZoneType.ORDERED:
            return self.I * self.J * self.K
        else:
            return self.I

    @property
    def num_elements(self) -> int:
        """Number of elements (same as nodes for ORDERED)."""
        if self.zone_type == ZoneType.ORDERED:
            return self.I * self.J * self.K
        else:
            return self.J

    @property
    def dimensions(self) -> tuple[int, int, int]:
        """``(I, J, K)`` dimensions for current zone."""
        return (self.I, self.J, self.K)

    @property
    def nodes_per_cell(self) -> int:
        """Number of nodes per cell based on zone type.

        Raises:
            ValueError: If ZoneType not known.
        """
        if self.zone_type == ZoneType.FELINESEG:
            return 2
        elif self.zone_type == ZoneType.FETRIANGLE:
            return 3
        elif (
            self.zone_type == ZoneType.FEQUADRILATERAL
            or self.zone_type == ZoneType.FETETRAHEDRON
        ):
            return 4
        elif self.zone_type == ZoneType.FEBRICK:
            return 8
        elif self.zone_type == ZoneType.ORDERED:
            # Check the dimension of the ordered dataset (1D, 2D, or 3D)
            dims = sum(1 for x in (self.I, self.J, self.K) if x > 1)
            return 2**dims
        else:
            raise ValueError("ZoneType does not have a consistent number of nodes")

    @property
    def solution_time(self) -> float:
        """Solution time (0.0 for stationary data)."""
        return libtecio.tec_zone_get_solution_time(self._handle, self.zone_index)

    @property
    def strand_id(self) -> int:
        """Strand ID (0 for stationary data)."""
        return libtecio.tec_zone_get_strand_id(self._handle, self.zone_index)

    @property
    def node_map(self) -> npt.NDArray[np.int64] | None:
        """Node connectivity array ``(num_elements, nodes_per_cell)``.

        Returns:
            (n x m) node map array for n-cells and m-nodes per cell.
        """
        is64bit = libtecio.is_64bit(self._handle, self.zone_index)
        if self.zone_type == ZoneType.ORDERED:
            return None
        elif is64bit:
            return libtecio.tec_zone_node_map_get_64(
                self._handle,
                self.zone_index,
                self.num_elements,
                self.nodes_per_cell,
            )
        else:
            return libtecio.tec_zone_node_map_get(
                self._handle,
                self.zone_index,
                self.num_elements,
                self.nodes_per_cell,
            ).astype(np.int64)

    @property
    def auxdata(self) -> ReadAuxData:
        """Zone-level auxiliary data."""
        if self._auxdata is None:
            self._auxdata = ReadAuxData(self._handle, "zone", self.zone_index)
        return self._auxdata


@dataclass
class ReadVariable:
    """Lazy variable reader for SZL files.

    Data is read from the C library only when :attr:`values` or
    :meth:`get_values` is called.

    Args:
        _handle (ctypes.c_void_p): C library file handle.
        zone_index (int): 1-based zone index.
        var_index (int): 1-based variable index.
    """

    _handle: ctypes.c_void_p
    zone_index: int
    var_index: int

    @property
    def name(self) -> str:
        """Variable name string."""
        return libtecio.tec_var_get_name(self._handle, self.var_index)

    def is_enabled(self) -> bool:
        """Return True if the variable is enabled."""
        return libtecio.tec_var_is_enabled(self._handle, self.var_index)

    @property
    def data_type(self) -> DataType:
        """Data type enum for this variable in this zone."""
        return libtecio.tec_zone_var_get_type(
            self._handle, self.zone_index, self.var_index
        )

    @property
    def value_location(self) -> ValueLocation:
        """Value location (NODAL or CELL_CENTERED)."""
        return libtecio.tec_zone_var_get_value_location(
            self._handle, self.zone_index, self.var_index
        )

    def is_passive(self) -> bool:
        """Return True if this variable is passive in this zone."""
        return libtecio.tec_zone_var_is_passive(
            self._handle, self.zone_index, self.var_index
        )

    @property
    def shared_zone(self) -> int | None:
        """Source zone index if shared, or None."""
        return libtecio.tec_zone_var_get_shared_zone(
            self._handle, self.zone_index, self.var_index
        )

    @property
    def num_values(self) -> int:
        """Number of values in the data array."""
        return libtecio.tec_zone_var_get_num_values(
            self._handle, self.zone_index, self.var_index
        )

    @property
    def values(
        self,
    ) -> (
        npt.NDArray[np.float32]
        | npt.NDArray[np.float64]
        | npt.NDArray[np.int32]
        | npt.NDArray[np.int16]
        | npt.NDArray[np.uint8]
        | None
    ):
        """All values as a NumPy array, or None if passive/shared."""
        return self.get_values()

    def get_values(
        self, value_range: tuple[int | None, int | None] = (None, None)
    ) -> (
        npt.NDArray[np.float32]
        | npt.NDArray[np.float64]
        | npt.NDArray[np.int32]
        | npt.NDArray[np.int16]
        | npt.NDArray[np.uint8]
        | None
    ):
        """Get variable values with optional range specification.

        For ordered zones a full read returns the array shaped ``(I, J, K)``
        for nodal variables, or ``(I-1, J-1, K-1)`` for cell-centered, so
        that zone dimensions can be inferred directly from the array shape.
        Partial reads always return a flat 1-D array.

        Args:
            value_range: Tuple of (start_index, end_index). If (None, None),
                         retrieves all values.

        Returns:
            NumPy array of values with appropriate dtype. Ordered zones return arrays
            reshaped to (I, J, K) or (I-1, J-1, K-1) for full reads. FE unstructured
            zones return flat 1-D arrays. Returns None if variable is passive or shared.

        Raises:
            ValueError: If only one of start/end is specified.
        """
        # First check if variable is passive or shared (no data to return)
        if self.is_passive() or (self.shared_zone is not None):
            return None

        data_type = self.data_type
        full_read = value_range == (None, None)

        if full_read:
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
                raise ValueError(f"Invalid value range: ({start_index}, {end_index})")

        if data_type == DataType.FLOAT:
            arr = libtecio.tec_zone_var_get_float_values(
                self._handle, self.zone_index, self.var_index, start_index, num_values
            )
        elif data_type == DataType.DOUBLE:
            arr = libtecio.tec_zone_var_get_double_values(
                self._handle, self.zone_index, self.var_index, start_index, num_values
            )
        elif data_type == DataType.INT32:
            arr = libtecio.tec_zone_var_get_int32_values(
                self._handle, self.zone_index, self.var_index, start_index, num_values
            )
        elif data_type == DataType.INT16:
            arr = libtecio.tec_zone_var_get_int16_values(
                self._handle, self.zone_index, self.var_index, start_index, num_values
            )
        elif data_type == DataType.BYTE:
            arr = libtecio.tec_zone_var_get_uint8_values(
                self._handle, self.zone_index, self.var_index, start_index, num_values
            )
        else:
            raise ValueError(f"Unknown data type: {data_type}")

        # Reshape to (I, J, K) / (I-1, J-1, K-1) for full reads of ordered zones.
        if full_read and arr is not None:
            ni, nj, nk = libtecio.tec_zone_get_ijk(self._handle, self.zone_index)
            zt = libtecio.tec_zone_get_type(self._handle, self.zone_index)
            if zt == ZoneType.ORDERED:
                if self.value_location == ValueLocation.CELL_CENTERED:
                    shape = (max(ni - 1, 1), max(nj - 1, 1), max(nk - 1, 1))
                else:
                    shape = (ni, nj, nk)
                if arr.size == shape[0] * shape[1] * shape[2]:
                    arr = arr.reshape(shape, order="F")

        return arr


class ReadAuxData:
    """Dict-like interface for Tecplot auxiliary data with type conversion.

    Values are accessed as strings in the SZL file but can be retrieved
    as integers or floats using the as_int() and as_float() methods.

    Args:
        handle (ctypes.c_void_p): C library file handle.
        aux_type (str): One of ``'dataset'``, ``'var'``, or ``'zone'``.
        index (int | None): 1-based variable or zone index (not needed
            for dataset).
    """

    def __init__(
        self,
        handle: ctypes.c_void_p,
        aux_type: str,
        index: int | None = None,
    ):
        """Initialize AuxData wrapper.

        Args:
            handle: File handle C pointer
            aux_type: Type of auxiliary data ('dataset', 'var', or 'zone')
            index: Variable or zone index (1-based), not needed for dataset
        """
        self._handle = handle
        self._aux_type = aux_type
        self._index = index
        self._data: dict[str, str] | None = None

    def _load_data(self) -> None:
        """Load auxiliary data from file into internal dictionary."""
        if self._data is not None:
            return

        self._data = {}

        if self._aux_type == "dataset":
            num_items = libtecio.tec_data_set_aux_data_get_num_items(self._handle)
            for i in range(num_items):
                name, value = libtecio.tec_data_set_aux_data_get_item(
                    self._handle, i + 1
                )
                self._data[name] = value

        elif self._aux_type == "var":
            if self._index is None:
                raise ValueError("Variable index required for variable aux data")
            num_items = libtecio.tec_var_aux_data_get_num_items(
                self._handle, self._index
            )
            for i in range(num_items):
                name, value = libtecio.tec_var_aux_data_get_item(
                    self._handle, self._index, i + 1
                )
                self._data[name] = value

        elif self._aux_type == "zone":
            if self._index is None:
                raise ValueError("Zone index required for zone aux data")
            num_items = libtecio.tec_zone_aux_data_get_num_items(
                self._handle, self._index
            )
            for i in range(num_items):
                name, value = libtecio.tec_zone_aux_data_get_item(
                    self._handle, self._index, i + 1
                )
                self._data[name] = value

        else:
            raise ValueError(f"Invalid aux_type: {self._aux_type}")

    @property
    def data(self) -> dict[str, str]:
        """Underlying dictionary of auxiliary data."""
        self._load_data()
        return self._data

    def __len__(self) -> int:
        """Number of auxiliary data items."""
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

    def as_int(self, key: str, default: int | None = None) -> int | None:
        """Get auxiliary data as int, or *default* on failure.

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

    def as_float(self, key: str, default: float | None = None) -> float | None:
        """Get auxiliary data value as float or *default* on failure.

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

    def as_bool(self, key: str, default: bool | None = None) -> bool | None:
        """Return value for *key* as bool, or *default* on failure.

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
            if value in ("true", "t", "yes", "y", "1"):
                return True
            elif value in ("false", "f", "no", "n", "0"):
                return False
            else:
                return default
        except (KeyError, AttributeError):
            return default

    def __repr__(self) -> str:
        """Return string representation of AuxData."""
        return f"ReadAuxData({self.data})"

    def __str__(self) -> str:
        """Return string representation of AuxData."""
