"""Higher level API for reading and writing SZPLT files."""

from __future__ import annotations

import ctypes
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from . import libtecio
from .libtecio import DataType, FileType, ValueLocation, ZoneType

# =======================================================================================
# SZL Reader class:
# - Leave as many fields as possible load-on demand such that the methods will call and
#   format data from the file, but not store in memory.
# - Aux data is stored in dictionaries since query functions require indices
# TODO: Fix ReadAuxData to separate out dataset, variable, and zone aux data functions
# =======================================================================================


class Read:
    """Read data from Tecplot szplt formatted binary files."""

    def __init__(self, file_name):
        """Initialize with a C-pointer file handle, metadata, and a list of zones."""
        self.handle = libtecio.tec_file_reader_open(file_name)
        self.zones = [
            ReadZone(self.handle, i + 1, self.num_vars) for i in range(self.num_zones)
        ]
        self._auxdata: ReadAuxData | None = None
        self._var_auxdata: list[ReadAuxData] | None = None

    @property
    def file_type(self) -> FileType:
        """Return file type (Full: 0, Grid: 1, Solution: 2)."""
        return libtecio.tec_file_get_type(self.handle)

    @property
    def title(self) -> str:
        """Return whole dataset title string."""
        return libtecio.tec_data_set_get_title(self.handle)

    @property
    def num_vars(self) -> int:
        """Return length of variable list."""
        return libtecio.tec_data_set_get_num_vars(self.handle)

    @property
    def var_list(self) -> list[str]:
        """Return list of variable names."""
        # Read list of var
        return [self.zones[0].variables[i].name for i in range(self.num_vars)]

    @property
    def num_zones(self) -> int:
        """Return number of zones in file."""
        return libtecio.tec_data_set_get_num_zones(self.handle)

    @property
    def num_auxdata_items(self) -> int:
        """Return the number of aux data items at the dataset level."""
        return libtecio.tec_data_set_aux_data_get_num_items(self.handle)

    @property
    def auxdata(self) -> ReadAuxData:
        """Return dictionary of dataset-level auxiliary data."""
        if self._auxdata is None:
            self._auxdata = ReadAuxData(self.handle, "dataset")
        return self._auxdata

    @property
    def var_auxdata(self) -> list[ReadAuxData]:
        """Get list of variable-level auxiliary data.

        Returns:
            List of AuxData objects, one per variable (1-indexed to match Tecplot)

        """
        if self._var_auxdata is None:
            # Create list with None at index 0 for 1-based indexing
            self._var_auxdata = [None]
            for i in range(self.num_vars):
                self._var_auxdata.append(ReadAuxData(self.handle, "var", i + 1))
        return self._var_auxdata

    def get_var_auxdata(self, var_index: int) -> ReadAuxData:
        """Get auxiliary data for a specific variable."""
        if var_index < 1 or var_index > self.num_vars:
            raise IndexError(
                f"Variable index {var_index} out of range [1, {self.num_vars}]"
            )
        return self.var_auxdata[var_index]

    def get_zone_auxdata(self, zone_index: int) -> ReadAuxData:
        """Get auxiliary data for a specific zone."""
        if zone_index < 1 or zone_index > self.num_zones:
            raise IndexError(
                f"Variable index {zone_index} out of range [1, {self.num_zones}]"
            )
        return ReadAuxData(self.handle, "zone", zone_index)


@dataclass
class ReadZone:
    """High level API with tecio functions to read szplt binary formatted zone data."""

    _handle: ctypes.c_void_p
    zone_index: int
    num_vars: int
    _auxdata: ReadAuxData | None = None
    _variables: list[ReadVariable] | None = None
    # Note: For simplicity in calling, lists of objects are initially set to none, then
    #       cached once called.

    def __post_init__(self) -> tuple[int, int, int]:
        """Set data dimensions as attributes."""
        self.I, self.J, self.K = libtecio.tec_zone_get_ijk(
            self._handle, self.zone_index
        )

    @property
    def variables(self) -> list[ReadVariable]:
        """Create list of variable-reader objects."""
        # Check cached private variables -> don't run C functions each time this is
        # called if already defined
        if self._variables is None:
            self._variables = [
                ReadVariable(self._handle, self.zone_index, i + 1)
                for i in range(self.num_vars)
            ]
        return self._variables

    @property
    def title(self) -> str:
        """Return the tile of the current zone."""
        return libtecio.tec_zone_get_title(self._handle, self.zone_index)

    @property
    def zone_type(self) -> ZoneType:
        """Return the type of the current zone as a ZoneType Enum object."""
        return ZoneType(libtecio.tec_zone_get_type(self._handle, self.zone_index))

    def is_enabled(self) -> bool:
        """Return boolean if zone is enabled."""
        return libtecio.tec_zone_is_enabled(self._handle, self.zone_index)

    @property
    def num_points(self) -> int:
        """Return number of nodes for the current zone."""
        if self.zone_type == ZoneType.ORDERED:
            return self.I * self.J * self.K
        else:
            return self.I

    @property
    def num_elements(self) -> int:
        """Return number of elements for the current zone.

        Note: same as nodes for ORDERED.
        """
        if self.zone_type == ZoneType.ORDERED:
            return self.I * self.J * self.K
        else:
            return self.J

    @property
    def dimensions(self) -> tuple[int, int, int]:
        """Returns I, J, K, dimensions for the current zone."""
        return (self.I, self.J, self.K)

    @property
    def nodes_per_cell(self) -> int:
        """Returns how many nodes per cell based on FE type."""
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
        """Returns the zone solution time for time dependent data.

        Note: stationary data use 0.
        """
        return libtecio.tec_zone_get_solution_time(self._handle, self.zone_index)

    @property
    def strand_id(self) -> int:
        """Returns the zone strand ID for time dependent data (0 for stationary)."""
        return libtecio.tec_zone_get_strand_id(self._handle, self.zone_index)

    @property
    def node_map(self) -> npt.NDArray[np.int64] | None:
        """Returns (n x m) node map array for n-cells and m-nodes per cell."""
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
        """Get zone-level auxiliary data."""
        if self._auxdata is None:
            self._auxdata = ReadAuxData(self._handle, "zone", self.zone_index)
        return self._auxdata


@dataclass
class ReadVariable:
    """High level API with tecio functions to read szplt binary zone data."""

    _handle: ctypes.c_void_p
    zone_index: int
    var_index: int

    @property
    def name(self) -> str:
        """Return variable name as string."""
        return libtecio.tec_var_get_name(self._handle, self.var_index)

    def is_enabled(self) -> bool:
        """Return true/false if variable is enabled."""
        return libtecio.tec_var_is_enabled(self._handle, self.var_index)

    @property
    def data_type(self) -> DataType:
        """Return DataType Enum corresponding the C-type of the value array."""
        return libtecio.tec_zone_var_get_type(
            self._handle, self.zone_index, self.var_index
        )

    @property
    def value_location(self) -> ValueLocation:
        """Return the location (cell or node centered) for the current variable."""
        return libtecio.tec_zone_var_get_value_location(
            self._handle, self.zone_index, self.var_index
        )

    def is_passive(self) -> bool:
        """Return if current variable does not exist for the parent zone (passive)."""
        return libtecio.tec_zone_var_is_passive(
            self._handle, self.zone_index, self.var_index
        )

    @property
    def shared_zone(self) -> int | None:
        """Outputs shared zone index (0 if none)."""
        return libtecio.tec_zone_var_get_shared_zone(
            self._handle, self.zone_index, self.var_index
        )

    @property
    def num_values(self) -> int:
        """Returns the number of values in the data array."""
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
    ):
        """Get all values for this variable."""
        return self.get_values()

    def get_values(
        self, value_range: tuple[int | None, int | None] = (None, None)
    ) -> (
        npt.NDArray[np.float32]
        | npt.NDArray[np.float64]
        | npt.NDArray[np.int32]
        | npt.NDArray[np.int16]
        | npt.NDArray[np.uint8]
    ):
        """Get variable values with optional range specification.

        Args:
            value_range: Tuple of (start_index, end_index). If (None, None),
                         retrieves all values.

        Returns:
            NumPy array of values with appropriate dtype

        """
        data_type = self.data_type

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
                raise ValueError(f"Invalid value range: ({start_index}, {end_index})")

        if data_type == DataType.FLOAT:
            return libtecio.tec_zone_var_get_float_values(
                self._handle, self.zone_index, self.var_index, start_index, num_values
            )

        elif data_type == DataType.DOUBLE:
            return libtecio.tec_zone_var_get_double_values(
                self._handle, self.zone_index, self.var_index, start_index, num_values
            )

        elif data_type == DataType.INT32:
            return libtecio.tec_zone_var_get_int32_values(
                self._handle, self.zone_index, self.var_index, start_index, num_values
            )

        elif data_type == DataType.INT16:
            return libtecio.tec_zone_var_get_int16_values(
                self._handle, self.zone_index, self.var_index, start_index, num_values
            )

        elif data_type == DataType.BYTE:
            return libtecio.tec_zone_var_get_uint8_values(
                self._handle, self.zone_index, self.var_index, start_index, num_values
            )

        raise ValueError(f"Unknown data type: {data_type}")


class ReadAuxData:
    """Dictionary-like interface for Tecplot aux data with auto type conversion.

    Values are accessed as strings in the SZL file but can be retrieved
    as integers or floats using the as_int() and as_float() methods.
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

    def as_int(self, key: str, default: int | None = None) -> int | None:
        """Get auxiliary data value as integer.

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
        """Get auxiliary data value as float.

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
        """Get auxiliary data value as boolean.

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


# =======================================================================================
# SZL Writer class:
# - Supports lazy loading such that file and aux data are buffered until first zone
#   assignment if variable list not provided when intialized.
# - Some writing steps that can easily be automated (such as writing variable data)
#   is exposed to users through public funcitons (eg write_data combines all libtecio
#   write functions and infers/casts data type).
#
# Notes:
# - add a call to open() in each zone writing function if self.handle==None
# - add a call to flush_aux() in each zone writing funcion if self.handle==None
# - flush_aux():
#   - write dataset aux if self.aux_dataset is not empty
#   - write variable aux if self.aux_var is not empty
#   - after writing set both to empty with self.aux_dataset.clear() and
#     self.aux_var.clear()
# - Zone writers:
#   - write_zone(): provide all parameters needed to fully write a zone. Will call the
#     appropriate header writer and data writer
#     - write_ijk_zone(): write zone header and optionally data for input ORDERED data
#       - set default value location as NODAL
#       - write default data as DOUBLE
#       - if no data provided, just write header and set self.current_zone
#       - support data provided as a list[npt.NDArrayLike, ...]
#       - if var list already defined for whole dataset, do not require, but if not
#         defined, throw error
#     - write_fe_zone():
# =======================================================================================


# FE zone types that use tec_zone_create_fe
_FE_SIMPLE: frozenset[ZoneType] = frozenset({
    ZoneType.FELINESEG,
    ZoneType.FETRIANGLE,
    ZoneType.FEQUADRILATERAL,
    ZoneType.FETETRAHEDRON,
    ZoneType.FEBRICK,
})


def _infer_data_type(dt: DataType | np.dtype) -> DataType:
    """Return a C-supported DataType for either DataType or any Numpy dtype input."""
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

    # Convert to NumPy dtype
    dt_np = np.dtype(dt)

    # Return exact match if in dtype map
    for key in closest_dtype_map:
        if dt_np == key:
            return closest_dtype_map[key]

    # If no exact match, handle broader type catagories
    if np.issubdtype(dt_np, np.floating):
        # Anything floating → DOUBLE or FLOAT depending on precision
        if dt_np.itemsize <= 4:
            return DataType.FLOAT
        else:
            return DataType.DOUBLE

    if np.issubdtype(dt_np, np.signedinteger):
        if dt_np.itemsize <= 2:
            return DataType.INT16
        else:
            return DataType.INT32

    if np.issubdtype(dt_np, np.unsignedinteger):
        return DataType.BYTE

    raise ValueError(f"Unsupported dtype: {dt_np}")


class Write:
    """Write Tecplot SZL (``.szplt``) files with a lazy-open file handle.

    The tecio library requires a list of variables when the output file is
    opened. However if writing data on the fly, it may be beneficial to store file
    outputs until the first zone is passed to the writer. Then file header will be
    garanteed to be consistent with the first zone variables.

    For the SZL API, file contents can be written out of order after creating zones.
    """

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
        self.file_type = file_type

        # Add created data to the buffer and flush once a file handle is created.
        # Dataset-level aux data buffer (flushed on first zone)
        self.auxdataset: dict[str, str] = {}
        # Variable-level aux data buffer: {var_name: {key: value}}
        self.var_auxdata: dict[int, dict[str, str]] = {}

        # Initialize if all needed info provided, else set to null
        if self.variables is not None:
            self.handle = libtecio.tec_file_writer_open(
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
        """Context manager to automatically close and flush file."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit contex manager regardless of exceptions."""
        self.close()

    def open(self, var_names: list[str]) -> None:
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
        """Finalise and flush the file.  Safe to call more than once."""
        if self.handle is not None:
            libtecio.tec_file_writer_close(self.handle)
            self.handle = None

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
        for name, value in self.dataset_aux.items():
            libtecio.tec_data_set_add_aux_data(self._handle, str(name), str(value))

        # Variable-level aux data keys can be variable index or variable names,
        # therefore need to normalize to index before writing to file
        for key, subdict in self.var_aux.items():
            if isinstance(key, int):
                var_idx = key - 1  # Convert to 0-based
                if var_idx not in range(len(self.variables)):
                    raise IndexError(
                        f"Index {var_idx + 1} out of bounds of "
                        f"number of variables ({len(self.variables)})"
                    )
            elif isinstance(key, str):
                try:
                    var_idx = self.variables.index(key)
                except ValueError as e:
                    raise KeyError(
                        f"Variable aux data key '{key}' not in available "
                        f"variable names ({self.variables})"
                    ) from e
            else:
                raise TypeError(
                    f"Key must be variable index or variable name. '{key}' not found."
                )

            # Write nested key/value to file
            for name, value in subdict.items():
                # Correct variable index to 1-based and call libtecio
                libtecio.tec_var_add_aux_data(
                    self._handle, var_idx + 1, str(name), str(value)
                )

        # Clear for future aux data definitions
        self.dataset_aux.clear()
        self.var_aux.clear()

    def write_zone(
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
        if self.handle is None:
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
                handle=self.handle,
                zone_title=title,
                I=I,
                J=J,
                K=K,
                var_types=var_types,
                value_locations=value_locations,
            )
        elif zone_type in _FE_SIMPLE:
            zone_num = libtecio.tec_zone_create_fe(
                handle=self.handle,
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
                handle=self.handle,
                zone=zone_num,
                strand=strand_id,
                solution_time=solution_time,
            )

        # Variable data
        for var_idx, (arr, dt) in enumerate(zip(arrays, var_types)):
            write_data(self.handle, zone_num, var_idx + 1, arr.ravel(), dt)

        # Connectivity
        if zone_type in _FE_SIMPLE and node_map is not None:
            flat = np.asarray(node_map).ravel()
            if flat.dtype == np.int64 or int(flat.max()) > np.iinfo(np.int32).max:
                arr64 = np.ascontiguousarray(flat, dtype=np.int64)
                libtecio.tec_zone_node_map_write64(self.handle, zone_num, arr64)
            else:
                arr32 = np.ascontiguousarray(flat, dtype=np.int32)
                libtecio.tec_zone_node_map_write32(self.handle, zone_num, arr32)

        # Zone aux data
        if auxdata:
            for name, value in auxdata.items():
                libtecio.tec_zone_add_aux_data(self.handle, zone_num, name, value)


def write_data(
    handle: ctypes.c_void_p,
    zone_num: int,
    var_num: int,
    data: npt.ArrayLike,
    dt: np.dtype | DataType = None,
) -> None:
    """Single simplified function to write data using NEW SZL API.

    Output defaults:
    1. Inferrs data_typeype for nummpy arrays
    2. Defaults to double precision for array-like (list, tuple, etc)
    3. Optionally casts to input DataType or numpy dtype.
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
        arr = np.ascontiguousarray(data, dtype=datatype_to_dtype[data_type]).ravel()
    else:
        arr = np.ascontiguousarray(data).ravel()
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
