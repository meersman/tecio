from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import numpy.typing as npt

from . import szlio
from . import szlfile
from .szlio import DataType, FileType, ValueLocation, ZoneType


@dataclass
class TecVariable:
    """
    Variable metadata.
    
    Mirrors variable properties from SZL files but mutable.
    All fields populated from input file during load.
    """

    name: str
    data_type: DataType = DataType.DOUBLE
    value_location: ValueLocation = ValueLocation.NODE_CENTERED
    auxdata: Dict[str, str] = field(default_factory=dict)


@dataclass
class TecZone:
    """
    Mutable zone with all data loaded in memory.
    
    Mirrors zone properties from SZL files but fully mutable.
    Data is stored in memory, not read on-demand.
    """

    title: str
    zone_type: ZoneType
    dimensions: tuple[int, int, int]  # (I, J, K)
    solution_time: float = 0.0
    strand_id: int = 0
    parent_zone: int = -1
    auxdata: Dict[str, str] = field(default_factory=dict)
    node_map: Optional[npt.NDArray[np.int64]] = None
    _data: Dict[int, npt.NDArray] = field(default_factory=dict)

    @property
    def num_points(self) -> int:
        """Calculate number of points based on zone type."""
        if self.zone_type == ZoneType.ORDERED:
            return self.dimensions[0] * self.dimensions[1] * self.dimensions[2]
        else:
            return self.dimensions[0]

    @property
    def num_elements(self) -> int:
        """Calculate number of elements based on zone type."""
        if self.zone_type == ZoneType.ORDERED:
            # For ordered zones, elements = cells
            i, j, k = self.dimensions
            return max(i - 1, 1) * max(j - 1, 1) * max(k - 1, 1)
        else:
            return self.dimensions[1]

    def get_variable_data(self, var_index: int) -> Optional[npt.NDArray]:
        """
        Get variable data by index (0-based).

        Args:
            var_index: Variable index (0-based)

        Returns:
            NumPy array of variable data, or None if not loaded
        """
        return self._data.get(var_index)

    def set_variable_data(self, var_index: int, values: npt.NDArray) -> None:
        """
        Set variable data by index (0-based).

        Args:
            var_index: Variable index (0-based)
            values: NumPy array of values
        """
        self._data[var_index] = values

    def has_variable_data(self, var_index: int) -> bool:
        """Check if variable data is loaded for this zone."""
        return var_index in self._data


class TecData:
    """
    Mutable in-memory Tecplot dataset.

    This class loads all requested data into memory (not on-demand caching).
    All properties are mutable and directly accessible.
    
    The structure mirrors SzlFile but with data fully loaded and mutable.
    """

    def __init__(
        self,
        file_path: Optional[str] = None,
        zones: Optional[Sequence[int]] = None,
        vars: Optional[Union[Sequence[int], Sequence[str]]] = None,
    ):
        """
        Initialize TecData, optionally loading from file.

        Args:
            file_path: Path to input file (.szplt, .plt, .dat). If None, creates empty.
            zones: Zone indices to load (0-based). If None, loads all zones.
                   If empty list, loads no zones (metadata only).
            vars: Variable indices (0-based) or names to load. If None, loads all.

        Examples:
            >>> data = TecData()  # Empty dataset
            >>> data = TecData("flow.szplt")  # Full load
            >>> data = TecData("flow.szplt", zones=[0, 1, 2])  # Zones 0-2 only
            >>> data = TecData("flow.szplt", vars=[0, 1, 2])  # Variables 0-2 only
            >>> data = TecData("flow.szplt", vars=["X", "Y", "Z"])  # By name
            >>> data = TecData("flow.szplt", zones=[])  # Metadata only, no zones
        """
        # All properties are public and mutable
        self.title: str = ""
        self.file_type: FileType = FileType.FULL
        self.num_vars: int = 0
        self.num_zones: int = 0
        self.variables: List[TecVariable] = []
        self.zones: List[TecZone] = []
        self.auxdata: Dict[str, str] = {}

        # Load from file if provided
        if file_path is not None:
            # Detect file type and load appropriately
            if file_path.endswith('.szplt'):
                self._load_from_szl(file_path, zones, vars)
            elif file_path.endswith('.plt'):
                self._load_from_plt(file_path, zones, vars)
            elif file_path.endswith('.dat'):
                self._load_from_dat(file_path, zones, vars)
            else:
                # Try SZL as default
                self._load_from_szl(file_path, zones, vars)

    def _load_from_szl(
        self,
        file_path: str,
        zone_filter: Optional[Sequence[int]],
        var_filter: Optional[Union[Sequence[int], Sequence[str]]],
    ) -> None:
        """
        Load data from SZL file into memory.

        All requested data is loaded immediately, not cached or read on-demand.

        Args:
            file_path: Path to .szplt file
            zone_filter: Zone indices to load (0-based), or None for all
            var_filter: Variable indices (0-based) or names, or None for all
        """
        # Open SZL file (read-only interface)
        szl = szlfile.Read(file_path)

        # Load dataset-level metadata
        self.title = szl.title
        self.file_type = szl.type
        self.auxdata = dict(szl.auxdata.items())

        # Process variable filter to get indices
        var_indices = self._resolve_var_filter(szl, var_filter)

        # Load variable metadata for requested variables
        self.variables = []
        for var_idx in var_indices:
            # Get variable info from first zone (all zones share variable names)
            szl_var = szl.zones[0].variables[var_idx]
            
            # Get variable-level auxiliary data
            var_aux = dict(szl.get_var_auxdata(var_idx + 1).items())
            
            # Create TecVariable with all metadata loaded
            tec_var = TecVariable(
                name=szl_var.name,
                data_type=szl_var.type,
                value_location=szl_var.value_location,
                auxdata=var_aux
            )
            self.variables.append(tec_var)

        # Update num_vars
        self.num_vars = len(self.variables)

        # Process zone filter
        if zone_filter is None:
            # Load all zones
            zone_indices = list(range(szl.num_zones))
        elif len(zone_filter) == 0:
            # Empty list = load no zones (metadata only)
            zone_indices = []
        else:
            # Load specified zones (0-based)
            zone_indices = list(zone_filter)

        # Load zones with all data into memory
        self.zones = []
        for zone_idx in zone_indices:
            if zone_idx < 0 or zone_idx >= szl.num_zones:
                raise IndexError(
                    f"Zone index {zone_idx} out of range [0, {szl.num_zones})"
                )

            szl_zone = szl.zones[zone_idx]

            # Create zone with all metadata
            zone = TecZone(
                title=szl_zone.title,
                zone_type=szl_zone.type,
                dimensions=szl_zone.dimensions,
                solution_time=szl_zone.solution_time,
                strand_id=szl_zone.strand_id,
                auxdata=dict(szl_zone.auxdata.items()),
            )

            # Load node map for FE zones
            if szl_zone.type != ZoneType.ORDERED:
                zone.node_map = szl_zone.node_map.copy()

            # Load ALL variable data into memory (not cached, fully loaded)
            for local_idx, global_idx in enumerate(var_indices):
                var_data = szl_zone.variables[global_idx].values.copy()
                zone.set_variable_data(local_idx, var_data)

            self.zones.append(zone)

        # Update num_zones
        self.num_zones = len(self.zones)

    def _load_from_plt(
        self,
        file_path: str,
        zone_filter: Optional[Sequence[int]],
        var_filter: Optional[Union[Sequence[int], Sequence[str]]],
    ) -> None:
        """
        Load data from PLT binary file into memory.

        Future implementation - will use pltfile module.

        Args:
            file_path: Path to .plt file
            zone_filter: Zone indices to load (0-based), or None for all
            var_filter: Variable indices (0-based) or names, or None for all
        """
        raise NotImplementedError("PLT file loading not yet implemented")

    def _load_from_dat(
        self,
        file_path: str,
        zone_filter: Optional[Sequence[int]],
        var_filter: Optional[Union[Sequence[int], Sequence[str]]],
    ) -> None:
        """
        Load data from ASCII DAT file into memory.

        Future implementation - will produce same TecData structure.

        Args:
            file_path: Path to .dat file
            zone_filter: Zone indices to load (0-based), or None for all
            var_filter: Variable indices (0-based) or names, or None for all
        """
        raise NotImplementedError("ASCII DAT file loading not yet implemented")

    def _resolve_var_filter(
        self,
        szl: szlfile.Read,
        var_filter: Optional[Union[Sequence[int], Sequence[str]]],
    ) -> List[int]:
        """
        Resolve variable filter to list of 0-based indices.

        Args:
            szl: SzlFile object
            var_filter: Variable indices (0-based) or names, or None for all

        Returns:
            List of 0-based variable indices to load
        """
        if var_filter is None:
            # Load all variables
            return list(range(szl.num_vars))

        # Check if filter is list of strings (names) or integers (indices)
        if len(var_filter) == 0:
            return []

        first_item = var_filter[0]

        if isinstance(first_item, str):
            # Filter by variable names
            var_indices = []
            all_var_names = [
                szl.zones[0].variables[i].name for i in range(szl.num_vars)
            ]

            for var_name in var_filter:
                if var_name not in all_var_names:
                    raise ValueError(f"Variable '{var_name}' not found in file")
                var_indices.append(all_var_names.index(var_name))

            return var_indices
        else:
            # Filter by indices (0-based)
            var_indices = list(var_filter)

            # Validate indices
            for idx in var_indices:
                if idx < 0 or idx >= szl.num_vars:
                    raise IndexError(
                        f"Variable index {idx} out of range [0, {szl.num_vars})"
                    )

            return var_indices

    def _infer_file_type(self) -> FileType:
        """
        Automatically determine FileType based on dataset content.

        Returns:
            FileType.FULL - Always returns FULL for complete datasets
        """
        # Could be enhanced to detect GRID vs SOLUTION based on variables
        return self.file_type if self.file_type else FileType.FULL

    def _infer_data_type(self, data: npt.NDArray) -> DataType:
        """
        Automatically determine DataType from numpy array dtype.

        Args:
            data: NumPy array

        Returns:
            Appropriate DataType enum

        Raises:
            ValueError: If dtype is not supported
        """
        dtype_map = {
            np.dtype(np.float64): DataType.DOUBLE,
            np.dtype(np.float32): DataType.FLOAT,
            np.dtype(np.int32): DataType.INT32,
            np.dtype(np.int16): DataType.INT16,
            np.dtype(np.uint8): DataType.BYTE,
        }

        if data.dtype in dtype_map:
            return dtype_map[data.dtype]
        
        # Try to match by kind and size
        if data.dtype.kind == 'f':
            if data.dtype.itemsize == 8:
                return DataType.DOUBLE
            elif data.dtype.itemsize == 4:
                return DataType.FLOAT
        elif data.dtype.kind == 'i':
            if data.dtype.itemsize == 4:
                return DataType.INT32
            elif data.dtype.itemsize == 2:
                return DataType.INT16
            elif data.dtype.itemsize == 1:
                return DataType.BYTE
        elif data.dtype.kind == 'u':
            if data.dtype.itemsize == 1:
                return DataType.BYTE

        # Default to DOUBLE for safety
        return DataType.DOUBLE

    def write_szl(self, file_path: str) -> None:
        """
        Write dataset to SZL (.szplt) file.

        Automatically determines FileType, ZoneType, DataType, and ValueLocation
        from the TecData object. Simply provide the output filename.

        Args:
            file_path: Output file path

        Example:
            >>> data = TecData("input.szplt")
            >>> data.write_szl("output.szplt")

        Raises:
            ValueError: If dataset has no variables or zones
        """
        # Validation
        if self.num_vars == 0 or len(self.variables) == 0:
            raise ValueError("Cannot write dataset with no variables")
        if self.num_zones == 0 or len(self.zones) == 0:
            raise ValueError("Cannot write dataset with no zones")

        # Automatically determine file type
        file_type = self._infer_file_type()

        # Prepare variable names as comma-separated string
        var_names_csv = ",".join([v.name for v in self.variables])

        # Open file for writing
        handle = szlio.tec_file_writer_open(
            file_name=file_path,
            dataset_title=self.title,
            var_names_csv=var_names_csv,
            file_type=file_type,
        )

        try:
            # Write each zone
            for zone in self.zones:
                self._write_zone(handle, zone)
        finally:
            # Always close the file
            szlio.tec_file_writer_close(handle)

    def _write_zone(self, handle, zone: TecZone) -> None:
        """
        Write a single zone to the output file.

        Automatically determines variable types and value locations from data.

        Args:
            handle: TecIO file handle
            zone: TecZone to write
        """
        # Automatically determine variable types from actual data
        var_types = []
        value_locations = []

        for i, var in enumerate(self.variables):
            # Get value location from variable metadata
            value_locations.append(var.value_location)

            # Determine data type from actual data
            if zone.has_variable_data(i):
                data = zone.get_variable_data(i)
                var_types.append(self._infer_data_type(data))
            else:
                raise ValueError(
                    f"Zone '{zone.title}' missing data for variable '{var.name}'"
                )

        # Create zone (automatically handles ORDERED, FE types, etc.)
        zone_num = szlio.tec_zone_create_ijk(
            handle=handle,
            zone_title=zone.title,
            I=zone.dimensions[0],
            J=zone.dimensions[1],
            K=zone.dimensions[2],
            var_types=var_types,
            value_locations=value_locations,
        )

        # Set unsteady options if needed (automatically detect from zone metadata)
        if zone.strand_id > 0 or zone.solution_time != 0.0:
            szlio.tec_zone_set_unsteady_options(
                handle=handle,
                zone=zone_num,
                strand=zone.strand_id,
                solution_time=zone.solution_time,
            )

        # Write variable data (automatically uses correct write function)
        for i in range(self.num_vars):
            data = zone.get_variable_data(i)
            data_type = var_types[i]
            self._write_variable_data(handle, zone_num, i + 1, data, data_type)

    def _write_variable_data(
        self,
        handle,
        zone_num: int,
        var_num: int,
        data: npt.NDArray,
        data_type: DataType,
    ) -> None:
        """
        Write variable data to file using appropriate data type.

        Args:
            handle: TecIO file handle
            zone_num: Zone number (1-based)
            var_num: Variable number (1-based)
            data: NumPy array of data
            data_type: DataType enum
        """
        if data_type == DataType.DOUBLE:
            szlio.tec_zone_var_write_double_values(handle, zone_num, var_num, data)
        elif data_type == DataType.FLOAT:
            szlio.tec_zone_var_write_float_values(handle, zone_num, var_num, data)
        elif data_type == DataType.INT32:
            szlio.tec_zone_var_write_int32_values(handle, zone_num, var_num, data)
        elif data_type == DataType.INT16:
            szlio.tec_zone_var_write_int16_values(handle, zone_num, var_num, data)
        elif data_type == DataType.BYTE:
            szlio.tec_zone_var_write_uint8_values(handle, zone_num, var_num, data)
        else:
            raise ValueError(f"Unsupported data type: {data_type}")

    def add_variable(
        self,
        name: str,
        data_type: DataType = DataType.DOUBLE,
        location: ValueLocation = ValueLocation.NODE_CENTERED,
    ) -> int:
        """
        Add a new variable to the dataset.

        Args:
            name: Variable name
            data_type: Data type for storage
            location: Value location (NODE_CENTERED or CELL_CENTERED)

        Returns:
            Index of added variable (0-based)

        Example:
            >>> data = TecData()
            >>> x_idx = data.add_variable("X")
            >>> p_idx = data.add_variable("Pressure", DataType.FLOAT)
        """
        var = TecVariable(name=name, data_type=data_type, value_location=location)
        self.variables.append(var)
        self.num_vars = len(self.variables)
        return self.num_vars - 1

    def get_variable_index(self, name: str) -> int:
        """
        Get variable index by name.

        Args:
            name: Variable name

        Returns:
            Variable index (0-based)

        Raises:
            ValueError: If variable not found
        """
        for i, var in enumerate(self.variables):
            if var.name == name:
                return i
        raise ValueError(f"Variable '{name}' not found")

    def add_zone(
        self,
        title: str,
        zone_type: ZoneType,
        dimensions: tuple[int, int, int],
        solution_time: float = 0.0,
        strand_id: int = 0,
    ) -> TecZone:
        """
        Add a new zone to the dataset.

        Args:
            title: Zone title
            zone_type: ZoneType enum
            dimensions: (I, J, K) dimensions
            solution_time: Solution time for unsteady data
            strand_id: Strand ID for unsteady data

        Returns:
            Created TecZone object

        Example:
            >>> data = TecData()
            >>> data.add_variable("X")
            >>> data.add_variable("Y")
            >>> zone = data.add_zone("Grid", ZoneType.ORDERED, (10, 10, 1))
        """
        zone = TecZone(
            title=title,
            zone_type=zone_type,
            dimensions=dimensions,
            solution_time=solution_time,
            strand_id=strand_id,
        )
        self.zones.append(zone)
        self.num_zones = len(self.zones)
        return zone

    def __repr__(self) -> str:
        """String representation of dataset."""
        return (
            f"TecData(title='{self.title}', "
            f"num_vars={self.num_vars}, "
            f"num_zones={self.num_zones})"
        )

    def summary(self) -> str:
        """
        Generate detailed summary of dataset for debugging/inspection.

        Returns:
            Multi-line string with dataset information

        Example:
            >>> data = TecData("flow.szplt")
            >>> print(data.summary())
        """
        lines = []
        lines.append("=" * 70)
        lines.append(f"TecData Summary: {self.title}")
        lines.append("=" * 70)
        
        # File type
        file_type = self._infer_file_type()
        lines.append(f"File Type: {file_type.name}")
        
        # Variables
        lines.append(f"\nVariables ({self.num_vars}):")
        for i, var in enumerate(self.variables):
            lines.append(
                f"  [{i}] {var.name:20s} {var.data_type.name:8s} {var.value_location.name}"
            )
            if var.auxdata:
                for name, value in var.auxdata.items():
                    lines.append(f"      aux: {name} = {value}")
        
        # Zones
        lines.append(f"\nZones ({self.num_zones}):")
        for i, zone in enumerate(self.zones):
            lines.append(f"  [{i}] {zone.title}")
            lines.append(f"      Type: {zone.zone_type.name}")
            lines.append(f"      Dimensions: {zone.dimensions}")
            lines.append(f"      Points: {zone.num_points:,}")
            lines.append(f"      Elements: {zone.num_elements:,}")
            
            if zone.strand_id > 0 or zone.solution_time != 0.0:
                lines.append(f"      Strand ID: {zone.strand_id}")
                lines.append(f"      Solution Time: {zone.solution_time}")
            
            # Check data types
            lines.append(f"      Variable Data:")
            for j, var in enumerate(self.variables):
                if zone.has_variable_data(j):
                    data = zone.get_variable_data(j)
                    data_type = self._infer_data_type(data)
                    lines.append(
                        f"        {var.name:20s} {data_type.name:8s} "
                        f"({data.dtype}, {data.nbytes:,} bytes)"
                    )
                else:
                    lines.append(f"        {var.name:20s} NOT LOADED")
            
            if zone.auxdata:
                lines.append(f"      Zone Auxiliary Data:")
                for name, value in zone.auxdata.items():
                    lines.append(f"        {name}: {value}")
        
        # Auxiliary data
        if self.auxdata:
            lines.append(f"\nDataset Auxiliary Data:")
            for name, value in self.auxdata.items():
                lines.append(f"  {name}: {value}")
        
        lines.append("=" * 70)
        return "\n".join(lines)
