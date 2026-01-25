from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import numpy as np
import numpy.typing as npt

from szlio import ValueLocation, ZoneType


class TecData:
    """
    Unified mutable Tecplot data structure.

    Can be:
    - Created empty and populated
    - Loaded from SZL files (binary)
    - Loaded from PLT files (binary) - future
    - Loaded from DAT files (ASCII) - future

    Can be written to:
    - SZL files (binary) - future
    - PLT files (binary) - future
    - DAT files (ASCII) - future
    """

    def __init__(self, title: str = ""):
        self.title = title
        self.variables: List[TecVariable] = []
        self.zones: List[TecZone] = []
        self.auxdata: Dict[str, str] = {}

    @classmethod
    def from_szl_file(
        cls,
        file_path: str,
        load_data: bool = True,
        zones: Optional[List[int]] = None,
        variables: Optional[List[int]] = None,
    ) -> TecData:
        """
        Create TecData from SZL file.

        Args:
            file_path: Path to .szplt file
            load_data: If True, load all variable data into memory.
                      If False, create structure but defer data loading.
            zones: List of zone indices to load (1-based). None = all zones.
            variables: List of variable indices to load (1-based). None = all variables.

        Returns:
            TecData object with data from file
        """
        from szlfile import SzlFile

        szl = SzlFile(file_path)
        tecdata = cls(title=szl.title)

        # Determine which zones and variables to load
        zone_indices = zones if zones is not None else list(range(1, szl.num_zones + 1))
        var_indices = (
            variables if variables is not None else list(range(1, szl.num_vars + 1))
        )

        # Load variable metadata
        for var_idx in var_indices:
            var_name = (
                szl.zones[0].variables[var_idx - 1].name
            )  # All zones share var names
            tecdata.add_variable(var_name)

        # Load dataset auxiliary data
        tecdata.auxdata = dict(szl.auxdata)

        # Load zones
        for zone_idx in zone_indices:
            szl_zone = szl.zones[zone_idx - 1]
            tec_zone = TecZone.from_szl_zone(szl_zone, var_indices, load_data)
            tecdata.zones.append(tec_zone)

        return tecdata

    def add_variable(
        self, name: str, location: ValueLocation = ValueLocation.NODAL
    ) -> int:
        """
        Add a new variable to the dataset.

        Args:
            name: Variable name
            location: Where values are stored (NODAL or CELL_CENTERED)

        Returns:
            Index of new variable (0-based)
        """
        var = TecVariable(name=name, location=location)
        self.variables.append(var)

        # Add placeholder to all existing zones
        for zone in self.zones:
            zone.add_variable_slot()

        return len(self.variables) - 1

    def get_variable_index(self, name: str) -> int:
        """Get variable index by name. Returns -1 if not found."""
        for i, var in enumerate(self.variables):
            if var.name == name:
                return i
        return -1

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
            zone_type: Type of zone (ORDERED, FETRIANGLE, etc.)
            dimensions: (I, J, K) for ordered or (num_points, num_elements, 0) for FE
            solution_time: Solution time for transient data
            strand_id: Strand ID for transient data

        Returns:
            The newly created zone
        """
        zone = TecZone(
            title=title,
            zone_type=zone_type,
            dimensions=dimensions,
            solution_time=solution_time,
            strand_id=strand_id,
            num_variables=len(self.variables),
        )
        self.zones.append(zone)
        return zone

    def normalize_variable(self, var_index: int, reference_value: float) -> None:
        """
        Normalize a variable by a reference value in all zones.

        Args:
            var_index: Variable index (0-based)
            reference_value: Value to divide by
        """
        for zone in self.zones:
            values = zone.get_variable_data(var_index)
            if values is not None:
                zone.set_variable_data(var_index, values / reference_value)

    def compute_magnitude(
        self, component_indices: tuple[int, int, int], result_name: str = "Magnitude"
    ) -> int:
        """
        Compute magnitude from vector components.

        Args:
            component_indices: (x_idx, y_idx, z_idx) - 0-based indices
            result_name: Name for the new magnitude variable

        Returns:
            Index of new magnitude variable
        """
        mag_idx = self.add_variable(result_name)

        for zone in self.zones:
            x = zone.get_variable_data(component_indices[0])
            y = zone.get_variable_data(component_indices[1])
            z = zone.get_variable_data(component_indices[2])

            if x is not None and y is not None and z is not None:
                magnitude = np.sqrt(x**2 + y**2 + z**2)
                zone.set_variable_data(mag_idx, magnitude)

        return mag_idx

    def __repr__(self) -> str:
        return (
            f"TecData(title='{self.title}', "
            f"num_vars={len(self.variables)}, "
            f"num_zones={len(self.zones)})"
        )


@dataclass
class TecVariable:
    """Metadata for a variable in the dataset."""

    name: str
    location: ValueLocation = ValueLocation.NODAL
    auxdata: Dict[str, str] = field(default_factory=dict)


@dataclass
class TecZone:
    """
    Mutable zone that stores all data in memory.
    """

    title: str
    zone_type: ZoneType
    dimensions: tuple[int, int, int]  # (I, J, K) or (num_points, num_elements, 0)
    solution_time: float = 0.0
    strand_id: int = 0
    num_variables: int = 0
    auxdata: Dict[str, str] = field(default_factory=dict)

    # Data storage
    _variable_data: List[Optional[npt.NDArray]] = field(default_factory=list)
    _node_map: Optional[npt.NDArray[np.int64]] = None

    def __post_init__(self):
        """Initialize variable data slots."""
        if len(self._variable_data) == 0:
            self._variable_data = [None] * self.num_variables

    @classmethod
    def from_szl_zone(
        cls, szl_zone, var_indices: List[int], load_data: bool = True
    ) -> TecZone:
        """
        Create TecZone from SzlFile Zone.

        Args:
            szl_zone: Zone object from SzlFile
            var_indices: List of variable indices to load (1-based)
            load_data: If True, load variable data. If False, defer loading.

        Returns:
            TecZone with data loaded or ready to load
        """
        zone = cls(
            title=szl_zone.title,
            zone_type=szl_zone.type,
            dimensions=szl_zone.dimensions,
            solution_time=szl_zone.solution_time,
            strand_id=szl_zone.strand_id,
            num_variables=len(var_indices),
            auxdata=dict(szl_zone.auxdata),
        )

        # Load variable data if requested
        if load_data:
            for i, var_idx in enumerate(var_indices):
                var = szl_zone.variables[var_idx - 1]
                zone._variable_data[i] = var.values.copy()

        # Load node map for FE zones
        if szl_zone.type != ZoneType.ORDERED:
            zone._node_map = szl_zone.node_map.copy()

        return zone

    @property
    def num_points(self) -> int:
        """Number of points/nodes in zone."""
        if self.zone_type == ZoneType.ORDERED:
            return self.dimensions[0] * self.dimensions[1] * self.dimensions[2]
        else:
            return self.dimensions[0]

    @property
    def num_elements(self) -> int:
        """Number of elements/cells in zone."""
        if self.zone_type == ZoneType.ORDERED:
            i_max = max(1, self.dimensions[0] - 1)
            j_max = max(1, self.dimensions[1] - 1)
            k_max = max(1, self.dimensions[2] - 1)
            return i_max * j_max * k_max
        else:
            return self.dimensions[1]

    def add_variable_slot(self) -> None:
        """Add a slot for a new variable (initialized to None)."""
        self._variable_data.append(None)
        self.num_variables += 1

    def get_variable_data(self, var_index: int) -> Optional[npt.NDArray]:
        """
        Get variable data array.

        Args:
            var_index: Variable index (0-based)

        Returns:
            NumPy array of values, or None if not loaded
        """
        if var_index < 0 or var_index >= self.num_variables:
            raise IndexError(
                f"Variable index {var_index} out of range [0, {self.num_variables})"
            )
        return self._variable_data[var_index]

    def set_variable_data(self, var_index: int, values: npt.NDArray) -> None:
        """
        Set variable data array.

        Args:
            var_index: Variable index (0-based)
            values: NumPy array of values
        """
        if var_index < 0 or var_index >= self.num_variables:
            raise IndexError(
                f"Variable index {var_index} out of range [0, {self.num_variables})"
            )

        expected_size = self.num_points  # Could be num_elements for cell-centered
        if len(values) != expected_size:
            raise ValueError(
                f"Array size {len(values)} doesn't match expected {expected_size}"
            )

        self._variable_data[var_index] = values.copy()

    @property
    def node_map(self) -> Optional[npt.NDArray[np.int64]]:
        """Get node connectivity map for FE zones."""
        return self._node_map

    @node_map.setter
    def node_map(self, connectivity: npt.NDArray[np.int64]) -> None:
        """Set node connectivity map for FE zones."""
        if self.zone_type == ZoneType.ORDERED:
            raise ValueError("Ordered zones do not use node maps")

        expected_shape = (self.num_elements, self._nodes_per_element())
        if connectivity.shape != expected_shape:
            raise ValueError(
                f"Node map shape {connectivity.shape} doesn't match "
                f"expected {expected_shape}"
            )

        self._node_map = connectivity.copy()

    def _nodes_per_element(self) -> int:
        """Get number of nodes per element based on zone type."""
        if self.zone_type == ZoneType.FELINESEG:
            return 2
        elif self.zone_type == ZoneType.FETRIANGLE:
            return 3
        elif self.zone_type == ZoneType.FEQUADRILATERAL:
            return 4
        elif self.zone_type == ZoneType.FETETRAHEDRON:
            return 4
        elif self.zone_type == ZoneType.FEBRICK:
            return 8
        else:
            raise ValueError(f"Cannot determine nodes per element for {self.zone_type}")

    def __repr__(self) -> str:
        return (
            f"TecZone(title='{self.title}', type={self.zone_type}, "
            f"dims={self.dimensions}, vars={self.num_variables})"
        )
