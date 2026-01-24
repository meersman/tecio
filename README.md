# Tecio

A Python library for reading, manipulating, and writing Tecplot data files.

## Overview

Tecio provides a comprehensive interface for working with Tecplot files in Python. It supports:

- **Reading** from Tecplot SZL (SZPLT) binary files
- **Manipulating** data in memory (normalize, compute derived variables, combine datasets)
- **Creating** datasets from scratch
- **(Planned)** Reading/writing PLT binary and DAT ASCII formats

The library is built around two main concepts:
- **Read-only file interfaces** (`SzlFile`, `Zone`, `Variable`) for efficient file access
- **Mutable in-memory data structures** (`TecData`, `TecZone`, `TecVariable`) for data manipulation

## Installation

### Requirements

- Python 3.10+
- NumPy
- Tecplot 360 installation (provides the required `libtecio` shared library)

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd pytecplot

# Install dependencies
pip install numpy

# Ensure Tecplot 360 is installed and accessible in your PATH
```

The library automatically locates your Tecplot installation using `tecutils.py`.

## Quick Start

### Reading a File

```python
from szlfile import SzlFile

# Open an SZL file (read-only)
szl = SzlFile("flow.szplt")

print(f"Title: {szl.title}")
print(f"Variables: {szl.num_vars}")
print(f"Zones: {szl.num_zones}")

# Access zone data
zone = szl.zones[0]
print(f"Zone title: {zone.title}")
print(f"Zone type: {zone.type}")
print(f"Dimensions: {zone.dimensions}")

# Access variable data
pressure = zone.variables[0].values  # NumPy array
print(f"Pressure range: [{pressure.min()}, {pressure.max()}]")
```

### Loading Data for Manipulation

```python
from tecdata import TecData

# Load entire file into memory
data = TecData.from_szl_file("flow.szplt", load_data=True)

# Load only specific zones and variables (memory efficient)
data = TecData.from_szl_file(
    "flow.szplt",
    load_data=True,
    zones=[1, 2, 3],      # Load only zones 1-3
    variables=[1, 2, 5]   # Load only variables 1, 2, 5
)
```

### Normalizing Variables

```python
from tecdata import TecData

data = TecData.from_szl_file("flow.szplt", load_data=True)

# Find pressure variable
p_idx = data.get_variable_index("Pressure")

# Normalize by reference pressure
p_ref = 101325.0  # Pa
data.normalize_variable(p_idx, p_ref)

# Update variable name
data.variables[p_idx].name = "Pressure_Nondimensional"
```

### Computing Derived Variables

```python
# Compute velocity magnitude
u_idx = data.get_variable_index("U-Velocity")
v_idx = data.get_variable_index("V-Velocity")
w_idx = data.get_variable_index("W-Velocity")

vmag_idx = data.compute_magnitude(
    (u_idx, v_idx, w_idx),
    result_name="Velocity_Magnitude"
)

# Compute Mach number (custom calculation)
import numpy as np

mach_idx = data.add_variable("Mach")
gamma = 1.4
R = 287.05  # J/(kg·K) for air

for zone in data.zones:
    u = zone.get_variable_data(u_idx)
    v = zone.get_variable_data(v_idx)
    w = zone.get_variable_data(w_idx)
    T = zone.get_variable_data(data.get_variable_index("Temperature"))
    
    velocity_mag = np.sqrt(u**2 + v**2 + w**2)
    speed_of_sound = np.sqrt(gamma * R * T)
    mach = velocity_mag / speed_of_sound
    
    zone.set_variable_data(mach_idx, mach)
```

### Combining Multiple Files

```python
from tecdata import TecData

# Create combined dataset
combined = TecData(title="Combined Analysis")

# Load and merge multiple files
for filepath in ["case1.szplt", "case2.szplt", "case3.szplt"]:
    data = TecData.from_szl_file(filepath, load_data=True)
    
    # First file: copy variable definitions
    if len(combined.variables) == 0:
        combined.variables = data.variables.copy()
    
    # Add zones from this file
    combined.zones.extend(data.zones)

print(f"Combined {len(combined.zones)} zones from 3 files")
```

### Creating Data from Scratch

```python
from tecdata import TecData
from szlio import ZoneType
import numpy as np

# Create empty dataset
data = TecData(title="Synthetic Data")

# Add variables
x_idx = data.add_variable("X")
y_idx = data.add_variable("Y")
p_idx = data.add_variable("Pressure")

# Create a structured grid zone
zone = data.add_zone(
    title="Grid",
    zone_type=ZoneType.ORDERED,
    dimensions=(100, 100, 1)  # 100x100x1 grid
)

# Generate synthetic data
nx, ny = 100, 100
x = np.linspace(0, 1, nx)
y = np.linspace(0, 1, ny)
X, Y = np.meshgrid(x, y, indexing='ij')

# Set coordinate data
zone.set_variable_data(x_idx, X.ravel())
zone.set_variable_data(y_idx, Y.ravel())

# Set pressure field
P = np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
zone.set_variable_data(p_idx, P.ravel())
```

### Working with Auxiliary Data

```python
from szlfile import SzlFile

szl = SzlFile("flow.szplt")

# Dataset-level auxiliary data
if len(szl.auxdata) > 0:
    print("Dataset auxiliary data:")
    for name, value in szl.auxdata.items():
        print(f"  {name}: {value}")
    
    # Type conversion helpers
    time = szl.auxdata.as_float("TimeValue", default=0.0)
    iteration = szl.auxdata.as_int("Iteration", default=0)

# Zone-level auxiliary data
zone = szl.zones[0]
if len(zone.auxdata) > 0:
    print(f"Zone '{zone.title}' auxiliary data:")
    for name, value in zone.auxdata.items():
        print(f"  {name}: {value}")

# Variable-level auxiliary data
var_aux = szl.get_var_auxdata(1)  # 1-based index
if len(var_aux) > 0:
    print("Variable auxiliary data:")
    for name, value in var_aux.items():
        print(f"  {name}: {value}")
```

## Architecture

### File Reading (Read-Only)

- **`szlfile.SzlFile`**: Main interface for reading SZL files
- **`szlfile.Zone`**: Read-only zone interface
- **`szlfile.Variable`**: Read-only variable interface
- **`szlio`**: Low-level C library bindings for Tecplot I/O
- **`auxdata.AuxData`**: Dictionary-like interface for auxiliary data

### Data Manipulation (Mutable)

- **`tecdata.TecData`**: Mutable in-memory dataset container
- **`tecdata.TecZone`**: Mutable zone with data storage
- **`tecdata.TecVariable`**: Variable metadata

### Utilities

- **`tecutils`**: Functions to locate Tecplot installation and libraries

## Module Reference

### `szlfile` - SZL File Reading

```python
class SzlFile:
    # Properties
    type: FileType              # FULL, GRID, or SOLUTION
    title: str                  # Dataset title
    num_vars: int              # Number of variables
    num_zones: int             # Number of zones
    zones: List[Zone]          # List of zones
    auxdata: AuxData           # Dataset auxiliary data
    
    # Methods
    get_var_auxdata(var_index: int) -> AuxData

class Zone:
    # Properties
    title: str                 # Zone title
    type: ZoneType            # Zone type (ORDERED, FETRIANGLE, etc.)
    dimensions: Tuple[int, int, int]  # (I, J, K) dimensions
    num_points: int           # Number of points
    num_elements: int         # Number of elements
    solution_time: float      # Solution time
    strand_id: int            # Strand ID for transient data
    variables: List[Variable] # List of variables
    node_map: NDArray         # Node connectivity (FE zones only)
    auxdata: AuxData          # Zone auxiliary data

class Variable:
    # Properties
    name: str                 # Variable name
    type: DataType           # FLOAT, DOUBLE, INT32, INT16, BYTE
    value_location: ValueLocation  # NODAL or CELL_CENTERED
    num_values: int          # Number of values
    values: NDArray          # All values (reads from file)
    
    # Methods
    get_values(value_range: Tuple[int, int]) -> NDArray
    is_enabled() -> bool
    is_passive() -> bool
```

### `tecdata` - Mutable Data Structures

```python
class TecData:
    # Properties
    title: str
    variables: List[TecVariable]
    zones: List[TecZone]
    auxdata: Dict[str, str]
    
    # Class Methods
    @classmethod
    from_szl_file(file_path, load_data=True, zones=None, variables=None) -> TecData
    
    # Methods
    add_variable(name: str, location: ValueLocation = NODAL) -> int
    get_variable_index(name: str) -> int
    add_zone(title, zone_type, dimensions, ...) -> TecZone
    normalize_variable(var_index: int, reference_value: float)
    compute_magnitude(component_indices: Tuple[int, int, int], result_name: str) -> int

class TecZone:
    # Properties
    title: str
    zone_type: ZoneType
    dimensions: Tuple[int, int, int]
    num_points: int
    num_elements: int
    solution_time: float
    strand_id: int
    auxdata: Dict[str, str]
    node_map: NDArray
    
    # Methods
    get_variable_data(var_index: int) -> NDArray
    set_variable_data(var_index: int, values: NDArray)
    add_variable_slot()
```

### `auxdata` - Auxiliary Data Interface

```python
class AuxData:
    # Dictionary-like interface
    __len__() -> int
    __getitem__(key: str) -> str
    __contains__(key: str) -> bool
    keys() -> Iterator[str]
    values() -> Iterator[str]
    items() -> Iterator[Tuple[str, str]]
    get(key: str, default=None) -> str
    
    # Type conversion helpers
    as_int(key: str, default=None) -> Optional[int]
    as_float(key: str, default=None) -> Optional[float]
    as_bool(key: str, default=None) -> Optional[bool]
```

## Enumerations

### `FileType`
- `FULL` - Full dataset with grid and solution
- `GRID` - Grid only
- `SOLUTION` - Solution only

### `ZoneType`
- `ORDERED` - Structured grid
- `FELINESEG` - Line segments
- `FETRIANGLE` - Triangles
- `FEQUADRILATERAL` - Quadrilaterals
- `FETETRAHEDRON` - Tetrahedra
- `FEBRICK` - Hexahedra (bricks)
- `FEPOLYGON` - Polygons
- `FEPOLYHEDRON` - Polyhedra

### `DataType`
- `FLOAT` - 32-bit float
- `DOUBLE` - 64-bit double
- `INT32` - 32-bit integer
- `INT16` - 16-bit integer
- `BYTE` - 8-bit unsigned integer

### `ValueLocation`
- `NODAL` - Values at nodes/points
- `CELL_CENTERED` - Values at cell centers

## Performance Considerations

### Memory Management

- **SzlFile** reads data from disk on each access - suitable for inspection
- **TecData** loads data into memory - suitable for manipulation
- Use partial loading for large files:
  ```python
  data = TecData.from_szl_file("large.szplt", zones=[1], variables=[1,2])
  ```

### Data Access Patterns

```python
# ❌ Inefficient - reads from file multiple times
for i in range(10):
    mean = szl.zones[0].variables[0].values.mean()

# ✅ Efficient - read once, cache in Python variable
values = szl.zones[0].variables[0].values
for i in range(10):
    mean = values.mean()

# ✅ Best - load into TecData for heavy manipulation
data = TecData.from_szl_file("file.szplt")
```

## Error Handling

All low-level C library errors are wrapped in `SzlioError` exceptions:

```python
from szlio import SzlioError

try:
    szl = SzlFile("nonexistent.szplt")
except SzlioError as e:
    print(f"Failed to open file: {e}")
```

## Development

### Code Style

The project follows PEP 8 with the following tools:
- **Black** - Code formatting
- **isort** - Import sorting
- **Pylint** - Linting
- **Pyflakes** - Error checking

Format code:
```bash
black .
isort .
```

Run linters:
```bash
pylint *.py
pyflakes *.py
```

### Type Hints

All code uses Python 3.10+ type hints for better IDE support and documentation.

## Roadmap

### Implemented
- ✅ Read SZL binary files
- ✅ Auxiliary data support (dataset, variable, zone levels)
- ✅ Mutable in-memory data structure
- ✅ Basic data manipulation (normalize, compute magnitude)
- ✅ Partial loading for memory efficiency
- ✅ Create datasets from scratch

### Planned
- ⬜ Write SZL binary files
- ⬜ Read PLT binary files
- ⬜ Write PLT binary files
- ⬜ Read ASCII DAT files
- ⬜ Write ASCII DAT files
- ⬜ Additional computation methods (Mach, recovery temperature, etc.)
- ⬜ Zone slicing and merging utilities
- ⬜ Parallel data loading for large files

## License

[Specify license here]

## Contributing

[Contribution guidelines here]

## Acknowledgments

This library uses the Tecplot TecIO library for reading binary SZL files. Tecplot 360 must be installed for the library to function.
