# Project README

This repository provides Python helpers for reading Tecplot SZL (.szplt) files using the TecIO C/C++ library shipped with Tecplot360EX.

Key modules
- szlio: thin Python wrappers around the TecIO C API.  See [`szlio.tec_file_reader_open`](szlio.py) and related wrappers such as [`szlio.tec_zone_var_get_float_values`](szlio.py). The module locates and loads the tecio shared library using [`tecutils.get_tecio_lib`](tecutils.py).
- szlfile: higher-level, Pythonic file API built on top of `szlio`. Use [`szlfile.SzlFile`](szlfile.py) to open a .szplt file and the [`szlfile.Zone`](szlfile.py) and [`szlfile.Variable`](szlfile.py) classes to inspect zones, variables and values.

Design notes
- szlio exposes direct, low-level wrappers for TecIO functions (I/O, zone/variable queries, node-maps, aux-data). It returns NumPy arrays for numeric data and Python strings for textual metadata.
- szlfile builds convenient abstractions (SzlFile, Zone, Variable, AuxData) that call into `szlio` and present a friendly API for common workflows.
- The project includes helper logic to find the TecIO shared library on macOS/Linux via [`tecutils.get_tecio_lib`](tecutils.py). If Tecplot360EX is not installed or the library cannot be found, that function raises a clear error.

Quick usage example
```py
from szlfile import SzlFile

szl = SzlFile("Onera.szplt")
print(szl.title)
for zi, zone in enumerate(szl.zones, start=1):
    print(zi, zone.title, zone.type)
    for var in zone.variables:
        print("  ", var.name, var.value_location)
    # read full values for variable 1:
    vals = zone.variables[0].values
    print("  values shape:", vals.shape)
```

Detailed module reference

szlio (low-level TecIO wrappers)
- Purpose: direct ctypes bindings to the TecIO C API shipped with Tecplot360EX.
- What it provides:
  - Library loading via tecutils.get_tecio_lib().
  - Enum mappings: FileType, ZoneType, DataType, ValueLocation.
  - Error type: SzlioError raised when an underlying tecio function returns non-zero.
  - File-level functions:
    - tec_file_reader_open(file_name) -> c_void_p
    - tec_file_get_type(handle) -> FileType
    - tec_data_set_get_title(handle) -> str
    - tec_data_set_get_num_vars(handle) -> int
    - tec_data_set_get_num_zones(handle) -> int
  - Zone-level functions:
    - tec_zone_get_ijk(handle, zone_index) -> (I, J, K)
    - tec_zone_get_title(handle, zone_index) -> str
    - tec_zone_get_type(handle, zone_index) -> ZoneType
    - tec_zone_is_enabled(handle, zone_index) -> bool
    - tec_zone_get_solution_time(handle, zone_index) -> float
    - tec_zone_get_strand_id(handle, zone_index) -> int
    - is_64bit(handle, zone_index) -> bool
    - tec_zone_node_map_get_64(...) -> numpy.ndarray[int64]
    - tec_zone_node_map_get(...) -> numpy.ndarray[int32]
  - Variable data functions (per-zone):
    - tec_var_get_name(...)
    - tec_var_is_enabled(...)
    - tec_zone_var_get_type(...)
    - tec_zone_var_get_value_location(...)
    - tec_zone_var_is_passive(...)
    - tec_zone_var_get_shared_zone(...)
    - tec_zone_var_get_num_values(...)
    - tec_zone_var_get_float_values(...) -> ndarray[float32]
    - tec_zone_var_get_double_values(...) -> ndarray[float64]
    - tec_zone_var_get_int32_values(...) -> ndarray[int32]
    - tec_zone_var_get_int16_values(...) -> ndarray[int16]
    - tec_zone_var_get_uint8_values(...) -> ndarray[uint8]
  - Aux-data:
    - tec_data_set_aux_data_get_num_items / tec_data_set_aux_data_get_item
    - tec_var_aux_data_get_num_items / tec_var_aux_data_get_item
    - tec_zone_aux_data_get_num_items / tec_zone_aux_data_get_item
- Notes: wrappers allocate ctypes buffers and convert to NumPy arrays with
  np.ctypeslib.as_array() to minimize copies. All wrappers raise SzlioError
  on non-zero tecio return codes.

szlfile (higher-level Python API)
- Purpose: provide a Pythonic object model (SzlFile, Zone, Variable, AuxData)
  built on top of szlio to make common tasks easy.
- Key classes:
  - SzlFile: opens file, exposes dataset metadata, list of Zone objects,
    and dataset/variable auxiliary data accessors.
  - Zone (dataclass): I,J,K, type, title, nodes/elements counts, node-map,
    solution time, strand id; creates Variable objects for per-variable access.
  - Variable: name, enabled, type, value_location, passivity, shared zone,
    number of values and methods to read values into numpy arrays.
  - AuxData: lazy-loaded dict-like wrapper exposing dataset/var/zone aux tokens
    and conversion helpers (as_int, as_float, as_bool).
- Indexing: SzlFile/Zone/Variable preserve TecIO 1-based indices for zone and
  variable arguments internally; Python lists exposed to callers use 0-based
  list positions (zones and variables are created in index order).
- Use the PDF refs/360-data-format.pdf for authoritative C API semantics.

Where to read more
- TecIO Data Format Guide (refs/360-data-format.pdf) — details each C/C++
  function, parameter expectations and return codes.
- See szlio.py docstring for a concise list of functions and behavior.
- Example usage: see quick usage section above.