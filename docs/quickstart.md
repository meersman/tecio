(quickstart)=
# Quickstart

This page introduces the `tecio` API with four self-contained examples.  For
installation and system requirements see {ref}`Installation <installation>`.

---

(qs-examples)=
## Examples

## 1. Structured IJK Zone (1-D Line)

The simplest possible file: a single ordered zone containing two variables
plotted against each other as a line.  Dimensions are inferred from the array
shape — a 1-D array of length *N* produces an ``(N, 1, 1)`` zone.

```python
import numpy as np
import tecio

x = np.linspace(0.0, 2.0 * np.pi, 256)
y = np.sin(x)

with tecio.open("line.szplt", "w", title="Sine Curve") as w:
    w.write_ijk_zone(
        title="sin(x)",
        variables=["x", "y"],
        data=[x, y],
    )
```

Reading the file back follows the same pattern:

```python
with tecio.open("line.szplt", "r") as r:
    print(r.title)       # 'Sine Curve'
    print(r.variables)   # ['x', 'y']

    zone = r.zone[0]
    x_read = zone.variable[0].values   # NumPy array, shape (256, 1, 1)
    y_read = zone.variable[1].values
```

:::{note}
**Python objects are 0-indexed.** The `zone` list and `variable` list on a
reader both use standard Python (zero-based) indexing: `r.zone[0]` is the
first zone, `zone.variable[1]` is the second variable.

**TecIO inputs and outputs are 1-indexed.** Whenever a function in
`tecio.libtecio` accepts a zone or variable index as an integer argument — for
example `tec_zone_var_write_float_values(handle, zone=1, var=2, ...)` — those
indices follow Tecplot's Fortran-style one-based convention.  The high-level
`tecio.open` API handles this translation automatically.
:::

---

## 2. Unstructured Finite-Element Zones

All five simple finite-element cell types supported by Tecplot are shown below
in a single file, one zone per element type.  The `node_map` argument is a
``(num_cells, nodes_per_cell)`` integer array of **1-based** node indices.

```python
import numpy as np
import tecio
from tecio.libtecio import ZoneType

# -- FELINESEG ---------------------------------------------------------------
# 4-node polyline: (0,0) → (1,0) → (2,1) → (3,0)
x_ls = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
y_ls = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
nm_ls = np.array([[1, 2], [2, 3], [3, 4]], dtype=np.int32)  # 3 segments

# --- FETRIANGLE -------------------------------------------------------------
x_tri = np.array([0.0, 1.0, 0.5], dtype=np.float32)
y_tri = np.array([0.0, 0.0, 1.0], dtype=np.float32)
nm_tri = np.array([[1, 2, 3]], dtype=np.int32)              # 1 triangle

# --- FEQUADRILATERAL --------------------------------------------------------
x_q = np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float32)
y_q = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
nm_q = np.array([[1, 2, 3, 4]], dtype=np.int32)             # 1 quad

# --- FETETRAHEDRON ----------------------------------------------------------
x_tet = np.array([0.0, 1.0, 0.5, 0.5], dtype=np.float32)
y_tet = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
z_tet = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
nm_tet = np.array([[1, 2, 3, 4]], dtype=np.int32)           # 1 tet

# --- FEBRICK ----------------------------------------------------------------
x_b = np.array([0,1,1,0,0,1,1,0], dtype=np.float32)
y_b = np.array([0,0,1,1,0,0,1,1], dtype=np.float32)
z_b = np.array([0,0,0,0,1,1,1,1], dtype=np.float32)
nm_b = np.array([[1,2,3,4,5,6,7,8]], dtype=np.int32)        # 1 hex brick


with tecio.open("fe_zones.szplt", "w", title="FE Zone Types") as w:

    w.write_fe_zone(
        zone_type=ZoneType.FELINESEG,
        title="LineSeg",
        variables=["x", "y"],
        data=[x_ls, y_ls],
        node_map=nm_ls,
    )
    w.write_fe_zone(
        zone_type=ZoneType.FETRIANGLE,
        title="Triangle",
        data=[x_tri, y_tri],
        node_map=nm_tri,
    )
    w.write_fe_zone(
        zone_type=ZoneType.FEQUADRILATERAL,
        title="Quad",
        data=[x_q, y_q],
        node_map=nm_q,
    )
    w.write_fe_zone(
        zone_type=ZoneType.FETETRAHEDRON,
        title="Tet",
        variables=["x", "y", "z"],
        data=[x_tet, y_tet, z_tet],
        node_map=nm_tet,
    )
    w.write_fe_zone(
        zone_type=ZoneType.FEBRICK,
        title="Brick",
        data=[x_b, y_b, z_b],
        node_map=nm_b,
    )
```

:::{note}
After the first `write_fe_zone` call the variable list is locked in. Subsequent
zones must supply the same set of variables (or a subset via `passive_vars`).
The `variables` keyword only needs to be passed once — on the first write call,
or at `tecio.open` time.
:::

---

## 3. Time-Dependent 2-D Field

Transient datasets are written by assigning each zone a `strand_id` and
`solution_time`.  Zones on the same strand animate together in Tecplot 360.
Grid coordinates that do not change between time steps can be **shared** from
the first zone to avoid writing duplicate arrays, which substantially reduces
file size for large grids.

```python
import numpy as np
import tecio

# 2-D grid
nx, ny = 128, 128
x = np.linspace(0.0, 1.0, nx)
y = np.linspace(0.0, 1.0, ny)
X, Y = np.meshgrid(x, y, indexing="ij")   # shape (nx, ny)

times = np.linspace(0.0, 4.0 * np.pi, 60)

with tecio.open("transient.szplt", "w", title="Travelling Wave") as w:

    # Set dataset auxiliary data before the first zone
    w.add_auxdataset_dict({
        "Common.XVar": 1,
        "Common.YVar": 2,
        "Common.CVar": 3,
    })

    for i, t in enumerate(times):
        phi = np.sin(2.0 * np.pi * X - t) * np.cos(2.0 * np.pi * Y)

        if i == 0:
            # First zone: write grid coordinates and solution
            w.write_ijk_zone(
                title=f"t = {t:.3f}",
                variables=["x", "y", "phi"],
                data=[X, Y, phi],
                strand_id=1,
                solution_time=t,
            )
        else:
            # Subsequent zones: share x and y from zone 1, write phi only
            w.write_ijk_zone(
                title=f"t = {t:.3f}",
                data=[phi],
                var_sharing=[1, 1, 0],   # x←zone1, y←zone1, phi=new
                strand_id=1,
                solution_time=t,
            )
```

Reading a transient file works identically to a steady file — the zones are
listed in the order they were written:

```python
with tecio.open("transient.szplt", "r") as r:
    print(r.num_zones)           # 60

    # Solution times across all zones
    times_read = [r.zone[i].solution_time for i in range(r.num_zones)]

    # Grid is only stored in zone 0; later zones return None for shared vars
    phi_t0 = r.zone[0].variable[2].values    # shape (128, 128, 1)
    phi_t1 = r.zone[1].variable[2].values    # shape (128, 128, 1)
    x_shared = r.zone[1].variable[0].values  # None — shared from zone 0
```

---

## 4. Low-Level `libtecio` API

The `tecio.libtecio` module exposes the TecIO C functions directly.  Using it
gives full control over data types, zone creation options, and the write
ordering required by the SZL API.  This example reproduces the 1-D sine curve
from Example 1 using the low-level SZL write functions.

```python
import numpy as np
from tecio import libtecio
from tecio.libtecio import DataType, FileType, ValueLocation, ZoneType

x = np.linspace(0.0, 2.0 * np.pi, 256, dtype=np.float32)
y = np.sin(x)

# 1. Open a writer handle (returns an opaque C pointer)
handle = libtecio.tec_file_writer_open(
    filename="line_lowlevel.szplt",
    variables=["x", "y"],
    title="Sine Curve (low-level)",
    file_type=FileType.FULL,
)

# 2. Create a zone and get its 1-based index
izone = libtecio.tec_zone_create_ijk(
    handle,
    zone_title="sin(x)",
    imax=256,
    jmax=1,
    kmax=1,
    var_types=[DataType.FLOAT, DataType.FLOAT],
    value_locations=[ValueLocation.NODAL, ValueLocation.NODAL],
)

# 3. Write each variable by 1-based index (zone=1, var=1 / var=2)
libtecio.tec_zone_var_write_float_values(handle, izone, 1, x)
libtecio.tec_zone_var_write_float_values(handle, izone, 2, y.astype(np.float32))

# 4. Close and flush
libtecio.tec_file_writer_close(handle)
```

:::{note}
**SZL vs. classic API.** The SZL functions (`tec_file_writer_open`,
`tec_zone_create_ijk`, `tec_zone_var_write_*`) return an explicit file handle
and allow variables to be written in any order after zone creation.  Multiple
files can be open simultaneously.

The classic PLT functions (`tecini142`, `teczne142`, `tecdat142`, `tecend142`)
maintain a single implicit global context: only one file is active at a time
and data must be written in strict zone → variable order.  Use the SZL API for
new code unless PLT format is specifically required.
:::

---

(qs-index-convention)=
## Indexing Conventions

| Context | Indexing | Example |
|---------|----------|---------|
| Python reader objects (`zone`, `variable`) | **0-based** | `r.zone[0]`, `zone.variable[2]` |
| `libtecio` function arguments (zone, var) | **1-based** | `tec_zone_var_write_float_values(h, 1, 3, arr)` |
| `var_sharing` list entries | **1-based** zone number | `var_sharing=[1, 1, 0]` → share from zone 1 |
| `node_map` connectivity arrays | **1-based** node indices | `np.array([[1, 2, 3]])` |

The high-level `tecio.open` API handles the translation between Python
zero-based indexing and TecIO one-based indexing automatically.  You only need
to think about one-based indices when calling `libtecio` functions directly or
when constructing `var_sharing` / `node_map` arrays.

---

(qs-getting-help)=
## Getting Help

All public classes and functions have docstrings accessible via Python's
built-in `help()` function:

```python
import tecio

help(tecio.open)               # top-level open function
help(tecio.szl.Write)          # SZL writer class and all its methods
help(tecio.libtecio.ZoneType)  # enum values and descriptions
```

Full API documentation is available in the {doc}`API Reference </api/index>`,
and runnable demos are provided in the `demos/` directory of the repository.
