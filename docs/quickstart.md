# Quickstart

(installation)=
## Installation

**Requirements:** Python 3.10+, NumPy, Tecplot 360

Install from the repository root:

```bash
pip install .
```

On import, `tecio` will attempt to locate the TecIO shared library
from your Tecplot 360 installation. However, if errors are occuring at
this step, or multiple versions of Tec360 are installed and youi wish
so specify a version, an environment variable, `TECIO_LIB` may be set
to override the search paths. See below for more info.

### Configuration

### `TECIO_LIB` environment variable

By default `tecio` searches for the TecIO shared library by locating the
Tecplot executable on `PATH` and resolving the library path relative to it.
Set `TECIO_LIB` to override this — useful when multiple Tecplot versions are
installed and you want to pin a specific one, or if automatic detection fails.

```bash
# Linux
export TECIO_LIB=/opt/tecplot/360ex_2025r1/bin/libtecio.so

# macOS
export TECIO_LIB=/Applications/Tecplot\ 360\ EX\ 2025\ R1.app/Contents/Frameworks/libtecio.dylib

# Windows (PowerShell)
$env:TECIO_LIB = "C:\Program Files\Tecplot\Tec360EX 2025 R1\bin\tecio.dll"
```

You can make this permanent by adding the export to your shell profile
(`~/.bashrc`, `~/.zshrc`, etc.) or to your project's `.env` file.

To verify which library `tecio` has resolved at runtime:

```python
from tecio import utils
print(utils.get_tecio_lib())
```

---

(example-usage)=
## Example Usage

### Reading a file

`tecio.open` returns a reader whose format is determined by the file
extension. Zone metadata is available immediately; variable data is read
lazily from disk on first access.

```python
import tecio

with tecio.open("flow.szplt", "r") as r:
    print(r.title)       # dataset title
    print(r.variables)   # ['x', 'y', 'z', 'pressure', 'velocity']
    print(r.num_zones)

    zone = r.zone[0]
    print(zone.title)
    print(zone.zone_type)
    print(zone.solution_time)

    # Variable data — loaded from disk here
    x = zone.variable[0].values   # NumPy array, shape (I, J, K)
    p = zone.variable[3].values

    # Dataset-level and zone-level auxiliary data
    print(r.auxdata)
    print(zone.auxdata)
```

The same interface works for `.plt` and `.dat` files — only the file
extension needs to change.

---

### Writing a file

Pass variable names and data arrays to `write_ijk_zone`. Dimensions are
inferred from the array shape; 1-D, 2-D, and 3-D arrays are all accepted.

```python
import numpy as np
import tecio

x = np.linspace(0, 1, 64)
y = np.linspace(0, 1, 32)
X, Y = np.meshgrid(x, y, indexing="ij")
P = np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)

with tecio.open("output.szplt", "w", title="My Dataset") as w:
    w.write_ijk_zone(
        data=[X, Y, P],
        variables=["x", "y", "pressure"],
        title="Zone 1",
    )
```

For time-dependent data, assign a `strand_id` and `solution_time` to each
zone. Share coordinate arrays across time steps with `var_sharing` to avoid
writing redundant grid data:

```python
times = np.linspace(0.0, 1.0, 50)

with tecio.open("transient.szplt", "w", title="Transient Flow") as w:
    for i, t in enumerate(times):
        P = compute_pressure(X, Y, t)
        w.write_ijk_zone(
            variables=["x", "y", "pressure"],
            data=[X, Y, P] if w.current_zone == 0 else [P],
            var_sharing=None if w.current_zone == 0 else [1, 1, 0],
            strand_id=1,
            solution_time=t,
        )
```

Unstructured finite-element zones use `write_fe_zone` and require a
`node_map` connectivity array:

```python
with tecio.open("mesh.szplt", "w") as w:
    w.write_fe_zone(
        zone_type=tecio.libtecio.ZoneType.FETRIANGLE,
        data=[nodes_x, nodes_y, pressure],
        variables=["x", "y", "pressure"],
        node_map=triangles,   # shape (num_cells, 3), 1-based
    )
```

Dataset-level auxiliary data (used by Tecplot to auto-configure axis
variables, velocity vectors, etc.) can be written before the first zone:

```python
with tecio.open("flow.szplt", "w") as w:
    w.add_auxdataset_dict({
        "Common.XVar": 1,
        "Common.YVar": 2,
        "Common.UVar": 3,
        "Common.VVar": 4,
    })
    w.write_ijk_zone(...)
```

---

### Appending to an existing file

`mode='a'` streams the existing file into a temporary copy, then leaves the
write handle open for new zones. On close the temporary file atomically
replaces the original.

```python
with tecio.open("flow.szplt", "a") as w:
    print(w.variables)      # variable list from the existing file
    print(w.current_zone)   # number of zones already copied

    w.write_ijk_zone(
        data=[x_new, y_new, p_new],
        solution_time=10.0,
        strand_id=1,
    )
```

`mode='a+'` additionally exposes the full read interface populated from the
original file, useful when new zones depend on existing data:

```python
with tecio.open("flow.szplt", "a+") as rw:
    # Read from the original zones
    p_avg = sum(
        rw.zone[i].variable["pressure"].values
        for i in range(rw.num_zones)
    ) / rw.num_zones

    # Append a new zone using the computed average
    x = rw.zone[0].variable[0].values
    y = rw.zone[0].variable[1].values
    rw.write_ijk_zone(
        data=[x, y, p_avg],
        title="Time-average",
        solution_time=rw.zone[-1].solution_time + 1.0,
        strand_id=2,
    )
```

---

## Command-Line Tools

After installation the following scripts are available directly from the shell.
See [Console Scripts](api/cli.md) for full documentation of each tool.

```bash
# Print all metadata and variable arrays for a file
tecdump flow.szplt

# Print per-variable statistics (min, max, mean, std)
tecstats flow.szplt

# Convert between formats
teconvert -szplt flow.plt
teconvert -dat flow.szplt

# Set NaN / Inf variable arrays to passive
tecfix blown_up.szplt

# Merge multiple files into one
tecmerge -o combined.szplt step_001.szplt step_002.szplt step_003.szplt

# Extract zones 1 and 3, variables 1–3
tecextract -zones 1,3 -variables 1,2,3 flow.szplt

# Thin a structured grid (every other point in I and J)
tecslice -i ::2 -j ::2 -o thinned.szplt flow.szplt

# Scale pressure variable from Pa to kPa
tecscale -variable pressure -scale 1e-3 flow.szplt
```
