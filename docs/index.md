# tecio

Python interface for reading and writing Tecplot data files.

```{toctree}
:maxdepth: 2
:caption: User Guide

quickstart
```

```{toctree}
:maxdepth: 2
:caption: Demos

demos/lorenz/lorenz
demos/gravity_waves/gravity_waves
demos/simple_spectral_solver/simple_spectral
```

```{toctree}
:maxdepth: 2
:caption: API Reference

api/io
api/szl
api/plt
api/dat
api/libtecio
api/tecutils
api/cli
```

## Installation

```bash
pip install -e .
```

**Requirements:** Python 3.10+, NumPy, Tecplot 360

## Quick Example

```python
import tecio

# Read an existing file
with tecio.open("flow.szplt", "r") as r:
    print(r.variables)
    x = r.zone[0].variable[0].values

# Write a new file
with tecio.open("out.szplt", "w") as w:
    w.write_ijk_zone(
        data=[x, y, p],
        variables=["x", "y", "pressure"],
        title="Zone 1",
    )
```
