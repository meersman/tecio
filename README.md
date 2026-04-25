# tecio

Python interface for reading and writing Tecplot data files.

## Overview

This package wraps Tecplot's TecIO C-library functions for working
with Tecplot binary formats (szplt and plt).

**Requirements:** Python 3.10+, NumPy, Tecplot 360

## Example Usage

```python
import tecio
from tecio.libtecio import ZoneType

# Create szplt reader object for access-on-demand
szl_in = tecio.open("flow.szplt", "r")

# List available variables in the dataset
print(szl_in.variables)
# ['x', 'y', 'z', 'uvel', 'vvel', 'wvel', 'pres', 'dens', 'temp']

# Extract single variable arrays from all zones
x = [szl_in.zone[i].x for i in range(len(szl_in.zone))]
y = [szl_in.zone[i].y for i in range(len(szl_in.zone))]
z = [szl_in.zone[i].z for i in range(len(szl_in.zone))]

# Create nondimensional pressure variable
pressure = [
    szl_in.zone[i].pres / szl_in.auxdata["Common.ReferencePressure"]
    for i in range(len(szl_in.zone))
]

# Write out grid and nodimensional pressure variable
with tecio.open("pres.szplt", "w") as szl_out:
    for i in range(len(x)):
        if szl_in.zone[i].zone_type == ZoneType.ORDERED:
            szl_out.write_ijk_zone(
                title=szl_in.zone[i].title,
                variables=["x", "y", "z", "pressure"],
                data=[x[i], y[i], z[i], pressure[i]],
            )
        else:
            szl_out.write_fe_zone(
                title=szl_in.zone[i].title,
                variables=["x", "y", "z", "pressure"],
                data=[x[i], y[i], z[i], pressure[i]],
                node_map=szl_in.zone[i].node_map,
            )
```

## Structure

```text
tecio
├── __init__.py
├── _io.py
├── cli
│   ├── tecdump.py
│   ├── tecfix.py
│   └── teconvert.py
├── dat               # High level Tecplot ascii API
│   ├── __init__.py
│   ├── read.py
│   └── write.py
├── libtecio.py       # TecIO C bindings
├── plt               # High level PLT file API
│   ├── __init__.py
│   ├── read.py
│   └── write.py
├── szl               # High level SZL file API
│   ├── __init__.py
│   ├── read.py
│   └── write.py
└── tecutils.py
```

## Development

```bash
make format  # Format with ruff
make check   # Lint with ruff + mypy
make test    # Run pytest
```

## Demos
 * [Lorenz attactor animation](demos/lorenz/lorenz.md)
 * [Gravity wave animation around binary black holes](demos/gravity_waves/gravity_waves.md)
 * [Spectral incompressible Navier Stokes animation](demos/simple_spectral_solver/simple_spectral.md)
 
## References

Tecplot data format specification: https://tecplot.azureedge.net/products/360/current/360-data-format.html

Get tecio: https://tecplot.com/products/tecio-library/