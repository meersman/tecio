# tecio

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![CI](https://github.com/meersman/tecio/actions/workflows/ci.yml/badge.svg)](https://github.com/meersman/tecio/actions/workflows/ci.yml)
[![Codecov](https://codecov.io/gh/meersman/tecio/branch/main/graph/badge.svg)](https://codecov.io/gh/meersman/tecio)
[![Docs](https://github.com/meersman/tecio/actions/workflows/docs.yml/badge.svg)](https://meersman.github.io/tecio)

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
szl = tecio.open("flow.szplt")

# List available variables in the dataset
print(szl.variables)
# ['x', 'y', 'z', 'uvel', 'vvel', 'wvel', 'pres', 'dens', 'temp']

# Create nondimensional pressure variable
pressure = [
    szl.zone[i].pres / szl.auxdata["Common.ReferencePressure"]
    for i in range(len(szl.zone))
]

# Write out grid and nodimensional pressure variable
with tecio.open("pres.szplt", "w") as writer:
    for i in range(len(x)):
        if szl.zone[i].zone_type == ZoneType.ORDERED:
            writer.write_ijk_zone(
                title=szl.zone[i].title,
                variables=["x", "y", "z", "pressure"],
                data=[szl.zone[i].x, szl.zone[i].y, szl.zone[i].z, pressure[i]],
            )
        else:
            writer.write_fe_zone(
                title=szl.zone[i].title,
                variables=["x", "y", "z", "pressure"],
                data=[x[i], y[i], z[i], pressure[i]],
                node_map=szl.zone[i].node_map,
            )
```

## Structure

```text
tecio
├── __init__.py
├── _io.py
├── cli                 # Console scripts
│   ├── __init__.py
│   ├── tecdump.py
│   ├── tecextract.py
│   ├── tecfix.py
│   ├── tecmerge.py
│   ├── teconvert.py
│   ├── tecscale.py
│   ├── tecslice.py
│   ├── tecsplit.py
│   └── tecstats.py
├── dat                 # Higher level Tecplot ascii API
│   ├── __init__.py
│   ├── _read.py
│   └── _write.py
├── plt
│   ├── __init__.py
│   ├── _read.py
│   └── _write.py
├── szl                 # Higher level SZL file API
│   ├── __init__.py
│   ├── _read.py
│   └── _write.py
├── libtecio.py         # TecIO Python wrapper functions
└── utils.py
```

## Development

```bash
make install-dev  # Install with development dependencies
make test         # Run unit tests
make check        # Run format, lint and typechecks
```
 
## References

Tecplot data format specification: https://tecplot.azureedge.net/products/360/current/360-data-format.html

Get tecio: https://tecplot.com/products/tecio-library/
