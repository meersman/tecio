# tecio

Python interface for reading and writing Tecplot data files.

## Overview

This package wraps Tecplot's TecIO C-library functions for working
with Tecplot binary formats (szplt and plt). Supports read-only file
access and in-memory data manipulation.

**Requirements:** Python 3.10+, NumPy, Tecplot 360

## Example Usage

```python
import numpy as np

import tecio
from tecio.libtecio import ZoneType

# Create szlplt reader object for access-on-demand
szl = tecio.Read("flow.szplt")
pressure = szl.zones[0].variables[0].values

# Load and manipulate
ds = tecio.Dataset("flow.szplt", zones=[0], vars=["X", "Y", "Pressure"])
p_idx = ds.get_variable_index("Pressure")
ds.normalize_variable(p_idx, 101325.0)
ds.write_szl("normalized.szplt")

# Create new Tecplot object from scratch
ds = tecio.Dataset(title="Grid")
zone = ds.add_zone("XGrid", ZoneType.ORDERED)
x_idx = ds.add_variable(zone, "X", np.linspace(0, 1, 10000))
ds.write_szl("output.szplt", file_type=FileType.GRID)
```

## Structure

- `szl.py`      - High level SZL file API
- `dataset.py`  - NOT YET IMPLEMENTED: Mutable in-memory data structures. Consistent for all initializations (plt, szplt, empty)
- `libtecio.py` - TecIO C bindings

## Development

```bash
make format  # Format with ruff
make check   # Lint with ruff + mypy
make test    # Run pytest
```

## References

Tecplot data format specification: https://tecplot.azureedge.net/products/360/current/360-data-format.html
Get tecio: https://tecplot.com/products/tecio-library/