# tecio

Python interface for reading and writing Tecplot data files.

## Overview

Tecio wraps Tecplot's TecIO C-library functions for working with
Tecplot binary formats (szplt and plt). Supports read-only file access
and in-memory data manipulation.

**Requirements:** Python 3.10+, NumPy, Tecplot 360

## Example

```python
from szlfile import SzlFile
from tecdata import TecData
from szlio import ZoneType
import numpy as np

# Create szlplt reader object for access-on-demand
szl = SzlFile("flow.szplt")
pressure = szl.zones[0].variables[0].values

# Load and manipulate
data = TecData("flow.szplt", zones=[0], vars=["X", "Y", "Pressure"])
p_idx = data.get_variable_index("Pressure")
data.normalize_variable(p_idx, 101325.0)
data.write_szl("normalized.szplt")

# Create new Tecplot object from scratch
data = TecData(title="Grid")
x_idx = data.add_variable("X")
zone = data.add_zone("Domain", ZoneType.ORDERED, dimensions=(100, 100, 1))
zone.set_variable_data(x_idx, np.linspace(0, 1, 10000))
data.write_szl("output.szplt")
```

## Structure

- `szlfile.py` - High level SZL file API
- `tecdata.py` - Mutable in-memory data structures. Consistent for all initializations (plt, szplt, empty)
- `szlio.py` - TecIO C bindings
- `capeio.py` - Binary I/O utilities for PLT read capability

## Development

```bash
make format  # Format with ruff
make check   # Lint with ruff + mypy
make test    # Run pytest
```

## References

Tecplot data format specification: https://tecplot.azureedge.net/products/360/current/360-data-format.html
