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
pressure = [szl_in.zone[i].pres for i in range(len(szl_in.zone))]

# Write out just pressure variable
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

- `dat.py`	- High level Tecplot ascii API (Not implemented yet)
- `plt.py`	- High level PLT file API (Only write implemented so far)
- `szl.py`      - High level SZL file API
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