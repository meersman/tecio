# Quickstart

## Reading Files

```python
import tecio

# Open any supported format — extension selects the handler
r = tecio.open("flow.szplt", "r")  # also .plt, .dat, .tec

print(r.title)           # dataset title
print(r.variables)       # ['x', 'y', 'z', 'pressure', ...]
print(r.num_zones)       # number of zones

# Access zone data
zone = r.zone[0]
print(zone.title, zone.zone_type, zone.dimensions)

# Read variable arrays (lazy — data loaded on access)
x = zone.variable[0].values     # NumPy array
p = zone.variable[3].values

# Auxiliary data
print(r.auxdata)                 # dataset-level
print(zone.auxdata)              # zone-level
```

## Writing Files

```python
import numpy as np
import tecio

x = np.linspace(0, 1, 100)
y = np.linspace(0, 1, 50)
X, Y = np.meshgrid(x, y, indexing="ij")
P = np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)

with tecio.open("output.szplt", "w") as w:
    w.write_ijk_zone(
        data=[X, Y, P],
        variables=["x", "y", "pressure"],
        title="Flow Field",
    )
```

## Variable Sharing

Share coordinate arrays across time steps to reduce file size:

```python
with tecio.open("transient.szplt", "w") as w:
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

## Format Conversion

```python
# Via Python API
with tecio.open("input.plt", "r") as r:
    with tecio.open("output.szplt", "w",
                    title=r.title, variables=r.variables) as w:
        # ... copy zones
        pass
```

```bash
# Via command line
teconvert -szplt flow.plt
teconvert -dat flow.szplt
```

## Command-Line Tools

```bash
tecdump flow.szplt           # Print file contents
tecstat flow.szplt            # Variable statistics
tecfix flow.szplt             # Remove NaN/Inf variables
tecslice -i ::2 -o thin.szplt flow.szplt  # Thin grid
tecmerge -o combined.szplt part1.szplt part2.szplt
tecextract -zones 1,3 flow.szplt
tecscale -variable Pressure -scale 1e-3 flow.szplt
```
