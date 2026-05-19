# tecio

A Python interface for reading and writing Tecplot data files.

This package wraps Tecplot's TecIO C library for reading and writing
binary SZL (`.szplt`) and PLT (`.plt`, `.bin`) formats as well as
providing read/write utilities for ASCII DAT (`.dat`, `.tec`) files.

The Tecplot data format is organised around *zones*, a flexible object
that can represent a structured or unstructured block, subdomain, or
boundary surface of a CFD solution, a single time step in a transient
dataset, or a line or scatter point in a plot. Zone data may be
*structured* (IJK-ordered matrix) or *unstructured* (nodes and cells
with an explicit connectivity map). A file may contain both grid and
solution data, or just one or the other via the GRID and SOLUTION file
types.

All zones in a dataset share the same variable list, though individual
variables may be marked *passive* (no data for that zone) or *shared*
from another zone to avoid storing redundant data, commonly used to
share grid coordinates across time steps. Auxiliary data can be
attached at the dataset, variable, or zone level as free-form
key-value strings. Special `Common.*` auxiliary data keys are
recognised by Tecplot 360 EX on load and can be used to automatically
configure axis variables, velocity vectors, the default contour
variable, and more.

The SZL (`.szplt`) format additionally uses a proprietary node-map
compression algorithm that produces significantly smaller files
compared to the classic PLT format, particularly for large
unstructured meshes.

> The TecIO C library is proprietary software distributed by Tecplot,
  Inc. under a permissive license agreement. This package is an
  independent utility and is not affiliated with or endorsed by
  Tecplot, Inc. Use of `tecio` requires an existing Tecplot
  installation to provide the shared library, but does not require an
  active Tecplot license.

---

## Contents

1. **[Quickstart](quickstart.md)**
   - {ref}`Installation <installation>`
   - {ref}`Example Usage <example-usage>`

2. **[API Reference](api/index.md)**
   - [`tecio.open`](api/index.md) — open a file for reading, writing, or appending
     - [`AppendWrite`](api/append_write.md) — append zones to an existing file (`mode='a'`)
     - [`AppendReadWrite`](api/append_read_write.md) — append and read in the same session (`mode='a+'`)
   - **[Submodules](api/submodules.md)**
     - [`tecio.szl`](api/szl.md) — read and write Tecplot SZL (`.szplt`) files
       - [`szl.Read`](api/szl_read.md)
       - [`szl.Write`](api/szl_write.md)
     - [`tecio.plt`](api/plt.md) — read and write Tecplot PLT (`.plt`) files
       - [`plt.Read`](api/plt_read.md)
       - [`plt.Write`](api/plt_write.md)
     - [`tecio.dat`](api/dat.md) — read and write Tecplot ASCII (`.dat`) files
       - [`dat.Read`](api/dat_read.md)
       - [`dat.Write`](api/dat_write.md)
     - [`tecio.libtecio`](api/libtecio.md) — low-level C library bindings and enums
       - [Enums](api/libtecio_enums.md)
       - [SZL Read Functions](api/libtecio_szl_read.md)
       - [SZL Write Functions](api/libtecio_szl_write.md)
       - [Classic API Functions](api/libtecio_classic.md)
     - [`tecio.utils`](api/utils.md) — locate Tecplot installations and the TecIO library
   - [Console Scripts](api/cli.md) — command-line tools (`tecdump`, `tecfix`, `tecmerge`, …)

3. **Demos**
   - [Lorenz Attractor Animation](_demos/lorenz/lorenz.md)
   - [Gravity Waves Around Binary Black Holes](_demos/gravity_waves/gravity_waves.md)
   - [Spectral Incompressible Navier–Stokes (Kelvin–Helmholtz)](_demos/simple_spectral_solver/simple_spectral.md)

---

```{toctree}
:hidden:
:maxdepth: 2
:caption: User Guide

quickstart
api/index
api/cli
genindex
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: Demos

_demos/lorenz/lorenz
_demos/gravity_waves/gravity_waves
_demos/simple_spectral_solver/simple_spectral
```
