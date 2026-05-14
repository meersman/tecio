# tecio

Python interface for reading and writing Tecplot data files.

Wraps Tecplot's TecIO C library for working with binary SZL (`.szplt`)
and PLT (`.plt`) formats as well as ASCII DAT (`.dat`) files.

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
api/submodules
api/cli
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: Demos

_demos/lorenz/lorenz
_demos/gravity_waves/gravity_waves
_demos/simple_spectral_solver/simple_spectral
```
