# API Reference

```{eval-rst}
.. autofunction:: tecio.open
```

## Append Handles

Returned by :func:`tecio.open` when using ``'a'`` or ``'a+'`` mode.

| Class | Mode | Description |
|---|---|---|
| {class}`~tecio._io.AppendWrite` | ``'a'`` | Append zones to an existing SZL file |
| {class}`~tecio._io.AppendReadWrite` | ``'a+'`` | Append zones and read existing zones |

## Submodules

| Module | Description |
|---|---|
| {mod}`tecio.szl` | Read and write Tecplot SZL (``.szplt``) files |
| {mod}`tecio.plt` | Read and write Tecplot PLT (``.plt``) files |
| {mod}`tecio.dat` | Read and write Tecplot ASCII (``.dat``) files |
| {mod}`tecio.libtecio` | Low-level C library bindings and enums |
| {mod}`tecio.utils` | Locate Tecplot installations and the TecIO library |

```{toctree}
:hidden:

tecio.open
append_write
append_read_write
szl
plt
dat
libtecio
utils
```
