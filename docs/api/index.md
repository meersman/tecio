# API Reference

{mod}`tecio` is designed around a single entry point, {func}`tecio.open`, which
opens a Tecplot file for reading, writing, or appending and returns the
appropriate handler based on the file extension. In most cases you will not
need to instantiate {class}`tecio.szl.Read`, {class}`tecio.szl.Write`, or any
other class directly; {func}`tecio.open` selects and returns the correct one
for you.

## Importing

For the {mod}`tecio.libtecio` enums and low-level functions, either of the
following styles is acceptable, both are explicit enough to be readable:

```python
# Import the module and reference through it
from tecio import libtecio
libtecio.ZoneType.ORDERED
libtecio.tec_file_writer_open(...)

# Or import specific names directly
from tecio.libtecio import ZoneType, DataType, ValueLocation
ZoneType.ORDERED
```

The submodules {mod}`tecio.szl`, {mod}`tecio.plt`, {mod}`tecio.dat`, and
{mod}`tecio.libtecio` are all part of the public API and are documented in
full below, but for typical use cases you should not need to import them
directly.

---

(api-open)=
## `tecio.open`

```{eval-rst}
.. autofunction:: tecio.open
```

## Append Handles

Returned by {func}`tecio.open` when using ``'a'`` or ``'a+'`` mode.

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

open
append_write
append_read_write
szl
plt
dat
libtecio
utils
```
