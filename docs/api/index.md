# API Reference

{mod}`tecio` is designed around a single entry point, {func}`tecio.open`,
which opens a Tecplot file for reading, writing, or appending and returns the
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

## Accessing Variable Data

Every reader's ``zone`` and ``variable`` properties return one of two
format-agnostic container types rather than a plain ``list``. Both support the
same indexing, iteration, and ``len()`` you would expect from a list, plus
name-based and slice-based access:

```python
with tecio.open("flow.szplt") as r:
    r.zone[0]                       # -> ReadZone
    r.zone[1:4]                     # -> ZoneList (sub-range, same kind)
    r.zone[0].variable               # -> VariableList
    r.zone[0].variable["x"]          # -> ReadVariable, by exact name
    r.zone[0].variable[2]            # -> ReadVariable, by 0-based index
```

Indexing a {class}`~tecio.ZoneList` or {class}`~tecio.VariableList` always
returns an element or a sub-collection of the *same* kind — never a raw
array. To pull the underlying NumPy data for one or more variables in a
single zone, use ``ReadZone.get_array``, available on every format's zone
object:

```python
    p = r.zone[0].get_array("p")                 # ndarray | None
    p = r.zone[0].get_array(2)                    # by 0-based index
    x, y, z = r.zone[0].get_array(["x", "y", "z"])  # tuple, for unpacking
```

A single key (index or name) returns one array; a list of names returns a
tuple of arrays in the order given, suitable for unpacking. There is
deliberately no cross-zone array accessor — to pull one variable across many
zones (e.g. a transient sequence), iterate explicitly so the outer axis stays
in your code, and stack only when you know the shapes match:

```python
    seq = [z.get_array("p") for z in r.zone]   # list[ndarray | None]
    stack = np.stack(seq)                       # only if every zone matches
```

``get_array`` returns ``None`` for a passive or shared variable, and raises
``KeyError`` for an unknown name or ``IndexError`` for an out-of-range index.

| Class | Returned by | Description |
|---|---|---|
| {class}`~tecio.ZoneList` | ``Read.zone`` | Sequence of zones; integer index, slice, or iterate |
| {class}`~tecio.VariableList` | ``ReadZone.variable`` | Sequence of variables; index by position or exact name |

## Append Handles

Returned by {func}`tecio.open` when using ``'a'`` or ``'a+'`` mode.

| Class | Mode | Description |
|---|---|---|
| {class}`~tecio.AppendWrite` | ``'a'`` | Append zones to an existing SZL file |
| {class}`~tecio.AppendReadWrite` | ``'a+'`` | Append zones and read existing zones |

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
containers
append_write
append_read_write
szl
plt
dat
libtecio
utils
```
