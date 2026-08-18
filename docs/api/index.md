# API Reference

{mod}`tecio` is designed around a single entry point, {func}`tecio.open`,
which opens a Tecplot file for reading, writing, or appending and returns the
appropriate reader or writer based on the file extension. In most cases you
will not need to instantiate {class}`~tecio.TecplotSzlReader`,
{class}`~tecio.TecplotSzlWriter`, or any other class directly;
{func}`tecio.open` selects and returns the correct one for you.

Every reader and every writer, regardless of format, implements the same
interface. That shared interface, defined by a small set of base classes, is
what most of this reference documents; the concrete per-format classes
mostly add a constructor and a handful of format-specific quirks on top.

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

Every other class documented here, readers, writers, zones, variables,
containers, is imported directly from {mod}`tecio`:

```python
from tecio import TecplotZoneReader, TecplotOrderedZoneReader

isinstance(zone, TecplotOrderedZoneReader)
```

---

(api-open)=
## `tecio.open`

The single entry point for reading, writing, and appending. See {doc}`open`
for the full signature and documentation.

---

## Core Classes

These five classes define the interface every reader and writer shares,
regardless of which of the three file formats produced or is producing the
data. This is the part of the API worth understanding first.

```{eval-rst}
.. currentmodule:: tecio

.. autosummary::
   :nosignatures:

   TecplotReader
   TecplotZoneReader
   TecplotOrderedZoneReader
   TecplotFEZoneReader
   TecplotVariableReader
   TecplotAuxDataReader
   TecplotWriter
```

{class}`~tecio.TecplotZoneReader` splits into
{class}`~tecio.TecplotOrderedZoneReader` and
{class}`~tecio.TecplotFEZoneReader` because the two zone topologies don't
share properties like dimensions or connectivity; a property that only makes
sense for one topology (``node_map`` on an FE zone, ``dimensions`` on an
ordered zone) exists only on that subclass. Accessing it on the other raises
``AttributeError`` rather than returning ``None``, so
``isinstance(zone, TecplotOrderedZoneReader)`` (or the FE equivalent) is the
way to branch on zone topology.

{class}`~tecio.TecplotVariableReader` and {class}`~tecio.TecplotAuxDataReader`
are not split this way, nodal vs. cell-centered data and dataset- vs.
zone-level auxiliary data don't remove any properties, they only change the
behavior of one already-existing method, so a single class covers every
case.

```{toctree}
:hidden:

reader
zone
variable
auxdata
writer
```

## Accessing Variable Data

Every reader's ``zone`` and ``variable`` properties return one of two
format-agnostic container types rather than a plain ``list``. Both support the
same indexing, iteration, and ``len()`` you would expect from a list, plus
name-based and slice-based access:

```python
with tecio.open("flow.szplt") as r:
    r.zone[0]  # -> TecplotOrderedZoneReader or TecplotFEZoneReader
    r.zone[1:4]  # -> ZoneList (sub-range, same kind)
    r.zone[0].variable  # -> VariableList
    r.zone[0].variable["x"]  # -> TecplotVariableReader, by exact name
    r.zone[0].variable[2]  # -> TecplotVariableReader, by 0-based index
```

Indexing a {class}`~tecio.ZoneList` or {class}`~tecio.VariableList` always
returns an element or a sub-collection of the *same* kind — never a raw
array. To pull the underlying NumPy data for one or more variables in a
single zone, use ``TecplotZoneReader.get_array``, shared by every zone type
and every format:

```python
p = r.zone[0].get_array("p")  # ndarray | None
p = r.zone[0].get_array(2)  # by 0-based index
x, y, z = r.zone[0].get_array(["x", "y", "z"])  # tuple, for unpacking
```

A single key (index or name) returns one array; a list of names returns a
tuple of arrays in the order given, suitable for unpacking. There is
deliberately no cross-zone array accessor — to pull one variable across many
zones (e.g. a transient sequence), iterate explicitly so the outer axis stays
in your code, and stack only when you know the shapes match:

```python
seq = [z.get_array("p") for z in r.zone]  # list[ndarray | None]
stack = np.stack(seq)  # only if every zone matches
```

``get_array`` returns ``None`` for a passive or shared variable, and raises
``KeyError`` for an unknown name or ``IndexError`` for an out-of-range index.

| Class | Returned by | Description |
|---|---|---|
| {class}`~tecio.ZoneList` | ``TecplotReader.zone`` | Sequence of zones; integer index, slice, or iterate |
| {class}`~tecio.VariableList` | ``TecplotZoneReader.variable`` | Sequence of variables; index by position or exact name |

See {doc}`containers` for the full container API.

```{toctree}
:hidden:

containers
```

---

## Format-Specific Classes

{func}`tecio.open` constructs one of these six classes for you, based on the
file extension. Construct one directly only if you want to bypass the
extension-based dispatch, e.g. to force a specific format regardless of file
name.

```{eval-rst}
.. currentmodule:: tecio

.. autosummary::
   :nosignatures:

   TecplotSzlReader
   TecplotPltReader
   TecplotDatReader
   TecplotSzlWriter
   TecplotPltWriter
   TecplotDatWriter
```

See {doc}`formats` for the full list with links to each class's own page.

```{toctree}
:hidden:

formats
```

---

## Append Handles

Returned by {func}`tecio.open` when using ``'a'`` or ``'a+'`` mode.

| Class | Mode | Description |
|---|---|---|
| {class}`~tecio.AppendWrite` | ``'a'`` | Append zones to an existing SZL file |
| {class}`~tecio.AppendReadWrite` | ``'a+'`` | Append zones and read existing zones |

## Submodules

| Module | Description |
|---|---|
| {mod}`tecio.cli` | Command-line tools built on the public reader/writer API |
| {mod}`tecio.libtecio` | Low-level C library bindings and enums |

```{toctree}
:hidden:

open
append_write
append_read_write
libtecio
```
