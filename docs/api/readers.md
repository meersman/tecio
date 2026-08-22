# Readers

{func}`tecio.open` (mode ``'r'``) returns a {class}`~tecio.TecplotSzlReader`,
{class}`~tecio.TecplotPltReader`, or {class}`~tecio.TecplotDatReader` based
on the file extension. All three implement the same interface, this page
documents that interface once; construct a specific class directly only if
you want to bypass the extension-based dispatch.

## What each format keeps in memory

This affects how expensive different operations are, not what you can do
with the reader, the interface below is identical either way.

| Format | On open | On first zone access | Per-value access |
|---|---|---|---|
| SZL | A live C file handle. Nothing else. | Zone headers resolved via C calls (cheap). | Live C call each time. |
| PLT | Whole header/zone-metadata section parsed eagerly; no file handle kept open. | Already available, it was parsed at open. | Read from disk on demand, using offsets recorded at open. |
| DAT | The entire file, header and every array, parsed into memory. | Already available, it was parsed at open. | Already in memory, no further disk access. |

Opening a PLT or DAT file (however large) does one pass over the file; opening SZL
does none, cost is deferred to whatever you actually touch.

## Zones and Variables: `ZoneList` / `VariableList`

`reader.zone` and `zone.variable` return one of two format-agnostic
container types, never a plain `list` and never a raw array:

```python
with tecio.open("flow.szplt") as r:
    r.zone[0]          # -> a zone reader (Ordered or FE, see below)
    r.zone[1:4]        # -> ZoneList, a sub-range, same kind
    r.zone[0].variable        # -> VariableList
    r.zone[0].variable["x"]   # -> TecplotVariableReader, by exact name
    r.zone[0].variable[2]     # -> TecplotVariableReader, by 0-based index
```

| Class | Returned by | Supports |
|---|---|---|
| {class}`~tecio.ZoneList` | `TecplotReader.zone` | `len()`, iteration, `int` index, `slice` (returns another `ZoneList`) |
| {class}`~tecio.VariableList` | `TecplotZoneReader.variable` | `len()`, iteration, `int` index, exact-name `str` index, `in` |

To pull NumPy arrays out of a zone, use `get_array`, shared by every zone
type and format:

```python
p = r.zone[0].get_array("p")             # ndarray | None
p = r.zone[0].get_array(2)               # by 0-based index
x, y, z = r.zone[0].get_array(["x", "y", "z"])   # tuple, for unpacking
```

A single key (index or name) returns one array; a list of names returns a
tuple in the order given. Returns `None` for a passive or shared variable
(shared resolves to the source zone's real data automatically, `None` only
means "no data at all"); raises `KeyError` for an unknown name or
`IndexError` for an out-of-range index. There's deliberately no cross-zone
accessor, to pull one variable across many zones, iterate explicitly so the
outer axis stays in your code:

```python
seq = [z.get_array("p") for z in r.zone]   # list[ndarray | None]
stack = np.stack(seq)                       # only once you know the shapes match
```

```{eval-rst}
.. currentmodule:: tecio

.. autoclass:: ZoneList
   :members:
   :show-inheritance:

.. autoclass:: VariableList
   :members:
   :show-inheritance:
```

## `TecplotReader`

One open file. Not constructed directly, {func}`tecio.open` returns the
concrete class for you.

```{eval-rst}
.. currentmodule:: tecio

.. autosummary::
   :nosignatures:

   TecplotReader.path
   TecplotReader.file_type
   TecplotReader.title
   TecplotReader.variables
   TecplotReader.num_vars
   TecplotReader.num_zones
   TecplotReader.zone
   TecplotReader.auxdata
   TecplotReader.num_auxdata_items
   TecplotReader.var_auxdata
   TecplotReader.get_var_auxdata
   TecplotReader.get_zone_auxdata
   TecplotReader.close
```

## Zones: `TecplotZoneReader`, split by topology

Only properties that mean the same thing for *every* zone, regardless of
topology, live on the shared base. Dimensions and connectivity don't: an
ordered zone has no node map, an FE zone has no `I`/`J`/`K`. Each lives only
on its own subclass, and accessing the wrong one raises `AttributeError`
rather than returning `None`:

```python
zone = r.zone[0]
zone.title, zone.zone_type          # always available
if isinstance(zone, tecio.TecplotOrderedZoneReader):
    zone.dimensions                  # (I, J, K)
elif isinstance(zone, tecio.TecplotFEZoneReader):
    zone.node_map                    # ndarray | None
```

```{eval-rst}
.. currentmodule:: tecio

.. autosummary::
   :nosignatures:

   TecplotZoneReader.zone_index
   TecplotZoneReader.title
   TecplotZoneReader.zone_type
   TecplotZoneReader.solution_time
   TecplotZoneReader.strand_id
   TecplotZoneReader.datapacking
   TecplotZoneReader.is_enabled
   TecplotZoneReader.variable
   TecplotZoneReader.auxdata
   TecplotZoneReader.get_array
```

`TecplotOrderedZoneReader` adds, for IJK-ordered zones:

```{eval-rst}
.. currentmodule:: tecio

.. autosummary::
   :nosignatures:

   TecplotOrderedZoneReader.I
   TecplotOrderedZoneReader.J
   TecplotOrderedZoneReader.K
   TecplotOrderedZoneReader.dimensions
   TecplotOrderedZoneReader.num_nodes
   TecplotOrderedZoneReader.num_elements
```

`TecplotFEZoneReader` adds, for finite-element zones (classic element types;
FEPOLYGON/FEPOLYHEDRON/FEMIXED metadata works too, connectivity for those
three doesn't yet):

```{eval-rst}
.. currentmodule:: tecio

.. autosummary::
   :nosignatures:

   TecplotFEZoneReader.num_nodes
   TecplotFEZoneReader.num_elements
   TecplotFEZoneReader.shared_connectivity
   TecplotFEZoneReader.nodes_per_cell
   TecplotFEZoneReader.node_map
```

```{eval-rst}
.. currentmodule:: tecio

.. autoclass:: TecplotZoneReader
   :members:
   :show-inheritance:

.. autoclass:: TecplotOrderedZoneReader
   :members:
   :show-inheritance:

.. autoclass:: TecplotFEZoneReader
   :members:
   :show-inheritance:
```

## `TecplotVariableReader`

One variable's metadata and data within a zone. Not split by value location
(nodal vs. cell-centered) or by the owning zone's topology, neither removes
a property, both only change the shape `get_values()`/`values` returns,
already computed for you from `value_location` and the owning zone.

```{eval-rst}
.. currentmodule:: tecio

.. autosummary::
   :nosignatures:

   TecplotVariableReader.name
   TecplotVariableReader.data_type
   TecplotVariableReader.value_location
   TecplotVariableReader.is_passive
   TecplotVariableReader.is_enabled
   TecplotVariableReader.shared_zone
   TecplotVariableReader.num_values
   TecplotVariableReader.values
   TecplotVariableReader.get_values

.. autoclass:: TecplotVariableReader
   :members:
   :show-inheritance:
```

## `TecplotAuxDataReader`

A read-only, dict-like view of one auxiliary-data mapping, dataset-, zone-,
or variable-level, same class for all three and every format. Implements
{class}`collections.abc.Mapping`, so `get`, `keys`, `values`, `items`, `in`,
and `len()` all work as expected, plus typed convenience accessors:

```{eval-rst}
.. currentmodule:: tecio

.. autosummary::
   :nosignatures:

   TecplotAuxDataReader.as_int
   TecplotAuxDataReader.as_float
   TecplotAuxDataReader.as_bool

.. autoclass:: TecplotAuxDataReader
   :members:
   :show-inheritance:
```

## Format-specific classes

Constructor and a handful of format quirks; everything else is the shared
interface above.

```{eval-rst}
.. currentmodule:: tecio

.. autoclass:: TecplotSzlReader
   :members:
   :show-inheritance:

.. autoclass:: TecplotPltReader
   :members:
   :show-inheritance:

.. autoclass:: TecplotDatReader
   :members:
   :show-inheritance:
```
