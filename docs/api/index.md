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

For {mod}`tecio.libtecio`'s low-level functions, either of the following
styles is acceptable, both are explicit enough to be readable:

```python
# Import the module and reference through it
from tecio import libtecio

libtecio.tec_file_writer_open(...)

# Or import the specific name directly
from tecio.libtecio import tec_file_writer_open
```

Every other name documented here, readers, writers, zones, variables,
containers, and constants, is imported directly from {mod}`tecio`:

```python
from tecio import TecplotZoneReader, TecplotOrderedZoneReader, ZoneType

isinstance(zone, TecplotOrderedZoneReader)
zone.zone_type == ZoneType.ORDERED
```

---

## Indexing

Tecplot itself is 1-based throughout, zones, variables, and auxiliary-data
items are all numbered starting from 1, in the C API, in the file formats, and
in the ASCII keywords. `tecio` follows Python convention instead wherever a
Python object is referenced, and follows Tecplot's convention wherever
Tecplot's own numbering is the thing being reported or requested. Knowing
which is which for a given property avoids a real class of off-by-one bugs.

**Container indexing is 0-based**, like any Python sequence:

```python
r.zone[0]              # the FIRST zone, not zone "0"
r.zone[0].variable[0]  # the FIRST variable in that zone
```

**`zone_index` reports Tecplot's own 1-based number**, and the two won't
match:

```python
r.zone[0].zone_index   # == 1, not 0
r.zone[2].zone_index   # == 3, not 2
```

**Methods that take an explicit index argument expect Tecplot's 1-based
numbering**, not a Python position, this is the case most likely to produce
a silent off-by-one rather than a clear error:

```python
r.get_var_auxdata(1)   # the FIRST variable's aux data
r.get_var_auxdata(0)   # raises IndexError -- there is no variable 0
```

**One asymmetry worth knowing**: `zone_index` is a real property on every
zone reader, there's no equivalent `var_index` on `TecplotVariableReader`.
If you need a variable's 1-based index (e.g. to call `get_var_auxdata`),
compute it from its position: `reader.variables.index(name) + 1`, or track
it while enumerating: `for i, var in enumerate(zone.variable, start=1): ...`.

---

## Core Classes

These classes define the interface every reader and writer shares,
regardless of which of the three file formats produced or is producing the
data. This is the part of the API worth understanding first; see
{doc}`readers` and {doc}`writers` for the full property/method reference.

**Reading**

| Class | Description |
|---|---|
| {class}`~tecio.TecplotReader` | One open file: title, variables, zones, dataset-level aux data |
| {class}`~tecio.TecplotZoneReader` | Shared zone interface: title, type, solution time, strand ID, variables, aux data |
| {class}`~tecio.TecplotOrderedZoneReader` | Adds `I`/`J`/`K`/`dimensions` for IJK-ordered zones |
| {class}`~tecio.TecplotFEZoneReader` | Adds `node_map`/`nodes_per_cell`/`shared_connectivity` for finite-element zones |
| {class}`~tecio.TecplotVariableReader` | One variable's metadata and data within a zone |
| {class}`~tecio.TecplotAuxDataReader` | Read-only, dict-like view of one auxiliary-data mapping |

**Writing**

| Class | Description |
|---|---|
| {class}`~tecio.TecplotWriter` | Shared writer interface: construction, aux-data staging, zone writing, closing |

{class}`~tecio.TecplotZoneReader` splits into
{class}`~tecio.TecplotOrderedZoneReader` and
{class}`~tecio.TecplotFEZoneReader` because the two zone topologies don't
share properties like dimensions or connectivity; a property that only makes
sense for one topology (`node_map` on an FE zone, `dimensions` on an
ordered zone) exists only on that subclass. Accessing it on the other raises
`AttributeError` rather than returning `None`, so
`isinstance(zone, TecplotOrderedZoneReader)` (or the FE equivalent) is the
way to branch on zone topology.

{class}`~tecio.TecplotVariableReader` and {class}`~tecio.TecplotAuxDataReader`
are not split this way, nodal vs. cell-centered data and dataset- vs.
zone-level auxiliary data don't remove any properties, they only change the
behavior of one already-existing method, so a single class covers every
case.

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

See {doc}`readers` and {doc}`writers` for each class's full documentation,
including per-format memory-model notes and parameter comparison tables.

---

## Accessing Variable Data

Every reader's `zone` and `variable` properties return one of two
format-agnostic container types rather than a plain `list`. Both support the
same indexing, iteration, and `len()` you would expect from a list, plus
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
single zone, use `TecplotZoneReader.get_array`, shared by every zone type
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

`get_array` returns `None` for a passive or shared variable, and raises
`KeyError` for an unknown name or `IndexError` for an out-of-range index.

| Class | Returned by | Description |
|---|---|---|
| {class}`~tecio.ZoneList` | `TecplotReader.zone` | Sequence of zones; integer index, slice, or iterate |
| {class}`~tecio.VariableList` | `TecplotZoneReader.variable` | Sequence of variables; index by position or exact name |

See {doc}`readers` for the full container API.

---

## Meaningful Constants

Zone types, data types, value locations, and the rest of Tecplot's
enumerated constants are plain Python enums with no dependency on the C
library, importable directly from {mod}`tecio` (e.g. `tecio.ZoneType`).

```{eval-rst}
.. currentmodule:: tecio

.. autosummary::
   :nosignatures:

   FileFormat
   FileType
   ZoneType
   DataType
   ValueLocation
   FaceNeighborMode
   FeCellShape
   DataPacking
   VarStatus
   Boolean
   Debug
```

See {doc}`constants` for each constant's full member listing.

---

## C Function Wrappers

{mod}`tecio.libtecio` provides the direct Python wrapper for every TecIO C
function tecio uses internally, kept public so the library can be called
directly, if desired. Importing it never raises, even with no Tecplot
installation at all, or an older one missing a newer function, see
{doc}`libtecio` for how that degradation works.

```{eval-rst}
.. currentmodule:: tecio.libtecio

.. autosummary::
   :nosignatures:

   TecioError
   TecioUnavailableError
   TecioAvailabilityWarning
```

**SZL Read Functions**

```{eval-rst}
.. currentmodule:: tecio.libtecio

.. autosummary::
   :nosignatures:

   tec_file_reader_open
   tec_file_reader_close
   tec_file_get_type
   tec_data_set_get_title
   tec_data_set_get_num_vars
   tec_data_set_get_num_zones
   tec_zone_get_ijk
   tec_zone_get_title
   tec_zone_get_type
   tec_zone_is_enabled
   tec_zone_get_solution_time
   tec_zone_get_strand_id
   is_64bit
   tec_zone_node_map_get_64
   tec_zone_node_map_get
   tec_var_get_name
   tec_var_is_enabled
   tec_zone_var_get_type
   tec_zone_var_get_value_location
   tec_zone_var_is_passive
   tec_zone_var_get_shared_zone
   tec_zone_var_get_num_values
   tec_zone_var_get_float_values
   tec_zone_var_get_double_values
   tec_zone_var_get_int32_values
   tec_zone_var_get_int16_values
   tec_zone_var_get_uint8_values
   tec_data_set_aux_data_get_num_items
   tec_data_set_aux_data_get_item
   tec_var_aux_data_get_num_items
   tec_var_aux_data_get_item
   tec_zone_aux_data_get_num_items
   tec_zone_aux_data_get_item
```

**SZL Write Functions**

```{eval-rst}
.. currentmodule:: tecio.libtecio

.. autosummary::
   :nosignatures:

   tec_file_writer_open
   tec_file_writer_close
   tec_file_writer_flush
   tec_zone_create_ijk
   tec_zone_create_fe
   tec_zone_set_unsteady_options
   tec_zone_var_write_double_values
   tec_zone_var_write_float_values
   tec_zone_var_write_int32_values
   tec_zone_var_write_int16_values
   tec_zone_var_write_uint8_values
   tec_zone_node_map_write32
   tec_zone_node_map_write64
   tec_zone_face_nbr_write_connections32
   tec_zone_face_nbr_write_connections64
   tec_data_set_add_aux_data
   tec_var_add_aux_data
   tec_zone_add_aux_data
```

**Classic API Functions**

```{eval-rst}
.. currentmodule:: tecio.libtecio

.. autosummary::
   :nosignatures:

   tecini142
   tecend142
   tecflush142
   tecfil142
   tecforeign142
   teczne142
   tecpolyzne142
   tecznefemixed142
   tecdat142
   tecnode142
   tecface142
   tecpolyface142
   tecpolybconn142
   tecauxstr142
   tecvauxstr142
   teczauxstr142
   tecusr142
```

See {doc}`libtecio` for the full module page, and {doc}`libtecio_szl_read`,
{doc}`libtecio_szl_write`, {doc}`libtecio_classic` for each function's full
signature and documentation.

---

## Append Handles

Returned by {func}`tecio.open` when using `'a'` or `'a+'` mode.

| Class | Mode | Description |
|---|---|---|
| {class}`~tecio.AppendWrite` | `'a'` | Append zones to an existing SZL file |
| {class}`~tecio.AppendReadWrite` | `'a+'` | Append zones and read existing zones |

```{toctree}
:hidden:

open
readers
writers
constants
libtecio
append_write
append_read_write
```
