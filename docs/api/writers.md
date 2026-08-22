# Writers

{func}`tecio.open` returns a {class}`~tecio.TecplotSzlWriter`,
{class}`~tecio.TecplotPltWriter`, or {class}`~tecio.TecplotDatWriter` for
modes ``'w'``/``'x'``, based on the output file extension, or an
{class}`~tecio.AppendWrite`/{class}`~tecio.AppendReadWrite` for modes
``'a'``/``'a+'``. All three concrete writers share the same five-method
lifecycle:

```python
with tecio.open("out.szplt", "w", variables=["x", "y", "p"]) as w:  # 1. open
    w.add_auxdataset_dict({"Solver": "MyCFD"})                       # 2. stage dataset level aux data
    w.write_ijk_zone(data=[x, y, p], title="Zone 1")                 # 3. write structured zones
    w.write_fe_zone(data=[...], zone_type=ZoneType.FETRIANGLE, ...)  # 4. write unstructured zones
    # 5. close happens automatically on context-manager exit
```

Opening with an explicit `variables` list opens the file immediately
(*eager* open); omitting it defers file creation until the first zone write,
which must then supply `variables` itself (*lazy* open). Either way, closing
(directly, or via the context manager, recommended) is required to produce a
valid, readable file.

## Order of operations

The SZL format allows for flexible order of operations, but PLT and DAT
require a sequential order of operations. However, the implementation of
writing via {func}`tecio.open` requires all three formats to write zones
sequentially. If a SZL file is desired to be written with arrays out of
sequence, {mod}`tecio.libtecio` wrapper functions must be called directly.

Zone-level aux data must be specified with the zone write methods, but
dataset aux data may be staged at any time. Upon file close, all buffered aux
data is written to the file before finalizing output.

## {class}`~tecio.TecplotWriter`

Set once at construction. These are plain attributes, not properties,
readable at any point, but reassigning one after the file header has
already been written (e.g. `title` under eager-open, once the first zone
exists) changes the Python object without changing the file, the header was
already committed.

| Attribute | Type | Description |
|---|---|---|
| `path` | `str` | Output file path |
| `title` | `str` | Dataset title |
| `variables` | `list[str] \| None` | Variable name list, `None` before the file is opened (lazy-open) |
| `file_type` | `FileType` | `FULL`, `GRID`, or `SOLUTION` |
| `current_zone` | `int` | 1-based index of the most recently written zone, `0` before any zone is written |

`meta` (below) is a running, read-only summary of everything committed to
the file so far (title, variables, aux-item counts, and a per-zone record
in write order), useful for a quick sanity check without re-opening the
file for reading.

```{eval-rst}
.. currentmodule:: tecio

.. autosummary::
   :toctree: writers

   TecplotWriter
```

## `write_ijk_zone`

Writes one complete IJK-ordered zone. Dimensions are inferred from the shape
of the first array (1-D → `(N, 1, 1)`, 2-D → `(I, J, 1)`, missing trailing
dimensions default to `1`).

| Parameter | Type | Default | SZL | PLT | DAT |
|---|---|---|---|---|---|
| `data` | `Sequence[ArrayLike]` | required | ✓ | ✓ | ✓ |
| `title` | `str \| None` | `None` | ✓ | ✓ | ✓ |
| `variables` | `list[str] \| None` | `None` | ✓ | ✓ | ✓ |
| `value_locations` | `Sequence[ValueLocation] \| None` | `None` | ✓ | ✓ | ✓ |
| `passive_vars` | `Sequence[bool \| int] \| None` | `None` | ✓ | ✓ | ✓ |
| `var_sharing` | `Sequence[int] \| None` | `None` | ✓ | ✓ | ✓ |
| `solution_time` | `float` | `0.0` | ✓ | ✓ | ✓ |
| `strand_id` | `int` | `0` | ✓ | ✓ | ✓ |
| `aux` | `dict[str, Any] \| None` | `None` | ✓ | ✓ | ✓ |
| `datapacking` | `DataPacking \| str` | `BLOCK` | ✓ | ✓ | ✓ |
| `flush` | `bool` | `False` | ✓ | — | — |

`flush` (SZL only) writes this zone's data to a temporary file and releases
it from memory immediately, rather than holding everything until `close()`;
useful for very large datasets. Every other parameter behaves identically
across all three formats.

## `write_fe_zone`

Writes one complete finite-element zone. All the `write_ijk_zone` parameters
apply here too (except `datapacking`, which only means something for
ordered zones), plus:

| Parameter | Type | Default | SZL | PLT | DAT |
|---|---|---|---|---|---|
| `zone_type` | `ZoneType` | required | ✓ | ✓ | ✓ |
| `node_map` | `ArrayLike \| None` | `None` | ✓ | ✓ | ✓ |
| `con_sharing` | `int \| None` | `None` | ✓ | ✓ | ✓ |
| `face_neighbors` | `ArrayLike \| None` | `None` | ✓ | ✓ | accepted, not yet written¹ |
| `face_nbr_mode` | `FaceNeighborMode` | `LOCAL_ONE_TO_ONE` | ✓ | ✓ | accepted, not yet written¹ |
| `flush` | `bool` | `False` | ✓ | — | — |

Node and cell counts are inferred from `node_map`, or, if `node_map` is
omitted, from the zone referenced by `con_sharing`.

¹ **Known limitation**: DAT's `face_neighbors`/`face_nbr_mode` parameters are
accepted without error but currently do nothing, the ASCII face-neighbor
representation isn't implemented yet. Passing them silently produces a file
with no face-neighbor data, planned work, not yet scheduled.

## Format-specific classes

Constructor and a handful of format quirks; everything else is the shared
interface above.

```{eval-rst}
.. currentmodule:: tecio

.. autosummary::
   :toctree: writers

   TecplotSzlWriter
   TecplotPltWriter
   TecplotDatWriter
```

## Precision

Each writer's `precision` constructor argument means something different,
this is the one constructor parameter worth understanding up front rather
than discovering by trial and error.

| | SZL | PLT | DAT |
|---|---|---|---|
| Default | `None` (per-variable, automatic) | `DataType.DOUBLE` | `DataType.FLOAT` |
| Scope | Optional whole-file override; without it, each variable's on-disk type is inferred from its own array's dtype | Whole file, unconditionally; the classic API has no per-variable type at all, even integers are stored as the chosen float type | Whole file for the *float-vs-double* choice and the printed significant-digit count; integer-inferred variables always keep their own declared type (`DT=` in the header) regardless of `precision` |

DAT is the one format where an integer variable's memory footprint on read
genuinely depends on what was written, not on `precision`, e.g. a `count`
array written alongside `float64` coordinates under `precision=DataType.DOUBLE`
still declares (and reads back) as `LONGINT`, only the printed digit count
for the *floating-point* variables changes.
