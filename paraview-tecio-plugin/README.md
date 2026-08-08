# Tecplot 360 Reader for ParaView (TecIO-backed)

A ParaView reader plugin for Tecplot 360 data files -- SZL (`.szplt`), classic
binary PLT (`.plt` / `.bin`), and ASCII DAT (`.dat` / `.tec`) -- built on top of
the `tecio` Python bindings for the TecIO C/C++ library. It exists to cover
three gaps in ParaView's built-in Tecplot reader:

1. **SZL is unsupported** by ParaView's native reader; this plugin reads it.
2. Some ASCII zone-header fields make ParaView's native reader **error out**;
   `tecio` already parses them.
3. ParaView's native reader **drops Tecplot auxiliary ("aux") data**. This
   plugin maps it onto VTK `vtkFieldData`, which ParaView surfaces in the
   Information panel and Spreadsheet View.

See the docstring at the top of `TecplotTecioReader.py` for the full mapping
from Tecplot concepts (zones, aux data, value locations) to VTK/ParaView
concepts (multiblock blocks, field data, point/cell data), and for the
current known limitations.

## Requirements

- ParaView 5.7 or newer (for the `paraview.util.vtkAlgorithm` Python-plugin
  decorators used here).
- The `tecio` package importable from **the Python interpreter ParaView
  itself uses** -- this is not necessarily your system Python. ParaView
  bundles its own Python; `pip install`-ing `tecio` into your regular
  environment will not make it visible inside ParaView.
- The TecIO shared library (`libtecio.so` / `libtecio.dylib`) locatable by
  `tecio` at import time -- via a local Tecplot 360 installation, a project
  build directory, or the `TECIO_LIB` environment variable (see
  `tecio/_utils.py::get_tecio_lib`).

### Getting `tecio` into ParaView's Python

Pick whichever matches how you run ParaView:

- **`pvpython`** (ships its own `pip`):
  ```bash
  pvpython -m pip install -e /path/to/tecio-project
  ```
- **ParaView desktop app**: use its Python Shell (`View > Python Shell`) to
  confirm the interpreter, then install into that interpreter's `site-packages`
  the same way, or set `PYTHONPATH` before launching ParaView so it can find
  an existing `tecio` install:
  ```bash
  export PYTHONPATH=/path/to/tecio-project:$PYTHONPATH
  export TECIO_LIB=/path/to/libtecio.so   # if not auto-discoverable
  paraview
  ```

If `tecio` can't be imported, the plugin still loads (so ParaView can show a
clear error) -- opening a file will raise a `RuntimeError` explaining what's
missing, rather than a bare traceback.

## Installing the plugin

1. In ParaView: **Tools > Manage Plugins > Load New...**
2. Select `TecplotTecioReader.py`.
3. Check **Auto Load** if you want it available on every ParaView launch.
4. **File > Open**, pick a `.szplt` / `.plt` / `.bin` / `.dat` / `.tec` file --
   it will show up under the "Tecplot 360 Data Files" reader.

You can also load it from a Python script / `pvpython`:

```python
from paraview.simple import LoadPlugin, OpenDataFile
LoadPlugin("/path/to/TecplotTecioReader.py", remote=False, ns=globals())
reader = OpenDataFile("/path/to/case.szplt")
```

## What you get

- **Output**: one `vtkMultiBlockDataSet`, one block per zone, named after the
  zone's title (falling back to `"Zone <n>"`). `ORDERED` zones are
  `vtkStructuredGrid`; classic finite-element zones (line, triangle,
  quadrilateral, tetrahedron, brick) are `vtkUnstructuredGrid`.
- **Aux data -> Field Data**: dataset-level and variable-level aux data (the
  latter prefixed `"<variable>::<key>"`) land on the root multiblock's field
  data; zone-level aux data lands on that zone's own block. All visible via
  the Information panel or a Spreadsheet View set to "Field Data".
- **Arrays property**: checkboxes to skip loading variables you don't need
  (helps with large files -- unselected variables are never read from disk).
- **Zones property**: checkboxes to skip loading whole zones.
- **XArray / YArray / ZArray properties**: override which variables are used
  as point coordinates. Left on `"(auto)"`, the reader looks for common
  spellings of `x`/`y`/`z` and falls back to positional guessing (see the
  module docstring's "Coordinates" section) -- **check these if your geometry
  looks wrong**, especially for non-CFD-style variable naming.
- **Time support**: if any zones have a non-zero Strand ID, their distinct
  solution times drive ParaView's time toolbar. Zones with Strand ID `0` are
  treated as static (Tecplot's own convention) and included at every time
  step alongside whichever transient zone matches the requested time.

## Known limitations (first pass)

- `FEPOLYGON`, `FEPOLYHEDRON`, and `FEMIXED` zones are not converted; they're
  skipped with a printed warning (visible in ParaView's Output Messages) and
  the rest of the file still loads. **Exception**: `tecio`'s ASCII DAT reader
  currently raises immediately on encountering one of these zone types rather
  than skipping just that zone, so a `.dat` file containing one won't load at
  all through this reader until `tecio` supports it upstream.
- SOLUTION-only files (grid coordinates stored in a separate, unopened GRID
  file) have no coordinate data in-file; this reader doesn't yet stitch a
  paired GRID file back in.
- Face-based connectivity, geometry/text annotations, and custom labels are
  out of scope (as they are for `tecio` itself).

## Development notes

The plugin is a single, dependency-light file (`numpy` + `vtkmodules` +
`paraview.util.vtkAlgorithm` + `tecio`) so it's easy to drop into any
ParaView install. It caches the open `tecio` reader across pipeline callbacks
(keyed on file path + modification time) rather than reopening per callback,
since ASCII DAT files are parsed eagerly and in full at open time -- this
avoids doubling that cost on every UI refresh.
