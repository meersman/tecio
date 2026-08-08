"""ParaView reader plugin for Tecplot 360 data files, built on ``tecio``.

Wraps the ``tecio`` Python bindings for the TecIO C/C++ library (see the `Tecplot 360
data format guide
<https://tecplot.azureedge.net/products/360/current/360-data-format.html>`_) so that
ParaView can load SZL (``.szplt``), classic binary PLT (``.plt`` / ``.bin``), and ASCII
DAT (``.dat`` / ``.tec``) files through one reader. This exists to close three gaps in
ParaView's built-in Tecplot reader: it cannot read ``.szplt`` at all, it errors out on
some ASCII zone-header fields tecio already handles, and it drops Tecplot's auxiliary
("aux") data on the floor instead of surfacing it as VTK
:class:`~vtkmodules.vtkCommonDataModel.  vtkFieldData`.

Output structure
    Each file is read into one ``vtkMultiBlockDataSet``, one block per zone, named after
    the zone's title (or ``"Zone <n>"`` if untitled):

    * ``ORDERED`` zones become ``vtkStructuredGrid`` blocks.
    * Classic finite-element zones (line, triangle, quadrilateral, tetrahedron, brick)
      become ``vtkUnstructuredGrid`` blocks.
    * ``FEPOLYGON``, ``FEPOLYHEDRON``, and ``FEMIXED`` zones are not yet supported --
      see "Known limitations" below -- and are skipped with a printed warning rather
      than aborting the whole read.

    Auxiliary data has no first-class equivalent in VTK, so it is mapped onto the
    nearest matching :class:`vtkFieldData`, each item as a one-element
    ``vtkStringArray`` named after the aux-data key:

    * Dataset-level aux data -> the root ``vtkMultiBlockDataSet``'s field data.
    * Variable-level aux data -> also the root field data, keyed ``"<variable
      name>::<aux key>"`` (there is no per-array metadata slot that ParaView's UI
      surfaces to users).
    * Zone-level aux data -> that zone's own block's field data.

Coordinates
    Tecplot data files do not tag which variables are spatial coordinates (that is a
    plot/frame setting, not a file-format concept), so this reader guesses: it looks for
    variables named some spelling of ``x``/``y``/``z`` (case-insensitively,
    e.g. ``"X"``, ``"x-coordinate"``), and falls back to the first, second, and third
    dataset variables if nothing matches. The ``XArray``/``YArray``/``ZArray``
    properties override the guess per axis; set ``ZArray`` explicitly for 3-D data whose
    variables aren't named conventionally, since there both is no third-variable
    enforcement and no reliable auto-detected result to fall back on beyond the naming
    convention. A dataset with no Z match at all is treated as 2-D (``Z=0``).

Known limitations (first pass)
    * ``FEPOLYGON`` / ``FEPOLYHEDRON`` / ``FEMIXED`` zones are skipped. Note that
      ``tecio``'s ASCII DAT reader currently raises immediately on encountering one of
      these zone types (rather than skipping just that zone), so a ``.dat`` file
      containing one won't load at all through this reader until ``tecio`` supports it.
    * SOLUTION-only files (grid coordinates stored in a separate, unopened GRID file)
      have no coordinate data to build geometry from and are not handled specially; they
      will fail coordinate resolution unless the variables happen to still be present.
    * Face-based connectivity, custom labels, geometry/text annotations, and Tecplot's
      classic-PLT auxiliary record types outside dataset/zone/ variable aux data are out
      of scope for this reader.

Requirements
    The ``tecio`` package (and the TecIO shared library it binds to) must be importable
    from the same Python interpreter ParaView uses to run this plugin -- ParaView's
    bundled Python, not necessarily the system Python.  See the accompanying README for
    setup notes.
"""

from __future__ import annotations

import math
import os
import sys
import weakref
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from paraview.util.vtkAlgorithm import smdomain, smhint, smproperty, smproxy
from vtkmodules.util import numpy_support
from vtkmodules.util.vtkAlgorithm import VTKPythonAlgorithmBase
from vtkmodules.vtkCommonCore import vtkDataArraySelection, vtkPoints, vtkStringArray
from vtkmodules.vtkCommonDataModel import (
    VTK_HEXAHEDRON,
    VTK_LINE,
    VTK_QUAD,
    VTK_TETRA,
    VTK_TRIANGLE,
    vtkCellArray,
    vtkCompositeDataSet,
    vtkFieldData,
    vtkMultiBlockDataSet,
    vtkStructuredGrid,
    vtkUnstructuredGrid,
)

# ParaView's bundled Python (vtkpython) has its own site-packages. The environment
# variable takes precedence over the hardcoded fallback; edit the fallback (or just set
# the environment variable and leave this alone) to match your local install.
# e.g. "/Users/you/tecio-for-paraview"
_TECIO_FALLBACK_PATH = os.environ.get("TECIO_PYTHONPATH", "")
if _TECIO_FALLBACK_PATH and _TECIO_FALLBACK_PATH not in sys.path:
    sys.path.insert(0, _TECIO_FALLBACK_PATH)

try:
    import tecio
    from tecio.libtecio import ValueLocation, ZoneType

    _TECIO_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - environment-dependent  # noqa: BLE001
    tecio = None  # type: ignore[assignment]
    ValueLocation = ZoneType = None  # type: ignore[assignment]
    _TECIO_IMPORT_ERROR = exc

if TYPE_CHECKING:
    from tecio.dat import Read as _DatRead
    from tecio.plt import Read as _PltRead
    from tecio.szl import Read as _SzlRead

    _TecioReader = _DatRead | _PltRead | _SzlRead

# --------------------------------------------------------------------------------------
# Module-level constants and small helpers
# --------------------------------------------------------------------------------------

_EXTENSIONS = "szplt plt bin dat tec"
_FILE_DESCRIPTION = "Tecplot 360 Data Files"

# Maps a supported classic finite-element ZoneType to its VTK cell type. Populated
# lazily (rather than at class-definition time) so that a missing `tecio` import doesn't
# prevent this module from loading -- the class still needs to be importable so ParaView
# can register it and show a clear error only once a file is actually opened.
_VTK_CELL_TYPE: dict[Any, int] = (
    {
        ZoneType.FELINESEG: VTK_LINE,
        ZoneType.FETRIANGLE: VTK_TRIANGLE,
        ZoneType.FEQUADRILATERAL: VTK_QUAD,
        ZoneType.FETETRAHEDRON: VTK_TETRA,
        ZoneType.FEBRICK: VTK_HEXAHEDRON,
    }
    if tecio is not None
    else {}
)

# Conservative, exact-match (case-insensitive, trimmed) spellings of each coordinate
# axis. Deliberately not a substring match: "contains x" would also catch unrelated
# variables such as "X Velocity".
_AXIS_SYNONYMS: dict[str, frozenset[str]] = {
    "x": frozenset({
        "x",
        "x-coordinate",
        "x coordinate",
        "coordinate x",
        "coord-x",
        "coordx",
        "xcoord",
        "x_coord",
    }),
    "y": frozenset({
        "y",
        "y-coordinate",
        "y coordinate",
        "coordinate y",
        "coord-y",
        "coordy",
        "ycoord",
        "y_coord",
    }),
    "z": frozenset({
        "z",
        "z-coordinate",
        "z coordinate",
        "coordinate z",
        "coord-z",
        "coordz",
        "zcoord",
        "z_coord",
    }),
}


class _UnsupportedZoneError(Exception):
    """Raised internally when a zone's geometry can't be represented in VTK.

    Caught by :meth:`TecplotReader.RequestData`, which prints a warning and skips the
    offending zone rather than failing the whole read.
    """


def _modified_callback(target: VTKPythonAlgorithmBase):
    """Return a callback that calls ``target.Modified()`` while it's alive.

    Wires a :class:`vtkDataArraySelection`'s ``ModifiedEvent`` back to the owning reader
    so that toggling an array/zone checkbox in the ParaView UI correctly marks the
    pipeline as needing re-execution. A weak reference avoids keeping the reader alive
    solely because the observer closure refers to it.
    """
    ref = weakref.ref(target)

    def _callback(*_args: Any) -> None:
        obj = ref()
        if obj is not None:
            obj.Modified()

    return _callback


def _autodetect_axis(names: list[str], axis: str) -> str | None:
    """Return the variable name in *names* that looks like *axis*, or ``None``."""
    synonyms = _AXIS_SYNONYMS[axis]
    for name in names:
        if name.strip().lower() in synonyms:
            return name
    return None


def _add_string_field_data(
    field_data: vtkFieldData,
    items: Any,
    key_fn: Any = None,
) -> None:
    """Attach each ``(name, value)`` pair in *items* to *field_data*.

    Each pair becomes a one-element :class:`vtkStringArray`, which is the closest
    practical VTK equivalent to Tecplot's string-valued auxiliary data -- there is no
    first-class "aux data" concept in VTK, but field data is visible in ParaView's
    Information panel and Spreadsheet View.

    Args:
        field_data: Target field data (root dataset, block, etc.).
        items: Iterable of ``(key, value)`` string pairs, e.g. a dict's ``.items()``.
        key_fn: Optional ``key -> array name`` transform, e.g. to prefix variable-level
            aux data with the variable name.
    """
    for key, value in items:
        name = key_fn(key) if key_fn is not None else key
        arr = vtkStringArray()
        arr.SetName(name)
        arr.SetNumberOfValues(1)
        arr.SetValue(0, value)
        field_data.AddArray(arr)


# ======================================================================================
# Reader
# ======================================================================================


@smproxy.reader(
    name="TecplotTecioReader",
    label="Tecplot 360 Reader (TecIO)",
    extensions=_EXTENSIONS,
    file_description=_FILE_DESCRIPTION,
)
class TecplotReader(VTKPythonAlgorithmBase):
    """Reads Tecplot SZL/PLT/DAT files into a ``vtkMultiBlockDataSet``.

    One block per zone; see the module docstring for the full mapping, including how
    auxiliary data becomes field data and how coordinate variables are chosen.
    """

    def __init__(self) -> None:
        """Initialize from VTK object."""
        super().__init__(
            nInputPorts=0, nOutputPorts=1, outputType="vtkMultiBlockDataSet"
        )

        self._filename: str | None = None
        self._reader: _TecioReader | None = None
        self._reader_path: str | None = None
        self._reader_mtime: float | None = None

        self._x_override: str | None = None
        self._y_override: str | None = None
        self._z_override: str | None = None

        self._array_selection = vtkDataArraySelection()
        self._array_selection.AddObserver("ModifiedEvent", _modified_callback(self))

        # Keyed "<1-based zone index>: <title>" -> whether that zone/block loads.
        self._zone_selection = vtkDataArraySelection()
        self._zone_selection.AddObserver("ModifiedEvent", _modified_callback(self))
        self._zone_keys: dict[int, str] = {}

    def __del__(self) -> None:  # pragma: no cover
        """Best-effort cleanup."""
        try:
            self._close_reader()
        except Exception:  # noqa: BLE001 - never raise from __del__
            pass

    # -- File handle management -------------------------------------------------------

    def _close_reader(self) -> None:
        """Release the cached reader handle, if any."""
        if self._reader is not None:
            close = getattr(self._reader, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # noqa: BLE001 - closing must not fail the pipeline
                    pass
        self._reader = None
        self._reader_path = None
        self._reader_mtime = None

    def _get_reader(self) -> _TecioReader | None:
        """Return a cached, open ``tecio`` reader for the current file.

        Reopens only when the path or on-disk modification time changes, so the
        frequent, metadata-only pipeline callbacks ParaView issues (array lists, time
        values, ...) don't repeatedly pay the cost of reopening the file -- which
        matters most for ASCII DAT files, whose entire contents ``tecio`` parses eagerly
        at open time.

        Returns:
            An open ``tecio`` reader, or ``None`` if no file has been set.

        Raises:
            RuntimeError: If the ``tecio`` package could not be imported.
            FileNotFoundError: If the configured file can't be found/read.
        """
        if self._filename is None:
            return None
        if tecio is None:
            raise RuntimeError(
                "The 'tecio' package is not importable in ParaView's Python "
                f"environment ({_TECIO_IMPORT_ERROR!r}). Install tecio (and "
                "make sure the TecIO shared library can be located, e.g. via "
                "the TECIO_LIB environment variable) in the Python "
                "interpreter ParaView uses, then reload this plugin."
            )

        try:
            mtime = os.path.getmtime(self._filename)
        except OSError as exc:
            raise FileNotFoundError(
                f"Cannot read Tecplot file: {self._filename}"
            ) from exc

        if (
            self._reader is not None
            and self._reader_path == self._filename
            and self._reader_mtime == mtime
        ):
            return self._reader

        self._close_reader()
        self._reader = tecio.open(self._filename, "r")
        self._reader_path = self._filename
        self._reader_mtime = mtime
        self._sync_selections(self._reader)
        return self._reader

    def _sync_selections(self, reader: _TecioReader) -> None:
        """(Re)populate the array and zone selections for a newly opened file."""
        self._array_selection.RemoveAllArrays()
        for name in reader.variables:
            self._array_selection.AddArray(name)

        self._zone_selection.RemoveAllArrays()
        self._zone_keys.clear()
        for zone in reader.zone:
            title = zone.title or f"Zone {zone.zone_index}"
            key = f"{zone.zone_index}: {title}"
            self._zone_keys[zone.zone_index] = key
            self._zone_selection.AddArray(key)

    # -- FileName --------------------------------------------------------------------

    @smproperty.stringvector(name="FileName")
    @smdomain.filelist()
    @smhint.filechooser(extensions=_EXTENSIONS, file_description=_FILE_DESCRIPTION)
    def SetFileName(self, name: str) -> None:
        """Set the path to the Tecplot file.

        Accepts ``.szplt``/``.plt``/``.bin``/``.dat``/``.tec``.
        """
        name = name or None
        if self._filename != name:
            self._filename = name
            self._close_reader()
            self.Modified()

    # -- Array / zone selection --------------------------------------------------------

    @smproperty.dataarrayselection(name="Arrays")
    def GetDataArraySelection(self) -> vtkDataArraySelection:
        """Which dataset variables load as point/cell data arrays."""
        return self._array_selection

    @smproperty.dataarrayselection(name="Zones")
    def GetZoneSelection(self) -> vtkDataArraySelection:
        """Which zones load as blocks -- useful to skip zones in large files."""
        return self._zone_selection

    # -- Coordinate variable overrides ------------------------------------------------

    @smproperty.stringvector(name="VariableInfo", information_only="1")
    def GetVariableInfo(self) -> list[str]:
        """Choices offered by the X/Y/Z array dropdowns.

        Eg ``"(auto)"`` plus every variable.
        """
        try:
            reader = self._get_reader()
        except Exception:  # noqa: BLE001 - domain refresh must not raise into the UI
            reader = None
        names = list(reader.variables) if reader is not None else []
        return ["(auto)", *names]

    def _set_axis_override(self, axis: str, value: str) -> None:
        resolved = None if value in ("", "(auto)") else value
        attr = f"_{axis}_override"
        if getattr(self, attr) != resolved:
            setattr(self, attr, resolved)
            self.Modified()

    @smproperty.stringvector(
        name="XArray", number_of_elements="1", default_values=["(auto)"]
    )
    @smdomain.xml("""
        <StringListDomain name="list">
          <RequiredProperties>
            <Property name="VariableInfo" function="StringInfo"/>
          </RequiredProperties>
        </StringListDomain>
    """)
    def SetXArray(self, value: str) -> None:
        """Override auto-detection of the X-coordinate variable."""
        self._set_axis_override("x", value)

    @smproperty.stringvector(
        name="YArray", number_of_elements="1", default_values=["(auto)"]
    )
    @smdomain.xml("""
        <StringListDomain name="list">
          <RequiredProperties>
            <Property name="VariableInfo" function="StringInfo"/>
          </RequiredProperties>
        </StringListDomain>
    """)
    def SetYArray(self, value: str) -> None:
        """Override auto-detection of the Y-coordinate variable."""
        self._set_axis_override("y", value)

    @smproperty.stringvector(
        name="ZArray", number_of_elements="1", default_values=["(auto)"]
    )
    @smdomain.xml("""
        <StringListDomain name="list">
          <RequiredProperties>
            <Property name="VariableInfo" function="StringInfo"/>
          </RequiredProperties>
        </StringListDomain>
    """)
    def SetZArray(self, value: str) -> None:
        """Override auto-detection of the Z-coordinate variable (else 2-D: Z=0)."""
        self._set_axis_override("z", value)

    def _resolve_coordinate_names(
        self, reader: _TecioReader
    ) -> tuple[str | None, str | None, str | None]:
        """Pick the dataset variables to use as X, Y, and Z point coordinates.

        A manual override wins when set and still names a real dataset
        variable. Otherwise each axis is matched by name (see :func:`_autodetect_axis`);
        X and Y additionally fall back to the first and second dataset variables, since
        almost every Tecplot CFD file leads with its coordinates. Z has no positional
        fallback beyond the third variable and defaults to all-zero (2-D data) when
        nothing is found.
        """
        names = list(reader.variables)

        def override(axis: str) -> str | None:
            value = getattr(self, f"_{axis}_override")
            return value if value in names else None

        x = override("x") or _autodetect_axis(names, "x")
        y = override("y") or _autodetect_axis(names, "y")
        z = override("z") or _autodetect_axis(names, "z")

        if x is None and names:
            x = names[0]
        if y is None and len(names) > 1:
            y = names[1]
        if z is None and len(names) > 2 and names[2] not in (x, y):
            z = names[2]

        return x, y, z

    # -- Time -----------------------------------------------------------------------

    def _compute_timesteps(self, reader: _TecioReader) -> list[float]:
        """Sorted, distinct solution times across all non-static zones.

        Zones with ``strand_id == 0`` are the standard Tecplot convention for
        static/always-present geometry, so they're excluded from the time axis and
        instead included at every requested time (see :meth:`RequestData`).
        """
        times = {zone.solution_time for zone in reader.zone if zone.strand_id != 0}
        return sorted(times)

    @smproperty.doublevector(
        name="TimestepValues", information_only="1", si_class="vtkSITimeStepsProperty"
    )
    def GetTimestepValues(self) -> list[float]:
        """Publish the dataset's discrete solution times to ParaView's time UI."""
        try:
            reader = self._get_reader()
        except Exception:  # noqa: BLE001 - domain refresh must not raise into the UI
            return []
        return self._compute_timesteps(reader) if reader is not None else []

    def _get_update_time(self, out_info: Any) -> float | None:
        """Resolve the pipeline's requested time to one of the dataset's own times."""
        reader = self._reader
        if reader is None:
            return None
        timesteps = self._compute_timesteps(reader)
        if not timesteps:
            return None

        executive = self.GetExecutive()
        if out_info.Has(executive.UPDATE_TIME_STEP()):
            requested = out_info.Get(executive.UPDATE_TIME_STEP())
            result = timesteps[0]
            for t in timesteps:
                if t <= requested + 1e-9:
                    result = t
                else:
                    break
            return result
        return timesteps[0]

    # -- Pipeline: RequestInformation --------------------------------------------------

    def RequestInformation(self, request, inInfoVec, outInfoVec) -> int:
        """Publish the dataset's time steps, if any, before ``RequestData`` runs."""
        try:
            reader = self._get_reader()
        except Exception as exc:  # noqa: BLE001 - report and produce empty output
            print(f"[TecplotTecioReader] {exc}")
            return 1
        if reader is None:
            return 1

        executive = self.GetExecutive()
        out_info = outInfoVec.GetInformationObject(0)
        out_info.Remove(executive.TIME_STEPS())
        out_info.Remove(executive.TIME_RANGE())

        timesteps = self._compute_timesteps(reader)
        if timesteps:
            for t in timesteps:
                out_info.Append(executive.TIME_STEPS(), t)
            out_info.Append(executive.TIME_RANGE(), timesteps[0])
            out_info.Append(executive.TIME_RANGE(), timesteps[-1])
        return 1

    # -- Pipeline: RequestData ---------------------------------------------------------

    def RequestData(self, request, inInfoVec, outInfoVec) -> int:
        """Build the ``vtkMultiBlockDataSet`` output for the requested time step."""
        output = vtkMultiBlockDataSet.GetData(outInfoVec, 0)

        try:
            reader = self._get_reader()
        except Exception as exc:  # noqa: BLE001 - report and produce empty output
            print(f"[TecplotTecioReader] {exc}")
            return 1
        if reader is None:
            return 1

        requested_time = self._get_update_time(outInfoVec.GetInformationObject(0))
        x_name, y_name, z_name = self._resolve_coordinate_names(reader)

        _add_string_field_data(output.GetFieldData(), reader.auxdata.items())
        self._add_variable_aux_field_data(output.GetFieldData(), reader)

        block_index = 0
        for zone in reader.zone:
            key = self._zone_keys.get(zone.zone_index)
            if key is not None and not self._zone_selection.ArrayIsEnabled(key):
                continue
            if (
                requested_time is not None
                and zone.strand_id != 0
                and not math.isclose(zone.solution_time, requested_time, abs_tol=1e-9)
            ):
                continue

            try:
                dataset = self._build_zone_dataset(zone, x_name, y_name, z_name)
            except _UnsupportedZoneError as exc:
                print(
                    f"[TecplotTecioReader] Skipping zone {zone.zone_index} "
                    f"({zone.title!r}): {exc}"
                )
                continue
            except Exception as exc:  # noqa: BLE001 - one bad zone shouldn't fail the read
                print(
                    f"[TecplotTecioReader] Error reading zone {zone.zone_index} "
                    f"({zone.title!r}), skipping: {exc}"
                )
                continue

            output.SetBlock(block_index, dataset)
            name = zone.title or f"Zone {zone.zone_index}"
            output.GetMetaData(block_index).Set(vtkCompositeDataSet.NAME(), name)
            block_index += 1

        if requested_time is not None:
            output.GetInformation().Set(output.DATA_TIME_STEP(), requested_time)
        return 1

    def _add_variable_aux_field_data(
        self, field_data: vtkFieldData, reader: _TecioReader
    ) -> None:
        """Attach variable-level aux data to *field_data*, prefixed by variable name."""
        names = reader.variables
        for i in range(1, reader.num_vars + 1):
            aux = reader.get_var_auxdata(i)
            if not len(aux):
                continue
            var_name = names[i - 1]
            _add_string_field_data(
                field_data, aux.items(), key_fn=lambda k, v=var_name: f"{v}::{k}"
            )

    # -- Zone -> VTK dataset conversion ------------------------------------------------

    def _build_zone_dataset(
        self,
        zone: Any,
        x_name: str | None,
        y_name: str | None,
        z_name: str | None,
    ) -> vtkStructuredGrid | vtkUnstructuredGrid:
        """Convert one Tecplot zone into a VTK type.

        Structured/Unstructured correspond to``vtkStructuredGrid``/
        ``vtkUnstructuredGrid`` respectively.
        """
        if zone.zone_type == ZoneType.ORDERED:
            dataset: vtkStructuredGrid | vtkUnstructuredGrid = (
                self._build_structured_grid(zone, x_name, y_name, z_name)
            )
        else:
            cell_type = _VTK_CELL_TYPE.get(zone.zone_type)
            if cell_type is None:
                raise _UnsupportedZoneError(
                    f"zone type {zone.zone_type.name} is not yet supported by this "
                    "reader (only ORDERED and classic FE zones -- line, triangle, "
                    "quadrilateral, tetrahedron, brick -- are converted; "
                    "polygon/polyhedron/mixed zones are not)"
                )
            dataset = self._build_unstructured_grid(
                zone, cell_type, x_name, y_name, z_name
            )

        _add_string_field_data(dataset.GetFieldData(), zone.auxdata.items())
        self._add_variable_arrays(dataset, zone)
        return dataset

    def _build_structured_grid(
        self, zone: Any, x_name: str | None, y_name: str | None, z_name: str | None
    ) -> vtkStructuredGrid:
        """Build a ``vtkStructuredGrid`` for an ORDERED zone."""
        ni, nj, nk = zone.dimensions
        shape = (ni, nj, nk)
        x = self._fetch_ordered_coordinate(zone, x_name, shape)
        y = self._fetch_ordered_coordinate(zone, y_name, shape)
        z = self._fetch_ordered_coordinate(zone, z_name, shape)

        # Tecplot's IJK arrays come back Fortran-ordered (I fastest); raveling the same
        # way lines the flat point list up with VTK's own I-fastest point ordering, so
        # no per-point reindexing is needed.
        points = np.empty((ni * nj * nk, 3), dtype=np.float64)
        points[:, 0] = x.ravel(order="F")
        points[:, 1] = y.ravel(order="F")
        points[:, 2] = z.ravel(order="F")

        vtk_points = vtkPoints()
        vtk_points.SetData(numpy_support.numpy_to_vtk(points, deep=True))

        grid = vtkStructuredGrid()
        grid.SetDimensions(ni, nj, nk)
        grid.SetPoints(vtk_points)
        return grid

    @staticmethod
    def _fetch_ordered_coordinate(
        zone: Any, name: str | None, shape: tuple[int, int, int]
    ) -> npt.NDArray[np.float64]:
        """Return an ``(I, J, K)`` nodal coordinate array for *name*, or zeros."""
        if name is None:
            return np.zeros(shape, dtype=np.float64)
        values = zone.get_array(name)
        if values is None:
            raise _UnsupportedZoneError(
                f"coordinate variable '{name}' has no data in this zone (it is passive)"
            )
        if values.shape != shape:
            raise _UnsupportedZoneError(
                f"coordinate variable '{name}' has shape {values.shape}, expected "
                f"nodal shape {shape} -- is it cell-centered?"
            )
        return values.astype(np.float64, copy=False)

    def _build_unstructured_grid(
        self,
        zone: Any,
        cell_type: int,
        x_name: str | None,
        y_name: str | None,
        z_name: str | None,
    ) -> vtkUnstructuredGrid:
        """Build a ``vtkUnstructuredGrid`` for a classic finite-element zone."""
        n_nodes = zone.num_nodes
        x = self._fetch_flat_coordinate(zone, x_name, n_nodes)
        y = self._fetch_flat_coordinate(zone, y_name, n_nodes)
        z = self._fetch_flat_coordinate(zone, z_name, n_nodes)

        points = np.column_stack((x, y, z)).astype(np.float64, copy=False)
        vtk_points = vtkPoints()
        vtk_points.SetData(numpy_support.numpy_to_vtk(points, deep=True))

        node_map = zone.node_map
        if node_map is None or node_map.size == 0:
            raise _UnsupportedZoneError("zone has no connectivity (empty node map)")

        n_cells, nodes_per_cell = node_map.shape
        # tecio node maps are 1-based (Tecplot convention); VTK point ids are 0-based.
        connectivity = (node_map.astype(np.int64, copy=False) - 1).ravel()
        offsets = np.arange(
            0, (n_cells + 1) * nodes_per_cell, nodes_per_cell, dtype=np.int64
        )

        cell_array = vtkCellArray()
        cell_array.SetData(
            numpy_support.numpy_to_vtkIdTypeArray(offsets, deep=True),
            numpy_support.numpy_to_vtkIdTypeArray(connectivity, deep=True),
        )

        grid = vtkUnstructuredGrid()
        grid.SetPoints(vtk_points)
        grid.SetCells(cell_type, cell_array)
        return grid

    @staticmethod
    def _fetch_flat_coordinate(
        zone: Any, name: str | None, n_nodes: int
    ) -> npt.NDArray[np.float64]:
        """Return a flat, length-``n_nodes`` coordinate array for *name*, or zeros."""
        if name is None:
            return np.zeros(n_nodes, dtype=np.float64)
        values = zone.get_array(name)
        if values is None:
            raise _UnsupportedZoneError(
                f"coordinate variable '{name}' has no data in this zone (it is passive)"
            )
        if values.size != n_nodes:
            raise _UnsupportedZoneError(
                f"coordinate variable '{name}' has {values.size} values, expected "
                f"{n_nodes} nodal values"
            )
        return values.astype(np.float64, copy=False)

    def _add_variable_arrays(
        self, dataset: vtkStructuredGrid | vtkUnstructuredGrid, zone: Any
    ) -> None:
        """Attach every selected, active variable to point or cell data.

        Node-located (``NODAL``) variables become point data; cell-located
        (``CELL_CENTERED``) variables become cell data. A variable whose array length
        doesn't match the dataset's point/cell count is skipped rather than raising, so
        one bad variable doesn't take an otherwise-good zone down with it.
        """
        n_points = dataset.GetNumberOfPoints()
        n_cells = dataset.GetNumberOfCells()

        for var in zone.variable:
            if not self._array_selection.ArrayIsEnabled(var.name):
                continue
            values = var.values
            if values is None:  # passive, or genuinely no data
                continue

            flat = values.ravel(order="F") if values.ndim == 3 else np.ravel(values)
            vtk_array = numpy_support.numpy_to_vtk(
                np.ascontiguousarray(flat), deep=True
            )
            vtk_array.SetName(var.name)

            if var.value_location == ValueLocation.NODAL:
                if flat.size != n_points:
                    continue
                dataset.GetPointData().AddArray(vtk_array)
            else:
                if flat.size != n_cells:
                    continue
                dataset.GetCellData().AddArray(vtk_array)
