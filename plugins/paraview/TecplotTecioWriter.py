"""ParaView writer plugin for Tecplot 360 data files, built on ``tecio``.

The counterpart to ``TecplotTecioReader``: writes whatever is on the input port out as
a Tecplot SZL/PLT/DAT file, so a pipeline (or a filter's output) can be exported without
going through ParaView's own, more limited Tecplot writer. As with the reader, the
output format is chosen by the saved file's extension.

Output structure
    The input may be a single ``vtkDataSet`` or a ``vtkMultiBlockDataSet``; every leaf
    block becomes one Tecplot zone, named after the block's composite-data name (or
    ``"Zone <n>"`` if unnamed). Geometry is mapped per block:

    * ``vtkStructuredGrid``, ``vtkRectilinearGrid``, and ``vtkImageData`` become
      ``ORDERED`` (IJK) zones.
    * ``vtkPolyData`` and ``vtkUnstructuredGrid`` (and any other dataset exposing
      explicit points and cells) become classic finite-element zones -- line,
      triangle, quadrilateral, tetrahedron, or brick -- one per VTK cell type present.
      A block mixing cell types (e.g. a surface with both triangles and quads) is
      split into one zone per type, each zone keeping only the points its own cells
      reference.
    * Any other cell type (polygon, polyhedron, triangle strip, vertex, wedge,
      pyramid, ...) is not representable by a classic Tecplot zone and is skipped with
      a printed warning -- the rest of the block is still written.

Coordinates
    The first three dataset variables are always named ``x``, ``y``, ``z`` and
    carry each zone's point coordinates (``z`` is all-zero for 2-D geometry). The
    dataset aux items ``Common.XVar``/``YVar``/``ZVar`` are set to ``1``/``2``/``3``
    so this plugin's own reader (and Tecplot 360 itself) auto-detect them on reopen.

Variables
    Every point-data and cell-data array across every block contributes to one
    dataset-wide variable list (in first-seen order, after the three coordinates); a
    block missing a given array has that variable marked passive for its own zone(s).
    A multi-component array (a vector or tensor) has no scalar equivalent in classic
    Tecplot zones, so each component becomes its own scalar variable: 2- and
    3-component arrays use ``"<name>_X"``/``"_Y"``/``"_Z"`` suffixes, matching how
    ``TecplotTecioReader`` reassembles ``Common.UVar``/``VVar``/``WVar`` into one
    vector array; wider arrays fall back to ``"<name>_0"``, ``"<name>_1"``, etc. An
    array named exactly ``"Velocity"`` or ``"Momentum"`` additionally gets
    ``Common.UVar``/``VVar``/``WVar`` (and, for ``"Velocity"``,
    ``Common.VectorVarsAreVelocity``) pointing at its decomposed components, so it
    round-trips back into one vector array when reopened through this plugin's reader.

Auxiliary data
    The input's field data has no first-class Tecplot equivalent, so it is mapped the
    same way ``TecplotTecioReader`` reads it back, in reverse: the root dataset's
    field data becomes dataset-level aux data (a ``"<variable name>::<key>"`` array
    name is split back into variable-level aux for that variable, if the name
    resolves), and each block's own field data becomes that zone's aux data. Only the
    first value of a multi-valued field-data array is kept, since Tecplot aux items
    are single strings.

Known limitations (first pass)
    * No coordinate, connectivity, or variable sharing is attempted between zones.
      Zone-to-zone sharing (as used elsewhere in this project, e.g. ``tecmerge``'s grid
      sharing) would reduce output size for such pipelines, but is not implemented here.
    * This plugin writes one static snapshot of its input. It does not loop over the
      pipeline's time steps itself; ParaView's own "write a file per time step"
      animation/file-series mechanism, where supported by the installed ParaView
      version, is the way to export a transient pipeline through this writer.
    * Polygonal/polyhedral zones (``FEPOLYGON``/``FEPOLYHEDRON``) are not produced,
      because ``tecio``'s Write API does not yet support them; polygon, triangle-strip,
      vertex, wedge, and pyramid cells are skipped rather than approximated.
    * Points not referenced by any kept cell are dropped from their zone rather than
      written as isolated, cell-less nodes (classic Tecplot FE zones have no concept
      of an unused node).

Requirements
    The ``tecio`` package must be importable from the same Python interpreter ParaView
    uses to run this plugin.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
from paraview.util.vtkAlgorithm import smdomain, smproperty, smproxy
from vtkmodules.util import numpy_support
from vtkmodules.util.vtkAlgorithm import VTKPythonAlgorithmBase
from vtkmodules.vtkCommonCore import vtkStringArray
from vtkmodules.vtkCommonDataModel import (
    VTK_HEXAHEDRON,
    VTK_LINE,
    VTK_QUAD,
    VTK_TETRA,
    VTK_TRIANGLE,
    vtkCompositeDataSet,
    vtkDataObject,
    vtkDataSet,
    vtkFieldData,
    vtkImageData,
    vtkMultiBlockDataSet,
    vtkPointSet,
    vtkRectilinearGrid,
    vtkStructuredGrid,
)

# Same escape hatch as TecplotTecioReader: ParaView's bundled Python can't have tecio
# installed into its own write-protected site-packages, so point at a local install
# via TECIO_PYTHONPATH (e.g. "/Users/you/tecio-for-paraview") if needed.
_TECIO_FALLBACK_PATH = os.environ.get("TECIO_PYTHONPATH", "")
if _TECIO_FALLBACK_PATH and _TECIO_FALLBACK_PATH not in sys.path:
    sys.path.insert(0, _TECIO_FALLBACK_PATH)

try:
    import tecio
    from tecio import FileType, ValueLocation, ZoneType

    _TECIO_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - environment-dependent  # noqa: BLE001
    tecio = None  # ty: ignore[invalid-assignment]
    FileType = ValueLocation = ZoneType = None  # ty: ignore[invalid-assignment]
    _TECIO_IMPORT_ERROR = exc

# --------------------------------------------------------------------------------------
# Module-level constants and small helpers
# --------------------------------------------------------------------------------------

_EXTENSIONS = "szplt plt bin dat tec"
_FILE_DESCRIPTION = "Tecplot 360 Data Files"

# Maps a VTK cell type to the classic Tecplot zone type it becomes. Only
# fixed-size, single-shape cell types are representable; anything else is skipped.
_ZONE_TYPE_FOR_VTK_CELL: dict[int, Any] = (
    {
        VTK_LINE: ZoneType.FELINESEG,
        VTK_TRIANGLE: ZoneType.FETRIANGLE,
        VTK_QUAD: ZoneType.FEQUADRILATERAL,
        VTK_TETRA: ZoneType.FETETRAHEDRON,
        VTK_HEXAHEDRON: ZoneType.FEBRICK,
    }
    if tecio is not None
    else {}
)

_COMPONENT_SUFFIXES: dict[int, tuple[str, ...]] = {2: ("X", "Y"), 3: ("X", "Y", "Z")}

# Arrays with one of these names are additionally advertised as a Common.*Var vector,
# matching how TecplotTecioReader names the vector it assembles on read.
_VECTOR_ARRAY_NAMES = frozenset({"Velocity", "Momentum"})


def _component_name(base_name: str, num_components: int, component: int) -> str:
    """Return the scalar variable name for one component of a multi-component array.

    Classic Tecplot zones only carry scalar variables, so each component of a vector
    or tensor array becomes its own variable. 2- and 3-component arrays use the
    conventional ``_X``/``_Y``/``_Z`` suffixes; wider arrays fall back to a numeric
    ``_<i>`` suffix.
    """
    if num_components == 1:
        return base_name
    suffixes = _COMPONENT_SUFFIXES.get(num_components)
    suffix = suffixes[component] if suffixes is not None else str(component)
    return f"{base_name}_{suffix}"


def _coerce_field_value(arr: Any) -> str | None:
    """Return the first value of a one-column field-data array, as a string.

    Mirrors ``TecplotTecioReader``'s ``_coerce_aux_value`` in reverse: a
    :class:`vtkStringArray` yields its own string, a numeric array yields ``int`` or
    ``float`` formatting depending on its dtype. Returns ``None`` for an empty array.
    """
    if isinstance(arr, vtkStringArray):
        return arr.GetValue(0) if arr.GetNumberOfValues() >= 1 else None

    values = numpy_support.vtk_to_numpy(arr)
    if values.size == 0:
        return None
    value = values.reshape(values.shape[0], -1)[0, 0]
    if np.issubdtype(values.dtype, np.integer):
        return str(int(value))
    return str(float(value))


def _field_data_to_aux(field_data: vtkFieldData) -> dict[str, str]:
    """Convert a ``vtkFieldData``'s arrays into a Tecplot-style ``{name: value}`` dict.

    Only the first value of each array is kept, since Tecplot aux items are single
    strings; an unnamed array is skipped.
    """
    aux: dict[str, str] = {}
    for i in range(field_data.GetNumberOfArrays()):
        arr = field_data.GetAbstractArray(i)
        name = arr.GetName()
        if not name:
            continue
        value = _coerce_field_value(arr)
        if value is not None:
            aux[name] = value
    return aux


def _split_variable_aux(
    root_aux: dict[str, str], var_names: list[str]
) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    """Split root field data's aux dict into dataset-level and per-variable aux.

    ``TecplotTecioReader`` writes variable-level aux back to the root field data as
    ``"<variable name>::<key>"``; this reverses that only for names whose prefix
    matches a real variable, so ordinary dataset aux (which may happen to contain
    "::") is left alone.
    """
    dataset_aux: dict[str, str] = {}
    var_aux: dict[str, dict[str, str]] = {}
    var_name_set = set(var_names)
    for key, value in root_aux.items():
        prefix, sep, rest = key.partition("::")
        if sep and prefix in var_name_set:
            var_aux.setdefault(prefix, {})[rest] = value
        else:
            dataset_aux[key] = value
    return dataset_aux, var_aux


# --------------------------------------------------------------------------------------
# Per-zone intermediate representation
# --------------------------------------------------------------------------------------


@dataclass
class _ZoneBuild:
    """Everything needed to emit one Tecplot zone via the ``tecio`` Write API."""

    title: str
    zone_type: Any | None  # ZoneType, or None for an ORDERED zone
    dims: tuple[int, int, int] | None  # (imax, jmax, kmax), ORDERED zones only
    node_map: npt.NDArray[np.int64] | None  # FE zones only
    arrays: dict[str, tuple[npt.NDArray, Any]] = field(default_factory=dict)
    aux: dict[str, str] = field(default_factory=dict)


def _decompose_array(arr: Any, flatten: Any) -> list[tuple[str, npt.NDArray]] | None:
    """Return ``[(component_name, flat_values), ...]`` for one VTK data array.

    *flatten* converts the array's natural VTK layout into the flat, per-node or
    per-cell order the caller needs (a no-op ``vtk_to_numpy`` for unstructured data,
    an IJK reshape for structured data). Returns ``None`` for an unnamed array.
    """
    name = arr.GetName()
    if not name:
        return None
    values = numpy_support.vtk_to_numpy(arr)
    num_components = 1 if values.ndim == 1 else values.shape[1]
    return [
        (
            _component_name(name, num_components, c),
            flatten(values if num_components == 1 else values[:, c]),
        )
        for c in range(num_components)
    ]


def _structured_coordinates(
    dataset: vtkImageData | vtkRectilinearGrid | vtkStructuredGrid,
    dims: tuple[int, int, int],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return per-axis ``(ni, nj, nk)`` nodal coordinate arrays for a structured block."""
    ni, nj, nk = dims

    if isinstance(dataset, vtkStructuredGrid):
        points = numpy_support.vtk_to_numpy(dataset.GetPoints().GetData())
        x = points[:, 0].reshape(dims, order="F")
        y = points[:, 1].reshape(dims, order="F")
        z = points[:, 2].reshape(dims, order="F")
        return (
            x.astype(np.float64, copy=False),
            y.astype(np.float64, copy=False),
            z.astype(np.float64, copy=False),
        )

    if isinstance(dataset, vtkRectilinearGrid):
        xs = numpy_support.vtk_to_numpy(dataset.GetXCoordinates())
        ys = numpy_support.vtk_to_numpy(dataset.GetYCoordinates())
        zs = numpy_support.vtk_to_numpy(dataset.GetZCoordinates())
    else:  # vtkImageData
        ox, oy, oz = dataset.GetOrigin()
        sx, sy, sz = dataset.GetSpacing()
        xs = ox + sx * np.arange(ni, dtype=np.float64)
        ys = oy + sy * np.arange(nj, dtype=np.float64)
        zs = oz + sz * np.arange(nk, dtype=np.float64)

    x, y, z = np.meshgrid(xs, ys, zs, indexing="ij")
    return (
        x.astype(np.float64, copy=False),
        y.astype(np.float64, copy=False),
        z.astype(np.float64, copy=False),
    )


def _structured_dims(
    dataset: vtkImageData | vtkRectilinearGrid | vtkStructuredGrid,
) -> tuple[int, int, int]:
    """Return ``(imax, jmax, kmax)``, via the output-argument overload.

    Some VTK Python binding versions only expose ``GetDimensions(dims: list)`` (no
    zero-argument, tuple-returning overload) for these classes, so the output-argument
    form is used unconditionally for portability.
    """
    dims = [0, 0, 0]
    dataset.GetDimensions(dims)
    return (dims[0], dims[1], dims[2])


def _build_ordered_zone(
    title: str, dataset: vtkImageData | vtkRectilinearGrid | vtkStructuredGrid
) -> _ZoneBuild:
    """Build an ORDERED zone from a structured VTK dataset."""
    dims = _structured_dims(dataset)
    x, y, z = _structured_coordinates(dataset, dims)
    cell_dims = tuple(max(n - 1, 1) for n in dims)

    zone = _ZoneBuild(title=title, zone_type=None, dims=dims, node_map=None)
    zone.arrays["x"] = (x, ValueLocation.NODAL)
    zone.arrays["y"] = (y, ValueLocation.NODAL)
    zone.arrays["z"] = (z, ValueLocation.NODAL)

    def nodal(values: npt.NDArray) -> npt.NDArray:
        return values.reshape(dims, order="F")

    def cell_centered(values: npt.NDArray) -> npt.NDArray:
        return values.reshape(cell_dims, order="F")

    point_data, cell_data = dataset.GetPointData(), dataset.GetCellData()
    for i in range(point_data.GetNumberOfArrays()):
        for name, values in _decompose_array(point_data.GetArray(i), nodal) or []:
            zone.arrays[name] = (values, ValueLocation.NODAL)
    for i in range(cell_data.GetNumberOfArrays()):
        for name, values in (
            _decompose_array(cell_data.GetArray(i), cell_centered) or []
        ):
            zone.arrays[name] = (values, ValueLocation.CELL_CENTERED)

    zone.aux = _field_data_to_aux(dataset.GetFieldData())
    return zone


def _group_cells_by_type(
    dataset: vtkDataSet,
) -> dict[int, tuple[list[int], list[tuple[int, ...]]]]:
    """Bucket every cell by VTK cell type -> ``(cell_indices, node_id_rows)``.

    ``cell_indices`` are 0-based indices into the dataset's own cell numbering (for
    subsetting cell-data arrays); ``node_id_rows`` are tuples of 0-based *global*
    point ids (for subsetting point-data arrays and building the node map).
    """
    buckets: dict[int, tuple[list[int], list[tuple[int, ...]]]] = {}
    for i in range(dataset.GetNumberOfCells()):
        cell = dataset.GetCell(i)
        ids = cell.GetPointIds()
        row = tuple(ids.GetId(j) for j in range(ids.GetNumberOfIds()))
        cell_indices, node_rows = buckets.setdefault(cell.GetCellType(), ([], []))
        cell_indices.append(i)
        node_rows.append(row)
    return buckets


def _build_fe_zones(title: str, dataset: vtkPointSet) -> list[_ZoneBuild]:
    """Build one classic FE zone per supported VTK cell type found in *dataset*.

    A block containing only one supported cell type yields a single zone named
    *title*; a block mixing types (e.g. triangles and quads on one surface) yields
    one zone per type, suffixed with the Tecplot zone type name so each is
    identifiable. Unsupported cell types (polygon, triangle strip, vertex, wedge,
    pyramid, ...) are skipped with a printed warning.
    """
    buckets = _group_cells_by_type(dataset)
    points = numpy_support.vtk_to_numpy(dataset.GetPoints().GetData())
    point_data, cell_data = dataset.GetPointData(), dataset.GetCellData()
    field_aux = _field_data_to_aux(dataset.GetFieldData())

    supported = {
        vt: rows for vt, rows in buckets.items() if vt in _ZONE_TYPE_FOR_VTK_CELL
    }
    unsupported = set(buckets) - set(supported)
    for vt in sorted(unsupported):
        print(
            f"[TecplotTecioWriter] '{title}': skipping {len(buckets[vt][0])} cell(s) "
            f"of unsupported VTK cell type {vt} (only line, triangle, quadrilateral, "
            "tetrahedron, and hexahedron cells become Tecplot FE zones)."
        )

    zones: list[_ZoneBuild] = []
    multiple = len(supported) > 1
    for vtk_cell_type, (cell_indices, node_rows) in supported.items():
        zone_type = _ZONE_TYPE_FOR_VTK_CELL[vtk_cell_type]
        zone_title = f"{title} ({zone_type.name})" if multiple else title

        global_ids = np.asarray(node_rows, dtype=np.int64)
        unique_global, inverse = np.unique(global_ids, return_inverse=True)
        node_map = inverse.reshape(global_ids.shape)
        cell_idx = np.asarray(cell_indices, dtype=np.int64)

        zone = _ZoneBuild(
            title=zone_title, zone_type=zone_type, dims=None, node_map=node_map
        )
        local_points = points[unique_global]
        zone.arrays["x"] = (local_points[:, 0].copy(), ValueLocation.NODAL)
        zone.arrays["y"] = (local_points[:, 1].copy(), ValueLocation.NODAL)
        zone.arrays["z"] = (local_points[:, 2].copy(), ValueLocation.NODAL)

        def nodal(
            values: npt.NDArray, _ids: npt.NDArray = unique_global
        ) -> npt.NDArray:
            return values[_ids]

        def cell_centered(
            values: npt.NDArray, _idx: npt.NDArray = cell_idx
        ) -> npt.NDArray:
            return values[_idx]

        for i in range(point_data.GetNumberOfArrays()):
            for name, values in _decompose_array(point_data.GetArray(i), nodal) or []:
                zone.arrays[name] = (values, ValueLocation.NODAL)
        for i in range(cell_data.GetNumberOfArrays()):
            for name, values in (
                _decompose_array(cell_data.GetArray(i), cell_centered) or []
            ):
                zone.arrays[name] = (values, ValueLocation.CELL_CENTERED)

        zone.aux = dict(field_aux)
        zones.append(zone)
    return zones


def _build_zones(name: str, dataset: vtkDataSet) -> list[_ZoneBuild]:
    """Build the Tecplot zone(s) for one input block, or ``[]`` if unsupported."""
    if isinstance(dataset, (vtkImageData, vtkRectilinearGrid, vtkStructuredGrid)):
        return [_build_ordered_zone(name, dataset)]
    if isinstance(dataset, vtkPointSet):
        return _build_fe_zones(name, dataset)
    print(
        f"[TecplotTecioWriter] '{name}': dataset type {type(dataset).__name__} is not "
        "supported (expected a structured grid, polydata, or unstructured grid); "
        "skipping."
    )
    return []


def _iter_blocks(data: vtkDataObject) -> list[tuple[str, vtkDataSet]]:
    """Return ``(title, dataset)`` for every non-empty leaf block of *data*.

    *data* may be a plain ``vtkDataSet`` (treated as a single block) or a
    (possibly nested) ``vtkMultiBlockDataSet``, whose leaves are visited via a
    composite-data iterator and named after their composite-data name, or
    ``"Zone <n>"`` if unnamed.
    """
    if isinstance(data, vtkMultiBlockDataSet):
        blocks: list[tuple[str, vtkDataSet]] = []
        it = data.NewIterator()
        it.UnRegister(None)
        it.VisitOnlyLeavesOn()
        it.InitTraversal()
        n = 0
        while not it.IsDoneWithTraversal():
            block = it.GetCurrentDataObject()
            n += 1
            if isinstance(block, vtkDataSet) and block.GetNumberOfPoints() > 0:
                meta = it.GetCurrentMetaData()
                name = (
                    meta.Get(vtkCompositeDataSet.NAME())
                    if meta is not None and meta.Has(vtkCompositeDataSet.NAME())
                    else f"Zone {n}"
                )
                blocks.append((name, block))
            it.GoToNextItem()
        return blocks
    if isinstance(data, vtkDataSet) and data.GetNumberOfPoints() > 0:
        return [("Zone 1", data)]
    return []


def _assemble_global_variables(
    zones: list[_ZoneBuild],
) -> list[str]:
    """Return the dataset-wide variable list: ``x``, ``y``, ``z``, then first-seen order."""
    names = ["x", "y", "z"]
    seen = set(names)
    for zone in zones:
        for name in zone.arrays:
            if name not in seen:
                seen.add(name)
                names.append(name)
    return names


def _add_vector_aux(dataset_aux: dict[str, str], var_names: list[str]) -> None:
    """Advertise ``Common.UVar``/``VVar``/``WVar`` for a ``Velocity``/``Momentum`` array.

    Lets this plugin's own reader (and Tecplot 360) reassemble the decomposed ``_X``/
    ``_Y``/``_Z`` components back into one vector array on reopen. Silently does
    nothing if no array with a recognized name was decomposed.
    """
    name_to_index = {name: i for i, name in enumerate(var_names)}
    for array_name in _VECTOR_ARRAY_NAMES:
        components = [f"{array_name}_{axis}" for axis in ("X", "Y", "Z")]
        if components[0] not in name_to_index:
            continue
        for key, comp in zip(
            ("Common.UVar", "Common.VVar", "Common.WVar"), components, strict=False
        ):
            if comp in name_to_index:
                dataset_aux[key] = str(name_to_index[comp] + 1)  # 1-based, per format
        if array_name == "Velocity":
            dataset_aux["Common.VectorVarsAreVelocity"] = "1"


# ======================================================================================
# Writer
# ======================================================================================


@smproxy.writer(
    name="TecplotTecioWriter",
    label="Tecplot 360 Writer (TecIO)",
    extensions=_EXTENSIONS,
    file_description=_FILE_DESCRIPTION,
    support_reload=False,
)
@smproperty.input(name="Input", port_index=0)
@smdomain.datatype(
    dataTypes=["vtkDataSet", "vtkMultiBlockDataSet"], composite_data_supported=True
)
class TecplotWriter(VTKPythonAlgorithmBase):
    """Writes the input's current data as one Tecplot SZL/PLT/DAT file.

    One zone per leaf block (or a single zone for a non-composite input); see the
    module docstring for the full mapping, including variable decomposition and
    auxiliary data.
    """

    def __init__(self) -> None:
        """Initialize from VTK object."""
        super().__init__(nInputPorts=1, nOutputPorts=0)
        self._filename: str | None = None

    def FillInputPortInformation(self, port: int, info: Any) -> int:
        """Accept any ``vtkDataObject`` so both plain and composite inputs connect."""
        info.Set(self.INPUT_REQUIRED_DATA_TYPE(), "vtkDataObject")
        return 1

    # -- FileName ------------------------------------------------------------------

    @smproperty.stringvector(name="FileName", panel_visibility="never")
    @smdomain.filelist()
    def SetFileName(self, name: str | None) -> None:
        """Set the output path. Accepts ``.szplt``/``.plt``/``.bin``/``.dat``/``.tec``."""
        name = name or None
        if self._filename != name:
            self._filename = name
            self.Modified()

    # -- Pipeline: RequestData ------------------------------------------------------

    def RequestData(self, request: Any, inInfoVec: Any, outInfoVec: Any) -> int:
        """Write the current input to :attr:`_filename`."""
        if tecio is None:
            print(
                "[TecplotTecioWriter] The 'tecio' package is not importable in "
                f"ParaView's Python environment ({_TECIO_IMPORT_ERROR!r}). Install "
                "tecio (and make sure the TecIO shared library can be located, e.g. "
                "via the TECIO_LIB environment variable) in the Python interpreter "
                "ParaView uses, then reload this plugin."
            )
            return 1
        if not self._filename:
            print("[TecplotTecioWriter] No output FileName set; nothing written.")
            return 1

        input0 = self.GetInputData(inInfoVec, 0, 0)
        blocks = _iter_blocks(input0)
        if not blocks:
            print(
                f"[TecplotTecioWriter] '{self._filename}': input has no data to write."
            )
            return 1

        zones: list[_ZoneBuild] = []
        for name, dataset in blocks:
            zones.extend(_build_zones(name, dataset))
        if not zones:
            print(
                f"[TecplotTecioWriter] '{self._filename}': no zone could be built from "
                "the input (see warnings above)."
            )
            return 1

        var_names = _assemble_global_variables(zones)
        root_dataset_aux, root_var_aux = _split_variable_aux(
            _field_data_to_aux(input0.GetFieldData()), var_names
        )
        root_dataset_aux.setdefault("Common.XVar", "1")
        root_dataset_aux.setdefault("Common.YVar", "2")
        root_dataset_aux.setdefault("Common.ZVar", "3")
        _add_vector_aux(root_dataset_aux, var_names)

        try:
            with tecio.open(
                self._filename,
                "w",
                title="ParaView export",
                variables=var_names,
                file_type=FileType.FULL,
            ) as writer:
                if root_dataset_aux:
                    writer.add_auxdataset_dict(root_dataset_aux)
                if root_var_aux:
                    name_to_index = {name: i for i, name in enumerate(var_names)}
                    writer.add_auxvar_dict({
                        name_to_index[name]: aux
                        for name, aux in root_var_aux.items()
                        if name in name_to_index
                    })

                for zone in zones:
                    passive_vars = {n for n in var_names if n not in zone.arrays}
                    data = [zone.arrays[n][0] for n in var_names if n in zone.arrays]
                    value_locations = [
                        zone.arrays[n][1] for n in var_names if n in zone.arrays
                    ]
                    common_kw: dict[str, Any] = {
                        "title": zone.title,
                        "value_locations": value_locations,
                        "passive_vars": passive_vars,
                        "aux": zone.aux or None,
                    }
                    if zone.zone_type is None:
                        writer.write_ordered_zone(data=data, **common_kw)
                    else:
                        writer.write_fe_zone(
                            data=data,
                            zone_type=zone.zone_type,
                            node_map=zone.node_map,
                            **common_kw,
                        )
        except Exception as exc:  # noqa: BLE001 - report and leave no half-written file
            print(f"[TecplotTecioWriter] Error writing '{self._filename}': {exc}")
            try:
                os.unlink(self._filename)
            except OSError:
                pass
            return 1

        return 1
