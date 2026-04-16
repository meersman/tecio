r"""
:mod:`dat`: Tecplot ASCII DAT file writer
=========================================

This module provides :class:`Write`, a context-manager-based writer for
Tecplot 360 ASCII data files (``.dat`` / ``.tec``).  The interface mirrors
:class:`szl.Write` and :class:`plt.Write` exactly so that calling code can
swap output formats by changing only the file extension::

    # SZL binary
    with tecio.open("result.szplt", "w", title="Demo",
                    variables=["X", "Y", "P"]) as w:
        w.write_ijk_zone(data=[x, y, p], title="Zone 1")

    # ASCII – identical call-site
    with tecio.open("result.dat", "w", title="Demo",
                    variables=["X", "Y", "P"]) as w:
        w.write_ijk_zone(data=[x, y, p], title="Zone 1")

Lazy-open support
-----------------
Like the binary writers, ``variables`` may be omitted from ``__init__``.
The file header is then deferred until the first call to
:meth:`write_ijk_zone` or :meth:`write_fe_zone`, which must supply
``variables``.  Dataset- and variable-level auxiliary data set via
:attr:`auxdataset` / :attr:`auxvar` are buffered and flushed automatically
at that point.

File format written
-------------------
::

    TITLE     = "<title>"
    [FILETYPE = GRID | SOLUTION]        (omitted when FULL)
    VARIABLES = "v1" "v2" ...
    DATASETAUXDATA <name> = "<value>"   (zero or more)
    VARAUXDATA <1-based> <name> = "<value>"  (zero or more)
    ZONE T="<title>", ZONETYPE=Ordered, I=i, J=j, K=k,
         DATAPACKING=BLOCK, STRANDID=s, SOLUTIONTIME=t
         [VARLOCATION=([n]=CELLCENTERED)]
         [PASSIVEVARLIST=[i,j,...]]
         [VARSHARELIST=([i=z,...])]
    <var 0 data block>
    <var 1 data block>
    ...

Format specification reference
-------------------------------
Tecplot 360 Data Format Guide 2025 R2, "ASCII Data" chapter.

Supported zone types
--------------------
* ``ORDERED``
* ``FELINESEG``, ``FETRIANGLE``, ``FEQUADRILATERAL``
* ``FETETRAHEDRON``, ``FEBRICK``

``FEPOLYGON`` and ``FEPOLYHEDRON`` raise :exc:`NotImplementedError`
(face-based connectivity is not yet implemented).
"""

from __future__ import annotations

# Standard library
import io
from collections.abc import Sequence
from typing import Any

# Third-party
import numpy as np
import numpy.typing as npt

# ---------------------------------------------------------------------------
# Local enums — imported from the project's libtecio bindings.
# A lightweight fallback is provided so the module can be imported and tested
# in isolation without a Tecplot installation.
# ---------------------------------------------------------------------------
try:
    from .libtecio import (
        DataType,
        FaceNeighborMode,
        FileType,
        ValueLocation,
        ZoneType,
    )
except ImportError:
    # Stand-alone / unit-test fallback
    from enum import IntEnum  # type: ignore[assignment]

    class FileType(IntEnum):  # type: ignore[no-redef]
        FULL = 0
        GRID = 1
        SOLUTION = 2

    class ZoneType(IntEnum):  # type: ignore[no-redef]
        ORDERED = 0
        FELINESEG = 1
        FETRIANGLE = 2
        FEQUADRILATERAL = 3
        FETETRAHEDRON = 4
        FEBRICK = 5
        FEPOLYGON = 6
        FEPOLYHEDRON = 7

    class ValueLocation(IntEnum):  # type: ignore[no-redef]
        NODAL = 0
        CELL_CENTERED = 1

    class DataType(IntEnum):  # type: ignore[no-redef]
        FLOAT = 1
        DOUBLE = 2
        INT32 = 3
        INT16 = 4
        BYTE = 5

    class FaceNeighborMode(IntEnum):  # type: ignore[no-redef]
        LOCAL_ONE_TO_ONE = 0


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: FE zone types fully supported by :meth:`Write.write_fe_zone`.
_FE_SIMPLE: frozenset[ZoneType] = frozenset({
    ZoneType.FELINESEG,
    ZoneType.FETRIANGLE,
    ZoneType.FEQUADRILATERAL,
    ZoneType.FETETRAHEDRON,
    ZoneType.FEBRICK,
})

#: Zone types whose connectivity is face-based (not yet supported).
_FE_POLY: frozenset[ZoneType] = frozenset({
    ZoneType.FEPOLYGON,
    ZoneType.FEPOLYHEDRON,
})

#: Nodes per element for each simple FE type.
_NODES_PER_ELEM: dict[ZoneType, int] = {
    ZoneType.FELINESEG: 2,
    ZoneType.FETRIANGLE: 3,
    ZoneType.FEQUADRILATERAL: 4,
    ZoneType.FETETRAHEDRON: 4,
    ZoneType.FEBRICK: 8,
}

#: ASCII keyword for each ZoneType.
_ZONETYPE_STR: dict[ZoneType, str] = {
    ZoneType.ORDERED: "Ordered",
    ZoneType.FELINESEG: "FELineSeg",
    ZoneType.FETRIANGLE: "FETriangle",
    ZoneType.FEQUADRILATERAL: "FEQuadrilateral",
    ZoneType.FETETRAHEDRON: "FETetrahedron",
    ZoneType.FEBRICK: "FEBrick",
    ZoneType.FEPOLYGON: "FEPolygon",
    ZoneType.FEPOLYHEDRON: "FEPolyhedron",
}

#: ASCII keyword for each FileType (FULL intentionally omitted from output).
_FILETYPE_STR: dict[FileType, str] = {
    FileType.GRID: "GRID",
    FileType.SOLUTION: "SOLUTION",
}

#: DataType → NumPy dtype string used for casting before formatting.
_DT_TO_DTYPE: dict[DataType, str] = {
    DataType.FLOAT: "f4",
    DataType.DOUBLE: "f8",
    DataType.INT32: "i4",
    DataType.INT16: "i2",
    DataType.BYTE: "u1",
}

#: Number of values written per line for variable data.
_VALUES_PER_LINE: int = 5


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _quote(s: str) -> str:
    """Wrap *s* in double-quotes, escaping embedded double-quotes."""
    return '"' + str(s).replace('"', '\\"') + '"'


def _infer_data_type(arr: npt.NDArray) -> DataType:
    """Return the most appropriate :class:`DataType` for *arr*'s dtype."""
    dt = arr.dtype
    if dt.kind == "f":
        return DataType.DOUBLE if dt.itemsize >= 8 else DataType.FLOAT
    if dt.kind in ("i", "u"):
        if dt.itemsize >= 4:
            return DataType.INT32
        if dt.itemsize == 2:
            return DataType.INT16
        return DataType.BYTE
    return DataType.FLOAT


def _write_array(fp: io.TextIOWrapper, arr: npt.NDArray, float_fmt: str) -> None:
    """Write a flat 1-D array to *fp*, :data:`_VALUES_PER_LINE` values per line.

    Floating-point arrays are formatted with *float_fmt*; integer arrays use
    plain ``%d``-style rendering.  A trailing newline is always written.
    """
    flat = np.asarray(arr).ravel()
    n = flat.size
    if n == 0:
        fp.write("\n")
        return

    is_float = np.issubdtype(flat.dtype, np.floating)
    vpl = _VALUES_PER_LINE

    for start in range(0, n, vpl):
        chunk = flat[start: start + vpl]
        if is_float:
            line = "\t".join(format(float(v), float_fmt) for v in chunk)
        else:
            line = "\t".join(str(int(v)) for v in chunk)
        fp.write(line + "\n")


# ---------------------------------------------------------------------------
# Write class
# ---------------------------------------------------------------------------

class Write:
    r"""Context-manager writer for Tecplot 360 ASCII (``.dat``) files.

    The public interface is identical to :class:`szl.Write` and
    :class:`plt.Write`.

    Parameters
    ----------
    path:
        Destination file path.
    title:
        Dataset title written to the file header.  Defaults to
        ``"untitled"``.
    variables:
        Ordered list of variable name strings.  When *None* the file
        header is deferred until the first zone-writing call (lazy open).
    file_type:
        :class:`~libtecio.FileType` enum value.  Defaults to
        :attr:`~libtecio.FileType.FULL`.

    Attributes
    ----------
    auxdataset : dict[str, str]
        Dataset-level auxiliary data buffer.  Assign entries before the
        first zone write; they will be flushed automatically.
    auxvar : dict[int | str, dict[str, str]]
        Variable-level auxiliary data buffer.  Keys are either 1-based
        integer variable indices or variable name strings.
    current_zone : int
        1-based index of the most recently written zone (0 before any zone
        is written).

    Examples
    --------
    Eager open (variables known up front)::

        with Write("out.dat", title="Demo", variables=["X", "Y", "P"]) as w:
            w.auxdataset["Author"] = "PyTecplot"
            w.write_ijk_zone(data=[x, y, p], title="Zone 1")

    Lazy open (variables supplied per zone)::

        with Write("out.dat", title="Demo") as w:
            w.write_ijk_zone(
                data=[x, y, p],
                variables=["X", "Y", "P"],
                title="Zone 1",
            )
    """

    #: Default float format (significant digits), matches Tecplot's own output.
    FLOAT_FMT: str = ".9g"

    def __init__(
        self,
        path: str,
        title: str = "untitled",
        variables: list[str] | None = None,
        file_type: FileType = FileType.FULL,
    ) -> None:
        """Store configuration; open the file immediately if *variables* given."""
        self.path: str = str(path)
        self.title: str = title
        self.variables: list[str] | None = variables
        self.file_type: FileType = file_type
        self.current_zone: int = 0

        # Buffered aux data — flushed to disk before the first zone.
        self.auxdataset: dict[str, str] = {}
        self.auxvar: dict[int | str, dict[str, str]] = {}

        # Internal state
        self._fp: io.TextIOWrapper | None = None
        self._opened: bool = False
        self._float_fmt: str = self.FLOAT_FMT

        if self.variables is not None:
            self._open(self.variables)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> Write:
        """Support ``with`` statement — returns *self*."""
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Close the file on context-manager exit."""
        try:
            self.close()
        except Exception:
            if exc_type is None:
                raise

    # ------------------------------------------------------------------
    # File lifecycle
    # ------------------------------------------------------------------

    def _open(self, var_names: list[str]) -> None:
        """Open the output file and write the dataset header.

        Called at most once per instance — either from ``__init__``
        (eager open) or on the first zone-writing call (lazy open).

        Args:
            var_names: Ordered list of variable name strings.

        Raises:
            ValueError: If *var_names* is empty.
        """
        if not var_names:
            raise ValueError(
                "Write requires at least one variable name."
            )
        self.variables = var_names
        self._fp = open(self.path, "w", encoding="utf-8", newline="\n")  # noqa: SIM115
        self._write_file_header()
        self._opened = True

    def close(self) -> None:
        """Flush and close the output file (safe to call more than once)."""
        if self._fp is not None:
            self._fp.flush()
            self._fp.close()
            self._fp = None
            self._opened = False

    def flush_aux(self) -> None:
        """Write buffered dataset- and variable-level aux data to the file.

        Called automatically before the first zone is created.  Explicit
        calls are only needed in unusual ordering scenarios.

        Raises:
            IOError: If the file has not been opened yet.
            IndexError: If an integer aux key is out of range.
            KeyError: If a string aux key does not match any variable.
        """
        if self._fp is None:
            raise IOError("flush_aux() called before file was opened.")

        # Dataset-level
        for name, value in self.auxdataset.items():
            self._fp.write(f"DATASETAUXDATA {name}={_quote(value)}\n")

        # Variable-level
        for key, subdict in self.auxvar.items():
            if isinstance(key, int):
                var_idx = key - 1  # convert 1-based → 0-based
                if var_idx not in range(len(self.variables)):
                    raise IndexError(
                        f"Variable index {key} out of bounds "
                        f"[1, {len(self.variables)}]"
                    )
            elif isinstance(key, str):
                try:
                    var_idx = self.variables.index(key)
                except ValueError as exc:
                    raise KeyError(
                        f"Variable aux data key {key!r} not found in "
                        f"variable list ({self.variables})"
                    ) from exc
            else:
                raise TypeError(
                    f"Aux data key must be a variable name (str) or "
                    f"1-based index (int), got {key!r}"
                )

            one_based = var_idx + 1
            for name, value in subdict.items():
                self._fp.write(
                    f"VARAUXDATA {one_based} {name}={_quote(value)}\n"
                )

        self.auxdataset.clear()
        self.auxvar.clear()

    # ------------------------------------------------------------------
    # Zone writers
    # ------------------------------------------------------------------

    def write_ijk_zone(
        self,
        data: Sequence[npt.NDArray],
        title: str | None = None,
        variables: list[str] | None = None,
        value_locations: Sequence[ValueLocation] | None = None,
        passive_vars: Sequence[bool | int] | None = None,
        var_sharing: Sequence[int] | None = None,
        solution_time: float = 0.0,
        strand_id: int = 0,
        aux: dict[str, Any] | None = None,
    ) -> None:
        """Write a complete IJK-ordered zone in BLOCK format.

        Zone dimensions ``imax × jmax × kmax`` are inferred from the shape
        of the first NODAL data array (or first CELL_CENTERED array when all
        variables are cell-centred).  1-D arrays produce a ``(N, 1, 1)``
        zone; 2-D arrays a ``(I, J, 1)`` zone; 3-D arrays a full
        ``(I, J, K)`` zone.

        Args:
            data:            Sequence of NumPy arrays, one per *active*
                             variable (i.e. not passive or shared).
            title:           Zone title.  Defaults to
                             ``"IJK_Zone_{current_zone + 1}"``.
            variables:       Variable names — required when the file has not
                             been opened yet (lazy-open path).  Ignored once
                             the file is initialised.
            value_locations: Per-variable :class:`ValueLocation`.  Defaults
                             to all :attr:`~ValueLocation.NODAL`.
            passive_vars:    Per-dataset-variable passive flags.  Length must
                             equal the total number of dataset variables.
            var_sharing:     Per-dataset-variable source-zone sharing index
                             (``0`` = no sharing).
            solution_time:   Solution time for transient data.
            strand_id:       Strand ID (``0`` = steady-state).
            aux:             Zone-level auxiliary data ``{name: value}``.

        Raises:
            ValueError: On variable-count or array-shape mismatch.
        """
        # Default title
        if title is None:
            title = f"IJK_Zone_{self.current_zone + 1}"

        # Auto-generate variable names for lazy open if not provided
        if variables is None:
            variables = [f"V{i}" for i in range(1, len(data) + 1)]

        # Lazy open + flush buffered aux data on the first zone
        if not self._opened:
            self._open(variables)
            self.flush_aux()

        # Infer per-variable data types from array dtypes
        variable_types = [_infer_data_type(np.asarray(arr)) for arr in data]

        # Default value locations — all nodal
        if value_locations is None:
            value_locations = [ValueLocation.NODAL] * len(data)

        # Default passive / sharing vectors (dataset-length)
        if passive_vars is None:
            passive_vars = [False] * len(self.variables)
        if var_sharing is None:
            var_sharing = [0] * len(self.variables)

        # ---- Validate active variable count ----
        if len(data) != len(self.variables):
            expected_vars = sum(
                1
                for is_passive, share_zone in zip(
                    passive_vars, var_sharing, strict=True
                )
                if not is_passive and not share_zone
            )
            if expected_vars == 0:
                raise ValueError(
                    "No active variables to write — all variables are either "
                    "passive or shared."
                )
            if len(data) == 0:
                raise ValueError(
                    f"No data arrays provided. Expected {expected_vars} active "
                    "variable arrays based on passive_vars and var_sharing."
                )
            if len(data) != expected_vars:
                raise ValueError(
                    f"Expected {expected_vars} data arrays for active variables, "
                    f"got {len(data)}."
                )

        # ---- Infer zone dimensions ----
        nodal_indices = [
            i for i, loc in enumerate(value_locations)
            if loc == ValueLocation.NODAL
        ]
        cell_indices = [
            i for i, loc in enumerate(value_locations)
            if loc == ValueLocation.CELL_CENTERED
        ]

        if nodal_indices:
            ndims = np.asarray(data[nodal_indices[0]]).ndim
            if ndims not in (1, 2, 3):
                raise ValueError(
                    f"Arrays must be 1D, 2D, or 3D; got {ndims}-D array.  "
                    "Write each time step as a separate zone."
                )
            nodal_shape = (
                np.asarray(data[nodal_indices[0]]).shape + (1,) * (3 - ndims)
            )
            cell_shape = tuple(max(d - 1, 1) for d in nodal_shape)
            imax, jmax, kmax = nodal_shape
        elif cell_indices:
            ndims = np.asarray(data[cell_indices[0]]).ndim
            if ndims not in (1, 2, 3):
                raise ValueError(
                    f"Arrays must be 1D, 2D, or 3D; got {ndims}-D array.  "
                    "Write each time step as a separate zone."
                )
            cell_shape = (
                np.asarray(data[cell_indices[0]]).shape + (1,) * (3 - ndims)
            )
            nodal_shape = tuple(max(d + 1, 1) for d in cell_shape)
            imax, jmax, kmax = nodal_shape
        else:
            raise ValueError(
                "Could not determine zone dimensions — no nodal or "
                "cell-centred variables found."
            )

        # ---- Validate individual array shapes ----
        for i, (arr, loc) in enumerate(zip(data, value_locations, strict=True)):
            arr = np.asarray(arr)
            if arr.ndim != ndims:
                raise ValueError(
                    f"Array {i} is {arr.ndim}D, expected {ndims}D."
                )
            shape = arr.shape + (1,) * (3 - arr.ndim)
            if loc == ValueLocation.NODAL and shape != nodal_shape:
                raise ValueError(
                    f"Array {i} is NODAL but has shape {shape}, "
                    f"expected {nodal_shape}."
                )
            if loc == ValueLocation.CELL_CENTERED and shape != cell_shape:
                raise ValueError(
                    f"Array {i} is CELL_CENTERED but has shape {shape}, "
                    f"expected {cell_shape}."
                )

        # ---- Determine active variable indices (1-based, dataset order) ----
        active_var_idx = [
            var_idx
            for var_idx, (is_passive, share_zone) in enumerate(
                zip(passive_vars, var_sharing, strict=True), start=1
            )
            if not is_passive and not share_zone
        ]

        # Build dataset-length value-location list
        value_locations_global = [ValueLocation.NODAL] * len(self.variables)
        for local_idx, var_idx in enumerate(active_var_idx):
            value_locations_global[var_idx - 1] = value_locations[local_idx]

        # ---- Write zone header ----
        self._write_zone_header(
            title=title,
            zone_type=ZoneType.ORDERED,
            imax=imax,
            jmax=jmax,
            kmax=kmax,
            solution_time=solution_time,
            strand_id=strand_id,
            passive_vars=passive_vars,
            var_sharing=var_sharing,
            value_locations_global=value_locations_global,
            aux=aux,
        )
        self.current_zone += 1

        # ---- Write active variable data blocks ----
        for arr, dt in zip(data, variable_types):
            cast = np.asarray(arr, dtype=_DT_TO_DTYPE[dt]).ravel(order="F")
            _write_array(self._fp, cast, self._float_fmt)

    def write_fe_zone(
        self,
        zone_type: ZoneType,
        data: Sequence[npt.NDArray],
        node_map: npt.ArrayLike | None = None,
        title: str | None = None,
        variables: list[str] | None = None,
        value_locations: Sequence[ValueLocation] | None = None,
        passive_vars: Sequence[bool | int] | None = None,
        var_sharing: Sequence[int] | None = None,
        con_sharing: int = 0,
        face_neighbors: npt.ArrayLike | None = None,
        face_nbr_mode: FaceNeighborMode = FaceNeighborMode.LOCAL_ONE_TO_ONE,
        solution_time: float = 0.0,
        strand_id: int = 0,
        aux: dict[str, Any] | None = None,
    ) -> None:
        """Write a complete finite-element zone in BLOCK format.

        ``FEPOLYGON`` and ``FEPOLYHEDRON`` are not yet supported.

        Args:
            zone_type:       FE zone type.  Must be one of :data:`_FE_SIMPLE`.
            data:            Sequence of NumPy arrays, one per active variable.
                             NODAL arrays must have length ``num_nodes``;
                             CELL_CENTERED arrays must have length
                             ``num_cells``.  Both counts are inferred from
                             *node_map*.
            node_map:        Integer array of shape
                             ``(num_cells, nodes_per_cell)`` with **1-based**
                             node indices (same convention as the binary
                             writers).
            title:           Zone title.  Defaults to
                             ``"FE_Zone_{current_zone + 1}"``.
            variables:       Variable names — required on the lazy-open first
                             zone call.
            value_locations: Per-variable :class:`ValueLocation`.  Defaults
                             to all NODAL.
            passive_vars:    Per-dataset-variable passive flags.
            var_sharing:     Per-dataset-variable sharing zone indices.
            con_sharing:     Source zone index for connectivity sharing
                             (``0`` = no sharing).  When non-zero the
                             connectivity block is omitted.
            face_neighbors:  Accepted for interface compatibility; ignored in
                             the ASCII format (face-neighbour data is not
                             representable in ASCII DAT files).
            face_nbr_mode:   Accepted for interface compatibility; ignored.
            solution_time:   Solution time for transient data.
            strand_id:       Strand ID.
            aux:             Zone-level auxiliary data ``{name: value}``.

        Raises:
            NotImplementedError: If *zone_type* is ``FEPOLYGON`` or
                                 ``FEPOLYHEDRON``.
            ValueError:          On variable-count or array-length mismatch.
        """
        # Guard unsupported types first so error surfaces before lazy-open
        if zone_type in _FE_POLY:
            raise NotImplementedError(
                f"Zone type {zone_type.name!r} is not supported by "
                "write_fe_zone.  FEPOLYGON and FEPOLYHEDRON zones require "
                "face-based connectivity which is not yet implemented for "
                "the ASCII writer."
            )
        if zone_type not in _FE_SIMPLE:
            raise NotImplementedError(
                f"Zone type {zone_type!r} is not supported by write_fe_zone."
            )

        # Default title
        if title is None:
            title = f"FE_Zone_{self.current_zone + 1}"

        # Auto-generate variable names for lazy open if not provided
        if variables is None:
            variables = [f"V{i}" for i in range(1, len(data) + 1)]

        # Lazy open + flush buffered aux data on the first zone
        if not self._opened:
            self._open(variables)
            self.flush_aux()

        # Infer per-variable data types
        variable_types = [_infer_data_type(np.asarray(arr)) for arr in data]

        # Default value locations — all nodal
        if value_locations is None:
            value_locations = [ValueLocation.NODAL] * len(data)

        # Default passive / sharing vectors (dataset-length)
        if passive_vars is None:
            passive_vars = [False] * len(self.variables)
        if var_sharing is None:
            var_sharing = [0] * len(self.variables)

        # ---- Validate active variable count ----
        if len(data) != len(self.variables):
            expected_vars = sum(
                1
                for is_passive, share_zone in zip(
                    passive_vars, var_sharing, strict=True
                )
                if not is_passive and not share_zone
            )
            if expected_vars == 0:
                raise ValueError(
                    "No active variables to write — all variables are either "
                    "passive or shared."
                )
            if len(data) == 0:
                raise ValueError(
                    f"No data arrays provided. Expected {expected_vars} active "
                    "variable arrays based on passive_vars and var_sharing."
                )
            if len(data) != expected_vars:
                raise ValueError(
                    f"Expected {expected_vars} data arrays for active variables, "
                    f"got {len(data)}."
                )

        # ---- Derive node/element counts from node_map ----
        node_map_arr = np.asarray(node_map)
        num_cells = int(node_map_arr.shape[0])
        num_nodes = int(node_map_arr.max())

        # ---- Validate per-variable array lengths ----
        for i, (arr, loc) in enumerate(zip(data, value_locations, strict=True)):
            arr_np = np.asarray(arr)
            if loc == ValueLocation.NODAL and arr_np.size != num_nodes:
                raise ValueError(
                    f"Array {i} is NODAL but has {arr_np.size} values; "
                    f"expected {num_nodes}."
                )
            if loc == ValueLocation.CELL_CENTERED and arr_np.size != num_cells:
                raise ValueError(
                    f"Array {i} is CELL_CENTERED but has {arr_np.size} values; "
                    f"expected {num_cells}."
                )

        # ---- Determine active variable indices (1-based, dataset order) ----
        active_var_idx = [
            var_idx
            for var_idx, (is_passive, share_zone) in enumerate(
                zip(passive_vars, var_sharing, strict=True), start=1
            )
            if not is_passive and not share_zone
        ]

        # Build dataset-length value-location list
        value_locations_global = [ValueLocation.NODAL] * len(self.variables)
        for local_idx, var_idx in enumerate(active_var_idx):
            value_locations_global[var_idx - 1] = value_locations[local_idx]

        # ---- Write zone header ----
        self._write_zone_header(
            title=title,
            zone_type=zone_type,
            num_nodes=num_nodes,
            num_elements=num_cells,
            solution_time=solution_time,
            strand_id=strand_id,
            passive_vars=passive_vars,
            var_sharing=var_sharing,
            value_locations_global=value_locations_global,
            con_sharing=con_sharing,
            aux=aux,
        )
        self.current_zone += 1

        # ---- Write active variable data blocks ----
        for arr, dt in zip(data, variable_types):
            cast = np.asarray(arr, dtype=_DT_TO_DTYPE[dt]).ravel()
            _write_array(self._fp, cast, self._float_fmt)

        # ---- Write connectivity (node map) ----
        # node_map is already 1-based (same as binary writers).
        if not con_sharing:
            conn = np.asarray(node_map, dtype=np.intp)
            conn = conn.reshape(num_cells, _NODES_PER_ELEM[zone_type])
            for row in conn:
                self._fp.write(
                    " ".join(str(int(n)) for n in row) + "\n"
                )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _write_file_header(self) -> None:
        """Write TITLE, optional FILETYPE, and VARIABLES lines."""
        fp = self._fp
        fp.write(f'TITLE     = {_quote(self.title)}\n')

        if self.file_type in _FILETYPE_STR:
            fp.write(f"FILETYPE  = {_FILETYPE_STR[self.file_type]}\n")

        var_strs = "\n".join(_quote(v) for v in self.variables)
        fp.write(f"VARIABLES = {var_strs}\n")

    def _write_zone_header(
        self,
        title: str,
        zone_type: ZoneType,
        imax: int = 0,
        jmax: int = 0,
        kmax: int = 0,
        num_nodes: int = 0,
        num_elements: int = 0,
        solution_time: float = 0.0,
        strand_id: int = 0,
        passive_vars: Sequence[bool | int] | None = None,
        var_sharing: Sequence[int] | None = None,
        value_locations_global: list[ValueLocation] | None = None,
        con_sharing: int = 0,
        aux: dict[str, Any] | None = None,
    ) -> None:
        """Emit a ``ZONE`` header followed by any zone-level ``AUXDATA`` lines."""
        fp = self._fp
        zt_str = _ZONETYPE_STR[zone_type]

        fp.write(f'ZONE T={_quote(title)}\n')
        fp.write(f' STRANDID={strand_id}, SOLUTIONTIME={solution_time}\n')

        if zone_type == ZoneType.ORDERED:
            fp.write(f' I={imax}, J={jmax}, K={kmax}\n')
            fp.write(f' ZONETYPE={zt_str},\n')
        else:
            fp.write(
                f' Nodes={num_nodes}, Elements={num_elements},'
                f' ZONETYPE={zt_str}\n'
            )

        fp.write(' DATAPACKING=BLOCK\n')

        # VARLOCATION — emit only cell-centred entries
        if value_locations_global:
            cc_indices = [
                i + 1
                for i, loc in enumerate(value_locations_global)
                if loc == ValueLocation.CELL_CENTERED
            ]
            if cc_indices:
                idx_str = ",".join(str(i) for i in cc_indices)
                fp.write(f' VARLOCATION=([{idx_str}]=CELLCENTERED)\n')

        # PASSIVEVARLIST
        if passive_vars:
            passive_indices = [
                str(i + 1)
                for i, flag in enumerate(passive_vars)
                if flag
            ]
            if passive_indices:
                fp.write(
                    f' PASSIVEVARLIST=[{",".join(passive_indices)}]\n'
                )

        # VARSHARELIST
        if var_sharing and any(var_sharing):
            share_entries = [
                f"[{i + 1}]={z}"
                for i, z in enumerate(var_sharing)
                if z
            ]
            if share_entries:
                fp.write(
                    f' VARSHARELIST=({",".join(share_entries)})\n'
                )

        # CONNECTIVITYSHAREZONE
        if con_sharing:
            fp.write(f' CONNECTIVITYSHAREZONE={con_sharing}\n')

        # Zone-level aux data
        if aux:
            for name, value in aux.items():
                fp.write(f' AUXDATA {name}={_quote(value)}\n')
