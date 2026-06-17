"""Module containing in-memory data container, ``Dataset``.

A :class:`Dataset` is a uniform, mutable, in-memory mirror of a Tecplot dataset:
dataset-level metadata and auxiliary data plus an ordered list of
:class:`~tecio._zone.Zone` objects, each holding an ordered list of
:class:`~tecio._variable.Variable` objects (one per dataset variable).

The public *read* interface mirrors the ``Read`` classes in :mod:`tecio.szl`,
:mod:`tecio.plt`, and :mod:`tecio.dat` (``title``, ``file_type``, ``num_vars``,
``variables``, ``num_zones``, ``zone``, ``auxdata``, ``num_auxdata_items``,
``var_auxdata``, ``get_var_auxdata``, ``get_zone_auxdata``) so a dataset can be handed
straight to the zone-copy routine used by the writers.

Unlike a reader, a dataset is fully mutable and can be:

* populated from any Tecplot file type, a list of files, an open reader, a
  :class:`dict`, or a :class:`pandas.DataFrame`, with optional ``zones`` / ``variables``
  subsetting;
* built up incrementally with :meth:`add_zone` / :meth:`add_variable`, where a new
  variable defined on one zone is immediately created dataset-wide (passive in every
  other zone) so the dataset always stays rectangular;
* written back out in any format via :meth:`write` / :meth:`to_szl` / :meth:`to_plt` /
  :meth:`to_dat`.

Indexing note:
    Zones and variables are addressed with **0-based** indices from the Python API
    (matching list indexing), while :meth:`get_var_auxdata` and :meth:`get_zone_auxdata`
    take **1-based** indices for parity with the read classes.

Roadmap:
    Analysis helpers (surface/volume slices, iso-surfaces, FE-to-structured resampling,
    etc.) are intended to build on this container in a future pass and are intentionally
    left out of this first version.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from ._variable import Variable
from ._zone import AuxData, Zone
from .libtecio import FileType, ValueLocation, ZoneType

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _own(arr: npt.ArrayLike) -> npt.NDArray:
    """Return an array that owns writeable memory, copying only if needed.

    Arrays produced by ``np.ctypeslib.as_array`` (the SZL read path) are views
    onto a ctypes buffer that becomes invalid once the file handle closes, so
    they must be copied.  Arrays produced by ``np.fromfile`` (PLT/DAT) already
    own their memory and are returned untouched to avoid wasted allocations.
    """
    a = np.asarray(arr)
    if a.flags["OWNDATA"] and a.flags["WRITEABLE"]:
        return a
    return np.array(a)


def _as_file_type(value: FileType | int | str) -> FileType:
    """Coerce an enum / int / UPPER-case name to a :class:`FileType`."""
    if isinstance(value, FileType):
        return value
    if isinstance(value, str):
        return FileType[value.strip().upper()]
    return FileType(int(value))


def _as_zone_type(value: ZoneType | int | str) -> ZoneType:
    """Coerce an enum / int / UPPER-case name to a :class:`ZoneType`."""
    if isinstance(value, ZoneType):
        return value
    if isinstance(value, str):
        return ZoneType[value.strip().upper()]
    return ZoneType(int(value))


def _as_value_location(value: ValueLocation | int | str) -> ValueLocation:
    """Coerce an enum / int / UPPER-case name to a :class:`ValueLocation`."""
    if isinstance(value, ValueLocation):
        return value
    if isinstance(value, str):
        return ValueLocation[value.strip().upper()]
    return ValueLocation(int(value))


def _require_pandas() -> pd:
    """Import and return :mod:`pandas`, raising a clear error if it is missing."""
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "pandas is required for this operation. "
            "Install it with 'pip install pandas'."
        ) from exc
    return pd


# ===========================================================================
# Dataset
# ===========================================================================


class Dataset:
    """A uniform, in-memory container of :class:`Zone` and :class:`Variable`.

    Args:
        source:    Optional data source to populate from.  May be a file path,
                   a list/tuple of file paths, an open ``Read`` instance, a
                   :class:`dict`, or a :class:`pandas.DataFrame`.  ``None``
                   creates an empty dataset.
        zones:     Optional 0-based zone indices to keep (file/reader sources
                   only).  ``None`` keeps all zones.
        variables: Optional variable filter (file/reader sources only) as a
                   list of 0-based indices and/or name strings.  ``None`` keeps
                   all variables.
        title:     Dataset title.  Defaults to the source file's title when
                   loading and left blank otherwise.
        file_type: :class:`~tecio.libtecio.FileType`.  Overridden by the source
                   file's type when loading.

    Example:
        >>> ds = Dataset("input.szplt")  # load every zone/variable
        >>> ds.to_plt()  # -> input.plt
        >>> ds.to_dat("output.dat")  # -> output.dat

        >>> empty = Dataset(title="demo")
        >>> z = empty.add_zone(title="line")
        >>> _ = z.add_variable("x", np.linspace(0, 1, 11))
        >>> _ = z.add_variable("p", np.random.rand(11))
        >>> empty.to_szl("demo.szplt")
    """

    def __init__(
        self,
        source: Any = None,
        *,
        zones: Sequence[int] | None = None,
        variables: Sequence[int | str] | None = None,
        title: str = "",
        file_type: FileType | int | str = FileType.FULL,
    ) -> None:
        self.title: str = str(title)
        self.file_type: FileType = _as_file_type(file_type)
        self.auxdata: AuxData = AuxData()

        self._variables: list[str] = []
        self._zones: list[Zone] = []
        self._var_auxdata: list[AuxData] = []
        self._source_path: Path | None = None

        if source is not None:
            self._ingest(source, zones=zones, variables=variables)

    # ------------------------------------------------------------------
    # Ingest dispatch
    # ------------------------------------------------------------------

    def _ingest(
        self,
        source: Any,
        *,
        zones: Sequence[int] | None,
        variables: Sequence[int | str] | None,
    ) -> None:
        """Dispatch *source* to the appropriate loader."""
        if isinstance(source, (str, os.PathLike)):
            self._load_files([source], zones=zones, variables=variables)
        elif isinstance(source, (list, tuple)):
            self._load_files(list(source), zones=zones, variables=variables)
        elif isinstance(source, dict):
            self._load_dict(source)
        elif hasattr(source, "columns") and hasattr(source, "to_numpy"):
            self._load_dataframe(source)
        elif hasattr(source, "zone") and hasattr(source, "variables"):
            self._load_reader(source, zones=zones, variables=variables)
        else:
            raise TypeError(
                "Unsupported Dataset source "
                f"{type(source).__name__!r}; expected a file path, a list of "
                "paths, a Read instance, a dict, or a pandas DataFrame."
            )

    # ------------------------------------------------------------------
    # Loading from files / readers
    # ------------------------------------------------------------------

    def _load_files(
        self,
        paths: Sequence[str | os.PathLike],
        *,
        zones: Sequence[int] | None,
        variables: Sequence[int | str] | None,
    ) -> None:
        """Load and merge one or more Tecplot files into this dataset."""
        from ._io import open as tecio_open  # local import avoids a cycle

        resolved = [Path(p) for p in paths]
        if resolved:
            self._source_path = resolved[0]
        for i, path in enumerate(resolved):
            if not path.exists():
                raise FileNotFoundError(f"No such file or directory: '{path}'")
            with tecio_open(str(path), "r") as reader:
                self._absorb_reader(
                    reader,
                    zones=zones,
                    variables=variables,
                    set_header=(i == 0),
                )

    def _load_reader(
        self,
        reader: Any,
        *,
        zones: Sequence[int] | None,
        variables: Sequence[int | str] | None,
    ) -> None:
        """Load from an already-open ``Read`` instance (caller keeps ownership)."""
        self._absorb_reader(reader, zones=zones, variables=variables, set_header=True)

    def _absorb_reader(
        self,
        reader: Any,
        *,
        zones: Sequence[int] | None,
        variables: Sequence[int | str] | None,
        set_header: bool,
    ) -> None:
        """Copy zones/variables from *reader* into this dataset."""
        all_names = list(reader.variables)
        var_idx = self._resolve_variable_filter(all_names, variables)
        zone_idx = self._resolve_zone_filter(reader.num_zones, zones)
        materialize = len(zone_idx) != reader.num_zones

        if set_header:
            if not self.title:
                self.title = reader.title
            self.file_type = reader.file_type
            if len(reader.auxdata) > 0:
                self.auxdata.update(reader.auxdata.items())

        for zi in zone_idx:
            self.add_zone(self._zone_from_reader(reader, zi, var_idx, materialize))

        if set_header:
            for new_i, orig in enumerate(var_idx):
                if new_i >= len(self._var_auxdata):
                    break
                try:
                    aux = reader.get_var_auxdata(orig + 1)
                except Exception:  # noqa: BLE001 - reader may not support it
                    aux = None
                if aux is not None and len(aux) > 0:
                    self._var_auxdata[new_i].update(aux.items())

    def _zone_from_reader(
        self,
        reader: Any,
        zi: int,
        var_idx: Sequence[int],
        materialize: bool,
    ) -> Zone:
        """Build an in-memory :class:`Zone` from one reader zone."""
        rz = reader.zone[zi]
        zt = rz.zone_type

        variables: list[Variable] = []
        for vidx in var_idx:
            rv = rz.variable[vidx]
            if rv.is_passive():
                var = Variable(
                    rv.name,
                    value_location=rv.value_location,
                    data_type=rv.data_type,
                    is_passive=True,
                )
            elif rv.shared_zone is not None:
                if materialize:
                    arr = self._resolve_shared(reader, zi, vidx)
                    if arr is None:
                        var = Variable(
                            rv.name,
                            value_location=rv.value_location,
                            is_passive=True,
                        )
                    else:
                        var = Variable(
                            rv.name,
                            values=_own(arr),
                            value_location=rv.value_location,
                        )
                else:
                    var = Variable(
                        rv.name,
                        value_location=rv.value_location,
                        data_type=rv.data_type,
                        shared_zone=rv.shared_zone,
                    )
            else:
                arr = rv.values
                if arr is None:
                    var = Variable(
                        rv.name,
                        value_location=rv.value_location,
                        is_passive=True,
                    )
                else:
                    var = Variable(
                        rv.name,
                        values=_own(arr),
                        value_location=rv.value_location,
                    )
            variables.append(var)

        aux = dict(rz.auxdata.items()) if len(rz.auxdata) > 0 else None

        if zt == ZoneType.ORDERED:
            return Zone(
                rz.title,
                zt,
                dimensions=tuple(int(d) for d in rz.dimensions),
                solution_time=rz.solution_time,
                strand_id=rz.strand_id,
                variables=variables,
                aux=aux,
            )

        node_map = rz.node_map
        return Zone(
            rz.title,
            zt,
            num_nodes=rz.num_nodes,
            num_elements=rz.num_elements,
            node_map=None if node_map is None else _own(np.asarray(node_map)),
            solution_time=rz.solution_time,
            strand_id=rz.strand_id,
            variables=variables,
            aux=aux,
        )

    @staticmethod
    def _resolve_shared(reader: Any, zi: int, vidx: int) -> npt.NDArray | None:
        """Read the source array for a shared variable when subsetting zones.

        The recorded ``shared_zone`` base is inconsistent across readers, so
        both the value and ``value - 1`` are tried defensively.  Returns
        ``None`` if a concrete source array cannot be located.
        """
        src = reader.zone[zi].variable[vidx].shared_zone
        if src is None:
            return None
        for cand in (src, src - 1):
            if 0 <= cand < reader.num_zones:
                try:
                    sv = reader.zone[cand].variable[vidx]
                except IndexError:
                    continue
                if sv.is_passive() or sv.shared_zone is not None:
                    continue
                arr = sv.values
                if arr is not None:
                    return np.asarray(arr)
        return None

    # ------------------------------------------------------------------
    # Filter resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_variable_filter(
        all_names: Sequence[str],
        variables: Sequence[int | str] | None,
    ) -> list[int]:
        """Resolve a variable filter to ordered 0-based indices."""
        if variables is None:
            return list(range(len(all_names)))
        result: list[int] = []
        for v in variables:
            if isinstance(v, (int, np.integer)):
                iv = int(v)
                if iv < 0 or iv >= len(all_names):
                    raise IndexError(
                        f"Variable index {iv} out of range [0, {len(all_names) - 1}]."
                    )
                result.append(iv)
                continue
            name = str(v)
            if name in all_names:
                result.append(all_names.index(name))
                continue
            low = name.lower()
            match = next((i for i, n in enumerate(all_names) if n.lower() == low), None)
            if match is None:
                raise KeyError(
                    f"Variable {v!r} not found. Available: {list(all_names)}"
                )
            result.append(match)
        return result

    @staticmethod
    def _resolve_zone_filter(
        num_zones: int,
        zones: Sequence[int] | None,
    ) -> list[int]:
        """Resolve a zone filter to 0-based indices."""
        if zones is None:
            return list(range(num_zones))
        result: list[int] = []
        for z in zones:
            iz = int(z)
            if iz < 0 or iz >= num_zones:
                raise IndexError(f"Zone index {iz} out of range [0, {num_zones - 1}].")
            result.append(iz)
        return result

    # ------------------------------------------------------------------
    # Building zones / variables
    # ------------------------------------------------------------------

    def add_variable(
        self,
        name: str,
        *,
        value_location: ValueLocation | int | str = ValueLocation.NODAL,
        data_type: Any = None,
        default: float | None = None,
    ) -> int:
        """Add a variable to the dataset (and to every existing zone).

        The call is idempotent: if *name* already exists its index is returned
        unchanged.  Otherwise the variable is appended to the dataset variable
        list and added to each existing zone -- passive when *default* is
        ``None`` (the usual case), or filled with *default* otherwise.

        Args:
            name:           Variable name.
            value_location: Location for the placeholders created in existing
                            zones.
            data_type:      Optional explicit :class:`~tecio.libtecio.DataType`
                            for the placeholders.
            default:        When given, existing zones receive a constant array
                            of this value instead of a passive placeholder.

        Returns:
            The 0-based index of the variable in the dataset.
        """
        name = str(name)
        if name in self._variables:
            return self._variables.index(name)

        loc = _as_value_location(value_location)
        self._variables.append(name)
        self._var_auxdata.append(AuxData())

        for zone in self._zones:
            if default is None:
                var = Variable(
                    name,
                    value_location=loc,
                    data_type=data_type,
                    is_passive=True,
                )
            else:
                count = (
                    zone.num_elements
                    if loc == ValueLocation.CELL_CENTERED
                    else zone.num_nodes
                )
                var = Variable(
                    name,
                    values=np.full(count, float(default)),
                    value_location=loc,
                )
            zone._attach_variable(var)

        return len(self._variables) - 1

    def add_zone(self, zone: Zone | None = None, **kwargs: Any) -> Zone:
        """Add a zone to the dataset, reconciling its variable list.

        Either pass an existing :class:`Zone` or keyword arguments forwarded to
        the :class:`Zone` constructor.  Any variables on the zone that are not
        yet in the dataset are introduced dataset-wide (passive in every other
        zone); the zone's own variable list is then rebuilt in dataset-variable
        order, with passive placeholders for any missing variables.

        Returns:
            The added :class:`Zone`.
        """
        if zone is None:
            zone = Zone(**kwargs)
        elif not isinstance(zone, Zone):
            raise TypeError(f"add_zone expected a Zone, got {type(zone).__name__!r}.")
        self._reconcile_zone(zone)
        self._zones.append(zone)
        return zone

    def _reconcile_zone(self, zone: Zone) -> None:
        """Make *zone* consistent with the dataset variable list."""
        zone._dataset = self

        zone_vars: dict[str, Variable] = {}
        for var in zone._variable:
            zone_vars.setdefault(var.name, var)

        # Introduce any new variables to the dataset (and to existing zones).
        for var in zone._variable:
            if var.name not in self._variables:
                self.add_variable(var.name, value_location=var.value_location)

        # Rebuild this zone's list in dataset-variable order.
        rebuilt: list[Variable] = []
        for name in self._variables:
            var = zone_vars.get(name)
            if var is None:
                var = Variable(name, is_passive=True)
            var._zone = zone
            rebuilt.append(var)
        zone._variable = rebuilt

    # ------------------------------------------------------------------
    # Variable management
    # ------------------------------------------------------------------

    def variable_index(self, key: int | str) -> int:
        """Return the 0-based dataset index for a variable name or index."""
        if isinstance(key, (int, np.integer)):
            iv = int(key)
            if iv < 0 or iv >= len(self._variables):
                raise IndexError(
                    f"Variable index {iv} out of range [0, {len(self._variables) - 1}]."
                )
            return iv
        name = str(key)
        if name in self._variables:
            return self._variables.index(name)
        low = name.lower()
        match = next(
            (i for i, n in enumerate(self._variables) if n.lower() == low), None
        )
        if match is None:
            raise KeyError(f"Variable {key!r} not found. Available: {self._variables}")
        return match

    def rename_variable(self, old: int | str, new: str) -> None:
        """Rename a variable consistently across the whole dataset."""
        idx = self.variable_index(old)
        self._variables[idx] = str(new)
        for zone in self._zones:
            zone.variable[idx].name = str(new)

    def delete_variable(self, key: int | str) -> None:
        """Remove a variable from the dataset and from every zone."""
        idx = self.variable_index(key)
        del self._variables[idx]
        del self._var_auxdata[idx]
        for zone in self._zones:
            del zone.variable[idx]

    def materialize_shared(self) -> None:
        """Replace shared variables with owned copies of their source data.

        Useful before reordering or deleting zones, since variable sharing
        references zones by position.  Variables whose source cannot be located
        are left shared.
        """
        for zone in self._zones:
            for vidx, var in enumerate(zone.variable):
                if var.shared_zone is not None:
                    arr = self._find_share_source(vidx, var.shared_zone)
                    if arr is not None:
                        var.values = np.array(arr)

    def _find_share_source(self, vidx: int, src: int) -> npt.NDArray | None:
        """Find a concrete source array for a shared variable, or ``None``."""
        for cand in (src, src - 1):
            if 0 <= cand < len(self._zones):
                sv = self._zones[cand].variable[vidx]
                if (
                    sv.shared_zone is None
                    and not sv.is_passive()
                    and sv.values is not None
                ):
                    return sv.values
        return None

    # ------------------------------------------------------------------
    # Read-parity properties
    # ------------------------------------------------------------------

    @property
    def num_vars(self) -> int:
        """Number of variables in the dataset."""
        return len(self._variables)

    @property
    def variables(self) -> list[str]:
        """Ordered list of variable name strings (a copy)."""
        return list(self._variables)

    @property
    def num_zones(self) -> int:
        """Number of zones in the dataset."""
        return len(self._zones)

    @property
    def zone(self) -> list[Zone]:
        """The list of :class:`Zone` objects (0-based)."""
        return self._zones

    @property
    def num_auxdata_items(self) -> int:
        """Number of dataset-level auxiliary data items."""
        return len(self.auxdata)

    @property
    def var_auxdata(self) -> list[AuxData | None]:
        """Per-variable aux data with a ``None`` placeholder at index 0.

        The leading ``None`` keeps 1-based indexing consistent with the read
        classes; use :meth:`get_var_auxdata` for direct 1-based access.
        """
        return [None, *self._var_auxdata]

    def get_var_auxdata(self, var_index: int) -> AuxData:
        """Return auxiliary data for variable *var_index* (1-based)."""
        if var_index < 1 or var_index > self.num_vars:
            raise IndexError(
                f"Variable index {var_index} out of range [1, {self.num_vars}]."
            )
        return self._var_auxdata[var_index - 1]

    def get_zone_auxdata(self, zone_index: int) -> AuxData:
        """Return auxiliary data for zone *zone_index* (1-based)."""
        if zone_index < 1 or zone_index > self.num_zones:
            raise IndexError(
                f"Zone index {zone_index} out of range [1, {self.num_zones}]."
            )
        return self._zones[zone_index - 1].auxdata

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def write(
        self,
        path: str | os.PathLike | None = None,
        *,
        file_type: FileType | int | str | None = None,
    ) -> Path:
        """Write the dataset to *path* (format chosen by its extension).

        Args:
            path:      Output path.  Defaults to the dataset's source path when
                       it was loaded from a file.
            file_type: Optional :class:`~tecio.libtecio.FileType` override.

        Returns:
            The :class:`~pathlib.Path` written.
        """
        from ._io import open as tecio_open  # local import avoids a cycle

        out = self._resolve_output(path)
        ftype = self.file_type if file_type is None else _as_file_type(file_type)
        with tecio_open(
            str(out),
            "w",
            title=self.title,
            variables=self.variables,
            file_type=ftype,
        ) as writer:
            self._write_aux(writer)
            self._write_zones(writer)
        return out

    def save(
        self,
        path: str | os.PathLike | None = None,
        *,
        file_type: FileType | int | str | None = None,
    ) -> Path:
        """Alias for :meth:`write`."""
        return self.write(path, file_type=file_type)

    def to_szl(self, path: str | os.PathLike | None = None) -> Path:
        """Write the dataset to a ``.szplt`` file."""
        return self.write(self._derive_path(path, ".szplt"))

    def to_plt(self, path: str | os.PathLike | None = None) -> Path:
        """Write the dataset to a ``.plt`` file."""
        return self.write(self._derive_path(path, ".plt"))

    def to_dat(self, path: str | os.PathLike | None = None) -> Path:
        """Write the dataset to an ASCII ``.dat`` file."""
        return self.write(self._derive_path(path, ".dat"))

    def _resolve_output(self, path: str | os.PathLike | None) -> Path:
        """Return the output path or fall back to the source path."""
        if path is not None:
            return Path(path)
        if self._source_path is not None:
            return self._source_path
        raise ValueError("No output path provided and dataset has no source file.")

    def _derive_path(self, path: str | os.PathLike | None, suffix: str) -> Path:
        """Return *path*, or the source path with *suffix* substituted."""
        if path is not None:
            return Path(path)
        if self._source_path is not None:
            return self._source_path.with_suffix(suffix)
        raise ValueError(
            "No output path provided and dataset has no source file to derive "
            f"a {suffix} path from."
        )

    def _write_aux(self, writer: Any) -> None:
        """Forward dataset- and variable-level aux data to *writer*."""
        if len(self.auxdata) > 0:
            writer.add_auxdataset_dict({
                str(k): str(v) for k, v in self.auxdata.items()
            })
        auxvar: dict[int, dict[str, str]] = {}
        for i, aux in enumerate(self._var_auxdata, start=1):
            if aux:
                auxvar[i] = {str(k): str(v) for k, v in aux.items()}
        if auxvar:
            writer.add_auxvar_dict(auxvar)

    def _write_zones(self, writer: Any) -> None:
        """Stream all zones to *writer* using the shared copy routine."""
        from ._io import _copy_zones  # local import avoids a cycle

        _copy_zones(self, writer)

    # ------------------------------------------------------------------
    # Construction from Python objects
    # ------------------------------------------------------------------

    def _load_dict(self, data: Mapping[str, Any]) -> None:
        """Populate the dataset from a :class:`dict`.

        Two schemas are accepted -- a structured multi-zone mapping::

            {
                "title": "demo",
                "file_type": "FULL",
                "auxdata": {"Solver": "MyCFD"},
                "zones": [
                    {
                        "title": "zone 1",
                        "zone_type": "ORDERED",
                        "solution_time": 0.0,
                        "strand_id": 0,
                        "data": {"x": x_arr, "p": p_arr},
                        "value_locations": {"p": "CELL_CENTERED"},
                        "aux": {"note": "first"},
                    },
                ],
            }

        or a flat ``{name: array}`` mapping that becomes a single ordered zone.
        """
        if "zones" in data:
            if "title" in data and not self.title:
                self.title = str(data["title"])
            if "file_type" in data:
                self.file_type = _as_file_type(data["file_type"])
            if "auxdata" in data and data["auxdata"]:
                self.auxdata.update(data["auxdata"])
            for zspec in data["zones"]:
                self.add_zone(self._zone_from_dict(zspec))
        elif "data" in data:
            self.add_zone(self._zone_from_dict(data))
        else:
            self.add_zone(self._zone_from_dict({"data": dict(data)}))

    def _zone_from_dict(self, spec: Mapping[str, Any]) -> Zone:
        """Build a :class:`Zone` from a single zone specification dict."""
        data: Mapping[str, Any] = spec.get("data", {})
        value_locations: Mapping[str, Any] = spec.get("value_locations", {})

        variables = [
            Variable(
                str(name),
                values=np.asarray(arr),
                value_location=_as_value_location(
                    value_locations.get(name, ValueLocation.NODAL)
                ),
            )
            for name, arr in data.items()
        ]

        kwargs: dict[str, Any] = dict(
            title=str(spec.get("title", "")),
            zone_type=_as_zone_type(spec.get("zone_type", ZoneType.ORDERED)),
            solution_time=float(spec.get("solution_time", 0.0)),
            strand_id=int(spec.get("strand_id", 0)),
            variables=variables,
            aux=spec.get("aux"),
        )
        if spec.get("dimensions") is not None:
            kwargs["dimensions"] = tuple(spec["dimensions"])
        if spec.get("num_nodes") is not None:
            kwargs["num_nodes"] = int(spec["num_nodes"])
        if spec.get("num_elements") is not None:
            kwargs["num_elements"] = int(spec["num_elements"])
        if spec.get("node_map") is not None:
            kwargs["node_map"] = np.asarray(spec["node_map"])
        return Zone(**kwargs)

    def _load_dataframe(
        self,
        df: Any,
        *,
        columns: Sequence[str] | None = None,
        by: str | None = None,
        value_locations: Mapping[str, Any] | None = None,
    ) -> None:
        """Populate the dataset from a :class:`pandas.DataFrame`.

        Each selected column becomes a nodal variable of a single ordered zone.
        When *by* is given the frame is grouped on that column and one zone is
        produced per group.
        """
        _require_pandas()  # surface a clear error if pandas is unavailable
        vlocs = value_locations or {}
        cols = (
            list(columns) if columns is not None else [c for c in df.columns if c != by]
        )

        def _make_zone(title: str, frame: Any) -> Zone:
            variables = [
                Variable(
                    str(c),
                    values=np.asarray(frame[c].to_numpy()),
                    value_location=_as_value_location(
                        vlocs.get(c, ValueLocation.NODAL)
                    ),
                )
                for c in cols
            ]
            return Zone(title=title, zone_type=ZoneType.ORDERED, variables=variables)

        if by is not None:
            for key, group in df.groupby(by):
                self.add_zone(_make_zone(str(key), group))
        else:
            self.add_zone(_make_zone("", df))

    def to_dataframe(
        self,
        zone: int | Zone = 0,
        *,
        all_zones: bool = False,
    ) -> Any:
        """Return a :class:`pandas.DataFrame` of a zone's nodal data.

        Active nodal variables whose length matches the node count become
        columns; passive, shared, cell-centered, or mismatched variables are
        filled with ``NaN``.  With ``all_zones=True`` every zone is concatenated
        with a leading ``zone`` column.

        Args:
            zone:      Zone index or :class:`Zone` (ignored when *all_zones*).
            all_zones: Concatenate all zones into one frame.
        """
        pd = _require_pandas()
        if all_zones:
            frames = []
            for zi, z in enumerate(self._zones):
                frame = self._zone_dataframe(pd, z)
                frame.insert(0, "zone", zi)
                frames.append(frame)
            if not frames:
                return pd.DataFrame()
            return pd.concat(frames, ignore_index=True)

        target = self._zones[zone] if isinstance(zone, int) else zone
        return self._zone_dataframe(pd, target)

    @staticmethod
    def _zone_dataframe(pd: Any, zone: Zone) -> Any:
        """Build a per-zone DataFrame of nodal variable columns."""
        n = zone.num_nodes
        data: dict[str, npt.NDArray] = {}
        for var in zone.variable:
            arr = var.values
            if (
                arr is not None
                and var.value_location == ValueLocation.NODAL
                and np.asarray(arr).size == n
            ):
                data[var.name] = np.asarray(arr).ravel(order="F")
            else:
                data[var.name] = np.full(n, np.nan)
        return pd.DataFrame(data)

    # ------------------------------------------------------------------
    # Classmethod constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_file(
        cls,
        path: str | os.PathLike,
        *,
        zones: Sequence[int] | None = None,
        variables: Sequence[int | str] | None = None,
        title: str = "",
        file_type: FileType | int | str = FileType.FULL,
    ) -> Dataset:
        """Create a dataset from a single Tecplot file."""
        return cls(
            path,
            zones=zones,
            variables=variables,
            title=title,
            file_type=file_type,
        )

    @classmethod
    def from_files(
        cls,
        paths: Sequence[str | os.PathLike],
        *,
        zones: Sequence[int] | None = None,
        variables: Sequence[int | str] | None = None,
        title: str = "",
    ) -> Dataset:
        """Create a dataset by merging several Tecplot files."""
        return cls(list(paths), zones=zones, variables=variables, title=title)

    @classmethod
    def from_reader(
        cls,
        reader: Any,
        *,
        zones: Sequence[int] | None = None,
        variables: Sequence[int | str] | None = None,
    ) -> Dataset:
        """Create a dataset from an already-open ``Read`` instance."""
        return cls(reader, zones=zones, variables=variables)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, title: str = "") -> Dataset:
        """Create a dataset from a :class:`dict` (see :meth:`_load_dict`)."""
        return cls(dict(data), title=title)

    @classmethod
    def from_dataframe(
        cls,
        df: Any,
        *,
        columns: Sequence[str] | None = None,
        by: str | None = None,
        value_locations: Mapping[str, Any] | None = None,
        title: str = "",
    ) -> Dataset:
        """Create a dataset from a :class:`pandas.DataFrame`."""
        obj = cls(title=title)
        obj._load_dataframe(df, columns=columns, by=by, value_locations=value_locations)
        return obj

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._zones)

    def __iter__(self) -> Iterator[Zone]:
        return iter(self._zones)

    def __repr__(self) -> str:
        return (
            f"Dataset(title={self.title!r}, file_type={self.file_type.name}, "
            f"num_zones={self.num_zones}, num_vars={self.num_vars})"
        )

    def __str__(self) -> str:
        return (
            f"Dataset: {self.title!r}\n"
            f"  File type : {self.file_type.name}\n"
            f"  Zones     : {self.num_zones}\n"
            f"  Variables : {self._variables}"
        )
