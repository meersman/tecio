"""Shared, format-independent base classes for Tecplot readers.

Every format's reader (SZL, PLT, DAT) exposes the same interface:
:class:`TecplotReader` (the open file/dataset), :class:`TecplotZoneReader` (one
zone, split into :class:`TecplotOrderedZoneReader` and
:class:`TecplotFEZoneReader` since the two topologies don't share properties
like dimensions or connectivity), :class:`TecplotVariableReader` (one variable
within a zone, not split by value location or zone topology, neither changes
which properties exist, only the shape ``get_values()`` returns, which the
method already computes from ``zone_type``/``value_location``), and
:class:`TecplotAuxDataReader` (a read-only auxiliary-data mapping). This module
defines that interface once so the three formats can no longer drift apart, and so
downstream code (the ParaView plugin, in particular) can be written against
``TecplotZoneReader``/``TecplotVariableReader`` instead of ``Any``.

Design notes:
    * Instances are immutable: attribute assignment raises :exc:`AttributeError`.
      Internal caches for lazily loaded data (a zone's variable list, node map, and
      aux data) are populated through ``object.__setattr__``, bypassing the
      blocked ``__setattr__`` on purpose, the same trick :mod:`~tecio._meta`
      already uses via ``frozen=True`` dataclasses, applied by hand here because
      these classes also need lazy loading, which a plain frozen dataclass can't
      combine with ``__slots__``.
    * :class:`TecplotZoneReader` and its subclasses split fields into two groups.
      Small, cheap scalar metadata (title, zone type, dimensions or node/element
      counts, solution time, strand ID) is resolved once at construction and
      frozen, since even a per-value C call is cheap and doing it once beats
      re-querying on every access. Data that can be large (the variable list, an
      FE node map that may hold millions of entries) or that a caller may never
      touch (aux data) stays lazily loaded on first access, exactly matching each
      format's existing behaviour.
    * Unlike :mod:`~tecio._meta`, this module imports :mod:`~tecio._constants`
      at runtime rather than only under ``TYPE_CHECKING``. ``_meta`` only ever
      needs the enums as type annotations (erased at runtime by ``from __future__
      import annotations``); ``nodes_per_cell`` below needs the actual
      ``ZoneType`` members to key a lookup table, so a runtime import is
      unavoidable. ``_constants`` is a dependency-free leaf module (no C library,
      no other ``tecio`` submodule), so this introduces no cycle regardless.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import ItemsView, Iterator, KeysView, Mapping, ValuesView
from typing import Any, overload

import numpy as np
import numpy.typing as npt

from ._constants import DataPacking, DataType, FileType, ValueLocation, ZoneType
from ._containers import VariableList, ZoneList, select_variable_arrays

# Nodes per element for the standard finite-element zone types. ORDERED zones are
# computed from active dimensions instead, see TecplotZoneReader.nodes_per_cell.
_NODES_PER_ELEM: dict[ZoneType, int] = {
    ZoneType.FELINESEG: 2,
    ZoneType.FETRIANGLE: 3,
    ZoneType.FEQUADRILATERAL: 4,
    ZoneType.FETETRAHEDRON: 4,
    ZoneType.FEBRICK: 8,
}


def _immutable_setattr(instance: object, name: str, value: object) -> None:
    """Raise, used as every reader class's ``__setattr__``."""
    raise AttributeError(f"{type(instance).__name__} instances are immutable")


def _immutable_delattr(instance: object, name: str) -> None:
    """Raise, used as every reader class's ``__delattr__``."""
    raise AttributeError(f"{type(instance).__name__} instances are immutable")


# ======================================================================================
# TecplotAuxDataReader
# ======================================================================================


class TecplotAuxDataReader(Mapping[str, str], ABC):
    """Read-only view of one Tecplot auxiliary-data mapping.

    Backs dataset-, zone-, and variable-level auxiliary data for all three formats.
    Subclasses supply the underlying ``{name: value}`` mapping through
    :meth:`_load_data`, called at most once, on first access, and cached for the
    life of the object.

    Implements :class:`collections.abc.Mapping`, so ``get``, ``keys``, ``values``,
    ``items``, ``__contains__``, and ``in`` all work without being redefined here.

    Note:
        A concrete subclass must call ``super().__init__()`` first, then store
        any private fields it needs (the data source to lazily load from) with
        ``object.__setattr__``, the same contract as :class:`TecplotZoneReader`
        and :class:`TecplotVariableReader`.
    """

    __slots__ = ("_cache",)
    _cache: dict[str, str] | None

    def __init__(self) -> None:
        object.__setattr__(self, "_cache", None)

    __setattr__ = _immutable_setattr
    __delattr__ = _immutable_delattr

    @abstractmethod
    def _load_data(self) -> dict[str, str]:
        """Return the underlying mapping. Called at most once, lazily."""

    @property
    def _data(self) -> dict[str, str]:
        data = self._cache
        if data is None:
            data = self._load_data()
            object.__setattr__(self, "_cache", data)
        return data

    def __getitem__(self, key: str) -> str:
        return self._data[key]

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._data!r})"

    # Mapping already provides get/keys/values/items via __getitem__ + __iter__ +
    # __len__, but typed overrides are kept here since the base implementations
    # return generic view types rather than the str-specific ones callers expect.

    def keys(self) -> KeysView[str]:
        """Return the auxiliary data names."""
        return self._data.keys()

    def values(self) -> ValuesView[str]:
        """Return the auxiliary data values."""
        return self._data.values()

    def items(self) -> ItemsView[str, str]:
        """Return the ``(name, value)`` pairs."""
        return self._data.items()

    def as_int(self, key: str, default: int | None = None) -> int | None:
        """Return the value for *key* as :class:`int`, or *default* on failure."""
        try:
            return int(self._data[key])
        except (KeyError, ValueError):
            return default

    def as_float(self, key: str, default: float | None = None) -> float | None:
        """Return the value for *key* as :class:`float`, or *default* on failure."""
        try:
            return float(self._data[key])
        except (KeyError, ValueError):
            return default

    def as_bool(self, key: str, default: bool | None = None) -> bool | None:
        """Return the value for *key* as :class:`bool`, or *default* on failure.

        Recognizes ``true``/``t``/``yes``/``y``/``1`` as True and
        ``false``/``f``/``no``/``n``/``0`` as False, case-insensitively.
        """
        try:
            value = self._data[key].strip().lower()
        except KeyError:
            return default
        if value in ("true", "t", "yes", "y", "1"):
            return True
        if value in ("false", "f", "no", "n", "0"):
            return False
        return default


# ======================================================================================
# TecplotVariableReader
# ======================================================================================


class TecplotVariableReader(ABC):
    """Read-only handle to one variable's metadata and data within a zone.

    Every metadata field is resolved by the concrete subclass on each access (a
    single libtecio call, or a lookup in already-parsed metadata), cheap enough
    in every format that this base does not add its own caching layer on top.
    Only :attr:`values` reads actual array data, and only when called.

    Note:
        ``__setattr__`` is blocked here, which also applies inside a subclass's
        own ``__init__``. A concrete subclass must store its private fields with
        ``object.__setattr__(self, "_name", value)`` rather than plain
        assignment, and should declare its own ``__slots__`` for them.
    """

    __slots__ = ()

    __setattr__ = _immutable_setattr
    __delattr__ = _immutable_delattr

    def __repr__(self) -> str:
        parts = [repr(self.name)]
        if self.is_passive():
            parts.append("passive")
        elif self.shared_zone is not None:
            parts.append(f"shared(zone={self.shared_zone})")
        else:
            parts.append(f"dtype={self.data_type.name}")
            values = self.values
            if values is not None:
                parts.append(f"shape={values.shape}")
            if self.value_location == ValueLocation.CELL_CENTERED:
                parts.append("CELL_CENTERED")
        return f"{type(self).__name__}({', '.join(parts)})"

    # -- Abstract: one C call / metadata lookup each, format-specific ------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Variable name string."""

    @property
    @abstractmethod
    def data_type(self) -> DataType:
        """Stored data type."""

    @property
    @abstractmethod
    def value_location(self) -> ValueLocation:
        """NODAL or CELL_CENTERED."""

    @abstractmethod
    def is_passive(self) -> bool:
        """True if this variable has no data in this zone."""

    @property
    @abstractmethod
    def shared_zone(self) -> int | None:
        """1-based source zone index if shared, else None."""

    @property
    @abstractmethod
    def num_values(self) -> int:
        """Number of stored values, 0 for a passive variable."""

    @abstractmethod
    def get_values(
        self,
        value_range: tuple[int | None, int | None] = (None, None),
    ) -> npt.NDArray[Any] | None:
        """Read values from disk or memory.

        Args:
            value_range: 1-based ``(start, end)``, half-open. ``(None, None)``
                reads all values.

        Returns:
            The data array, or ``None`` if the variable is passive. A shared
            variable resolves to its source zone's array.

        Raises:
            ValueError: If only one of ``start``/``end`` is given.
        """

    # -- Shared ------------------------------------------------------------------------

    def is_enabled(self) -> bool:
        """True unless the variable is passive.

        SZL overrides this with the dataset-level enabled flag it actually
        stores; PLT and DAT have no such flag and use this default.
        """
        return not self.is_passive()

    @property
    def values(self) -> npt.NDArray[Any] | None:
        """All values as a NumPy array, or None if passive."""
        return self.get_values()


# ======================================================================================
# TecplotZoneReader
# ======================================================================================


class TecplotZoneReader(ABC):
    """Read-only handle to one zone's metadata in an open Tecplot file.

    This is the common base for :class:`TecplotOrderedZoneReader` and
    :class:`TecplotFEZoneReader`. Only properties that mean the same thing for
    every zone topology live here, dimensions, node maps, and everything else
    that's specific to one topology live on the corresponding subclass instead,
    so accessing e.g. ``.node_map`` on an ordered zone is a plain
    :exc:`AttributeError` rather than a silent ``None``.

    Scalar metadata (title, zone type, solution time, strand ID, datapacking)
    is resolved once at construction and frozen. The variable list and aux
    data stay lazily loaded on first access, since either can be large enough
    that eagerly building them for every zone in a file would be wasteful.

    Args:
        zone_index: 1-based zone index within the dataset.
        title: Zone title.
        zone_type: The zone's :class:`~tecio.libtecio.ZoneType`.
        solution_time: Solution time, 0.0 for static data.
        strand_id: Strand ID, 0 for static data.
        datapacking: On-disk value layout. SZL always supplies
            :attr:`~tecio.libtecio.DataPacking.BLOCK`, binary files have no
            packing distinction on disk; the field exists purely so code can
            treat every zone uniformly regardless of source format.
    """

    __slots__ = (
        "_zone_index",
        "_title",
        "_zone_type",
        "_solution_time",
        "_strand_id",
        "_datapacking",
        "_variable_cache",
        "_auxdata_cache",
    )
    _zone_index: int
    _title: str
    _zone_type: ZoneType
    _solution_time: float
    _strand_id: int
    _datapacking: DataPacking
    _variable_cache: VariableList[TecplotVariableReader] | None
    _auxdata_cache: TecplotAuxDataReader | None

    def __init__(
        self,
        zone_index: int,
        title: str,
        zone_type: ZoneType,
        solution_time: float,
        strand_id: int,
        datapacking: DataPacking,
    ) -> None:
        object.__setattr__(self, "_zone_index", zone_index)
        object.__setattr__(self, "_title", title)
        object.__setattr__(self, "_zone_type", zone_type)
        object.__setattr__(self, "_solution_time", solution_time)
        object.__setattr__(self, "_strand_id", strand_id)
        object.__setattr__(self, "_datapacking", datapacking)
        object.__setattr__(self, "_variable_cache", None)
        object.__setattr__(self, "_auxdata_cache", None)

    __setattr__ = _immutable_setattr
    __delattr__ = _immutable_delattr

    def __repr__(self) -> str:
        title = self.title
        if len(title) > 30:
            title = title[:29] + "\u2026"
        extra = f", aux={len(self.auxdata)}" if len(self.auxdata) else ""
        cls = type(self).__name__
        return f"{cls}({title!r}, {self.zone_type.name}, {self._repr_size()}{extra})"

    @abstractmethod
    def _repr_size(self) -> str:
        """Return the shape portion of __repr__, e.g. ``'I=4, J=3, K=1'``."""

    # -- Frozen scalar metadata --------------------------------------------------------

    @property
    def zone_index(self) -> int:
        """1-based zone index within the dataset."""
        return self._zone_index

    @property
    def title(self) -> str:
        """Zone title string."""
        return self._title

    @property
    def zone_type(self) -> ZoneType:
        """The zone's :class:`~tecio.libtecio.ZoneType`."""
        return self._zone_type

    @property
    def solution_time(self) -> float:
        """Solution time, 0.0 for static data."""
        return self._solution_time

    @property
    def strand_id(self) -> int:
        """Strand ID, 0 for static data."""
        return self._strand_id

    @property
    def datapacking(self) -> DataPacking:
        """On-disk value layout for this zone."""
        return self._datapacking

    def is_enabled(self) -> bool:
        """True unless the zone reports itself disabled.

        Only SZL can report False here; PLT and DAT have no such flag.
        """
        return True

    # -- Lazy: variable list, aux data ------------------------------------------------

    @property
    def variable(self) -> VariableList[TecplotVariableReader]:
        """Variables in this zone, by 0-based index or exact name."""
        variables = self._variable_cache
        if variables is None:
            variables = VariableList(self._load_variables())
            object.__setattr__(self, "_variable_cache", variables)
        return variables

    @abstractmethod
    def _load_variables(self) -> list[TecplotVariableReader]:
        """Construct this zone's variable readers. Called at most once, lazily."""

    @property
    def auxdata(self) -> TecplotAuxDataReader:
        """Zone-level auxiliary data."""
        aux = self._auxdata_cache
        if aux is None:
            aux = self._load_auxdata()
            object.__setattr__(self, "_auxdata_cache", aux)
        return aux

    @abstractmethod
    def _load_auxdata(self) -> TecplotAuxDataReader:
        """Construct this zone's aux-data reader. Called at most once, lazily."""

    # -- Fully generic -----------------------------------------------------------------

    @overload
    def get_array(self, key: int | str) -> npt.NDArray[Any] | None: ...
    @overload
    def get_array(self, key: list[str]) -> tuple[npt.NDArray[Any] | None, ...]: ...

    def get_array(
        self, key: int | str | list[str]
    ) -> npt.NDArray[Any] | None | tuple[npt.NDArray[Any] | None, ...]:
        """Return variable data array(s) for this zone.

        A single key (0-based index or exact name) returns one array. A list of
        exact names returns a tuple of arrays in the order given::

            p = zone.get_array("p")
            x, y, z = zone.get_array(["x", "y", "z"])

        Returns:
            One array (or ``None`` only if the variable is passive) for a scalar
            key; a tuple of such arrays for a list of names.

        Raises:
            KeyError: If a name does not exist.
            IndexError: If an index is out of range.
        """
        return select_variable_arrays(self.variable, key)


# ======================================================================================
# TecplotOrderedZoneReader
# ======================================================================================


class TecplotOrderedZoneReader(TecplotZoneReader):
    """Read-only handle to one ORDERED (IJK-structured) zone.

    Args:
        zone_index: 1-based zone index within the dataset.
        title: Zone title.
        solution_time: Solution time, 0.0 for static data.
        strand_id: Strand ID, 0 for static data.
        datapacking: On-disk value layout.
        i: Nodal I dimension.
        j: Nodal J dimension.
        k: Nodal K dimension.
    """

    __slots__ = ("_i", "_j", "_k")
    _i: int
    _j: int
    _k: int

    def __init__(
        self,
        zone_index: int,
        title: str,
        solution_time: float,
        strand_id: int,
        datapacking: DataPacking,
        i: int,
        j: int,
        k: int,
    ) -> None:
        super().__init__(
            zone_index=zone_index,
            title=title,
            zone_type=ZoneType.ORDERED,
            solution_time=solution_time,
            strand_id=strand_id,
            datapacking=datapacking,
        )
        object.__setattr__(self, "_i", i)
        object.__setattr__(self, "_j", j)
        object.__setattr__(self, "_k", k)

    def _repr_size(self) -> str:
        return f"I={self._i}, J={self._j}, K={self._k}"

    @property
    def I(self) -> int:  # noqa: E743 - match Tecplot IJK convention
        """Nodal I dimension."""
        return self._i

    @property
    def J(self) -> int:
        """Nodal J dimension."""
        return self._j

    @property
    def K(self) -> int:
        """Nodal K dimension."""
        return self._k

    @property
    def dimensions(self) -> tuple[int, int, int]:
        """``(I, J, K)`` dimensions."""
        return (self._i, self._j, self._k)

    @property
    def num_nodes(self) -> int:
        """Number of nodes/points, ``I * J * K``."""
        return self._i * self._j * self._k

    @property
    def num_elements(self) -> int:
        """Number of cells."""
        # Normalize to proper dimension
        if self._k == 1 and self._j == 1:
            # 1D case
            return self._i - 1
        elif self._k == 1 and self._j > 1:
            # 2D case
            return (self._i - 1) * (self._j - 1)
        else:
            # 3D case
            return (self._i - 1) * (self._j - 1) * (self._k - 1)


# ======================================================================================
# TecplotFEZoneReader
# ======================================================================================


class TecplotFEZoneReader(TecplotZoneReader):
    """Read-only handle to one finite-element zone.

    Covers every non-ORDERED zone type, including FEPOLYGON, FEPOLYHEDRON, and FEMIXED:
    :attr:`num_nodes`/:attr:`num_elements`/:attr:`variable`/ :attr:`auxdata` are
    meaningful for all of them today. :attr:`nodes_per_cell` only has a fixed answer for
    the classic types (FELINESEG through FEBRICK) and raises for the rest, and
    :attr:`node_map` may legitimately be None for a zone type whose connectivity a
    format can't yet resolve (e.g. PLT's FEPOLYGON/FEPOLYHEDRON, face-map reading isn't
    implemented there), the same contract this class already had before the Ordered/FE
    split. A dedicated poly reader with real face-map support (a `facemap` property,
    variable-length per-face connectivity) is a reasonable future split once some format
    actually implements it, informed by a real implementation rather than a guess, not
    before.

    Args:
        zone_index: 1-based zone index within the dataset.
        title: Zone title.
        zone_type: The zone's :class:`~tecio.libtecio.ZoneType`.
        solution_time: Solution time, 0.0 for static data.
        strand_id: Strand ID, 0 for static data.
        datapacking: On-disk value layout.
        num_nodes: Number of nodes/points.
        num_elements: Number of elements/cells.
        shared_connectivity: 1-based source zone index if this zone's
            connectivity is shared, else None.
    """

    __slots__ = (
        "_num_nodes",
        "_num_elements",
        "_shared_connectivity",
        "_node_map_cache",
        "_node_map_loaded",
    )
    _num_nodes: int
    _num_elements: int
    _shared_connectivity: int | None
    _node_map_cache: npt.NDArray[np.int64] | None
    _node_map_loaded: bool

    def __init__(
        self,
        zone_index: int,
        title: str,
        zone_type: ZoneType,
        solution_time: float,
        strand_id: int,
        datapacking: DataPacking,
        num_nodes: int,
        num_elements: int,
        shared_connectivity: int | None,
    ) -> None:
        super().__init__(
            zone_index=zone_index,
            title=title,
            zone_type=zone_type,
            solution_time=solution_time,
            strand_id=strand_id,
            datapacking=datapacking,
        )
        object.__setattr__(self, "_num_nodes", num_nodes)
        object.__setattr__(self, "_num_elements", num_elements)
        object.__setattr__(self, "_shared_connectivity", shared_connectivity)
        object.__setattr__(self, "_node_map_cache", None)
        object.__setattr__(self, "_node_map_loaded", False)

    def _repr_size(self) -> str:
        return f"N={self._num_nodes}, E={self._num_elements}"

    @property
    def num_nodes(self) -> int:
        """Number of nodes/points."""
        return self._num_nodes

    @property
    def num_elements(self) -> int:
        """Number of elements/cells."""
        return self._num_elements

    @property
    def shared_connectivity(self) -> int | None:
        """1-based source zone index if connectivity is shared, else None."""
        return self._shared_connectivity

    @property
    def nodes_per_cell(self) -> int:
        """Nodes per cell, fixed by zone type.

        Raises:
            ValueError: For a zone type without a fixed nodes-per-cell count (not
                expected here, :class:`TecplotFEZoneReader` is only ever constructed for
                the classic FE types, but kept as a defensive check rather than an
                unchecked KeyError).
        """
        zt = self.zone_type
        if zt in _NODES_PER_ELEM:
            return _NODES_PER_ELEM[zt]
        raise ValueError(f"ZoneType {zt} does not have a fixed nodes-per-cell count.")

    @property
    def node_map(self) -> npt.NDArray[np.int64] | None:
        """Node connectivity array ``(num_elements, nodes_per_cell)``.

        None only if a format cannot yet resolve connectivity for this zone's type
        (e.g. PLT's FEPOLYGON/FEPOLYHEDRON, face-map reading not yet implemented there;
        such zones would need a future, dedicated poly reader, not this class, once
        that's built).
        """
        if not self._node_map_loaded:
            node_map = self._load_node_map()
            object.__setattr__(self, "_node_map_cache", node_map)
            object.__setattr__(self, "_node_map_loaded", True)
            return node_map
        return self._node_map_cache

    @abstractmethod
    def _load_node_map(self) -> npt.NDArray[np.int64] | None:
        """Read this zone's node connectivity. Called at most once, lazily."""


# ======================================================================================
# TecplotReader
# ======================================================================================


class TecplotReader(ABC):
    """Shared interface for all Tecplot file readers (SZL, PLT, DAT).

    Concrete subclasses (:class:`TecplotSzlReader`, ...) differ in how a file is opened
    and parsed, SZL keeps a live C file handle and queries it on demand, PLT and DAT
    parse eagerly and hold no handle, but expose an identical interface once open, so
    code (including the ParaView plugin) can be written against this base without caring
    which format produced a given file.
    """

    __slots__ = ()

    @abstractmethod
    def __init__(self, path: str) -> None:
        """Open *path* for reading.

        Declared here so that ``ReaderCls(path)`` type-checks against
        ``type[TecplotReader]``, e.g. in :func:`~tecio._io.open`'s dispatch table,
        without needing a cast. Each format's actual constructor may accept a broader
        path type (PLT also takes ``os.PathLike``); ``str`` is the narrowest common
        signature, matching how every call site in this package already invokes it.
        """

    # -- Abstract: resolved differently per format -------------------------------------

    @property
    @abstractmethod
    def path(self) -> str:
        """Source file path."""

    @property
    @abstractmethod
    def file_type(self) -> FileType:
        """FULL, GRID, or SOLUTION."""

    @property
    @abstractmethod
    def title(self) -> str:
        """Dataset title string."""

    @property
    @abstractmethod
    def variables(self) -> list[str]:
        """Ordered list of variable name strings."""

    @property
    @abstractmethod
    def zone(self) -> ZoneList[TecplotZoneReader]:
        """Zones in this file, by 0-based index or slice."""

    @property
    @abstractmethod
    def auxdata(self) -> TecplotAuxDataReader:
        """Dataset-level auxiliary data."""

    @abstractmethod
    def _var_auxdata_at(self, var_index: int) -> TecplotAuxDataReader:
        """Return aux data for variable *var_index* (1-based, unchecked)."""

    def close(self) -> None:  # noqa: B027
        """Release any open resources. No-op unless overridden.

        SZL overrides this to close its C file handle; PLT and DAT hold no handle
        between accesses and use this default.
        """

    # -- Shared ------------------------------------------------------------------------

    def __enter__(self) -> TecplotReader:
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        try:
            self.close()
        except Exception:
            if exc_type is None:
                raise

    @property
    def num_vars(self) -> int:
        """Number of variables in the dataset."""
        return len(self.variables)

    @property
    def num_zones(self) -> int:
        """Number of zones in the file."""
        return len(self.zone)

    @property
    def num_auxdata_items(self) -> int:
        """Number of dataset-level auxiliary data items."""
        return len(self.auxdata)

    @property
    def var_auxdata(self) -> list[TecplotAuxDataReader | None]:
        """Per-variable auxiliary data, 1-based (index 0 is a placeholder).

        Derived from :meth:`get_var_auxdata`, a subclass never implements this
        separately.
        """
        return [None] + [self.get_var_auxdata(i) for i in range(1, self.num_vars + 1)]

    def get_var_auxdata(self, var_index: int) -> TecplotAuxDataReader:
        """Return auxiliary data for variable *var_index* (1-based).

        Raises:
            IndexError: If *var_index* is out of range.
        """
        if var_index < 1 or var_index > self.num_vars:
            raise IndexError(
                f"Variable index {var_index} out of range [1, {self.num_vars}]"
            )
        return self._var_auxdata_at(var_index)

    def get_zone_auxdata(self, zone_index: int) -> TecplotAuxDataReader:
        """Return auxiliary data for zone *zone_index* (1-based).

        Raises:
            IndexError: If *zone_index* is out of range.
        """
        if zone_index < 1 or zone_index > self.num_zones:
            raise IndexError(
                f"Zone index {zone_index} out of range [1, {self.num_zones}]"
            )
        return self.zone[zone_index - 1].auxdata

    def __repr__(self) -> str:
        cls = type(self).__name__
        name = self.path.replace("\\", "/").rsplit("/", 1)[-1]
        try:
            parts = [f"path={name!r}"]
            title = self.title
            if title:
                if len(title) > 40:
                    title = title[:25] + "\u2026"
                parts.append(f"title={title!r}")
            parts += [
                f"file_type={self.file_type.name}",
                f"zones={self.num_zones}",
                f"variables={self.num_vars}",
            ]
            if self.num_auxdata_items:
                parts.append(f"aux={self.num_auxdata_items}")
            return f"{cls}({', '.join(parts)})"
        except Exception:
            return f"{cls}(path={name!r}, <unavailable>)"
