"""In-memory ``Zone`` container and the shared ``AuxData`` mapping.

A :class:`Zone` groups one block of grid/solution data: its topology
(``zone_type`` and, for FE zones, a ``node_map``), unsteady metadata
(``solution_time``, ``strand_id``), zone-level auxiliary data, and an ordered
list of :class:`~tecio._variable.Variable` objects -- one per dataset variable.

The public *read* interface matches the ``ReadZone`` classes in
:mod:`tecio.szl`, :mod:`tecio.plt`, and :mod:`tecio.dat` (``title``,
``zone_type``, ``dimensions``, ``num_nodes``, ``num_elements``,
``nodes_per_cell``, ``solution_time``, ``strand_id``, ``node_map``,
``shared_connectivity``, ``auxdata``, ``variable``, ``get_array``,
``is_enabled()``, ``datapacking``) so a zone can be fed straight into the
writers.  Dimensions for ordered zones and node/element counts for FE zones are
inferred from the variable arrays / node map when not supplied explicitly.

Connectivity sharing:
    Like the readers, an FE zone may inherit its connectivity from an earlier
    zone rather than owning a :attr:`node_map` of its own.  Sharing is stored as
    a direct reference to the source :class:`Zone`, so :attr:`node_map` reads
    through to that source and :attr:`shared_connectivity` reports the source
    zone's 1-based index (derived from its current dataset position, so it
    survives zone reordering).  :meth:`tecio.Dataset.branch_connectivity` turns
    a shared node map into an owned copy.

Zones can be built directly from ``{"name": array}`` dictionaries via
:meth:`Zone.ijk_from_dict` (ordered) and :meth:`Zone.fe_from_dict` (finite
element, with the node map supplied separately).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from .libtecio import DataPacking, DataType, ValueLocation, ZoneType
from ._variable import Variable, coerce_value_location

if TYPE_CHECKING:
    from ._dataset import Dataset

# ======================================================================================
# Module-level constants and helpers
# ======================================================================================

#: Nodes per element for the simple (cell-based) FE zone types.
_NODES_PER_ELEM: dict[ZoneType, int] = {
    ZoneType.FELINESEG: 2,
    ZoneType.FETRIANGLE: 3,
    ZoneType.FEQUADRILATERAL: 4,
    ZoneType.FETETRAHEDRON: 4,
    ZoneType.FEBRICK: 8,
}

#: FE zone types supported end-to-end by the writers.
_FE_SIMPLE: frozenset[ZoneType] = frozenset(_NODES_PER_ELEM)

#: Default FE zone type inferred from a node map's nodes-per-cell.  A count of 4
#: is ambiguous (tetrahedron vs. quadrilateral); tetrahedron is assumed, so pass
#: ``zone_type`` explicitly for quad-surface meshes.
_NODES_TO_FE_TYPE: dict[int, ZoneType] = {
    2: ZoneType.FELINESEG,
    3: ZoneType.FETRIANGLE,
    4: ZoneType.FETETRAHEDRON,
    8: ZoneType.FEBRICK,
}


def _coerce_zone_type(value: ZoneType | int | str) -> ZoneType:
    """Coerce an enum / int / case-insensitive name to a :class:`ZoneType`."""
    if isinstance(value, ZoneType):
        return value
    if isinstance(value, str):
        return ZoneType[value.strip().upper()]
    return ZoneType(int(value))


def _infer_fe_zone_type(node_map: npt.NDArray) -> ZoneType:
    """Infer a simple FE :class:`ZoneType` from a node map's nodes-per-cell."""
    npc = int(node_map.shape[1]) if node_map.ndim == 2 else int(node_map.shape[-1])
    try:
        return _NODES_TO_FE_TYPE[npc]
    except KeyError:
        raise ValueError(
            f"Cannot infer an FE zone type from nodes-per-cell={npc}; "
            "pass zone_type explicitly."
        ) from None


def _variables_from_dict(
    data: Mapping[str, npt.ArrayLike],
    value_locations: Mapping[str, ValueLocation | int | str] | None,
) -> list[Variable]:
    """Build a list of active :class:`Variable` objects from a name->array map."""
    vlocs = value_locations or {}
    return [
        Variable(
            str(name),
            values=np.asarray(arr),
            value_location=coerce_value_location(
                vlocs.get(name, ValueLocation.NODAL)
            ),
        )
        for name, arr in data.items()
    ]


# ======================================================================================
# AuxData
# ======================================================================================


class AuxData(dict):
    """Mutable ``dict`` of Tecplot auxiliary data with typed accessors.

    Tecplot stores auxiliary data as ``name -> value`` string pairs at the
    dataset, variable, and zone levels.  This subclass keeps the full ``dict``
    API (so it is trivial to build and update) and adds the same typed
    converters exposed by the read-only ``ReadAuxData`` classes.

    Example:
        >>> aux = AuxData({"Iteration": "42", "Converged": "true"})
        >>> aux.as_int("Iteration")
        42
        >>> aux.as_bool("Converged")
        True
    """

    def as_int(self, key: str, default: int | None = None) -> int | None:
        """Return value for *key* as :class:`int`, or *default* on failure."""
        try:
            return int(self[key])
        except (KeyError, ValueError, TypeError):
            return default

    def as_float(self, key: str, default: float | None = None) -> float | None:
        """Return value for *key* as :class:`float`, or *default* on failure."""
        try:
            return float(self[key])
        except (KeyError, ValueError, TypeError):
            return default

    def as_bool(self, key: str, default: bool | None = None) -> bool | None:
        """Return value for *key* as :class:`bool`, or *default* on failure.

        Recognises ``true/t/yes/y/1`` and ``false/f/no/n/0`` case-insensitively.
        """
        try:
            value = str(self[key]).strip().lower()
        except KeyError:
            return default
        if value in ("true", "t", "yes", "y", "1"):
            return True
        if value in ("false", "f", "no", "n", "0"):
            return False
        return default


# ======================================================================================
# Zone
# ======================================================================================


class Zone:
    """A single zone: topology, metadata, and a list of variables.

    Args:
        title:          Zone title.
        zone_type:      :class:`~tecio.libtecio.ZoneType`.  Defaults to
                        ``ORDERED``.
        dimensions:     ``(i, j, k)`` for ordered zones.  When ``None`` the
                        dimensions are inferred from the first nodal variable
                        array (or a cell-centered array, padded by one).
        num_nodes:      Explicit node count for FE zones (otherwise inferred
                        from ``node_map`` / variable lengths).
        num_elements:   Explicit element count for FE zones (otherwise inferred
                        from ``node_map``).
        solution_time:  Solution time for transient data.
        strand_id:      Strand ID grouping related time steps.
        node_map:       FE connectivity of shape ``(num_elements,
                        nodes_per_cell)`` with 1-based node indices.  May be
                        ``None`` for an FE zone that shares its connectivity via
                        *connectivity_source*.
        connectivity_source: Source :class:`Zone` this FE zone inherits its
                        connectivity from.  When set, :attr:`node_map` reads
                        through to that zone and :attr:`shared_connectivity`
                        reports its 1-based index.
        variables:      Initial list of :class:`Variable` objects.  When the
                        zone is added to a dataset the list is reconciled with
                        the dataset variable list.
        aux:            Zone-level auxiliary data.
        dataset:        Back-reference to the owning :class:`~tecio.Dataset`
                        (normally set by :meth:`tecio.Dataset.add_zone`).

    Example:
        >>> import numpy as np
        >>> x = np.linspace(0, 1, 11)
        >>> z = Zone("line", variables=[Variable("x", x)])
        >>> z.dimensions
        (11, 1, 1)
    """

    def __init__(
        self,
        title: str = "",
        zone_type: ZoneType | int | str = ZoneType.ORDERED,
        *,
        dimensions: tuple[int, int, int] | None = None,
        num_nodes: int | None = None,
        num_elements: int | None = None,
        solution_time: float = 0.0,
        strand_id: int = 0,
        node_map: npt.ArrayLike | None = None,
        connectivity_source: Zone | None = None,
        variables: list[Variable] | None = None,
        aux: Mapping[str, Any] | None = None,
        dataset: Dataset | None = None,
    ) -> None:
        self._dataset: Dataset | None = dataset
        self.title: str = str(title)
        self.zone_type: ZoneType = _coerce_zone_type(zone_type)
        self.solution_time: float = float(solution_time)
        self.strand_id: int = int(strand_id)
        self._node_map: npt.NDArray | None = (
            None if node_map is None else np.asarray(node_map)
        )
        self._connectivity_source: Zone | None = connectivity_source
        self.auxdata: AuxData = AuxData(aux or {})

        self._dimensions: tuple[int, int, int] | None = (
            tuple(int(d) for d in dimensions) if dimensions is not None else None
        )
        self._num_nodes: int | None = (
            int(num_nodes) if num_nodes is not None else None
        )
        self._num_elements: int | None = (
            int(num_elements) if num_elements is not None else None
        )

        self._variable: list[Variable] = []
        for var in variables or []:
            self._attach_variable(var)

    # ----------------------------------------------------------------------------------
    # Alternative constructors
    # ----------------------------------------------------------------------------------

    @classmethod
    def ijk_from_dict(
        cls,
        data: Mapping[str, npt.ArrayLike],
        *,
        title: str = "",
        value_locations: Mapping[str, ValueLocation | int | str] | None = None,
        dimensions: tuple[int, int, int] | None = None,
        solution_time: float = 0.0,
        strand_id: int = 0,
        aux: Mapping[str, Any] | None = None,
    ) -> Zone:
        """Build an ordered (IJK) zone from a ``{"name": array}`` mapping.

        Each entry becomes a variable; dimensions are inferred from the arrays
        unless *dimensions* is given.  Use *value_locations* to mark individual
        variables as cell-centered (any array/enum/name accepted).

        Example:
            >>> z = Zone.ijk_from_dict({"x": np.arange(5.0), "p": np.arange(5.0)})
            >>> z.dimensions
            (5, 1, 1)
        """
        return cls(
            title=title,
            zone_type=ZoneType.ORDERED,
            dimensions=dimensions,
            solution_time=solution_time,
            strand_id=strand_id,
            variables=_variables_from_dict(data, value_locations),
            aux=aux,
        )

    @classmethod
    def fe_from_dict(
        cls,
        data: Mapping[str, npt.ArrayLike],
        node_map: npt.ArrayLike,
        *,
        zone_type: ZoneType | int | str | None = None,
        title: str = "",
        value_locations: Mapping[str, ValueLocation | int | str] | None = None,
        solution_time: float = 0.0,
        strand_id: int = 0,
        aux: Mapping[str, Any] | None = None,
    ) -> Zone:
        """Build a finite-element zone from a mapping plus a *node_map*.

        Args:
            data:            ``{"name": array}`` of nodal/cell-centered arrays.
            node_map:        Connectivity of shape ``(num_elements,
                             nodes_per_cell)`` with 1-based node indices.
            zone_type:       FE :class:`~tecio.libtecio.ZoneType`.  When ``None``
                             it is inferred from the node map's nodes-per-cell
                             (a count of 4 defaults to ``FETETRAHEDRON``; pass
                             ``FEQUADRILATERAL`` explicitly for quad meshes).
            value_locations: Optional per-variable value-location overrides.

        Example:
            >>> nm = np.array([[1, 2, 3], [2, 3, 4]])
            >>> z = Zone.fe_from_dict({"x": np.arange(4.0)}, nm)
            >>> z.zone_type.name
            'FETRIANGLE'
        """
        nm = np.asarray(node_map)
        zt = _infer_fe_zone_type(nm) if zone_type is None else _coerce_zone_type(
            zone_type
        )
        return cls(
            title=title,
            zone_type=zt,
            node_map=nm,
            solution_time=solution_time,
            strand_id=strand_id,
            variables=_variables_from_dict(data, value_locations),
            aux=aux,
        )

    # ----------------------------------------------------------------------------------
    # Parent-child relationships
    # ----------------------------------------------------------------------------------

    @property
    def dataset(self) -> Dataset | None:
        """Owning :class:`~tecio.Dataset`, or ``None`` if detached."""
        return self._dataset

    @property
    def variable(self) -> list[Variable]:
        """Ordered list of :class:`Variable` objects (0-based)."""
        return self._variable

    def _attach_variable(self, var: Variable) -> None:
        """Take ownership of *var* and append it to this zone."""
        var._zone = self
        self._variable.append(var)

    # ----------------------------------------------------------------------------------
    # Variable access / mutation
    # ----------------------------------------------------------------------------------

    def get_variable(self, key: int | str) -> Variable:
        """Return a variable by 0-based index or by name (case-insensitive).

        Raises:
            IndexError: If an integer index is out of range.
            KeyError:   If a name does not match any variable.
        """
        if isinstance(key, (int, np.integer)):
            return self._variable[int(key)]
        low = str(key).lower()
        for var in self._variable:
            if var.name.lower() == low:
                return var
        raise KeyError(f"Variable {key!r} not found in zone {self.title!r}.")

    def set_variable_values(self, key: int | str, values: npt.ArrayLike) -> Variable:
        """Set the data array for an existing variable in this zone."""
        var = self.get_variable(key)
        var.values = np.asarray(values)
        return var

    def add_variable(
        self,
        name: str,
        values: npt.ArrayLike | None = None,
        *,
        value_location: ValueLocation | int | str = ValueLocation.NODAL,
        data_type: DataType | int | None = None,
    ) -> Variable:
        """Add (or update) a variable in this zone.

        When the zone is attached to a dataset, the variable is first created
        across the whole dataset (passive in every other zone) so the dataset
        stays rectangular; the data, if given, is then assigned to this zone.

        Returns:
            The :class:`Variable` instance for this zone.
        """
        if self._dataset is not None:
            self._dataset.add_variable(
                name, value_location=value_location, data_type=data_type
            )
            var = self.get_variable(name)
            var.value_location = value_location
            if data_type is not None:
                var.data_type = data_type
            if values is not None:
                var.values = np.asarray(values)
            return var

        # Detached zone: keep it local.
        try:
            var = self.get_variable(name)
        except KeyError:
            var = Variable(
                name,
                value_location=value_location,
                data_type=data_type,
                zone=self,
            )
            self._variable.append(var)
        if values is not None:
            var.values = np.asarray(values)
        return var

    def get_array(
        self, key: int | str | list[str]
    ) -> npt.NDArray | None | tuple[npt.NDArray | None, ...]:
        """Return variable data array(s) for this zone (reader parity).

        A single key (0-based index or exact name) returns one array; a list of
        names returns a tuple of arrays in order, suitable for unpacking::

            p = zone.get_array("p")
            x, y, z = zone.get_array(["x", "y", "z"])

        A passive variable resolves to ``None``; a shared variable reads through
        to its source array, mirroring the readers.

        Raises:
            KeyError:   If a name does not exist.
            IndexError: If an index is out of range.
        """
        if isinstance(key, list):
            return tuple(self.get_variable(k).values for k in key)
        return self.get_variable(key).values

    def __getattr__(self, name: str) -> Any:
        """Access a variable's data array as ``zone.<variable_name>``.

        Only consulted when normal attribute lookup fails.  Names are matched
        case-insensitively against the zone's variables.
        """
        if name.startswith("_"):
            raise AttributeError(name)
        variables = self.__dict__.get("_variable")
        if variables:
            low = name.lower()
            for var in variables:
                if var.name.lower() == low:
                    return var.values
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}"
        )

    # ----------------------------------------------------------------------------------
    # Connectivity
    # ----------------------------------------------------------------------------------

    @property
    def node_map(self) -> npt.NDArray | None:
        """FE connectivity ``(num_elements, nodes_per_cell)``, 1-based indices.

        Reads through to the source zone when the connectivity is shared; assign
        an array to make the zone own its connectivity (which clears any share).
        """
        if self._connectivity_source is not None:
            return self._connectivity_source.node_map
        return self._node_map

    @node_map.setter
    def node_map(self, value: npt.ArrayLike | None) -> None:
        self._node_map = None if value is None else np.asarray(value)
        if value is not None:
            self._connectivity_source = None

    @property
    def shared_connectivity(self) -> int | None:
        """1-based index of the source zone this zone's connectivity is shared from.

        Derived from the source zone's current dataset position (so it survives
        reordering).  ``None`` when the zone owns its :attr:`node_map` (always
        ``None`` for ordered zones, which have no explicit connectivity).
        """
        source = self._connectivity_source
        if source is None:
            return None
        dataset = source.dataset
        if dataset is None:
            return None
        try:
            return dataset.zone.index(source) + 1
        except ValueError:
            return None

    @shared_connectivity.setter
    def shared_connectivity(self, value: int | Zone | None) -> None:
        if value is None or value == 0:
            self._connectivity_source = None
            return
        if isinstance(value, Zone):
            self.share_connectivity_from(value)
            return
        dataset = self.dataset
        if dataset is None:
            raise ValueError(
                "Cannot resolve shared_connectivity by index on a detached "
                "zone; pass the source Zone to share_connectivity_from()."
            )
        self.share_connectivity_from(dataset.zone[int(value) - 1])

    def share_connectivity_from(self, source: Zone) -> None:
        """Share this zone's connectivity from *source* (a read-through reference)."""
        self._connectivity_source = source
        self._node_map = None

    @property
    def connectivity_source(self) -> Zone | None:
        """The source :class:`Zone` connectivity is shared from, or ``None``."""
        return self._connectivity_source

    def shares_connectivity(self) -> bool:
        """Return ``True`` if this zone reads its connectivity from a source zone."""
        return self._connectivity_source is not None

    # ----------------------------------------------------------------------------------
    # Dimensions / counts (read parity with ReadZone)
    # ----------------------------------------------------------------------------------

    def _first_with_data(
        self, *, nodal: bool = False, cell: bool = False
    ) -> Variable | None:
        """Return the first variable resolving to data of the requested location."""
        for var in self._variable:
            if var.values is None:
                continue
            loc = var.value_location
            if nodal and loc != ValueLocation.NODAL:
                continue
            if cell and loc != ValueLocation.CELL_CENTERED:
                continue
            return var
        return None

    def _infer_ordered_dimensions(self) -> tuple[int, int, int]:
        """Infer ``(i, j, k)`` from a variable array for an ordered zone."""
        var = self._first_with_data(nodal=True)
        if var is not None and var.values is not None:
            shp = var.values.shape
            return tuple(int(shp[d]) if d < len(shp) else 1 for d in range(3))
        var = self._first_with_data(cell=True)
        if var is not None and var.values is not None:
            shp = var.values.shape
            return tuple(int(shp[d]) + 1 if d < len(shp) else 1 for d in range(3))
        return (0, 0, 0)

    @property
    def dimensions(self) -> tuple[int, int, int]:
        """``(i, j, k)`` for ordered zones, ``(nodes, elements, 0)`` for FE."""
        if self.zone_type == ZoneType.ORDERED:
            if self._dimensions is not None:
                return self._dimensions
            return self._infer_ordered_dimensions()
        return (self.num_nodes, self.num_elements, 0)

    def set_dimensions(self, i: int, j: int = 1, k: int = 1) -> None:
        """Set explicit ``(i, j, k)`` dimensions for an ordered zone."""
        self._dimensions = (int(i), int(j), int(k))

    @property
    def num_nodes(self) -> int:
        """Number of nodes/points in the zone."""
        if self.zone_type == ZoneType.ORDERED:
            i, j, k = self.dimensions
            return max(i, 1) * max(j, 1) * max(k, 1)
        if self._num_nodes is not None:
            return self._num_nodes
        node_map = self.node_map
        if node_map is not None:
            return int(np.asarray(node_map).max())
        var = self._first_with_data(nodal=True)
        return int(var.values.size) if var is not None else 0

    @property
    def num_elements(self) -> int:
        """Number of elements (equals ``num_nodes`` for ordered zones)."""
        if self.zone_type == ZoneType.ORDERED:
            return self.num_nodes
        if self._num_elements is not None:
            return self._num_elements
        node_map = self.node_map
        if node_map is not None:
            return int(np.asarray(node_map).shape[0])
        var = self._first_with_data(cell=True)
        return int(var.values.size) if var is not None else 0

    @property
    def nodes_per_cell(self) -> int:
        """Nodes per cell, fixed for FE types and inferred for ordered zones.

        Raises:
            ValueError: For zone types without a fixed nodes-per-cell count.
        """
        zt = self.zone_type
        if zt in _NODES_PER_ELEM:
            return _NODES_PER_ELEM[zt]
        if zt == ZoneType.ORDERED:
            active_dims = sum(1 for d in self.dimensions if d > 1)
            return 2**active_dims
        raise ValueError(f"ZoneType {zt} has no fixed nodes-per-cell count.")

    @property
    def datapacking(self) -> DataPacking:
        """Always :attr:`~tecio.libtecio.DataPacking.BLOCK` for parity."""
        return DataPacking.BLOCK

    def is_enabled(self) -> bool:
        """Always ``True`` for an in-memory zone."""
        return True

    # ----------------------------------------------------------------------------------
    # Misc
    # ----------------------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._variable)

    def __repr__(self) -> str:
        if self.zone_type == ZoneType.ORDERED:
            size = f"dimensions={self.dimensions}"
        else:
            size = f"num_nodes={self.num_nodes}, num_elements={self.num_elements}"
            shared = self.shared_connectivity
            if shared is not None:
                size += f", shared_connectivity={shared}"
        return (
            f"Zone(title={self.title!r}, zone_type={self.zone_type.name}, "
            f"{size}, num_variables={len(self._variable)})"
        )
