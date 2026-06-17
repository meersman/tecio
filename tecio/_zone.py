"""In-memory ``Zone`` container and the shared ``AuxData`` mapping.

A :class:`Zone` groups one block of grid/solution data: its topology
(``zone_type`` and, for FE zones, a ``node_map``), unsteady metadata
(``solution_time``, ``strand_id``), zone-level auxiliary data, and an ordered
list of :class:`~tecio._variable.Variable` objects -- one per dataset variable.

The public *read* interface matches the ``ReadZone`` classes in
:mod:`tecio.szl`, :mod:`tecio.plt`, and :mod:`tecio.dat` (``title``,
``zone_type``, ``dimensions``, ``num_nodes``, ``num_elements``,
``nodes_per_cell``, ``solution_time``, ``strand_id``, ``node_map``,
``auxdata``, ``variable``, ``is_enabled()``, ``datapacking``) so a zone can be
fed straight into the writers.  Dimensions for ordered zones and node/element
counts for FE zones are inferred from the variable arrays / node map when not
supplied explicitly.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from ._variable import Variable
from .libtecio import DataPacking, DataType, ValueLocation, ZoneType

if TYPE_CHECKING:
    from ._dataset import Dataset

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

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


# ===========================================================================
# AuxData
# ===========================================================================


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


# ===========================================================================
# Zone
# ===========================================================================


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
                        nodes_per_cell)`` with 1-based node indices.
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
        zone_type: ZoneType | int = ZoneType.ORDERED,
        *,
        dimensions: tuple[int, int, int] | None = None,
        num_nodes: int | None = None,
        num_elements: int | None = None,
        solution_time: float = 0.0,
        strand_id: int = 0,
        node_map: npt.ArrayLike | None = None,
        variables: list[Variable] | None = None,
        aux: Mapping[str, Any] | None = None,
        dataset: Dataset | None = None,
    ) -> None:
        self._dataset: Dataset | None = dataset
        self.title: str = str(title)
        self.zone_type: ZoneType = ZoneType(zone_type)
        self.solution_time: float = float(solution_time)
        self.strand_id: int = int(strand_id)
        self.node_map: npt.NDArray | None = (
            None if node_map is None else np.asarray(node_map)
        )
        self.auxdata: AuxData = AuxData(aux or {})

        self._dimensions: tuple[int, int, int] | None = (
            tuple(int(d) for d in dimensions) if dimensions is not None else None
        )
        self._num_nodes: int | None = int(num_nodes) if num_nodes is not None else None
        self._num_elements: int | None = (
            int(num_elements) if num_elements is not None else None
        )

        self._variable: list[Variable] = []
        for var in variables or []:
            self._attach_variable(var)

    # ------------------------------------------------------------------
    # Parent-child relationships
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Variable access / mutation
    # ------------------------------------------------------------------

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
        value_location: ValueLocation | int = ValueLocation.NODAL,
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
            var.value_location = ValueLocation(value_location)
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

    # ------------------------------------------------------------------
    # Dimensions / counts (read parity with ReadZone)
    # ------------------------------------------------------------------

    def _first_active(
        self, *, nodal: bool = False, cell: bool = False
    ) -> Variable | None:
        """Return the first variable holding data of the requested location."""
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
        var = self._first_active(nodal=True)
        if var is not None and var.values is not None:
            shp = var.values.shape
            return tuple(int(shp[d]) if d < len(shp) else 1 for d in range(3))
        var = self._first_active(cell=True)
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
        if self.node_map is not None:
            return int(np.asarray(self.node_map).max())
        var = self._first_active(nodal=True)
        return int(var.values.size) if var is not None else 0

    @property
    def num_elements(self) -> int:
        """Number of elements (equals ``num_nodes`` for ordered zones)."""
        if self.zone_type == ZoneType.ORDERED:
            return self.num_nodes
        if self._num_elements is not None:
            return self._num_elements
        if self.node_map is not None:
            return int(np.asarray(self.node_map).shape[0])
        var = self._first_active(cell=True)
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

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._variable)

    def __repr__(self) -> str:
        if self.zone_type == ZoneType.ORDERED:
            size = f"dimensions={self.dimensions}"
        else:
            size = f"num_nodes={self.num_nodes}, num_elements={self.num_elements}"
        return (
            f"Zone(title={self.title!r}, zone_type={self.zone_type.name}, "
            f"{size}, num_variables={len(self._variable)})"
        )
