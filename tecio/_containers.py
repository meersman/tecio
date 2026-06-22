"""Index- and name-based container types for Tecplot data collections.

These containers are shared by the ``tecio`` readers: ``Read.zone`` returns a
:class:`ZoneList` of ``ReadZone`` and ``ReadZone.variable`` returns a
:class:`VariableList` of ``ReadVariable`` for every supported format (SZL, PLT, DAT).
They depend only on small structural protocols (``.name`` for variables, ``.variable``
for zones) so they import nothing from either hierarchy and cannot introduce a circular
dependency.

Access model:

    reader.zone                  # ZoneList
    reader.zone[0]               # ReadZone            (element)
    reader.zone[1:4]             # ZoneList            (sub-collection, same kind)
    reader.zone[0].variable      # VariableList
    reader.zone[0].variable["x"] # ReadVariable        (object: .values, .is_passive)
    reader.zone[0].variable[2]   # ReadVariable        (0-based index)

Subscripting always returns an element or a sub-collection *of the same kind* (never a
raw array). The underlying NumPy data is pulled with ``get_array`` on a single zone,
which mirrors the pandas ``df[...]`` split: a scalar key returns one array, a list of
names returns a tuple of arrays (for unpacking)::

    p = reader.zone[0].get_array("p")  # ndarray | None
    p = reader.zone[0].get_array(2)  # ndarray | None  (0-based index)
    x, y, z = reader.zone[0].get_array(["x", "y", "z"])  # tuple, one per name

There is deliberately **no** cross-zone array accessor. To pull one variable across
many zones (e.g. a transient sequence), iterate explicitly so the outer axis is owned by
your code, and stack only when you know the result is rectangular::

    seq = [z.get_array("p") for z in reader.zone]  # list[ndarray | None]
    stack = np.stack(seq)  # only if shapes all match

Name lookup is exact and case-sensitive throughout, so distinct variables such
as ``"x"`` (a local coordinate) and ``"X"`` (a global coordinate) never
collide, and names that are not valid Python identifiers (``"X [ft]"``,
``"p'"``) resolve like any other key.

Note:
    ``get_array`` returns ``None`` for passive or shared variables, mirroring
    ``ReadVariable.values``. A list of length 1 (``get_array(["p"])``) returns a 1-tuple
    ``(array,)``, not a bare array, sequence-in always yields tuple-out, matching
    ``df[["x"]]`` staying 2-D. A missing variable *name* raises ``KeyError``; an
    out-of-range *index* raises ``IndexError``.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Generic, Protocol, TypeVar, overload

import numpy as np
import numpy.typing as npt

# --------------------------------------------------------------------------------------
# Structural protocols
# --------------------------------------------------------------------------------------


class _HasName(Protocol):
    """Minimal protocol satisfied by every format's variable element."""

    @property
    def name(self) -> str: ...


class _HasNameAndValues(Protocol):
    """A variable element that can surface both its name and its data array."""

    @property
    def name(self) -> str: ...
    @property
    def values(self) -> npt.NDArray | None: ...


class _HasVariableList(Protocol):
    """A zone element exposing a name/index-addressable variable container."""

    @property
    def variable(self) -> VariableList[Any]: ...


_VarT = TypeVar("_VarT", bound=_HasName)
_ZoneT = TypeVar("_ZoneT", bound=_HasVariableList)


# ---------------------------------------------------------------------------
# Shared dispatch helper
# ---------------------------------------------------------------------------


@overload
def select_variable_arrays(
    variables: VariableList[Any], key: int | str
) -> npt.NDArray | None: ...
@overload
def select_variable_arrays(
    variables: VariableList[Any], key: list[str]
) -> tuple[npt.NDArray | None, ...]: ...


def select_variable_arrays(
    variables: VariableList[Any],
    key: int | str | list[str],
) -> npt.NDArray | None | tuple[npt.NDArray | None, ...]:
    """Resolve *key* against *variables* and return the underlying array(s).

    Backs ``ReadZone.get_array`` (and any future ``Zone.get_array``) so the
    scalar-vs-sequence dispatch is defined exactly once.

    Args:
        variables: The zone's :class:`VariableList`.
        key: A single 0-based index or exact name (→ one array), or a list of exact
            names (→ a tuple of arrays, in the order given).

    Returns:
        One array (or ``None`` for a passive/shared variable) for a scalar key; a tuple
        of such arrays for a list of names.

    Raises:
        KeyError:   If a name does not exist.
        IndexError: If an index is out of range.
    """
    if isinstance(key, np.integer):
        key = int(key)
    if isinstance(key, (str, int)):
        return variables[key].values
    return tuple(variables[k].values for k in key)


# ======================================================================================
# VariableList
# ======================================================================================


class VariableList(Generic[_VarT]):
    """Read-only sequence of variables with positional *and* named access.

    Drop-in for the ``list`` previously returned by ``ReadZone.variable``:
    iteration, ``len()``, and integer indexing are unchanged.  A string key
    resolves a variable by its exact, case-sensitive name.

    Subscripting returns the variable *object*; use its ``.values`` (or the
    zone's ``get_array``) to obtain the underlying array.

    Args:
        variables: Ordered list of variable elements (each exposing ``.name``).

    Note:
        If two variables share a name (rare), the first occurrence wins for
        name-based lookup.
    """

    __slots__ = ("_items", "_name_index")

    def __init__(self, variables: list[_VarT]) -> None:
        self._items: list[_VarT] = variables
        # Built lazily on first name lookup so purely positional use pays no
        # cost.  For SZL this also avoids issuing a C call per variable name
        # until a name is actually requested.
        self._name_index: dict[str, int] | None = None

    def _index(self) -> dict[str, int]:
        """Return (building once) the ``name -> position`` lookup table."""
        if self._name_index is None:
            index: dict[str, int] = {}
            for i, var in enumerate(self._items):
                index.setdefault(var.name, i)  # first occurrence wins
            self._name_index = index
        return self._name_index

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[_VarT]:
        return iter(self._items)

    def __getitem__(self, key: int | str) -> _VarT:
        """Return a variable object by 0-based index or exact name.

        Args:
            key: A 0-based integer index or an exact, case-sensitive variable
                name.

        Returns:
            The matching variable element.

        Raises:
            KeyError:   If *key* is a name that does not exist.
            IndexError: If *key* is an out-of-range index.
        """
        if isinstance(key, str):
            try:
                return self._items[self._index()[key]]
            except KeyError:
                raise KeyError(
                    f"No variable named {key!r}. Available: {self.names()}"
                ) from None
        return self._items[key]

    def __contains__(self, key: object) -> bool:
        if isinstance(key, str):
            return key in self._index()
        return key in self._items

    def names(self) -> list[str]:
        """Return the variable names in dataset order."""
        return [var.name for var in self._items]

    def __repr__(self) -> str:
        return f"VariableList({self.names()!r})"


# ======================================================================================
# ZoneList
# ======================================================================================


class ZoneList(Generic[_ZoneT]):
    """Read-only sequence of zones: positional access and slicing only.

    Drop-in for the ``list`` previously returned by ``Read.zone``: iteration, ``len()``,
    and integer indexing are unchanged. Slicing returns another :class:`ZoneList` (not a
    plain ``list``) so navigation composes.

    This container deliberately exposes **no** data-extraction method. Pulling one
    variable across many zones is an explicit loop over the zones, keeping the outer
    (zone) axis owned by the caller (see the module docs).

    Args:
        zones: Ordered list of zone elements (each exposing ``.variable``).
    """

    __slots__ = ("_items",)

    def __init__(self, zones: list[_ZoneT]) -> None:
        self._items: list[_ZoneT] = zones

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[_ZoneT]:
        return iter(self._items)

    def __contains__(self, item: object) -> bool:
        return item in self._items

    @overload
    def __getitem__(self, key: int) -> _ZoneT: ...
    @overload
    def __getitem__(self, key: slice) -> ZoneList[_ZoneT]: ...

    def __getitem__(self, key: int | slice) -> _ZoneT | ZoneList[_ZoneT]:
        """Index a single zone or slice a sub-collection of zones.

        Args:
            key: A 0-based zone index, or a ``slice``.

        Returns:
            A zone element (``int`` key) or a new :class:`ZoneList` (``slice``
            key).

        Raises:
            IndexError: If an integer *key* is out of range.
        """
        if isinstance(key, slice):
            return ZoneList(self._items[key])
        return self._items[key]

    def __repr__(self) -> str:
        return f"ZoneList({len(self._items)} zones)"
