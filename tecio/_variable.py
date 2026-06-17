"""In-memory ``Variable`` container for a single zone-variable data array.

A :class:`Variable` holds the data array and per-zone metadata (data type,
value location, passive flag, and sharing source) for one variable within one
:class:`~tecio._zone.Zone`.  Together ``Dataset -> Zone -> Variable`` form an
in-memory, mutable mirror of the on-disk Tecplot data hierarchy.

The public *read* interface intentionally matches the ``ReadVariable`` classes
in :mod:`tecio.szl`, :mod:`tecio.plt`, and :mod:`tecio.dat` (``name``,
``data_type``, ``value_location``, ``is_passive()``, ``is_enabled()``,
``shared_zone``, ``num_values``, ``values``, ``get_values()``) so that a
:class:`Variable` can be used anywhere a reader's variable is expected --
including the zone-copy routines used by the writers.  Unlike the read-only
variants, the metadata and data here are mutable.

A variable is in exactly one of three states:

    ===========  =============================================================
    State        Meaning
    ===========  =============================================================
    *active*     Owns a NumPy array in :attr:`values`.
    *passive*    Has no data in this zone (:meth:`is_passive` is ``True``).
    *shared*     Borrows data from another zone (:attr:`shared_zone` is set).
    ===========  =============================================================
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from .libtecio import DataType, ValueLocation

if TYPE_CHECKING:
    from ._dataset import Dataset
    from ._zone import Zone

# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

#: Canonical NumPy dtype -> Tecplot :class:`DataType` for the supported types.
_DTYPE_TO_DATATYPE: dict[np.dtype, DataType] = {
    np.dtype(np.float64): DataType.DOUBLE,
    np.dtype(np.float32): DataType.FLOAT,
    np.dtype(np.int32): DataType.INT32,
    np.dtype(np.int16): DataType.INT16,
    np.dtype(np.uint8): DataType.BYTE,
}


def infer_data_type(dtype: npt.DTypeLike) -> DataType:
    """Return the closest Tecplot :class:`DataType` for a NumPy dtype.

    Args:
        dtype: A NumPy dtype or anything :func:`numpy.dtype` accepts.

    Returns:
        The nearest supported :class:`DataType`.  64-bit integers are promoted
        to :attr:`DataType.INT32`; ``float16`` is promoted to
        :attr:`DataType.FLOAT`; signed 8-bit and all unsigned integers map to
        :attr:`DataType.BYTE`.

    Raises:
        ValueError: If *dtype* cannot be mapped to a Tecplot data type.

    Example:
        >>> infer_data_type(np.float32)
        <DataType.FLOAT: 1>
    """
    dt = np.dtype(dtype)
    mapped = _DTYPE_TO_DATATYPE.get(dt)
    if mapped is not None:
        return mapped
    if np.issubdtype(dt, np.floating):
        return DataType.FLOAT if dt.itemsize <= 4 else DataType.DOUBLE
    if np.issubdtype(dt, np.signedinteger):
        return DataType.INT16 if dt.itemsize <= 2 else DataType.INT32
    if np.issubdtype(dt, np.unsignedinteger):
        return DataType.BYTE
    raise ValueError(f"Unsupported dtype for Tecplot data: {dt!r}")


# ===========================================================================
# Variable
# ===========================================================================


class Variable:
    """A single zone-variable data array plus its per-zone metadata.

    Args:
        name:           Variable name.  Within a :class:`~tecio.Dataset` the
                        name is kept consistent across all zones; the canonical
                        list lives on the parent dataset.
        values:         Optional NumPy array for this zone-variable pair.  When
                        provided the variable becomes *active* and any
                        ``is_passive`` / ``shared_zone`` arguments are ignored.
        data_type:      Optional explicit :class:`~tecio.libtecio.DataType`
                        override.  When ``None`` the type is inferred from the
                        array dtype (and falls back to ``FLOAT`` when there is
                        no data, matching the read classes).
        value_location: :class:`~tecio.libtecio.ValueLocation` (NODAL or
                        CELL_CENTERED).  Defaults to ``NODAL``.
        is_passive:     Mark the variable passive (no data in this zone).
        shared_zone:    Source-zone index this variable borrows data from, or
                        ``None``.  The index base matches whatever the source
                        reader/writer uses (it is passed through unchanged).
        zone:           Optional back-reference to the owning :class:`Zone`.

    Example:
        >>> import numpy as np
        >>> v = Variable("pressure", np.linspace(0, 1, 10))
        >>> v.is_passive()
        False
        >>> v.num_values
        10
    """

    def __init__(
        self,
        name: str,
        values: npt.ArrayLike | None = None,
        *,
        data_type: DataType | int | None = None,
        value_location: ValueLocation | int = ValueLocation.NODAL,
        is_passive: bool = False,
        shared_zone: int | None = None,
        zone: Zone | None = None,
    ) -> None:
        self._zone: Zone | None = zone
        self._name: str = str(name)
        self._value_location: ValueLocation = ValueLocation(value_location)
        self._data_type_override: DataType | None = (
            DataType(data_type) if data_type is not None else None
        )
        self._shared_zone: int | None = shared_zone
        self._data: npt.NDArray | None = None
        self._is_passive: bool = bool(is_passive)

        if values is not None:
            # Data always wins: an explicit array makes the variable active.
            self._data = np.asarray(values)
            self._is_passive = False
            self._shared_zone = None
        elif shared_zone is not None:
            self._is_passive = False
        else:
            # No data and not shared -> the only valid state is passive.
            self._is_passive = True

    # ------------------------------------------------------------------
    # Parent-child relationships
    # ------------------------------------------------------------------

    @property
    def zone(self) -> Zone | None:
        """Parent :class:`Zone`, or ``None`` if detached."""
        return self._zone

    @property
    def dataset(self) -> Dataset | None:
        """Owning :class:`~tecio.Dataset`, or ``None`` if detached."""
        return self._zone.dataset if self._zone is not None else None

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Variable name.

        Renaming a single variable directly does not propagate to sibling
        zones; use :meth:`tecio.Dataset.rename_variable` to rename a variable
        consistently across the whole dataset.
        """
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = str(value)

    # ------------------------------------------------------------------
    # Metadata (read parity with ReadVariable)
    # ------------------------------------------------------------------

    @property
    def data_type(self) -> DataType:
        """On-disk :class:`~tecio.libtecio.DataType` for this zone-variable.

        Returns the explicit override when set, otherwise the type inferred
        from the array dtype.  Passive/shared variables with no array report
        :attr:`DataType.FLOAT` as a placeholder.
        """
        if self._data_type_override is not None:
            return self._data_type_override
        if self._data is not None:
            return infer_data_type(self._data.dtype)
        return DataType.FLOAT

    @data_type.setter
    def data_type(self, value: DataType | int | None) -> None:
        self._data_type_override = DataType(value) if value is not None else None

    @property
    def value_location(self) -> ValueLocation:
        """:class:`~tecio.libtecio.ValueLocation` (NODAL or CELL_CENTERED)."""
        return self._value_location

    @value_location.setter
    def value_location(self, value: ValueLocation | int) -> None:
        self._value_location = ValueLocation(value)

    @property
    def shared_zone(self) -> int | None:
        """Source-zone index this variable borrows data from, or ``None``."""
        return self._shared_zone

    @shared_zone.setter
    def shared_zone(self, value: int | None) -> None:
        self._shared_zone = value
        if value is not None:
            self._data = None
            self._is_passive = False

    @property
    def passive(self) -> bool:
        """Whether this variable is passive (writable).

        Setting this to ``True`` clears any data and sharing.  ``is_passive()``
        is the read-only method form kept for parity with the reader classes.
        """
        return self._is_passive

    @passive.setter
    def passive(self, value: bool) -> None:
        self._is_passive = bool(value)
        if self._is_passive:
            self._data = None
            self._shared_zone = None

    def is_passive(self) -> bool:
        """Return ``True`` if the variable has no data in this zone."""
        return self._is_passive

    def is_enabled(self) -> bool:
        """Return ``True`` unless the variable is passive."""
        return not self._is_passive

    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------

    @property
    def values(self) -> npt.NDArray | None:
        """The data array, or ``None`` for passive/shared variables.

        Assigning an array makes the variable active and clears the passive
        flag and any sharing.  Assigning ``None`` clears the data.
        """
        return self._data

    @values.setter
    def values(self, array: npt.ArrayLike | None) -> None:
        if array is None:
            self._data = None
            return
        self._data = np.asarray(array)
        self._is_passive = False
        self._shared_zone = None

    @property
    def num_values(self) -> int:
        """Number of values stored (``0`` for passive/shared variables)."""
        return 0 if self._data is None else int(self._data.size)

    @property
    def shape(self) -> tuple[int, ...] | None:
        """Shape of the data array, or ``None`` if there is no data."""
        return None if self._data is None else self._data.shape

    def get_values(
        self,
        value_range: tuple[int | None, int | None] = (None, None),
    ) -> npt.NDArray | None:
        """Return all values, or a 1-based half-open slice of the flat array.

        Args:
            value_range: ``(None, None)`` returns the full array as stored.
                Otherwise ``(start, end)`` is a 1-based, half-open range over
                the Fortran-flattened values.

        Returns:
            The requested array, or ``None`` for passive/shared variables.

        Raises:
            ValueError: If only one of *start* / *end* is given.
        """
        if self._data is None:
            return None
        start, end = value_range
        if start is None and end is None:
            return self._data
        if start is None or end is None:
            raise ValueError("Both start and end indices must be specified.")
        flat = self._data.ravel(order="F")
        return flat[start - 1 : end - 1]

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def copy(self) -> Variable:
        """Return a detached deep copy of this variable (no parent zone)."""
        return Variable(
            self._name,
            values=None if self._data is None else np.array(self._data),
            data_type=self._data_type_override,
            value_location=self._value_location,
            is_passive=self._is_passive,
            shared_zone=self._shared_zone,
        )

    def __repr__(self) -> str:
        if self._is_passive:
            state = "passive"
        elif self._shared_zone is not None:
            state = f"shared<-{self._shared_zone}"
        else:
            state = f"num_values={self.num_values}"
        return (
            f"Variable(name={self._name!r}, "
            f"data_type={self.data_type.name}, "
            f"value_location={self._value_location.name}, {state})"
        )
