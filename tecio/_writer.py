"""Shared, format-independent base class for Tecplot writers.

Every format's writer (SZL, PLT, DAT) exposes the same lifecycle: construct
with a path and optional variable list (eager or lazy open), buffer auxiliary
data with :meth:`~TecplotWriter.add_auxdataset_dict`/
:meth:`~TecplotWriter.add_auxvar_dict`, write zones with ``write_ijk_zone``/
``write_fe_zone``, and close (directly or via context manager). This module
defines the parts of that lifecycle that are identical across formats once,
so the three writers can no longer drift apart the way the readers had.

Design notes:
    * Unlike the reader hierarchy, writer instances are not made immutable.
      A writer is inherently a mutable, sequentially-appended-to object
      (``current_zone`` grows, aux buffers fill and drain, a file handle
      opens and closes), there's no "already-read, now frozen" state to
      protect the way there was for reader Zone/Variable objects.
    * :meth:`~TecplotWriter.flush_aux` centralizes the aux-data buffer
      draining and key resolution (a 1-based int or a variable name), which
      was previously duplicated nearly verbatim in all three formats. Each
      format only implements the two small hooks that actually differ,
      writing one dataset-level item and one variable-level item.
    * ``_open``, ``close``, ``write_ijk_zone``, and ``write_fe_zone`` stay
      abstract: how a file is opened/closed and how a zone is actually
      written are genuinely format-specific (a live C handle for SZL, a
      global implicit context for PLT's classic API, a plain text file for
      DAT). Their signatures here are the common shape for documentation;
      Python doesn't enforce exact signature matching on ``abstractmethod``,
      so a format may add its own extra keyword-only parameters (SZL's
      ``flush``, for subzone flushing mid-write) without conflict.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import numpy.typing as npt

from ._constants import DataPacking, DataType, FileType, ValueLocation, ZoneType
from ._meta import WriterMeta

_STR_TO_PRECISION: dict[str, DataType] = {
    "single": DataType.FLOAT,
    "float": DataType.FLOAT,
    "double": DataType.DOUBLE,
}


def normalize_precision(
    precision: DataType | str | None, *, allow_none: bool
) -> DataType | None:
    """Return the :class:`~tecio.libtecio.DataType` for *precision*.

    Args:
        precision: ``None``, the enum directly, or a case-insensitive string
            (``"single"``/``"float"``/``"double"``).
        allow_none: Whether ``None`` is a valid result. SZL: yes, per-variable
            data type is inferred automatically from each array, with no
            file-wide override to normalize. PLT and DAT: no, ``precision``
            must resolve to a concrete FLOAT/DOUBLE, but for different
            reasons and with different scope:

            * PLT's classic API has one ``VIsDouble`` flag for the entire
              file (set once at ``tecini142``); every variable, including
              integer-valued data, is declared and stored at that single
              type. There is no per-variable type in PLT's zone header at
              all.
            * DAT's ``precision`` only decides two things: the ASCII
              significant-digit count used to format floating values, and
              which of FLOAT/DOUBLE is declared for variables whose own
              array is itself float-typed. A variable inferred as an integer
              type keeps its own INT32/INT16/BYTE in the zone's ``DT=``
              declaration regardless of ``precision`` (see
              :func:`~tecio._dat_write._resolve_written_type`), since that
              declaration is what Tecplot uses to allocate memory on read,
              not the printed digit count. A value can be written as
              ``1.000000000e0`` and still be declared (and read back) as an
              integer.

    Raises:
        ValueError: If *precision* is ``None`` and *allow_none* is False, or
            if it's neither ``None`` nor FLOAT/DOUBLE (or a recognized string
            alias for one of them).
    """
    if precision is None:
        if allow_none:
            return None
        raise ValueError(
            "precision=None is not supported by this format; use "
            "DataType.FLOAT/DataType.DOUBLE (or 'single'/'double')."
        )
    if isinstance(precision, str):
        try:
            precision = _STR_TO_PRECISION[precision.strip().lower()]
        except KeyError:
            raise ValueError(
                f"precision={precision!r} is not recognized; use 'single' or "
                "'double' (or DataType.FLOAT / DataType.DOUBLE)."
            ) from None
    if precision not in (DataType.FLOAT, DataType.DOUBLE):
        raise ValueError(
            f"precision={precision!r} is not supported; precision only "
            "applies to floating-point variables -- use DataType.FLOAT, "
            "DataType.DOUBLE, or None."
        )
    return precision


class TecplotWriter(ABC):
    """Shared interface and lifecycle for all Tecplot file writers.

    Concrete subclasses (:class:`~tecio.TecplotSzlWriter`, ...) differ in how
    a file is actually opened, closed, and written to, but share the same
    aux-data buffering, variable-list handling, and context-manager
    lifecycle, so application code can be written against this base without
    caring which format is being produced.

    Args:
        path: Output file path.
        title: Dataset title.
        variables: Variable name list. ``None`` defers file creation until
            the first zone-writing call (lazy open).
        file_type: File type enum (FULL, GRID, or SOLUTION).

    Attributes:
        path: Output file path.
        title: Dataset title string.
        variables: Variable name list, or ``None`` if the file has not been
            opened yet.
        file_type: File type (FULL, GRID, or SOLUTION).
        current_zone: Index of the most recently written zone. ``0`` before
            any zone has been written; incremented only after a zone-writing
            method successfully completes.
        auxdataset: Buffered dataset-level auxiliary data, flushed before the
            first zone.
        auxvar: Buffered variable-level auxiliary data, flushed before the
            first zone.
    """

    def __init__(
        self,
        path: str,
        title: str,
        variables: list[str] | None,
        file_type: FileType,
    ) -> None:
        self.path: str = str(path)
        self.title: str = title
        self.variables: list[str] | None = variables
        self.file_type: FileType = file_type
        self.current_zone: int = 0
        self.auxdataset: dict[str, str] = {}
        self.auxvar: dict[int, dict[str, str]] = {}
        self._meta = WriterMeta(
            path=self.path,
            title=self.title,
            file_type=self.file_type,
            file_format=self._file_format,
        )
        if self.variables is not None:
            self._open(self.variables)

    @property
    @abstractmethod
    def _file_format(self) -> str:
        """Format tag for :class:`~tecio._meta.WriterMeta`, e.g. ``'szplt'``."""

    # -- Context manager ---------------------------------------------------------------

    def __enter__(self) -> TecplotWriter:
        """Support ``with`` statement, returns *self*."""
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        """Close the file on context-manager exit.

        The file is closed regardless of whether an exception was raised in
        the ``with`` block. If closing itself raises, that secondary
        exception is only re-raised when the ``with`` block completed
        without error; otherwise the original exception takes precedence.
        """
        try:
            self.close()
        except Exception:
            if exc_type is None:
                raise

    # -- Validation --------------------------------------------------------------------

    def _check_variables(self) -> list[str]:
        """Return the variable list, raising if the file has not been opened yet."""
        if self.variables is None:
            raise RuntimeError(
                "Attempted to access variable name list before they were set. "
                "Ensure variables are set on initialization or zone write."
            )
        return self.variables

    @property
    def meta(self) -> WriterMeta:
        """Read-only record of everything written to this file so far."""
        return self._meta

    # -- File lifecycle: format-specific -----------------------------------------------

    @abstractmethod
    def _open(self, var_names: list[str]) -> None:
        """Open the file/write the header. Called at most once per instance."""

    @abstractmethod
    def close(self) -> None:
        """Finalize and close the file. Safe to call more than once."""

    # -- Aux data: buffering and key resolution are shared, the actual write isn't -----

    def add_auxdataset_dict(self, auxdict: dict[str, Any]) -> None:
        """Buffer dataset-level auxiliary data from a dictionary."""
        self.auxdataset.update(auxdict)

    def add_auxvar_dict(self, auxdict: dict[int, dict[str, Any]]) -> None:
        """Buffer variable-level auxiliary data from a dictionary."""
        self.auxvar.update(auxdict)

    def _resolve_var_index(self, key: int | str) -> int:
        """Return the 0-based variable index for an aux-data key.

        Args:
            key: A 1-based variable index, or an exact variable name.

        Raises:
            IndexError: If a 1-based index key is out of range.
            KeyError: If a name key doesn't match any variable.
            TypeError: If *key* is neither ``int`` nor ``str``.
        """
        if isinstance(key, bool):
            raise TypeError(
                f"Aux data key must be a variable name (str) or 1-based "
                f"index (int), got {key!r}"
            )
        if isinstance(key, int):
            var_idx = key - 1
            if var_idx not in range(len(self._check_variables())):
                raise IndexError(
                    f"Variable index {key} out of bounds "
                    f"[1, {len(self._check_variables())}]"
                )
            return var_idx
        if isinstance(key, str):
            try:
                return self._check_variables().index(key)
            except ValueError as exc:
                raise KeyError(
                    f"Variable aux data key {key!r} not found in variable "
                    f"list ({self.variables})"
                ) from exc
        raise TypeError(
            f"Aux data key must be a variable name (str) or 1-based index "
            f"(int), got {key!r}"
        )

    def flush_aux(self) -> None:
        """Write buffered dataset- and variable-level aux data to the file.

        Called automatically before the first zone is written. Only needed
        directly if you want to flush explicitly, e.g. before checking
        :attr:`meta`.
        """
        for name, value in self.auxdataset.items():
            self._write_dataset_aux_item(str(name), str(value))

        for key, subdict in self.auxvar.items():
            var_idx = self._resolve_var_index(key)
            for name, value in subdict.items():
                self._write_var_aux_item(var_idx + 1, str(name), str(value))

        # Record counts, then clear buffers -- each item is written exactly once.
        self._meta.note_dataset_aux(len(self.auxdataset))
        self._meta.note_var_aux(sum(len(subdict) for subdict in self.auxvar.values()))
        self.auxdataset.clear()
        self.auxvar.clear()

    @abstractmethod
    def _write_dataset_aux_item(self, name: str, value: str) -> None:
        """Write one dataset-level aux item. Called only from :meth:`flush_aux`."""

    @abstractmethod
    def _write_var_aux_item(
        self, one_based_var_index: int, name: str, value: str
    ) -> None:
        """Write one variable-level aux item. Called only from :meth:`flush_aux`."""

    # -- Zone writers: fully format-specific ------------------------------------------

    @abstractmethod
    def write_ijk_zone(
        self,
        data: Sequence[npt.ArrayLike],
        *,
        title: str | None = None,
        variables: list[str] | None = None,
        value_locations: Sequence[ValueLocation] | None = None,
        passive_vars: Sequence[bool | int] | None = None,
        var_sharing: Sequence[int] | None = None,
        solution_time: float = 0.0,
        strand_id: int = 0,
        aux: dict[str, Any] | None = None,
        datapacking: DataPacking | str = DataPacking.BLOCK,
    ) -> None:
        """Write a complete IJK-ordered zone. See the concrete subclass for details."""

    @abstractmethod
    def write_fe_zone(
        self,
        data: Sequence[npt.ArrayLike],
        zone_type: ZoneType,
        *,
        node_map: npt.ArrayLike | None = None,
        title: str | None = None,
        variables: list[str] | None = None,
        passive_vars: Sequence[bool | int] | None = None,
        var_sharing: Sequence[int] | None = None,
        con_sharing: int | None = None,
        solution_time: float = 0.0,
        strand_id: int = 0,
        aux: dict[str, Any] | None = None,
    ) -> None:
        """Write a complete finite-element zone.

        See the concrete subclass for details.
        """
