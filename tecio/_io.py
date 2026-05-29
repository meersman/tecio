"""Entry point for tecio such that IO capability can be called from a single function.

Supported modes mirror Python's built-in :func:`open`:

    ======  ================================================================
    Mode    Behaviour
    ======  ================================================================
    ``r``   Open existing file for reading.  Raises :exc:`FileNotFoundError`
            if the file does not exist.
    ``w``   Open file for writing.  Overwrites any existing file.
    ``x``   Exclusive creation — open for writing only if the file does
            **not** already exist.  Raises :exc:`FileExistsError` otherwise.
    ``a``   Append — stream all zones from the existing file into a new
            temporary file, then continue writing new zones into that same
            file.  On :meth:`~AppendWrite.close` the temporary file
            atomically replaces the original path.  Returns an
            :class:`AppendWrite` object whose write API is identical to the
            format's ``Write`` class.
    ``a+``  Append-read — same streaming copy as ``a``, but the returned
            :class:`AppendReadWrite` object also exposes the full ``Read``
            API populated from the *original* file so that existing zone
            data can be inspected while new zones are written.
    ======  ================================================================

Note:
    ``"a"`` and ``"a+"`` work by reading the source file in full and re-writing
    it zone-by-zone into a temporary sibling file, then leaving the ``Write``
    handle open for the caller to add further zones.  On close the temporary
    file is atomically renamed over the original path (POSIX) or replaced
    (Windows).  This is the only safe approach because PLT and SZL are
    sequential binary formats with no in-place editing capability.

Note:
    FEPOLYGON and FEPOLYHEDRON zones are not copied during append operations
    because the ``Write`` API does not yet expose a poly-zone writer.  A
    :exc:`NotImplementedError` is raised if such zones are encountered in the
    source file.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from . import dat, plt, szl
from .libtecio import FileType, ValueLocation, ZoneType

# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

_HANDLERS: dict[str, dict[str, Any]] = {
    ".szplt": {
        "r": szl.Read,
        "w": szl.Write,
        "x": None,  # filled below after class definitions
        "a": None,
        "a+": None,
    },
    ".plt": {
        "r": plt.Read,
        "w": plt.Write,
        "x": None,
        "a": None,
        "a+": None,
    },
    ".bin": {
        "r": plt.Read,
        "w": plt.Write,
        "x": None,
        "a": None,
        "a+": None,
    },
    ".dat": {
        "r": dat.Read,
        "w": dat.Write,
        "x": None,
        "a": None,
        "a+": None,
    },
    ".tec": {
        "r": dat.Read,
        "w": dat.Write,
        "x": None,
        "a": None,
        "a+": None,
    },
}

# FE zone types that the Write API supports for zone copying.
_FE_SIMPLE: frozenset[ZoneType] = frozenset({
    ZoneType.FELINESEG,
    ZoneType.FETRIANGLE,
    ZoneType.FEQUADRILATERAL,
    ZoneType.FETETRAHEDRON,
    ZoneType.FEBRICK,
})


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _reader_for_ext(ext: str) -> type:
    """Return the Read class for *ext*."""
    return _HANDLERS[ext]["r"]


def _writer_for_ext(ext: str) -> type:
    """Return the Write class for *ext*."""
    return _HANDLERS[ext]["w"]


def _copy_zones(reader: szl.Read | plt.Read, writer: szl.Write | plt.Write) -> None:
    """Stream all zones from *reader* into the open *writer*.

    Each zone is copied variable-by-variable at its original data type and
    value location.  Connectivity (node maps) is copied for FE zones.

    Args:
        reader: An already-opened ``Read`` instance.
        writer: An already-opened ``Write`` instance (file handle live).

    Raises:
        NotImplementedError: If a FEPOLYGON or FEPOLYHEDRON zone is
            encountered, as the ``Write`` API does not yet support them.
    """
    for zone in reader.zone:
        zt = zone.zone_type

        if zt in (ZoneType.FEPOLYGON, ZoneType.FEPOLYHEDRON):
            raise NotImplementedError(
                f"Zone '{zone.title}' is {zt.name} — poly zone copying is not "
                "supported by the Write API yet."
            )

        # Collect per-variable metadata and data arrays.
        data: list[npt.NDArray] = []
        value_locations: list[ValueLocation] = []
        passive_vars: list[bool] = []
        var_sharing: list[int] = []

        for var in zone.variable:
            passive_vars.append(var.is_passive())
            sv = var.shared_zone  # None or 0-based zone number (szl convention)
            # Write API expects 0 = no sharing, positive = 1-based zone source.
            var_sharing.append((sv) if sv is not None else 0)
            value_locations.append(var.value_location)

            if var.is_passive() or sv is not None:
                # No array needed; append a placeholder so indices stay aligned.
                data.append(np.array([], dtype=np.float32))
            else:
                data.append(var.values)

        # Strip placeholder arrays — Write infers active variables from
        # passive_vars / var_sharing.
        active_data = [
            arr
            for arr, is_p, sv in zip(data, passive_vars, var_sharing, strict=False)
            if not is_p and sv == 0
        ]
        active_locs = [
            loc
            for loc, is_p, sv in zip(
                value_locations, passive_vars, var_sharing, strict=False
            )
            if not is_p and sv == 0
        ]

        common_kw: dict[str, Any] = dict(
            title=zone.title,
            value_locations=active_locs,
            passive_vars=passive_vars,
            var_sharing=var_sharing,
            solution_time=zone.solution_time,
            strand_id=zone.strand_id,
            aux=dict(zone.auxdata.items()) or None,
        )

        if zt == ZoneType.ORDERED:
            writer.write_ijk_zone(data=active_data, **common_kw)
        else:
            writer.write_fe_zone(
                zone_type=zt,
                data=active_data,
                node_map=zone.node_map,
                **common_kw,
            )


# ---------------------------------------------------------------------------
# AppendWrite — returned by open(..., "a")
# ---------------------------------------------------------------------------


class AppendWrite:
    """Write handle returned by :func:`tecio.open` with ``mode='a'``.

    Appends new zones to an existing Tecplot file.  Internally the existing
    file is stream-copied zone-by-zone into a temporary sibling file, and
    the write handle is left open past all copied zones so new zones can be
    appended.  On :meth:`close` (or context-manager exit) the temporary file
    is atomically renamed over the original path — a POSIX rename on Unix,
    :func:`Path.replace` on Windows.

    The copy step preserves the original title, variable list, file type,
    all zone data, all zone metadata (strand, solution time, aux data), and
    variable sharing.  Connectivity is copied for FE zones.

    Do not instantiate this class directly — use :func:`tecio.open`.

    Note:
        FEPOLYGON and FEPOLYHEDRON zones cannot be copied because the Write
        API does not yet expose a poly-zone writer.  A
        :exc:`NotImplementedError` is raised during ``open()`` if such zones
        are present in the source file.

    Example:
        >>> with tecio.open("flow.szplt", "a") as w:
        ...     print(w.variables)  # variable list from the existing file
        ...     print(w.current_zone)  # number of zones already copied
        ...     w.write_ijk_zone(
        ...         data=[x_new, y_new, p_new],
        ...         solution_time=10.0,
        ...         strand_id=1,
        ...     )
        >>> # flow.szplt now contains the original zones plus the new one
    """

    def __init__(
        self,
        original_path: str | os.PathLike,
        tmp_path: str | os.PathLike,
        writer: szl.Write | plt.Write,
    ) -> None:
        self._original = Path(original_path)
        self._tmp = Path(tmp_path)
        self._writer = writer
        self._closed = False

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> AppendWrite:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        try:
            self.close()
        except Exception:
            if exc_type is None:
                raise

    # ------------------------------------------------------------------
    # Write delegation
    # ------------------------------------------------------------------

    def write_ijk_zone(self, *args: Any, **kwargs: Any) -> None:
        """Append a structured IJK-ordered zone.

        Delegates to the underlying format writer's ``write_ijk_zone``
        (e.g. :meth:`tecio.szl.Write.write_ijk_zone`).  All parameters
        from that method are accepted here with one exception: ``variables``
        is not meaningful because the variable list is fixed at open time
        from the existing file.

        Args:
            data (Sequence[ndarray]): One NumPy array per **active**
                variable (non-passive, non-shared).  Shape
                ``(imax[, jmax[, kmax]])`` is inferred automatically.
                Fortran (column-major) order is assumed.
            title (str | None): Zone title.  Defaults to
                ``"IJK_Zone_{n}"``.
            value_locations (Sequence[ValueLocation] | None): Per-variable
                :class:`~tecio.libtecio.ValueLocation` for active variables.
                Defaults to all :attr:`~tecio.libtecio.ValueLocation.NODAL`.
            passive_vars (Sequence[bool | int] | None): Per-variable passive
                flag for the **full** variable list (length = total variables
                in the file, not just active ones).  Defaults to all
                ``False``.
            var_sharing (Sequence[int] | None): Per-variable share-from zone
                index (1-based, counting all zones including those copied
                from the original file).  ``0`` = no sharing.  Defaults to
                all ``0``.  Sharing grid coordinates from zone 1 is a common
                pattern for transient data to avoid duplicating large arrays.
            solution_time (float): Solution time for transient data.
                Defaults to ``0.0``.
            strand_id (int): Strand ID grouping related time steps.  Zones
                with the same strand ID animate together in Tecplot 360.
                Defaults to ``0`` (static).
            aux (dict[str, str] | None): Zone-level auxiliary data as
                ``{name: value}`` pairs.  Defaults to ``None``.

        Example:
            Append a time step, sharing the grid from zone 1:

            >>> n = len(w.variables)  # e.g. ["x", "y", "pressure"]
            >>> w.write_ijk_zone(
            ...     data=[p_new],  # only the non-shared variable
            ...     passive_vars=[False] * n,
            ...     var_sharing=[1, 1, 0],  # x and y shared from zone 1
            ...     solution_time=5.0,
            ...     strand_id=1,
            ... )
        """
        self._writer.write_ijk_zone(*args, **kwargs)

    def write_fe_zone(self, *args: Any, **kwargs: Any) -> None:
        """Append an unstructured finite-element zone.

        Delegates to the underlying format writer's ``write_fe_zone``
        (e.g. :meth:`tecio.szl.Write.write_fe_zone`).  The ``variables``
        parameter is not meaningful here — the variable list is fixed from
        the existing file.  FEPOLYGON and FEPOLYHEDRON are not supported.

        Args:
            zone_type (ZoneType): Element topology.  One of:
                :attr:`~tecio.libtecio.ZoneType.FELINESEG`,
                :attr:`~tecio.libtecio.ZoneType.FETRIANGLE`,
                :attr:`~tecio.libtecio.ZoneType.FEQUADRILATERAL`,
                :attr:`~tecio.libtecio.ZoneType.FETETRAHEDRON`, or
                :attr:`~tecio.libtecio.ZoneType.FEBRICK`.
            data (Sequence[ndarray]): One 1-D array per active variable.
                NODAL arrays must have length ``num_nodes``; CELL_CENTERED
                arrays must have length ``num_cells``.  Both are inferred
                from ``node_map``.
            node_map (array-like | None): Integer array of shape
                ``(num_cells, nodes_per_cell)`` with **1-based** node
                indices.  ``num_nodes = node_map.max()``.  The 32- or
                64-bit write path is chosen automatically.
            title (str | None): Zone title.  Defaults to ``"FE_Zone_{n}"``.
            value_locations (Sequence[ValueLocation] | None): Per-variable
                value location for active variables.  Defaults to all
                :attr:`~tecio.libtecio.ValueLocation.NODAL`.
            passive_vars (Sequence[bool | int] | None): Per-variable passive
                flags for the full variable list.  Defaults to all ``False``.
            var_sharing (Sequence[int] | None): Per-variable share-from zone
                index (1-based).  ``0`` = no sharing.  Defaults to all ``0``.
            con_sharing (int): Zone index to share connectivity from
                (1-based).  ``0`` = write connectivity directly.  Cannot be
                used with GLOBAL face-neighbor modes.  Defaults to ``0``.
            face_neighbors (array-like | None): Face-neighbor connectivity.
                When provided, ``num_face_cons`` is set automatically.
            face_nbr_mode (FaceNeighborMode | None): Face-neighbor mode.
                Defaults to
                :attr:`~tecio.libtecio.FaceNeighborMode.LOCAL_ONE_TO_ONE`.
                Ignored when ``face_neighbors`` is ``None``.
            solution_time (float): Solution time.  Defaults to ``0.0``.
            strand_id (int): Strand ID.  Defaults to ``0``.
            aux (dict[str, str] | None): Zone-level auxiliary data.

        Raises:
            NotImplementedError: If ``zone_type`` is FEPOLYGON or
                FEPOLYHEDRON.
        """
        self._writer.write_fe_zone(*args, **kwargs)

    def add_auxdataset_dict(self, auxdict: dict[str, Any]) -> None:
        """Buffer dataset-level auxiliary data."""
        self._writer.add_auxdataset_dict(auxdict)

    def add_auxvar_dict(self, auxdict: dict[int, dict[str, Any]]) -> None:
        """Buffer variable-level auxiliary data."""
        self._writer.add_auxvar_dict(auxdict)

    def flush_aux(self) -> None:
        """Flush buffered auxiliary data to disk."""
        self._writer.flush_aux()

    # Expose title / variables / current_zone so callers can inspect state.
    @property
    def title(self) -> str:
        """Dataset title of the output file."""
        return self._writer.title

    @property
    def variables(self) -> list[str] | None:
        """Variable name list of the output file."""
        return self._writer.variables

    @property
    def current_zone(self) -> int:
        """Index of the most recently written zone (1-based)."""
        return self._writer.current_zone

    # ------------------------------------------------------------------
    # Close / finalise
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Finalise the temporary file and replace the original.

        Safe to call more than once.
        """
        if self._closed:
            return
        self._closed = True
        self._writer.close()
        # Atomic replace: on POSIX this is a rename; on Windows replace()
        # also works across drives by falling back to copy+delete.
        self._tmp.replace(self._original)


# ---------------------------------------------------------------------------
# AppendReadWrite — returned by open(..., "a+")
# ---------------------------------------------------------------------------


class AppendReadWrite(AppendWrite):
    """Read + write handle returned by :func:`tecio.open` with ``mode='a+'``.

    Inherits all write methods from :class:`AppendWrite` and additionally
    exposes the full read interface populated from the **original** file.
    The read data reflects the file as it existed before any new zones were
    appended — zones written during the current session are not visible via
    :attr:`zone` until the file is closed and re-opened.

    Useful when new zones depend on existing zone data — for example,
    computing a time-average over all previous time steps and writing it
    as an additional zone.

    Do not instantiate this class directly — use :func:`tecio.open`.

    Example:
        >>> with tecio.open("flow.szplt", "a+") as rw:
        ...     # Read from all zones present before this session
        ...     times = [rw.zone[i].solution_time for i in range(rw.num_zones)]
        ...     p_avg = (
        ...         sum(rw.zone[i].variable["p"].values for i in range(rw.num_zones))
        ...         / rw.num_zones
        ...     )
        ...
        ...     # Append a new zone using the computed average
        ...     x = rw.zone[0].variable["x"].values
        ...     y = rw.zone[0].variable["y"].values
        ...     rw.write_ijk_zone(
        ...         data=[x, y, p_avg],
        ...         title="Time-average",
        ...         solution_time=max(times) + 1.0,
        ...         strand_id=2,
        ...     )
    """

    def __init__(
        self,
        original_path: str | os.PathLike,
        tmp_path: str | os.PathLike,
        writer: szl.Write | plt.Write,
        reader: szl.Read | plt.Read,
    ) -> None:
        super().__init__(original_path, tmp_path, writer)
        self._reader = reader

    # ------------------------------------------------------------------
    # Read delegation — mirrors szl.Read / plt.Read public API
    # ------------------------------------------------------------------

    @property
    def file_type(self) -> FileType:
        """File type of the *original* file (FULL / GRID / SOLUTION)."""
        return self._reader.file_type

    @property
    def num_vars(self) -> int:
        """Number of variables in the *original* file."""
        return self._reader.num_vars

    @property
    def num_zones(self) -> int:
        """Number of zones in the *original* file (before appending)."""
        return self._reader.num_zones

    @property
    def zone(self) -> list:
        """Zone list from the *original* file."""
        return self._reader.zone

    @property
    def auxdata(self) -> Any:
        """Dataset-level auxiliary data from the *original* file."""
        return self._reader.auxdata

    @property
    def num_auxdata_items(self) -> int:
        """Number of dataset-level aux data items in the *original* file."""
        return self._reader.num_auxdata_items

    @property
    def var_auxdata(self) -> list:
        """Per-variable auxiliary data list from the *original* file."""
        return self._reader.var_auxdata

    def get_var_auxdata(self, var_index: int) -> Any:
        """Return variable aux data for *var_index* (1-based)."""
        return self._reader.get_var_auxdata(var_index)

    def get_zone_auxdata(self, zone_index: int) -> Any:
        """Return zone aux data for *zone_index* (1-based)."""
        return self._reader.get_zone_auxdata(zone_index)


# ---------------------------------------------------------------------------
# _open_append — shared implementation for "a" and "a+"
# ---------------------------------------------------------------------------


def _open_append(
    path: str | os.PathLike,
    ext: str,
    read_write: bool,
    **writer_kwargs: Any,
) -> AppendWrite | AppendReadWrite:
    """Stream-copy *path* into a temp file and return an open write handle.

    Args:
        path:          Original file path.
        ext:           Lowercase file extension (e.g. ``".szplt"``).
        read_write:    If ``True`` return :class:`AppendReadWrite`; otherwise
                       :class:`AppendWrite`.
        **writer_kwargs: Extra keyword arguments forwarded to the ``Write``
                       constructor (e.g. ``title``, ``file_type``).

    Returns:
        An :class:`AppendWrite` or :class:`AppendReadWrite` instance with the
        ``Write`` handle already past all original zones and ready for new
        zones to be written.

    Raises:
        FileNotFoundError: If *path* does not exist (nothing to append to).
    """
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(
            f"Cannot append to '{src}': file does not exist.  "
            "Use mode='w' to create a new file."
        )

    ReadCls = _reader_for_ext(ext)
    WriteCls = _writer_for_ext(ext)

    # Open the original file for reading.
    reader: szl.Read | plt.Read = ReadCls(str(src))

    # Resolve writer kwargs from the source file when not explicitly supplied.
    title: str = writer_kwargs.pop("title", reader.title)
    file_type: FileType = writer_kwargs.pop("file_type", reader.file_type)
    variables: list[str] = writer_kwargs.pop("variables", reader.variables)

    # Create a sibling temporary file so the rename at close stays on the
    # same filesystem (required for an atomic os.rename on most OSes).
    tmp_fd, tmp_path_str = tempfile.mkstemp(
        dir=src.parent,
        prefix=f".{src.stem}_append_",
        suffix=src.suffix,
    )
    os.close(tmp_fd)  # WriteCls will open it independently
    tmp_path = Path(tmp_path_str)

    try:
        writer = WriteCls(
            str(tmp_path),
            title=title,
            variables=variables,
            file_type=file_type,
            **writer_kwargs,
        )
        # Replay all existing zones into the new file.
        _copy_zones(reader, writer)
    except Exception:
        # Close writer and clean up the temp file if anything goes wrong during the
        # copy.
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
        tmp_path.unlink(missing_ok=True)
        raise

    if read_write:
        return AppendReadWrite(
            original_path=src,
            tmp_path=tmp_path,
            writer=writer,
            reader=reader,
        )
    return AppendWrite(
        original_path=src,
        tmp_path=tmp_path,
        writer=writer,
    )


# ---------------------------------------------------------------------------
# _open_exclusive — implementation for "x"
# ---------------------------------------------------------------------------


def _open_exclusive(
    path: str | os.PathLike,
    ext: str,
    **writer_kwargs: Any,
) -> szl.Write | plt.Write:
    """Open *path* for writing only if it does not already exist.

    Args:
        path: Target file path.
        ext:  Lowercase file extension.
        **writer_kwargs: Forwarded to the ``Write`` constructor.

    Returns:
        A ``Write`` instance, identical to ``open(path, 'w')``.

    Raises:
        FileExistsError: If *path* already exists.
    """
    target = Path(path)
    if target.exists():
        raise FileExistsError(
            f"Cannot create '{target}': file already exists.  "
            "Use mode='w' to overwrite or mode='a' to append."
        )
    WriteCls = _writer_for_ext(ext)
    return WriteCls(str(target), **writer_kwargs)


# ---------------------------------------------------------------------------
# Public open()
# ---------------------------------------------------------------------------


def open(
    path: str | os.PathLike,
    mode: str = "r",
    **kwargs: Any,
) -> (
    szl.Read
    | szl.Write
    | plt.Read
    | plt.Write
    | dat.Read
    | dat.Write
    | AppendWrite
    | AppendReadWrite
):
    """Open a Tecplot file for reading, writing, or appending.

    Selects the correct format handler from the file extension and returns
    the appropriate reader or writer.  All handles are context managers.

    Args:
        path (str | os.PathLike): File path. Extension determines the
            format handler:

            - ``.szplt`` → :mod:`tecio.szl`
            - ``.plt`` / ``.bin`` → :mod:`tecio.plt`
            - ``.dat`` / ``.tec`` → :mod:`tecio.dat`

        mode (str): Access mode — mirrors Python's built-in :func:`open`:

            - ``'r'`` -- Read an existing file. Raises
              :exc:`FileNotFoundError` if it does not exist.
            - ``'w'`` -- Write a new file, overwriting any existing file.
            - ``'x'`` -- Exclusive creation: write a new file only if it
              does not already exist.  Raises :exc:`FileExistsError` if
              it does.
            - ``'a'`` -- Append new zones to an existing file.  The file
              is stream-copied to a temporary file; on close the temporary
              atomically replaces the original.  Raises
              :exc:`FileNotFoundError` if the file does not exist (use
              ``'w'`` to create a new file).
            - ``'a+'`` -- Same as ``'a'`` but the returned handle also
              exposes the full read API populated from the original file.

        **kwargs: Forwarded to the underlying handler constructor.  The
            accepted keyword arguments differ by mode — see the tables
            below.

    Keyword Args (modes ``'r'``, ``'w'``, ``'x'``):
        Forwarded directly to the format's ``Read`` or ``Write``
        constructor.

        For **reading** — no keyword arguments are accepted; all metadata
        is read from the file.

        For **writing** (``'w'`` and ``'x'``):

        .. list-table::
           :header-rows: 1
           :widths: 20 15 65

           * - Keyword
             - Default
             - Description
           * - ``title``
             - ``""``
             - Dataset title string embedded in the file header.
           * - ``variables``
             - ``None``
             - Variable name list.  For SZL and PLT this may be deferred
               to the first :meth:`~tecio.szl.Write.write_ijk_zone` or
               :meth:`~tecio.szl.Write.write_fe_zone` call.  Required at
               open time for DAT.
           * - ``file_type``
             - ``FileType.FULL``
             - One of :attr:`~tecio.libtecio.FileType.FULL`,
               :attr:`~tecio.libtecio.FileType.GRID`, or
               :attr:`~tecio.libtecio.FileType.SOLUTION`.

    Keyword Args (modes ``'a'``, ``'a+'``):
        ``title``, ``variables``, and ``file_type`` default to the values
        read from the existing file.  In most cases you should **not**
        pass these — they are documented here for completeness.

        .. list-table::
           :header-rows: 1
           :widths: 20 15 65

           * - Keyword
             - Default
             - Description
           * - ``title``
             - From file
             - Override the dataset title in the output file.  Rarely
               needed — the existing title is preserved by default.
           * - ``variables``
             - From file
             - **Do not override.** The variable list is fixed by the
               existing file.  Passing a different list will cause a
               mismatch between the copied zones and any new zones.
           * - ``file_type``
             - From file
             - Override the file type.  Rarely needed.

    Returns:
        - ``'r'`` → :class:`tecio.szl.Read`, :class:`tecio.plt.Read`,
          or :class:`tecio.dat.Read`
        - ``'w'`` / ``'x'`` → :class:`tecio.szl.Write`,
          :class:`tecio.plt.Write`, or :class:`tecio.dat.Write`
        - ``'a'`` → :class:`~tecio.AppendWrite`
        - ``'a+'`` → :class:`~tecio.AppendReadWrite`

    Raises:
        ValueError: Unsupported file extension or unrecognised mode.
        FileNotFoundError: ``'r'``, ``'a'``, or ``'a+'`` on a missing file.
        FileExistsError: ``'x'`` on an existing file.
        NotImplementedError: Append mode on a file containing FEPOLYGON
            or FEPOLYHEDRON zones.

    Examples:
        Read a file:

        >>> with tecio.open("flow.szplt") as r:
        ...     x = r.zone[0].variable["x"].values

        Write a new file, deferring variable names to the first zone:

        >>> with tecio.open("out.szplt", "w", title="Run 1") as w:
        ...     w.write_ijk_zone(data=[x, y, p], variables=["x", "y", "p"])

        Append a new zone to an existing file:

        >>> with tecio.open("out.szplt", "a") as w:
        ...     w.write_ijk_zone(data=[x2, y2, p2], solution_time=2.0)

        Append and read in the same session:

        >>> with tecio.open("out.szplt", "a+") as rw:
        ...     prev_p = rw.zone[-1].variable["p"].values
        ...     rw.write_ijk_zone(data=[x2, y2, prev_p * 0.9])
    """
    ext = Path(path).suffix.lower()

    if ext not in _HANDLERS:
        raise ValueError(
            f"Unsupported file extension: '{ext}'.  Supported: {sorted(_HANDLERS)}"
        )

    if mode == "r":
        ReadCls = _HANDLERS[ext]["r"]
        return ReadCls(str(path), **kwargs)

    elif mode == "w":
        WriteCls = _HANDLERS[ext]["w"]
        return WriteCls(str(path), **kwargs)

    elif mode == "x":
        return _open_exclusive(path, ext, **kwargs)

    elif mode == "a":
        return _open_append(path, ext, read_write=False, **kwargs)

    elif mode == "a+":
        return _open_append(path, ext, read_write=True, **kwargs)

    else:
        raise ValueError(
            f"Unrecognised mode '{mode}'.  Supported modes: 'r', 'w', 'x', 'a', 'a+'"
        )
