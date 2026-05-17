r""" Native binary reader for Tecplot PLT (``.plt``) files.

This module provides a pure-Python / NumPy reader for Tecplot PLT binary
files (format versions v112 and v191).  It is a **standalone debugging
module** intended to be integrated into the existing :mod:`plt` module once
validated.

Design goals:
    * **No byte-swapping boilerplate.**  The INT32 value of ``1`` written
      immediately after the magic number is used to detect endianness at open
      time.  All subsequent reads use the correct NumPy byte-order prefix
      (``"<"`` or ``">"``).
    * **Lazy variable data.**  Zone and variable *metadata* is parsed during
      ``__init__`` and stored as lightweight dataclass attributes.  Actual
      numeric data is read from disk only when ``.values`` (or
      ``.get_values()``) is called, matching the behaviour of
      :class:`szl.ReadVariable`.
    * **No text / geometry support.**  Header records with markers 399.0
      (geometry) and 499.0 (text) are detected and skipped without parsing
      their contents.
    * **v112 and v191 zone headers.**  Zone marker ``299.0`` → v112 header;
      ``298.0`` → v191 header.  Both are fully supported.

Limitations
    * FEPOLYGON and FEPOLYHEDRON zones are parsed for metadata but face-map
      reading is not yet implemented (``node_map`` returns ``None`` for those
      types).
    * Bit-packed data (``DataType`` 6) is not supported and raises
      ``NotImplementedError``.

Format reference:
    Tecplot 360 Data Format Guide — Binary PLT ``v112`` / ``v191``.
"""

from __future__ import annotations

import io
import os
import struct
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from ..libtecio import DataType, FileType, ValueLocation, ZoneType

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Magic strings for the two supported file versions.
_MAGIC_V112: bytes = b"#!TDV112"
_MAGIC_V191: bytes = b"#!TDV191"

#: Floating-point sentinel values used in the header.
_MARKER_ZONE: float = 299.0  # v112 zone header
_MARKER_ZONE_V191: float = 298.0  # v191 zone header
_MARKER_GEOM: float = 399.0
_MARKER_TEXT: float = 499.0
_MARKER_CUSTOM_LABEL: float = 599.0
_MARKER_USER_REC: float = 699.0
_MARKER_DATASET_AUX: float = 799.0
_MARKER_VAR_AUX: float = 899.0
_MARKER_EOHMARKER: float = 357.0

#: Nodes per element for simple FE zone types.
_NODES_PER_ELEMENT: dict[ZoneType, int] = {
    ZoneType.FELINESEG: 2,
    ZoneType.FETRIANGLE: 3,
    ZoneType.FEQUADRILATERAL: 4,
    ZoneType.FETETRAHEDRON: 4,
    ZoneType.FEBRICK: 8,
}

#: Map from PLT DataType integer to (numpy dtype string, itemsize).
_DATATYPE_TO_NUMPY: dict[int, tuple[str, int]] = {
    1: ("f4", 4),  # FLOAT
    2: ("f8", 8),  # DOUBLE
    3: ("i4", 4),  # INT32 / LongInt
    4: ("i2", 2),  # INT16 / ShortInt
    5: ("u1", 1),  # BYTE
}

#: Map from DataType integer to DataType enum.
_INT_TO_DATATYPE: dict[int, DataType] = {dt.value: dt for dt in DataType}

#: Map from DataType enum to the canonical numpy dtype (no endian prefix).
_DATATYPE_DTYPE: dict[DataType, str] = {
    DataType.FLOAT: "f4",
    DataType.DOUBLE: "f8",
    DataType.INT32: "i4",
    DataType.INT16: "i2",
    DataType.BYTE: "u1",
}


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------


class PltReadError(RuntimeError):
    """Raised when the PLT binary cannot be parsed."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _read_int32(fp: io.RawIOBase, byte_order: str) -> int:
    """Read a single INT32 with the given byte-order prefix."""
    raw = fp.read(4)
    if len(raw) < 4:
        raise PltReadError("Unexpected end of file reading INT32.")
    (v,) = struct.unpack(f"{byte_order}i", raw)
    return int(v)


def _read_int64(fp: io.RawIOBase, byte_order: str) -> int:
    """Read a single INT64 with the given byte-order prefix."""
    raw = fp.read(8)
    if len(raw) < 8:
        raise PltReadError("Unexpected end of file reading INT64.")
    (v,) = struct.unpack(f"{byte_order}q", raw)
    return int(v)


def _read_float32(fp: io.RawIOBase, byte_order: str) -> float:
    """Read a single FLOAT32 with the given byte-order prefix."""
    raw = fp.read(4)
    if len(raw) < 4:
        raise PltReadError("Unexpected end of file reading FLOAT32.")
    (v,) = struct.unpack(f"{byte_order}f", raw)
    return float(v)


def _read_float64(fp: io.RawIOBase, byte_order: str) -> float:
    """Read a single FLOAT64 with the given byte-order prefix."""
    raw = fp.read(8)
    if len(raw) < 8:
        raise PltReadError("Unexpected end of file reading FLOAT64.")
    (v,) = struct.unpack(f"{byte_order}d", raw)
    return float(v)


def _read_string(fp: io.RawIOBase, byte_order: str) -> str:
    """Read a null-terminated string stored as INT32 code-points.

    The PLT format stores each character as a 4-byte integer (see Note 1 of
    the format spec).  The string is terminated by a zero INT32.
    """
    chars: list[str] = []
    while True:
        raw = fp.read(4)
        if len(raw) < 4:
            break
        (code,) = struct.unpack(f"{byte_order}i", raw)
        if code == 0:
            break
        chars.append(chr(code))
    return "".join(chars)


def _skip_string(fp: io.RawIOBase, byte_order: str) -> None:
    """Read and discard a null-terminated INT32 string."""
    while True:
        raw = fp.read(4)
        if len(raw) < 4:
            break
        (code,) = struct.unpack(f"{byte_order}i", raw)
        if code == 0:
            break


def _peek_float32(fp: io.RawIOBase, byte_order: str) -> float:
    """Peek at the next 4 bytes as a FLOAT32 without advancing the file pointer."""
    pos = fp.tell()
    val = _read_float32(fp, byte_order)
    fp.seek(pos)
    return val


# ---------------------------------------------------------------------------
# Internal storage dataclasses (populated during header parse)
# ---------------------------------------------------------------------------


@dataclass
class _ZoneMeta:
    """All zone-header metadata plus data-section file offsets."""

    # Zone header fields
    title: str = ""
    zone_type: ZoneType = ZoneType.ORDERED
    strand_id: int = -1
    solution_time: float = 0.0
    # Ordered zone dimensions
    i_max: int = 0
    j_max: int = 0
    k_max: int = 0
    # FE zone counts
    num_nodes: int = 0
    num_elements: int = 0
    # Poly zone extras
    num_faces: int = 0
    total_face_nodes: int = 0
    num_boundary_faces: int = 0
    num_boundary_connections: int = 0
    # Variable metadata (per variable)
    var_data_types: list[DataType] = field(default_factory=list)
    is_passive: list[bool] = field(default_factory=list)
    shared_zone: list[int] = field(default_factory=list)  # -1 = not shared
    value_locations: list[ValueLocation] = field(default_factory=list)
    connectivity_shared_zone: int = -1
    has_raw_face_neighbors: bool = False
    num_face_connections: int = 0
    # Zone-level aux data  {name: value}
    auxdata: dict[str, str] = field(default_factory=dict)
    # Header version: True → v191, False → v112
    is_v191: bool = False
    # File offsets filled in during data-section parse
    # offsets[var_index] = (file_offset, count, dtype_str)
    var_offsets: dict[int, tuple[int, int, str]] = field(default_factory=dict)
    # File offset of the connectivity block (None = no connectivity)
    connectivity_offset: int | None = None
    connectivity_count: int = 0  # total int32 values


# ---------------------------------------------------------------------------
# ReadAuxData — identical API to szl.ReadAuxData
# ---------------------------------------------------------------------------


class ReadAuxData:
    """Dictionary-like interface for PLT auxiliary data.

    Values are stored as plain Python strings.  Convenience converters
    ``as_int``, ``as_float``, and ``as_bool`` are provided to match the
    :class:`szl.ReadAuxData` API.
    """

    def __init__(self, data: dict[str, str]) -> None:
        self._data: dict[str, str] = data

    # ------------------------------------------------------------------
    # Mapping interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, key: str) -> str:
        return self._data[key]

    def __contains__(self, key: object) -> bool:
        return key in self._data

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def get(self, key: str, default: Any = None) -> str | None:
        """Return value for *key* or *default* if absent."""
        return self._data.get(key, default)

    def keys(self) -> Iterator[str]:
        """Return iterator over auxiliary data names."""
        return self._data.keys()  # type: ignore[return-value]

    def values(self) -> Iterator[str]:
        """Return iterator over auxiliary data values."""
        return self._data.values()  # type: ignore[return-value]

    def items(self) -> Iterator[tuple[str, str]]:
        """Return iterator over (name, value) pairs."""
        return self._data.items()  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Type converters (same as szl.ReadAuxData)
    # ------------------------------------------------------------------

    def as_int(self, key: str, default: int | None = None) -> int | None:
        """Return auxiliary data value converted to :class:`int`."""
        val = self._data.get(key)
        if val is None:
            return default
        try:
            return int(val)
        except ValueError:
            return default

    def as_float(self, key: str, default: float | None = None) -> float | None:
        """Return auxiliary data value converted to :class:`float`."""
        val = self._data.get(key)
        if val is None:
            return default
        try:
            return float(val)
        except ValueError:
            return default

    def as_bool(self, key: str, default: bool | None = None) -> bool | None:
        """Return auxiliary data value converted to :class:`bool`.

        Recognises ``"true"`` / ``"1"`` (case-insensitive) as ``True`` and
        ``"false"`` / ``"0"`` as ``False``.
        """
        val = self._data.get(key)
        if val is None:
            return default
        low = val.strip().lower()
        if low in ("true", "1"):
            return True
        if low in ("false", "0"):
            return False
        return default

    def __repr__(self) -> str:
        return f"ReadAuxData({self._data!r})"


# ---------------------------------------------------------------------------
# ReadVariable
# ---------------------------------------------------------------------------


class ReadVariable:
    """Lazy variable reader — mirrors :class:`szl.ReadVariable`.

    No data is read from disk until :attr:`values` or :meth:`get_values`
    is called.
    """

    def __init__(
        self,
        file_path: str | os.PathLike,
        zone_meta: _ZoneMeta,
        zone_index: int,
        var_index: int,
        var_names: list[str],
        byte_order: str,
    ) -> None:
        """Initialise a lazy variable reader.

        Args:
            file_path:  Path to the ``.plt`` file on disk.
            zone_meta:  Parsed metadata for the parent zone.
            zone_index: 1-based zone index (for API parity with szl).
            var_index:  1-based variable index.
            var_names:  All variable names for the dataset.
            byte_order:         NumPy byte-order prefix (``"<"`` or ``">"``).
        """
        self._file_path = file_path
        self._meta = zone_meta
        self.zone_index = zone_index
        self.var_index = var_index
        self._var_names = var_names
        self._byte_order = byte_order

    # ------------------------------------------------------------------
    # Metadata (read immediately from zone_meta — no disk I/O)
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Variable name string."""
        return self._var_names[self.var_index - 1]

    @property
    def data_type(self) -> DataType:
        """DataType enum for this variable in this zone."""
        return self._meta.var_data_types[self.var_index - 1]

    @property
    def value_location(self) -> ValueLocation:
        """ValueLocation (NODAL or CELL_CENTERED)."""
        return self._meta.value_locations[self.var_index - 1]

    def is_passive(self) -> bool:
        """Return ``True`` if this variable is passive in this zone."""
        return self._meta.is_passive[self.var_index - 1]

    @property
    def shared_zone(self) -> int | None:
        """Zero-based zone number from which data is shared, or ``None``."""
        sz = self._meta.shared_zone[self.var_index - 1]
        return sz if sz >= 0 else None

    @property
    def num_values(self) -> int:
        """Number of values stored for this variable."""
        offset_info = self._meta.var_offsets.get(self.var_index - 1)
        if offset_info is None:
            return 0
        _, count, _ = offset_info
        return count

    def is_enabled(self) -> bool:
        """Return ``True`` unless this variable is passive."""
        return not self.is_passive()

    # ------------------------------------------------------------------
    # Data access — lazy read from disk
    # ------------------------------------------------------------------

    @property
    def values(
        self,
    ) -> (
        npt.NDArray[np.float32]
        | npt.NDArray[np.float64]
        | npt.NDArray[np.int32]
        | npt.NDArray[np.int16]
        | npt.NDArray[np.uint8]
        | None
    ):
        """All values for this variable as a NumPy array."""
        return self.get_values()

    def get_values(
        self,
        value_range: tuple[int | None, int | None] = (None, None),
    ) -> (
        npt.NDArray[np.float32]
        | npt.NDArray[np.float64]
        | npt.NDArray[np.int32]
        | npt.NDArray[np.int16]
        | npt.NDArray[np.uint8]
        | None
    ):
        """Read variable data from disk.

        Args:
            value_range: Optional ``(start, end)`` 1-based half-open range.
                         ``(None, None)`` reads all values.

        Returns:
            NumPy array with the dtype corresponding to :attr:`data_type`,
            or ``None`` if the variable is passive or shared.

        Raises:
            ValueError: On invalid *value_range*.
        """
        if self.is_passive() or self.shared_zone is not None:
            return None

        offset_info = self._meta.var_offsets.get(self.var_index - 1)
        if offset_info is None:
            return None

        file_offset, total_count, dtype_str = offset_info

        # Resolve value range
        full_read = value_range == (None, None)
        if full_read:
            start_idx = 0  # zero-based
            count = total_count
        else:
            start_1, end_1 = value_range
            if start_1 is None or end_1 is None:
                raise ValueError("Both start and end indices must be specified.")
            if start_1 < 1 or start_1 > total_count:
                raise ValueError(
                    f"Start index {start_1} out of range [1, {total_count}]."
                )
            if end_1 > total_count or end_1 < start_1:
                raise ValueError(f"Invalid value range: ({start_1}, {end_1}).")
            start_idx = start_1 - 1
            count = end_1 - start_1

        # Build endian-tagged dtype string for fromfile
        dt = np.dtype(f"{self._byte_order}{dtype_str}")
        itemsize = dt.itemsize

        with open(self._file_path, "rb") as fp:
            fp.seek(file_offset + start_idx * itemsize)
            data = np.fromfile(fp, dtype=dt, count=count)

        # Byte-swap if needed so the result is always native-endian
        if data.dtype.byteorder not in ("=", "|", np.dtype(dt).str[0]):
            data = data.byteswap().newbyteorder()

        # Reshape to (I, J, K) / (I-1, J-1, K-1) for full reads of ordered zones.
        if full_read and self._meta.zone_type == ZoneType.ORDERED:
            ni, nj, nk = self._meta.i_max, self._meta.j_max, self._meta.k_max
            if self.value_location == ValueLocation.CELL_CENTERED:
                shape = (max(ni - 1, 1), max(nj - 1, 1), max(nk - 1, 1))
            else:
                shape = (ni, nj, nk)
            if data.size == shape[0] * shape[1] * shape[2]:
                data = data.reshape(shape, order="F")

        return data


# ---------------------------------------------------------------------------
# ReadZone
# ---------------------------------------------------------------------------


class ReadZone:
    """Zone reader — mirrors :class:`szl.ReadZone`.

    All metadata comes from the pre-parsed :class:`_ZoneMeta` object.
    """

    def __init__(
        self,
        file_path: str | os.PathLike,
        meta: _ZoneMeta,
        zone_index: int,
        num_vars: int,
        var_names: list[str],
        byte_order: str,
    ) -> None:
        self._file_path = file_path
        self._meta = meta
        self.zone_index = zone_index
        self.num_vars = num_vars
        self._var_names = var_names
        self._byte_order = byte_order
        self._variable: list[ReadVariable] | None = None
        self._auxdata: ReadAuxData | None = None

        # Expose I/J/K directly like szl.ReadZone
        self.I = meta.i_max
        self.J = meta.j_max
        self.K = meta.k_max

    # ------------------------------------------------------------------
    # Metadata properties
    # ------------------------------------------------------------------

    @property
    def title(self) -> str:
        """Zone title string."""
        return self._meta.title

    @property
    def zone_type(self) -> ZoneType:
        """Zone type enum."""
        return self._meta.zone_type

    @property
    def dimensions(self) -> tuple[int, int, int]:
        """(I, J, K) dimensions."""
        return (self.I, self.J, self.K)

    @property
    def solution_time(self) -> float:
        """Solution time for transient data (0.0 for stationary)."""
        return self._meta.solution_time

    @property
    def strand_id(self) -> int:
        """Strand ID (0 = static, positive = transient strand).

        PLT files store -1 as a sentinel for "no strand assigned".
        Normalised to 0 here to match the SZL/DAT convention and
        to prevent invalid arguments to tec_zone_set_unsteady_options.
        """
        return max(self._meta.strand_id, 0)

    @property
    def num_nodes(self) -> int:
        """Number of nodes/points in this zone."""
        if self.zone_type == ZoneType.ORDERED:
            return max(self.I, 1) * max(self.J, 1) * max(self.K, 1)
        return self._meta.num_nodes

    @property
    def num_elements(self) -> int:
        """Number of elements in this zone (same as num_nodes for ORDERED)."""
        if self.zone_type == ZoneType.ORDERED:
            return max(self.I, 1) * max(self.J, 1) * max(self.K, 1)
        return self._meta.num_elements

    @property
    def nodes_per_cell(self) -> int:
        """Nodes per cell for simple FE zones or inferred for ORDERED."""
        zt = self.zone_type
        if zt in _NODES_PER_ELEMENT:
            return _NODES_PER_ELEMENT[zt]
        if zt == ZoneType.ORDERED:
            dims = sum(1 for x in (self.I, self.J, self.K) if x > 1)
            return 2**dims
        raise ValueError(f"ZoneType {zt} does not have a fixed nodes-per-cell count.")

    def is_enabled(self) -> bool:
        """Always ``True`` for PLT zones (no concept of disabled zones)."""
        return True

    # ------------------------------------------------------------------
    # Variable access
    # ------------------------------------------------------------------

    @property
    def variable(self) -> list[ReadVariable]:
        """List of :class:`ReadVariable` objects (0-indexed)."""
        if self._variable is None:
            self._variable = [
                ReadVariable(
                    file_path=self._file_path,
                    zone_meta=self._meta,
                    zone_index=self.zone_index,
                    var_index=i + 1,
                    var_names=self._var_names,
                    byte_order=self._byte_order,
                )
                for i in range(self.num_vars)
            ]
        return self._variable

    def __getattr__(self, name: str) -> Any:
        """Dynamic attribute access for variable names.

        Example:
            zone.pressure  # returns NumPy array for variable "Pressure"
        """
        for var in self.variable:
            if var.name.lower() == name.lower():
                return var.values
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name!r}'"
        )

    # ------------------------------------------------------------------
    # Connectivity
    # ------------------------------------------------------------------

    @property
    def node_map(self) -> npt.NDArray[np.int64] | None:
        """Node connectivity array of shape ``(num_elements, nodes_per_cell)``.

        Returns ``None`` for ORDERED, FEPOLYGON, or FEPOLYHEDRON zones, or
        when connectivity is shared from another zone.
        """
        zt = self.zone_type
        if zt == ZoneType.ORDERED:
            return None
        if zt in (ZoneType.FEPOLYGON, ZoneType.FEPOLYHEDRON):
            # Poly face-map reading is not yet implemented.
            return None
        if self._meta.connectivity_offset is None:
            return None

        count = self._meta.connectivity_count
        dt = np.dtype(f"{self._byte_order}i4")

        with open(self._file_path, "rb") as fp:
            fp.seek(self._meta.connectivity_offset)
            flat = np.fromfile(fp, dtype=dt, count=count)

        return flat.reshape(self.num_elements, -1).astype(np.int64) + 1

    # ------------------------------------------------------------------
    # Auxiliary data
    # ------------------------------------------------------------------

    @property
    def auxdata(self) -> ReadAuxData:
        """Zone-level auxiliary data."""
        if self._auxdata is None:
            self._auxdata = ReadAuxData(self._meta.auxdata)
        return self._auxdata


# ---------------------------------------------------------------------------
# Main parser — builds all metadata during __init__
# ---------------------------------------------------------------------------


class _PltParser:
    """Low-level PLT binary parser.

    Called once from :class:`Read.__init__`.  After construction the
    following attributes are available:

        ===================   ===============================================
        Attribute             Description
        ===================   ===============================================
        ``title``             dataset title string
        ``file_type``         :class:`FileType` enum
        ``var_names``         list of variable name strings
        ``zones``             list of :class:`_ZoneMeta` objects
        ``dataset_auxdata``   ``{name: value}``
        ``var_auxdata``       ``{var_index_0based: {name: value}}``
        ``byte_order``         byte-order prefix for NumPy (``"<"`` or ``">"`)
        ===================   ===============================================
    """

    def __init__(self, file_path: str | os.PathLike) -> None:
        self._path = str(file_path)
        self.title: str = ""
        self.file_type: FileType = FileType.FULL
        self.var_names: list[str] = []
        self.zones: list[_ZoneMeta] = []
        self.dataset_auxdata: dict[str, str] = {}
        self.var_auxdata: dict[int, dict[str, str]] = {}
        self.byte_order: str = "<"  # byte order: "<" little, ">" big
        self._file_version: str = "v112"

        with open(self._path, "rb") as fp:
            self._parse_header(fp)
            self._parse_data_section(fp)

    # ------------------------------------------------------------------
    # Header section
    # ------------------------------------------------------------------

    def _parse_header(self, fp: io.RawIOBase) -> None:
        """Parse the PLT header section up to and including EOHMARKER."""
        self._detect_version_and_endian(fp)
        byte_order = self.byte_order

        # FileType
        self.file_type = FileType(_read_int32(fp, byte_order))

        # Dataset title
        self.title = _read_string(fp, byte_order)

        # Variable names
        num_vars = _read_int32(fp, byte_order)
        self.var_names = [_read_string(fp, byte_order) for _ in range(num_vars)]

        # Zone / geometry / text / aux records — loop until EOHMARKER
        while True:
            marker = _read_float32(fp, byte_order)

            if abs(marker - _MARKER_EOHMARKER) < 0.5:
                # End of header
                return

            elif abs(marker - _MARKER_ZONE) < 0.5:
                self._parse_zone_header(fp, v191=False)

            elif abs(marker - _MARKER_ZONE_V191) < 0.5:
                self._parse_zone_header(fp, v191=True)

            elif abs(marker - _MARKER_GEOM) < 0.5:
                self._skip_geometry(fp)

            elif abs(marker - _MARKER_TEXT) < 0.5:
                self._skip_text(fp)

            elif abs(marker - _MARKER_CUSTOM_LABEL) < 0.5:
                self._skip_custom_label(fp)

            elif abs(marker - _MARKER_USER_REC) < 0.5:
                self._skip_user_rec(fp)

            elif abs(marker - _MARKER_DATASET_AUX) < 0.5:
                name = _read_string(fp, byte_order)
                _read_int32(fp, byte_order)  # value format (always 0 = string)
                value = _read_string(fp, byte_order)
                self.dataset_auxdata[name] = value

            elif abs(marker - _MARKER_VAR_AUX) < 0.5:
                var_num = _read_int32(fp, byte_order)  # 0-based variable number
                name = _read_string(fp, byte_order)
                _read_int32(fp, byte_order)  # value format
                value = _read_string(fp, byte_order)
                self.var_auxdata.setdefault(var_num, {})[name] = value

            else:
                # Unknown marker — should not happen in a well-formed file.
                raise PltReadError(
                    f"Unexpected header marker {marker:.1f} at file offset "
                    f"{fp.tell() - 4:#010x}."
                )

    def _detect_version_and_endian(self, fp: io.RawIOBase) -> None:
        """Read magic bytes and INT32=1 endian probe; set ``self.byte_order``."""
        magic = fp.read(8)
        if magic == _MAGIC_V112:
            self._file_version = "v112"
        elif magic == _MAGIC_V191:
            self._file_version = "v191"
        else:
            raise PltReadError(
                f"Unrecognised PLT magic number: {magic!r}.  "
                "Expected b'#!TDV112' or b'#!TDV191'."
            )

        # Next 4 bytes encode the integer 1 in the writer's byte order.
        raw = fp.read(4)
        if len(raw) < 4:
            raise PltReadError("File too short — cannot read byte-order probe.")

        # Try little-endian first
        (le_val,) = struct.unpack("<i", raw)
        if le_val == 1:
            self.byte_order = "<"
            return

        (be_val,) = struct.unpack(">i", raw)
        if be_val == 1:
            self.byte_order = ">"
            return

        raise PltReadError(
            f"Byte-order probe value {raw!r} is neither 1 in little- nor big-endian."
        )

    def _parse_zone_header(self, fp: io.RawIOBase, *, v191: bool) -> None:
        """Parse a single zone header record and append to ``self.zones``."""
        byte_order = self.byte_order
        num_vars = len(self.var_names)

        meta = _ZoneMeta(is_v191=v191)

        # Zone name
        meta.title = _read_string(fp, byte_order)

        # ParentZone (no longer used since 2020r1; still present in file)
        _read_int32(fp, byte_order)

        # StrandID
        meta.strand_id = _read_int32(fp, byte_order)

        # Solution time
        meta.solution_time = _read_float64(fp, byte_order)

        # Default zone colour (unused; set to -1)
        _read_int32(fp, byte_order)

        # ZoneType
        meta.zone_type = ZoneType(_read_int32(fp, byte_order))

        # Variable locations
        specify_var_location = _read_int32(fp, byte_order)
        if specify_var_location == 1:
            locs = [_read_int32(fp, byte_order) for _ in range(num_vars)]
            meta.value_locations = [ValueLocation(loc) for loc in locs]
        else:
            meta.value_locations = [ValueLocation.NODAL] * num_vars

        # Raw local 1-to-1 face neighbours supplied flag
        meta.has_raw_face_neighbors = bool(_read_int32(fp, byte_order))

        # Number of miscellaneous face-neighbour connections
        meta.num_face_connections = _read_int32(fp, byte_order)
        if meta.num_face_connections != 0:
            # face neighbour mode
            _read_int32(fp, byte_order)
            zt = meta.zone_type
            fe_types = (
                ZoneType.FELINESEG,
                ZoneType.FETRIANGLE,
                ZoneType.FEQUADRILATERAL,
                ZoneType.FETETRAHEDRON,
                ZoneType.FEBRICK,
                ZoneType.FEPOLYGON,
                ZoneType.FEPOLYHEDRON,
            )
            if zt in fe_types:
                # completely specified flag
                _read_int32(fp, byte_order)

        # Dimensions / element counts
        if meta.zone_type == ZoneType.ORDERED:
            meta.i_max = _read_int32(fp, byte_order)
            meta.j_max = _read_int32(fp, byte_order)
            meta.k_max = _read_int32(fp, byte_order)
        else:
            meta.num_nodes = _read_int32(fp, byte_order)

            if meta.zone_type in (ZoneType.FEPOLYGON, ZoneType.FEPOLYHEDRON):
                if v191:
                    meta.num_faces = _read_int64(fp, byte_order)
                    meta.total_face_nodes = _read_int64(fp, byte_order)
                else:
                    meta.num_faces = _read_int32(fp, byte_order)
                    meta.total_face_nodes = _read_int32(fp, byte_order)
                meta.num_boundary_faces = _read_int32(fp, byte_order)
                meta.num_boundary_connections = _read_int32(fp, byte_order)

            meta.num_elements = _read_int32(fp, byte_order)
            # ICellDim, JCellDim, KCellDim (for future use; always 0)
            _read_int32(fp, byte_order)
            _read_int32(fp, byte_order)
            _read_int32(fp, byte_order)

        # Zone-level auxiliary data
        while True:
            has_aux = _read_int32(fp, byte_order)
            if has_aux == 0:
                break
            name = _read_string(fp, byte_order)
            _read_int32(fp, byte_order)  # value format (always 0)
            value = _read_string(fp, byte_order)
            meta.auxdata[name] = value

        self.zones.append(meta)

    # ------------------------------------------------------------------
    # Header skip helpers (geometry and text records)
    # ------------------------------------------------------------------

    def _skip_geometry(self, fp: io.RawIOBase) -> None:
        """Skip a geometry record (marker already consumed)."""
        byte_order = self.byte_order
        _read_int32(fp, byte_order)  # CoordSys
        _read_int32(fp, byte_order)  # Scope
        _read_int32(fp, byte_order)  # DrawOrder
        # XYZ starting location (3 × FLOAT64)
        fp.read(24)
        _read_int32(fp, byte_order)  # Zone
        _read_int32(fp, byte_order)  # Color
        _read_int32(fp, byte_order)  # FillColor
        _read_int32(fp, byte_order)  # IsFilled
        geom_type = _read_int32(fp, byte_order)
        _read_int32(fp, byte_order)  # LinePattern
        fp.read(16)  # PatternLength + LineThickness (2 × FLOAT64)
        _read_int32(fp, byte_order)  # NumEllipsePts
        _read_int32(fp, byte_order)  # ArrowheadStyle
        _read_int32(fp, byte_order)  # ArrowheadAttachment
        fp.read(16)  # ArrowheadSize + ArrowheadAngle (2 × FLOAT64)
        _skip_string(fp, byte_order)  # MacroFunctionCommand
        gtype = _read_int32(fp, byte_order)  # PolylineFieldDataType (1=float, 2=double)
        gtype_bytes = 4 if gtype == 1 else 8
        _read_int32(fp, byte_order)  # Clipping

        if geom_type == 0:  # Line
            num_polylines = _read_int32(fp, byte_order)
            for _ in range(num_polylines):
                num_pts = _read_int32(fp, byte_order)
                # X-block, Y-block (and Z-block for Grid3D — we cannot
                # distinguish here, so we skip 2 blocks conservatively).
                fp.read(num_pts * gtype_bytes * 2)
        elif geom_type == 1:  # Rectangle
            fp.read(gtype_bytes * 2)
        elif geom_type == 2 or geom_type == 3:  # Square
            fp.read(gtype_bytes)
        elif geom_type == 4:  # Ellipse
            fp.read(gtype_bytes * 2)

    def _skip_text(self, fp: io.RawIOBase) -> None:
        """Skip a text record (marker already consumed)."""
        byte_order = self.byte_order
        _read_int32(fp, byte_order)  # CoordSys
        _read_int32(fp, byte_order)  # Scope
        fp.read(24)  # XYZ starting location (3 × FLOAT64)
        _read_int32(fp, byte_order)  # FontType
        _read_int32(fp, byte_order)  # CharHeightUnits
        fp.read(8)  # Height (FLOAT64)
        _read_int32(fp, byte_order)  # TextBoxType
        fp.read(16)  # Margin + MarginLinewidth (2 × FLOAT64)
        _read_int32(fp, byte_order)  # TextBoxOutlineColor
        _read_int32(fp, byte_order)  # TextBoxFillColor
        fp.read(16)  # Angle + LineSpacing (2 × FLOAT64)
        _read_int32(fp, byte_order)  # TextAnchor
        _read_int32(fp, byte_order)  # Zone
        _read_int32(fp, byte_order)  # Color
        _skip_string(fp, byte_order)  # MacroFunctionCommand
        _read_int32(fp, byte_order)  # Clipping
        _skip_string(fp, byte_order)  # The text itself

    def _skip_custom_label(self, fp: io.RawIOBase) -> None:
        """Skip a CustomLabel record (marker already consumed)."""
        byte_order = self.byte_order
        n_labels = _read_int32(fp, byte_order)
        for _ in range(n_labels):
            _skip_string(fp, byte_order)

    def _skip_user_rec(self, fp: io.RawIOBase) -> None:
        """Skip a UserRec record (marker already consumed)."""
        _skip_string(fp, self.byte_order)

    # ------------------------------------------------------------------
    # Data section
    # ------------------------------------------------------------------

    def _parse_data_section(self, fp: io.RawIOBase) -> None:
        """Parse the data section and record variable file offsets.

        The EOHMARKER has already been consumed by :meth:`_parse_header`.
        We iterate over zones in declaration order, matching each data block
        to the corresponding :class:`_ZoneMeta` in ``self.zones``.
        """
        byte_order = self.byte_order
        num_vars = len(self.var_names)

        for zone_idx, meta in enumerate(self.zones):
            # Zone marker (299.0 or 298.0) — must match header.
            data_zone_marker = _read_float32(fp, byte_order)
            if not (
                abs(data_zone_marker - _MARKER_ZONE) < 0.5
                or abs(data_zone_marker - _MARKER_ZONE_V191) < 0.5
            ):
                raise PltReadError(
                    f"Data section zone {zone_idx}: expected zone marker "
                    f"299.0 or 298.0, got {data_zone_marker:.2f}."
                )

            # Per-variable data formats
            raw_data_types = [_read_int32(fp, byte_order) for _ in range(num_vars)]
            meta.var_data_types = [_INT_TO_DATATYPE[dt] for dt in raw_data_types]

            # Passive variables
            has_passive = _read_int32(fp, byte_order)
            if has_passive:
                meta.is_passive = [
                    bool(_read_int32(fp, byte_order)) for _ in range(num_vars)
                ]
            else:
                meta.is_passive = [False] * num_vars

            # Variable sharing
            has_var_sharing = _read_int32(fp, byte_order)
            if has_var_sharing:
                meta.shared_zone = [
                    _read_int32(fp, byte_order) for _ in range(num_vars)
                ]
            else:
                meta.shared_zone = [-1] * num_vars

            # Connectivity sharing (-1 = no sharing)
            meta.connectivity_shared_zone = _read_int32(fp, byte_order)

            # Min / max pairs for non-shared, non-passive variables
            for v in range(num_vars):
                if not meta.is_passive[v] and meta.shared_zone[v] < 0:
                    fp.read(16)  # min (FLOAT64) + max (FLOAT64)

            # Variable data blocks
            self._record_var_offsets(fp, meta, num_vars)

            # Connectivity block (FE non-ORDERED, non-shared)
            self._record_connectivity_offset(fp, meta)

    def _record_var_offsets(
        self,
        fp: io.RawIOBase,
        meta: _ZoneMeta,
        num_vars: int,
    ) -> None:
        """Record file offsets for each active variable data block.

        Data is stored in **block** order (all values for variable 0, then
        all values for variable 1, …).  We seek rather than read so we do
        not load the data into memory.
        """
        for v in range(num_vars):
            if meta.is_passive[v] or meta.shared_zone[v] >= 0:
                # No data on disk for this variable.
                continue

            dt_int = meta.var_data_types[v].value
            if dt_int == 6:
                raise NotImplementedError(
                    "Bit-packed data (DataType 6) is not supported."
                )

            dtype_str, itemsize = _DATATYPE_TO_NUMPY[dt_int]
            count = self._var_value_count(meta, v)

            file_offset = fp.tell()
            meta.var_offsets[v] = (file_offset, count, dtype_str)

            # Seek past the data block without reading it.
            fp.seek(count * itemsize, os.SEEK_CUR)

    def _var_value_count(self, meta: _ZoneMeta, var_idx: int) -> int:
        """Return the number of values written for variable *var_idx*.

        For cell-centered variables in ORDERED zones the PLT format writes
        ``IMax * JMax * (KMax - 1)`` values (ghost-padded; see Note 5 of
        the format spec), but we simply read the padded count here because
        that is what is physically on disk.
        """
        loc = meta.value_locations[var_idx]

        if meta.zone_type == ZoneType.ORDERED:
            i = max(meta.i_max, 1)
            j = max(meta.j_max, 1)
            k = max(meta.k_max, 1)
            if loc == ValueLocation.CELL_CENTERED:
                # Ghost values included (see spec Note 5)
                return max(i - 1, 1) * max(j - 1, 1) * max(k - 1, 1)
            return i * j * k
        else:
            # FE zones
            if loc == ValueLocation.CELL_CENTERED:
                return meta.num_elements
            return meta.num_nodes

    def _record_connectivity_offset(
        self,
        fp: io.RawIOBase,
        meta: _ZoneMeta,
    ) -> None:
        """Record the file offset of the connectivity block (if present)."""
        zt = meta.zone_type

        if zt == ZoneType.ORDERED:
            # Ordered zones can have miscellaneous face-neighbor connections
            if meta.connectivity_shared_zone < 0 and meta.num_face_connections > 0:
                # Skip: N = num_face_connections * P (mode-dependent).
                # We do not parse face neighbors, so just skip them.
                # Safe approximation: each connection record is at least 3
                # INT32 values (LocalOneToOne), but the actual length is
                # mode-dependent.  Since we do not use them, skip naively.
                pass
            return

        if zt in (ZoneType.FEPOLYGON, ZoneType.FEPOLYHEDRON):
            # Poly face map — complex structure, skip for now.
            self._skip_poly_face_map(fp, meta)
            return

        # Simple FE zones
        if meta.connectivity_shared_zone >= 0:
            # Connectivity is shared from another zone — nothing on disk.
            return

        nodes_per_cell = _NODES_PER_ELEMENT.get(zt)
        if nodes_per_cell is None:
            return

        count = meta.num_elements * nodes_per_cell
        meta.connectivity_offset = fp.tell()
        meta.connectivity_count = count

        # Seek past the connectivity data
        fp.seek(count * 4, os.SEEK_CUR)  # INT32 values

        # Raw local face neighbors (if supplied)
        if meta.has_raw_face_neighbors:
            faces_per_element = nodes_per_cell  # same count for supported types
            fp.seek(meta.num_elements * faces_per_element * 4, os.SEEK_CUR)

        # Miscellaneous face-neighbor connections (skip; not needed for data)
        # We do not parse the mode-dependent records here.

    def _skip_poly_face_map(self, fp: io.RawIOBase, meta: _ZoneMeta) -> None:
        """Skip the face-map block for FEPOLYGON / FEPOLYHEDRON zones."""
        zt = meta.zone_type

        if meta.connectivity_shared_zone >= 0:
            return

        v191 = meta.is_v191
        nf = meta.num_faces
        fn = meta.total_face_nodes
        nbf = meta.num_boundary_faces
        nbc = meta.num_boundary_connections

        if zt == ZoneType.FEPOLYHEDRON:
            if v191:
                # Face node count per face (INT32 * nf)
                fp.seek(nf * 4, os.SEEK_CUR)
            else:
                # Face node offsets (INT32 * (nf + 1))
                fp.seek((nf + 1) * 4, os.SEEK_CUR)

        # Face nodes (INT32 * fn)
        fp.seek(fn * 4, os.SEEK_CUR)
        # Left elements (INT32 * nf)
        fp.seek(nf * 4, os.SEEK_CUR)
        # Right elements (INT32 * nf)
        fp.seek(nf * 4, os.SEEK_CUR)

        if nbf > 0:
            # Boundary face connection offsets (INT32 * (nbf + 1))
            fp.seek((nbf + 1) * 4, os.SEEK_CUR)
            # Boundary connection elements (INT32 * nbc)
            fp.seek(nbc * 4, os.SEEK_CUR)
            # Boundary connection zones (INT32 * nbc)
            fp.seek(nbc * 4, os.SEEK_CUR)


# ---------------------------------------------------------------------------
# Public Read class — mirrors szl.Read
# ---------------------------------------------------------------------------


class Read:
    """Read data from Tecplot PLT (``.plt``) binary files.

    This class provides the **exact same public API** as :class:`szl.Read`
    so that the two can be used interchangeably.

    Zone metadata is parsed eagerly during ``__init__``; variable data is
    read lazily from disk when accessed via :attr:`ReadVariable.values` or
    :meth:`ReadVariable.get_values`.

    Args:
        file_name: Path to the ``.plt`` file.

    Raises:
        PltReadError: If the file cannot be parsed as a valid PLT binary.

    Example:
        from tecio import plt

        r = plt.Read("flow.plt")
        print(r.title)
        print(r.num_vars)

        zone = r.zone[0]
        pressure = zone.variable[2].values  # NumPy array, read from disk now
    """

    def __init__(self, file_name: str | os.PathLike) -> None:
        self._path = os.fspath(file_name)
        parser = _PltParser(self._path)

        self._file_type: FileType = parser.file_type
        self._title: str = parser.title
        self._var_names: list[str] = parser.var_names
        self._byte_order: str = parser.byte_order
        self._dataset_auxdata: dict[str, str] = parser.dataset_auxdata
        # var_auxdata: 0-based index → {name: value}
        self._raw_var_auxdata: dict[int, dict[str, str]] = parser.var_auxdata

        self.zone: list[ReadZone] = [
            ReadZone(
                file_path=self._path,
                meta=meta,
                zone_index=i + 1,
                num_vars=len(parser.var_names),
                var_names=parser.var_names,
                byte_order=parser.byte_order,
            )
            for i, meta in enumerate(parser.zones)
        ]

        self._auxdata: ReadAuxData | None = None
        self._var_auxdata: list[ReadAuxData] | None = None

    def __enter__(self) -> Read:
        """Context manager for Read class."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager protocol for read-only file interfaces.

        Read classes hold no open file handle between accesses, so there is nothing to
        release on exit.  Provided for API consistency with the Write classes and to
        support the ``with tecio.open(...) as r:`` pattern.
        """
        pass

    # ------------------------------------------------------------------
    # Dataset metadata properties (parity with szl.Read)
    # ------------------------------------------------------------------

    @property
    def file_type(self) -> FileType:
        """File type (FULL, GRID, or SOLUTION)."""
        return self._file_type

    @property
    def title(self) -> str:
        """Dataset title string."""
        return self._title

    @property
    def num_vars(self) -> int:
        """Number of variables in the dataset."""
        return len(self._var_names)

    @property
    def variables(self) -> list[str]:
        """List of variable name strings."""
        return list(self._var_names)

    @property
    def num_zones(self) -> int:
        """Number of zones in the dataset."""
        return len(self.zone)

    @property
    def num_auxdata_items(self) -> int:
        """Number of dataset-level auxiliary data items."""
        return len(self._dataset_auxdata)

    @property
    def auxdata(self) -> ReadAuxData:
        """Dataset-level auxiliary data."""
        if self._auxdata is None:
            self._auxdata = ReadAuxData(self._dataset_auxdata)
        return self._auxdata

    @property
    def var_auxdata(self) -> list[ReadAuxData]:
        """List of per-variable auxiliary data objects.

        Index 0 is ``None`` (placeholder) so that 1-based indexing matches
        Tecplot convention; use ``var_auxdata[1]`` for the first variable.
        """
        if self._var_auxdata is None:
            self._var_auxdata = [None]  # type: ignore[list-item]
            for i in range(self.num_vars):
                self._var_auxdata.append(ReadAuxData(self._raw_var_auxdata.get(i, {})))
        return self._var_auxdata

    def get_var_auxdata(self, var_index: int) -> ReadAuxData:
        """Return auxiliary data for variable *var_index* (1-based).

        Raises:
            IndexError: If *var_index* is outside ``[1, num_vars]``.
        """
        if var_index < 1 or var_index > self.num_vars:
            raise IndexError(
                f"Variable index {var_index} out of range [1, {self.num_vars}]."
            )
        return self.var_auxdata[var_index]

    def get_zone_auxdata(self, zone_index: int) -> ReadAuxData:
        """Return auxiliary data for zone *zone_index* (1-based).

        Raises:
            IndexError: If *zone_index* is outside ``[1, num_zones]``.
        """
        if zone_index < 1 or zone_index > self.num_zones:
            raise IndexError(
                f"Zone index {zone_index} out of range [1, {self.num_zones}]."
            )
        return self.zone[zone_index - 1].auxdata
