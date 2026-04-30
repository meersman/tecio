r""":mod:`dat`: Tecplot ASCII DAT file reader and writer.
=====================================================

This module provides :class:`Read` for parsing and :class:`Write` for
producing Tecplot 360 ASCII data files (``.dat`` / ``.tec``).  Both
classes mirror the interfaces of :class:`szl.Read` / :class:`szl.Write`
and :class:`plt.Read` / :class:`plt.Write` so that downstream code can
switch between file formats by changing only the file extension passed to
:func:`tecio.open`.

Reading
-------
:class:`Read` parses the entire file on construction and stores all data
in memory::

    dat = tecio.open("result.dat", "r")

    print(dat.title)
    print(dat.variables)  # list of variable name strings
    print(dat.num_zones)

    zone = dat.zone[0]
    print(zone.title, zone.zone_type, zone.solution_time)

    var = zone.variable[0]
    print(var.name, var.data_type, var.values)

    if zone.zone_type != ZoneType.ORDERED:
        print(zone.node_map)  # (num_elements, nodes_per_cell) int64 array

Supported read features
~~~~~~~~~~~~~~~~~~~~~~~
* ``FULL``, ``GRID``, and ``SOLUTION`` file types
* Ordered and simple FE zones (FELINESEG through FEBRICK)
* ``DATAPACKING=BLOCK`` only (POINT packing raises :exc:`ValueError`)
* ``VARLOCATION`` (cell-centred variables)
* ``PASSIVEVARLIST`` and ``VARSHARELIST``
* ``CONNECTIVITYSHAREZONE``
* Dataset-level ``DATASETAUXDATA`` and variable-level ``VARAUXDATA``
* Zone-level ``AUXDATA``

Writing
-------
:class:`Write` is a context-manager writer that supports lazy-open,
buffered aux data, and atomic (all-or-nothing) zone writes::

    with tecio.open("result.dat", "w", title="Demo",
                    variables=["X", "Y", "P"]) as w:
        w.write_ijk_zone(data=[x, y, p], title="Zone 1")

All floating-point variable data is written in scientific notation with a
configurable number of significant digits (default 9).

Format specification reference
-------------------------------
Tecplot 360 Data Format Guide 2025 R2, "ASCII Data" chapter.
"""

from __future__ import annotations

import contextlib

# Standard library
import re
from collections.abc import Iterator
from typing import Any

# Third-party
import numpy as np
import numpy.typing as npt

from ..libtecio import (
    DataType,
    FileType,
    ValueLocation,
    ZoneType,
)

# ---------------------------------------------------------------------------
# Shared module-level constants
# ---------------------------------------------------------------------------

#: FE zone types fully supported for reading and writing.
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

#: Nodes per element for each supported simple FE type.
_NODES_PER_ELEM: dict[ZoneType, int] = {
    ZoneType.FELINESEG: 2,
    ZoneType.FETRIANGLE: 3,
    ZoneType.FEQUADRILATERAL: 4,
    ZoneType.FETETRAHEDRON: 4,
    ZoneType.FEBRICK: 8,
}

#: ASCII keyword → ZoneType (lower-cased at parse time).
_STR_TO_ZONETYPE: dict[str, ZoneType] = {
    "ordered": ZoneType.ORDERED,
    "felineseg": ZoneType.FELINESEG,
    "fetriangle": ZoneType.FETRIANGLE,
    "fequadrilateral": ZoneType.FEQUADRILATERAL,
    "fetetrahedron": ZoneType.FETETRAHEDRON,
    "febrick": ZoneType.FEBRICK,
    "fepolygon": ZoneType.FEPOLYGON,
    "fepolyhedron": ZoneType.FEPOLYHEDRON,
}

#: ASCII keyword → FileType.
_STR_TO_FILETYPE: dict[str, FileType] = {
    "full": FileType.FULL,
    "grid": FileType.GRID,
    "solution": FileType.SOLUTION,
}

#: ZoneType → ASCII keyword (for writing).
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

#: FileType → ASCII keyword (FULL omitted from output).
_FILETYPE_STR: dict[FileType, str] = {
    FileType.GRID: "GRID",
    FileType.SOLUTION: "SOLUTION",
}

#: DataType → NumPy dtype string (used by Write for casting).
_DT_TO_DTYPE: dict[DataType, str] = {
    DataType.FLOAT: "f4",
    DataType.DOUBLE: "f8",
    DataType.INT32: "i4",
    DataType.INT16: "i2",
    DataType.BYTE: "u1",
}

#: Values per line for Write data blocks.
_VALUES_PER_LINE: int = 5


# ===========================================================================
# Shared internal helpers
# ===========================================================================


def _quote(s: str) -> str:
    """Wrap *s* in double-quotes, escaping embedded double-quotes.

    :Call:
        >>> q = _quote("hello world")
    :Versions:
        * 2025-01-01 ``@user``: Version 1.0
    """
    return '"' + str(s).replace('"', '\\"') + '"'


def _unquote(s: str) -> str:
    r"""Remove surrounding double-quotes and unescape internal ``\\"``.

    :Call:
        >>> s = _unquote('"hello \\"world\\""')
    :Versions:
        * 2025-01-01 ``@user``: Version 1.0
    """
    s = s.strip()
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        return s[1:-1].replace('\\"', '"')
    return s


def _strip_comment(line: str) -> str:
    """Remove a Tecplot ``#`` comment and trailing whitespace.

    :Call:
        >>> clean = _strip_comment(line)
    :Versions:
        * 2025-01-01 ``@user``: Version 1.0
    """
    idx = line.find("#")
    return line[:idx].rstrip() if idx >= 0 else line.rstrip()


def _infer_data_type(arr: npt.NDArray) -> DataType:
    """Return the most appropriate :class:`DataType` for *arr*'s dtype.

    :Call:
        >>> dt = _infer_data_type(arr)
    :Versions:
        * 2025-01-01 ``@user``: Version 1.0
    """
    dtype = arr.dtype
    if dtype.kind == "f":
        return DataType.DOUBLE if dtype.itemsize >= 8 else DataType.FLOAT
    if dtype.kind in ("i", "u"):
        if dtype.itemsize >= 4:
            return DataType.INT32
        if dtype.itemsize == 2:
            return DataType.INT16
        return DataType.BYTE
    return DataType.FLOAT


# ===========================================================================
# Parsing helpers (used only by Read)
# ===========================================================================


def _extract_quoted_strings(text: str) -> list[str]:
    """Return all double-quoted strings found in *text* (content unescaped).

    :Call:
        >>> names = _extract_quoted_strings('"X" "Y" "Pressure"')
    :Versions:
        * 2025-01-01 ``@user``: Version 1.0
    """
    return [
        m.group(1).replace('\\"', '"')
        for m in re.finditer(r'"((?:[^"\\]|\\.)*)"', text)
    ]


def _kv_split(text: str) -> dict[str, str]:
    """Parse a loose ``KEY=VALUE`` string into an upper-cased-key dict.

    Handles quoted values, parenthesised blocks (VARLOCATION, VARSHARELIST),
    and bracketed lists (PASSIVEVARLIST).

    :Call:
        >>> d = _kv_split("I=3, J=4, K=1")
    :Versions:
        * 2025-01-01 ``@user``: Version 1.0
    """
    result: dict[str, str] = {}
    text = text.replace("\n", " ").strip().rstrip(",")
    i, n = 0, len(text)

    while i < n:
        # Skip separators
        while i < n and text[i] in " \t,":
            i += 1
        if i >= n:
            break

        # Key
        key_start = i
        while i < n and text[i] not in "= \t,":
            i += 1
        key = text[key_start:i].strip().upper()
        if not key:
            i += 1
            continue

        # Skip whitespace and '='
        while i < n and text[i] in " \t":
            i += 1
        if i >= n or text[i] != "=":
            result[key] = ""
            continue
        i += 1
        while i < n and text[i] in " \t":
            i += 1

        # Value
        if i >= n:
            result[key] = ""
            continue

        if text[i] == '"':
            j = i + 1
            while j < n and not (text[j] == '"' and text[j - 1] != "\\"):
                j += 1
            value = text[i + 1 : j].replace('\\"', '"')
            i = j + 1
        elif text[i] == "(":
            depth, j = 0, i
            while j < n:
                if text[j] == "(":
                    depth += 1
                elif text[j] == ")":
                    depth -= 1
                    if depth == 0:
                        j += 1
                        break
                j += 1
            value = text[i:j]
            i = j
        elif text[i] == "[":
            j = text.find("]", i)
            value = text[i : j + 1] if j >= 0 else text[i:]
            i = j + 1 if j >= 0 else n
        else:
            j = i
            while j < n and text[j] not in " \t,":
                j += 1
            value = text[i:j]
            i = j

        result[key] = value.strip()

    return result


def _parse_index_list(text: str) -> list[int]:
    """Parse ``[1,3,5]`` into a list of 0-based integers.

    :Call:
        >>> indices = _parse_index_list("[1,3,5]")
    :Versions:
        * 2025-01-01 ``@user``: Version 1.0
    """
    text = text.strip().lstrip("[").rstrip("]")
    result = []
    for tok in re.split(r"[,\s]+", text):
        tok = tok.strip()
        if tok:
            try:
                result.append(int(tok) - 1)  # 1-based → 0-based
            except ValueError:
                pass
    return result


def _parse_varlocation(text: str) -> dict[int, ValueLocation]:
    """Parse ``VARLOCATION=([i,j,...]=CELLCENTERED)`` → ``{0-based: loc}``.

    :Call:
        >>> locs = _parse_varlocation("([3,4]=CELLCENTERED)")
    :Versions:
        * 2025-01-01 ``@user``: Version 1.0
    """
    result: dict[int, ValueLocation] = {}
    for m in re.finditer(r"\[([^\]]+)\]\s*=\s*(\w+)", text):
        indices_str = m.group(1)
        loc_str = m.group(2).upper()
        loc = (
            ValueLocation.CELL_CENTERED
            if loc_str == "CELLCENTERED"
            else ValueLocation.NODAL
        )
        for tok in re.split(r"[,\s]+", indices_str):
            tok = tok.strip()
            if tok:
                with contextlib.suppress(ValueError):
                    result[int(tok) - 1] = loc
    return result


def _parse_varsharelist(text: str) -> dict[int, int]:
    """Parse ``VARSHARELIST=([i]=z,[j]=z)`` → ``{0-based var: 1-based zone}``.

    :Call:
        >>> share = _parse_varsharelist("([1]=2,[2]=2)")
    :Versions:
        * 2025-01-01 ``@user``: Version 1.0
    """
    result: dict[int, int] = {}
    for m in re.finditer(r"\[(\d+)\]\s*=\s*(\d+)", text):
        result[int(m.group(1)) - 1] = int(m.group(2))  # 0-based var, 1-based zone
    return result


def _parse_auxdata_line(line: str) -> tuple[str, str]:
    """Parse ``DATASETAUXDATA name="value"`` → ``(name, value)``.

    :Call:
        >>> name, val = _parse_auxdata_line('DATASETAUXDATA Solver="MyCFD"')
    :Versions:
        * 2025-01-01 ``@user``: Version 1.0
    """
    m = re.match(r"(?i)DATASETAUXDATA\s+", line)
    rest = line[m.end() :] if m else line
    eq = rest.find("=")
    if eq < 0:
        return "", ""
    return rest[:eq].strip(), _unquote(rest[eq + 1 :].strip())


def _apply_varauxdata(line: str, var_auxdata_list: list) -> None:
    """Parse ``VARAUXDATA 1-based-idx name="value"`` and store in list.

    :Call:
        >>> _apply_varauxdata(line, var_auxdata_list)
    :Versions:
        * 2025-01-01 ``@user``: Version 1.0
    """
    m = re.match(r"(?i)VARAUXDATA\s+(\d+)\s+", line)
    if not m:
        return
    var_idx = int(m.group(1))  # 1-based
    rest = line[m.end() :]
    eq = rest.find("=")
    if eq < 0:
        return
    name = rest[:eq].strip()
    value = _unquote(rest[eq + 1 :].strip())
    if 1 <= var_idx < len(var_auxdata_list) and var_auxdata_list[var_idx] is not None:
        var_auxdata_list[var_idx]._data[name] = value


class _LineBuffer:
    """Peekable iterator over stripped, comment-free text lines.

    :Call:
        >>> buf = _LineBuffer(lines)
    :Versions:
        * 2025-01-01 ``@user``: Version 1.0
    """

    def __init__(self, lines: list[str]) -> None:
        self._lines: list[str] = lines
        self._pos: int = 0

    def has_more(self) -> bool:
        """Return ``True`` if there are unconsumed lines."""
        return self._pos < len(self._lines)

    def peek_stripped(self) -> str:
        """Return the next non-blank stripped line without consuming it."""
        pos = self._pos
        while pos < len(self._lines):
            raw = _strip_comment(self._lines[pos])
            stripped = raw.strip()
            if stripped:
                return stripped
            pos += 1
        return ""

    def next_stripped(self) -> str:
        """Consume and return the next non-blank stripped line."""
        while self._pos < len(self._lines):
            raw = _strip_comment(self._lines[self._pos])
            self._pos += 1
            stripped = raw.strip()
            if stripped:
                return stripped
        return ""


# ===========================================================================
# ReadAuxData
# ===========================================================================


class ReadAuxData:
    """Dictionary-like container for Tecplot auxiliary data strings.

    Interface matches :class:`szl.ReadAuxData` exactly.

    :Call:
        >>> aux = ReadAuxData({"Solver": "MyCFD"})
    :Versions:
        * 2025-01-01 ``@user``: Version 1.0
    """

    def __init__(self, data: dict[str, str] | None = None) -> None:
        self._data: dict[str, str] = data if data is not None else {}

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, key: str) -> str:
        return self._data[key]

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __repr__(self) -> str:
        return f"ReadAuxData({self._data!r})"

    def __str__(self) -> str:
        return repr(self)

    def get(self, key: str, default: Any = None) -> str | None:
        """Return value for *key* or *default*."""
        return self._data.get(key, default)

    def keys(self) -> Iterator[str]:
        """Return iterator over auxiliary data names."""
        return self._data.keys()

    def values(self) -> Iterator[str]:
        """Return iterator over auxiliary data values."""
        return self._data.values()

    def items(self) -> Iterator[tuple[str, str]]:
        """Return iterator over (name, value) pairs."""
        return self._data.items()

    def as_int(self, key: str, default: int | None = None) -> int | None:
        """Return value for *key* as :class:`int`, or *default* on failure.

        :Call:
            >>> n = aux.as_int("Iteration", default=0)
        :Versions:
            * 2025-01-01 ``@user``: Version 1.0
        """
        try:
            return int(self._data[key])
        except (KeyError, ValueError):
            return default

    def as_float(self, key: str, default: float | None = None) -> float | None:
        """Return value for *key* as :class:`float`, or *default* on failure.

        :Call:
            >>> t = aux.as_float("TimeValue", default=0.0)
        :Versions:
            * 2025-01-01 ``@user``: Version 1.0
        """
        try:
            return float(self._data[key])
        except (KeyError, ValueError):
            return default

    def as_bool(self, key: str, default: bool | None = None) -> bool | None:
        """Return value for *key* as :class:`bool`, or *default* on failure.

        Recognises ``"true"``/``"false"``, ``"yes"``/``"no"``,
        ``"1"``/``"0"`` (case-insensitive).

        :Call:
            >>> flag = aux.as_bool("IsBoundaryZone", default=False)
        :Versions:
            * 2025-01-01 ``@user``: Version 1.0
        """
        try:
            v = self._data[key].lower().strip()
            if v in ("true", "t", "yes", "y", "1"):
                return True
            if v in ("false", "f", "no", "n", "0"):
                return False
        except (KeyError, AttributeError):
            pass
        return default


# ===========================================================================
# ReadVariable
# ===========================================================================


class ReadVariable:
    """Variable metadata and data for one zone parsed from an ASCII DAT file.

    Interface matches :class:`szl.ReadVariable`.

    :Call:
        >>> var = ReadVariable(name, data, value_location, ...)
    :Versions:
        * 2025-01-01 ``@user``: Version 1.0
    """

    def __init__(
        self,
        name: str,
        data: npt.NDArray | None,
        value_location: ValueLocation = ValueLocation.NODAL,
        is_passive: bool = False,
        shared_zone: int | None = None,
    ) -> None:
        self._name: str = name
        self._data: npt.NDArray | None = data
        self._value_location: ValueLocation = value_location
        self._is_passive: bool = is_passive
        self._shared_zone: int | None = shared_zone

    @property
    def name(self) -> str:
        """Return variable name string."""
        return self._name

    @property
    def data_type(self) -> DataType:
        """Return :class:`DataType` inferred from the stored NumPy dtype.

        Passive and shared variables return :attr:`DataType.FLOAT` as a
        placeholder since no data array is stored.
        """
        if self._data is None:
            return DataType.FLOAT
        dt = self._data.dtype
        if dt == np.float64:
            return DataType.DOUBLE
        if dt == np.float32:
            return DataType.FLOAT
        if dt == np.int32:
            return DataType.INT32
        if dt == np.int16:
            return DataType.INT16
        if dt == np.uint8:
            return DataType.BYTE
        return DataType.DOUBLE if dt.itemsize >= 8 else DataType.FLOAT

    @property
    def value_location(self) -> ValueLocation:
        """Return :class:`ValueLocation` for this variable."""
        return self._value_location

    def is_enabled(self) -> bool:
        """Return ``True`` unless the variable is passive."""
        return not self._is_passive

    def is_passive(self) -> bool:
        """Return ``True`` if the variable is passive (no data in this zone)."""
        return self._is_passive

    @property
    def shared_zone(self) -> int | None:
        """Return 1-based source-zone index if shared, else ``None``."""
        return self._shared_zone

    @property
    def num_values(self) -> int:
        """Return number of values stored for this variable."""
        return 0 if self._data is None else self._data.size

    @property
    def values(self) -> npt.NDArray | None:
        """Return the data array, or ``None`` for passive/shared variables."""
        return self._data

    def get_values(
        self,
        value_range: tuple[int | None, int | None] = (None, None),
    ) -> npt.NDArray | None:
        """Return a slice of the data array using a 1-based range.

        :Call:
            >>> arr = var.get_values((1, 100))
        :Inputs:
            *value_range*: ``(None, None)`` | ``(start, end)``
                1-based start (inclusive) and end (exclusive).
        :Outputs:
            *arr*: :class:`numpy.ndarray` | ``None``
        :Raises:
            :exc:`ValueError`: If only one of start/end is given.
        :Versions:
            * 2025-01-01 ``@user``: Version 1.0
        """
        if self._data is None:
            return None
        start, end = value_range
        if start is None and end is None:
            return self._data
        if start is None or end is None:
            raise ValueError("Both start and end indices must be specified.")
        return self._data[start - 1 : end - 1]

    def __repr__(self) -> str:
        return (
            f"ReadVariable(name={self._name!r}, "
            f"data_type={self.data_type.name}, "
            f"num_values={self.num_values})"
        )


# ===========================================================================
# ReadZone
# ===========================================================================


class ReadZone:
    """Zone data parsed from a Tecplot ASCII DAT file.

    Interface matches :class:`szl.ReadZone`.

    :Call:
        >>> zone = ReadZone(title, zone_type, I, J, K, ...)
    :Versions:
        * 2025-01-01 ``@user``: Version 1.0
    """

    def __init__(
        self,
        title: str,
        zone_type: ZoneType,
        I: int,
        J: int,
        K: int,
        solution_time: float,
        strand_id: int,
        variables: list[ReadVariable],
        auxdata: ReadAuxData,
        node_map: npt.NDArray | None = None,
    ) -> None:
        self.title: str = title
        self.zone_type: ZoneType = zone_type
        self.I: int = I
        self.J: int = J
        self.K: int = K
        self.solution_time: float = solution_time
        self.strand_id: int = strand_id
        self.variable: list[ReadVariable] = variables
        self.auxdata: ReadAuxData = auxdata
        self.node_map: npt.NDArray | None = node_map

    @property
    def dimensions(self) -> tuple[int, int, int]:
        """Return ``(I, J, K)`` dimensions."""
        return (self.I, self.J, self.K)

    @property
    def num_nodes(self) -> int:
        """Return number of nodes/points."""
        if self.zone_type == ZoneType.ORDERED:
            return self.I * self.J * self.K
        return self.I  # FE: I stores num_nodes

    @property
    def num_elements(self) -> int:
        """Return number of elements (equals num_points for ORDERED)."""
        if self.zone_type == ZoneType.ORDERED:
            return self.I * self.J * self.K
        return self.J  # FE: J stores num_cells

    def is_enabled(self) -> bool:
        """Always ``True`` for zones successfully parsed from a file."""
        return True

    def __repr__(self) -> str:
        return (
            f"ReadZone(title={self.title!r}, "
            f"zone_type={self.zone_type.name}, "
            f"I={self.I}, J={self.J}, K={self.K})"
        )


# ===========================================================================
# Read — top-level file reader
# ===========================================================================


class Read:
    """Read a Tecplot ASCII DAT file into memory.

    The entire file is parsed on construction.  All data is then available
    through the same attributes and methods as :class:`szl.Read`.

    :Call:
        >>> dat = Read("Onera.dat")
        >>> dat = tecio.open("Onera.dat", "r")
    :Inputs:
        *path*: :class:`str`
            Path to the ``.dat`` file.
    :Raises:
        :exc:`FileNotFoundError`: If *path* does not exist.
        :exc:`ValueError`: On unsupported format features.
    :Versions:
        * 2025-01-01 ``@user``: Version 1.0
    """

    def __init__(self, path: str) -> None:
        self._path: str = str(path)

        self._title: str = ""
        self._file_type: FileType = FileType.FULL
        self._variable_names: list[str] = []
        self._zones: list[ReadZone] = []
        self._auxdata: ReadAuxData = ReadAuxData()
        # Index 0 is a None placeholder so that 1-based indexing works directly.
        self._var_auxdata: list[ReadAuxData | None] = [None]

        self._parse()

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

    # -- public API (mirrors szl.Read) ----------------------------------------

    @property
    def file_type(self) -> FileType:
        """Return :class:`FileType` enum for this file."""
        return self._file_type

    @property
    def title(self) -> str:
        """Return the dataset title string."""
        return self._title

    @property
    def num_vars(self) -> int:
        """Return the number of variables in the dataset."""
        return len(self._variable_names)

    @property
    def variables(self) -> list[str]:
        """Return the ordered list of variable name strings."""
        return list(self._variable_names)

    @property
    def num_zones(self) -> int:
        """Return the number of zones in the file."""
        return len(self._zones)

    @property
    def zone(self) -> list[ReadZone]:
        """Return the list of :class:`ReadZone` objects (0-based)."""
        return self._zones

    @property
    def num_auxdata_items(self) -> int:
        """Return the number of dataset-level auxiliary data items."""
        return len(self._auxdata)

    @property
    def auxdata(self) -> ReadAuxData:
        """Return the dataset-level :class:`ReadAuxData` container."""
        return self._auxdata

    @property
    def var_auxdata(self) -> list[ReadAuxData | None]:
        """Return the per-variable aux data list (index 0 is ``None``)."""
        return self._var_auxdata

    def get_var_auxdata(self, var_index: int) -> ReadAuxData:
        """Return auxiliary data for variable *var_index* (1-based).

        :Call:
            >>> aux = dat.get_var_auxdata(1)
        :Raises:
            :exc:`IndexError`: If *var_index* is out of range.
        :Versions:
            * 2025-01-01 ``@user``: Version 1.0
        """
        if var_index < 1 or var_index > self.num_vars:
            raise IndexError(
                f"Variable index {var_index} out of range [1, {self.num_vars}]"
            )
        return self._var_auxdata[var_index]

    def get_zone_auxdata(self, zone_index: int) -> ReadAuxData:
        """Return auxiliary data for zone *zone_index* (1-based).

        :Call:
            >>> aux = dat.get_zone_auxdata(1)
        :Raises:
            :exc:`IndexError`: If *zone_index* is out of range.
        :Versions:
            * 2025-01-01 ``@user``: Version 1.0
        """
        if zone_index < 1 or zone_index > self.num_zones:
            raise IndexError(
                f"Zone index {zone_index} out of range [1, {self.num_zones}]"
            )
        return self._zones[zone_index - 1].auxdata

    # -- parser ---------------------------------------------------------------

    def _parse(self) -> None:
        """Read and parse the entire DAT file.

        :Call:
            >>> self._parse()
        :Versions:
            * 2025-01-01 ``@user``: Version 1.0
        """
        with open(self._path, encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()

        tokens = _LineBuffer(lines)
        self._parse_file_header(tokens)

        # Build per-variable aux data slots now that num_vars is known.
        self._var_auxdata = [None] + [ReadAuxData() for _ in range(self.num_vars)]

        while tokens.has_more():
            line = tokens.peek_stripped()
            upper = line.upper()
            if upper.startswith("ZONE"):
                self._parse_zone(tokens)
            elif upper.startswith("DATASETAUXDATA"):
                raw = tokens.next_stripped()
                name, value = _parse_auxdata_line(raw)
                if name:
                    self._auxdata._data[name] = value
            elif upper.startswith("VARAUXDATA"):
                raw = tokens.next_stripped()
                _apply_varauxdata(raw, self._var_auxdata)
            else:
                tokens.next_stripped()  # skip unrecognised lines

    def _parse_file_header(self, tokens: _LineBuffer) -> None:
        """Parse TITLE, FILETYPE, VARIABLES, and dataset-level aux data.

        Stops (without consuming) when a ZONE keyword is seen.

        :Call:
            >>> self._parse_file_header(tokens)
        :Versions:
            * 2025-01-01 ``@user``: Version 1.0
        """
        while tokens.has_more():
            line = tokens.peek_stripped()
            upper_key = (
                line.split("=")[0].strip().upper()
                if "=" in line
                else line.strip().upper()
            )

            if upper_key.startswith("ZONE"):
                break

            tokens.next_stripped()  # consume

            if upper_key == "TITLE":
                rhs = line.split("=", 1)[1].strip() if "=" in line else ""
                self._title = _unquote(rhs)

            elif upper_key == "FILETYPE":
                rhs = line.split("=", 1)[1].strip() if "=" in line else ""
                self._file_type = _STR_TO_FILETYPE.get(
                    rhs.strip().lower(), FileType.FULL
                )

            elif upper_key == "VARIABLES":
                # Names may be on the same line or on subsequent quoted lines.
                rhs = line.split("=", 1)[1].strip() if "=" in line else ""
                names = _extract_quoted_strings(rhs)

                while tokens.has_more():
                    nxt = tokens.peek_stripped()
                    if not nxt:
                        tokens.next_stripped()
                        continue
                    nxt_upper = nxt.lstrip().upper()
                    # A continuation line is a bare quoted string with no '=' keyword.
                    if nxt.lstrip().startswith('"'):
                        more = _extract_quoted_strings(tokens.next_stripped())
                        names.extend(more)
                    elif (
                        "=" not in nxt.split("#")[0]
                        and not nxt_upper.startswith("ZONE")
                        and not nxt_upper.startswith("DATASETAUXDATA")
                        and not nxt_upper.startswith("VARAUXDATA")
                    ):
                        # Non-quoted, non-keyword line — could be a bare name, skip.
                        tokens.next_stripped()
                    else:
                        break

                self._variable_names = names

            elif upper_key.startswith("DATASETAUXDATA"):
                name, value = _parse_auxdata_line(line)
                if name:
                    self._auxdata._data[name] = value

            # VARAUXDATA lines before the first ZONE are deferred — they need
            # the variable list, which may not yet be complete.  They are
            # processed in the main _parse() loop after the header finishes.

    def _parse_zone(self, tokens: _LineBuffer) -> None:
        """Parse one ZONE block (header + data blocks + connectivity).

        :Call:
            >>> self._parse_zone(tokens)
        :Versions:
            * 2025-01-01 ``@user``: Version 1.0
        """
        # ------------------------------------------------------------------ #
        # 1. Collect header lines                                             #
        # ------------------------------------------------------------------ #
        header_lines: list[str] = [tokens.next_stripped()]  # ZONE T=...

        while tokens.has_more():
            nxt = tokens.peek_stripped()
            if not nxt:
                tokens.next_stripped()
                continue
            nxt_upper = nxt.lstrip().upper()
            # Stop at a new zone, top-level keyword, or the first data line.
            if nxt_upper.startswith("ZONE"):
                break
            if nxt_upper.startswith("DATASETAUXDATA") or nxt_upper.startswith(
                "VARAUXDATA"
            ):
                break
            first_ch = nxt.lstrip()[0] if nxt.lstrip() else ""
            if first_ch in "0123456789+-":
                break
            header_lines.append(tokens.next_stripped())

        header_text = " ".join(header_lines)

        # Strip the leading ZONE keyword from the header text.
        m_zone = re.match(r"(?i)^ZONE\s*", header_text)
        if m_zone:
            header_text = header_text[m_zone.end() :]

        # ------------------------------------------------------------------ #
        # 2. Parse header key=value pairs                                     #
        # ------------------------------------------------------------------ #
        kv = _kv_split(header_text)

        zone_title = _unquote(kv.get("T", ""))
        strand_id = int(kv.get("STRANDID", "0") or "0")
        solution_time = float(kv.get("SOLUTIONTIME", "0.0") or "0.0")

        zt_raw = kv.get("ZONETYPE", "ordered").rstrip(",").strip().lower()
        zone_type = _STR_TO_ZONETYPE.get(zt_raw, ZoneType.ORDERED)

        if zone_type in _FE_POLY:
            raise ValueError(
                f"Zone type {zone_type.name!r} is not yet supported by the "
                "ASCII reader."
            )

        if zone_type == ZoneType.ORDERED:
            I = int(kv.get("I", "1") or "1")
            J = int(kv.get("J", "1") or "1")
            K = int(kv.get("K", "1") or "1")
            num_nodes = I * J * K
            num_cells = num_nodes
        else:
            num_nodes = int(kv.get("NODES", kv.get("N", "0")) or "0")
            num_cells = int(kv.get("ELEMENTS", kv.get("E", "0")) or "0")
            I, J, K = num_nodes, num_cells, 0

        packing = kv.get("DATAPACKING", "BLOCK").upper()
        if packing != "BLOCK":
            raise ValueError(
                f"DATAPACKING={packing!r} is not supported; only BLOCK "
                "is implemented in the ASCII reader."
            )

        # Variable locations (0-based index → ValueLocation)
        var_locs: dict[int, ValueLocation] = {}
        if "VARLOCATION" in kv:
            var_locs = _parse_varlocation(kv["VARLOCATION"])

        # Passive variables (0-based)
        passive_set: set[int] = set()
        if "PASSIVEVARLIST" in kv:
            passive_set = set(_parse_index_list(kv["PASSIVEVARLIST"]))

        # Shared variables {0-based → 1-based source zone}
        share_map: dict[int, int] = {}
        if "VARSHARELIST" in kv:
            share_map = _parse_varsharelist(kv["VARSHARELIST"])

        # Connectivity sharing
        con_share_zone = int(kv.get("CONNECTIVITYSHAREZONE", "0") or "0")

        # Zone-level aux data (AUXDATA key="value" lines inside the header)
        zone_aux: dict[str, str] = {}
        for m in re.finditer(
            r"(?i)AUXDATA\s+(\S+)\s*=\s*(\"[^\"]*\"|[^\s,\"]+)", header_text
        ):
            zone_aux[m.group(1)] = _unquote(m.group(2))

        # ------------------------------------------------------------------ #
        # 3. Read variable data blocks                                        #
        # ------------------------------------------------------------------ #
        var_arrays: list[npt.NDArray | None] = [None] * self.num_vars

        for var_idx in range(self.num_vars):
            if var_idx in passive_set or var_idx in share_map:
                continue  # no data block for passive/shared variables
            loc = var_locs.get(var_idx, ValueLocation.NODAL)
            n_vals = num_cells if loc == ValueLocation.CELL_CENTERED else num_nodes
            var_arrays[var_idx] = self._read_float_block(tokens, n_vals)

        # ------------------------------------------------------------------ #
        # 4. Read connectivity (FE zones only)                                #
        # ------------------------------------------------------------------ #
        node_map: npt.NDArray | None = None

        if zone_type != ZoneType.ORDERED:
            if con_share_zone and self._zones:
                node_map = self._zones[con_share_zone - 1].node_map
            else:
                nodes_per_cell = _NODES_PER_ELEM[zone_type]
                flat = self._read_int_block(tokens, num_cells * nodes_per_cell)
                node_map = flat.reshape(num_cells, nodes_per_cell)

        # ------------------------------------------------------------------ #
        # 5. Build ReadVariable and ReadZone objects                          #
        # ------------------------------------------------------------------ #
        # For ordered zones, reshape each variable array from flat 1-D to
        # (I, J, K) for nodal variables or (I-1, J-1, K-1) for cell-centered
        # so that zone dimensions can be inferred from array shape downstream.
        def _shaped(arr, loc):
            if arr is None or zone_type != ZoneType.ORDERED:
                return arr
            if loc == ValueLocation.CELL_CENTERED:
                shape = (max(I - 1, 1), max(J - 1, 1), max(K - 1, 1))
            else:
                shape = (I, J, K)
            if arr.size == shape[0] * shape[1] * shape[2]:
                return arr.reshape(shape, order="F")
            return arr

        read_vars = [
            ReadVariable(
                name=name,
                data=_shaped(
                    var_arrays[idx],
                    var_locs.get(idx, ValueLocation.NODAL),
                ),
                value_location=var_locs.get(idx, ValueLocation.NODAL),
                is_passive=(idx in passive_set),
                shared_zone=share_map.get(idx, None),
            )
            for idx, name in enumerate(self._variable_names)
        ]

        self._zones.append(
            ReadZone(
                title=zone_title,
                zone_type=zone_type,
                I=I,
                J=J,
                K=K,
                solution_time=solution_time,
                strand_id=strand_id,
                variables=read_vars,
                auxdata=ReadAuxData(zone_aux),
                node_map=node_map,
            )
        )

    # -- low-level block readers ----------------------------------------------

    @staticmethod
    def _read_float_block(tokens: _LineBuffer, n_values: int) -> npt.NDArray:
        """Read exactly *n_values* floats from *tokens* into a float64 array.

        :Call:
            >>> arr = Read._read_float_block(tokens, 100)
        :Versions:
            * 2025-01-01 ``@user``: Version 1.0
        """
        values: list[float] = []
        while len(values) < n_values and tokens.has_more():
            nxt = tokens.peek_stripped()
            upper = nxt.lstrip().upper()
            if (
                upper.startswith("ZONE")
                or upper.startswith("DATASETAUXDATA")
                or upper.startswith("VARAUXDATA")
                or (upper.startswith("TITLE") and "=" in upper)
                or (upper.startswith("VARIABLES") and "=" in upper)
            ):
                break
            line = tokens.next_stripped()
            for tok in line.split():
                try:
                    values.append(float(tok))
                    if len(values) == n_values:
                        break
                except ValueError:
                    pass
        return np.array(values[:n_values], dtype=np.float64)

    @staticmethod
    def _read_int_block(tokens: _LineBuffer, n_values: int) -> npt.NDArray:
        """Read exactly *n_values* integers from *tokens* into an int64 array.

        :Call:
            >>> arr = Read._read_int_block(tokens, 24)
        :Versions:
            * 2025-01-01 ``@user``: Version 1.0
        """
        values: list[int] = []
        while len(values) < n_values and tokens.has_more():
            nxt = tokens.peek_stripped()
            upper = nxt.lstrip().upper()
            if upper.startswith("ZONE") or upper.startswith("DATASETAUXDATA"):
                break
            line = tokens.next_stripped()
            for tok in line.split():
                try:
                    values.append(int(tok))
                    if len(values) == n_values:
                        break
                except ValueError:
                    pass
        return np.array(values[:n_values], dtype=np.int64)
