r"""Tecplot ASCII DAT file reader API.

Supported ``DATAPACKING`` modes:
    * ``BLOCK`` — one contiguous value block per variable (Tecplot default).
    * ``POINT`` — one row of all variable values per node, followed by a separate
      row-per-cell section for any cell-centred variables. This is the format most
      commonly produced by third-party exporters and tools that treat the file like a
      CSV with a header.
"""

from __future__ import annotations

import contextlib
import re
from collections.abc import ItemsView, Iterator, KeysView, ValuesView
from typing import Any, overload

import numpy as np
import numpy.typing as npt

from .._containers import VariableList, ZoneList, select_variable_arrays
from ..libtecio import (
    DataPacking,
    DataType,
    FileType,
    ValueLocation,
    ZoneType,
)

# --------------------------------------------------------------------------------------
# Shared module-level constants
# --------------------------------------------------------------------------------------

# FE zone types fully supported for reading and writing
_FE_SIMPLE: frozenset[ZoneType] = frozenset({
    ZoneType.FELINESEG,
    ZoneType.FETRIANGLE,
    ZoneType.FEQUADRILATERAL,
    ZoneType.FETETRAHEDRON,
    ZoneType.FEBRICK,
})

# Zone types whose connectivity is face-based
_FE_POLY: frozenset[ZoneType] = frozenset({
    ZoneType.FEPOLYGON,
    ZoneType.FEPOLYHEDRON,
})

# Nodes per element for each supported simple FE type
_NODES_PER_ELEM: dict[ZoneType, int] = {
    ZoneType.FELINESEG: 2,
    ZoneType.FETRIANGLE: 3,
    ZoneType.FEQUADRILATERAL: 4,
    ZoneType.FETETRAHEDRON: 4,
    ZoneType.FEBRICK: 8,
}

# ASCII keyword to ZoneType
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

# ASCII keyword to FileType
_STR_TO_FILETYPE: dict[str, FileType] = {
    "full": FileType.FULL,
    "grid": FileType.GRID,
    "solution": FileType.SOLUTION,
}

# ASCII keyword to DataPacking
_STR_TO_DATAPACKING: dict[str, DataPacking] = {
    "point": DataPacking.POINT,
    "block": DataPacking.BLOCK,
}

# Legacy F= (format) keyword to (is_fe, DataPacking)
_STR_TO_LEGACY_FORMAT: dict[str, tuple[bool, DataPacking]] = {
    "POINT": (False, DataPacking.POINT),
    "BLOCK": (False, DataPacking.BLOCK),
    "FEPOINT": (True, DataPacking.POINT),
    "FEBLOCK": (True, DataPacking.BLOCK),
}

# Legacy ``ET=`` (element type) keyword to ZoneType
_STR_TO_ELEMENT_TYPE: dict[str, ZoneType] = {
    "LINESEG": ZoneType.FELINESEG,
    "TRIANGLE": ZoneType.FETRIANGLE,
    "QUADRILATERAL": ZoneType.FEQUADRILATERAL,
    "TETRAHEDRON": ZoneType.FETETRAHEDRON,
    "BRICK": ZoneType.FEBRICK,
}

# ZoneType to ASCII keyword
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

# FileType to ASCII keyword
_FILETYPE_STR: dict[FileType, str] = {
    FileType.GRID: "GRID",
    FileType.SOLUTION: "SOLUTION",
}

# DataType to NumPy dtype string
_DT_TO_DTYPE: dict[DataType, str] = {
    DataType.FLOAT: "f4",
    DataType.DOUBLE: "f8",
    DataType.INT32: "i4",
    DataType.INT16: "i2",
    DataType.BYTE: "u1",
}

# ASCII DT= keyword (and common aliases) to DataType
_STR_TO_DATATYPE: dict[str, DataType] = {
    "single": DataType.FLOAT,
    "float": DataType.FLOAT,
    "double": DataType.DOUBLE,
    "longint": DataType.INT32,
    "shortint": DataType.INT16,
    "byte": DataType.BYTE,
}

#: Values per line for Write data blocks.
_VALUES_PER_LINE: int = 5

# Safety cap on the number of grow-and-reparse iterations the vectorized block readers
# will attempt before giving up and falling back to the tolerant token-by-token parser
_MAX_FAST_BLOCK_ITERATIONS: int = 64


# ======================================================================================
# Shared internal helpers
# ======================================================================================


def _quote(s: str) -> str:
    """Wrap *s* in double-quotes, escaping embedded double-quotes.

    Example:
        >>> q = _quote("hello world")
    """
    return '"' + str(s).replace('"', '\\"') + '"'


def _unquote(s: str) -> str:
    r"""Remove surrounding double-quotes and unescape internal ``\\"``.

    Example:
        >>> s = _unquote('"hello \\"world\\""')
    """
    s = s.strip()
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        return s[1:-1].replace('\\"', '"')
    return s


def _strip_comment(line: str) -> str:
    """Remove a Tecplot ``#`` comment and trailing whitespace.

    Example:
        >>> clean = _strip_comment(line)
    """
    idx = line.find("#")
    return line[:idx].rstrip() if idx >= 0 else line.rstrip()


def _is_float_block_boundary(line: str) -> bool:
    r"""Return ``True`` if a stripped *line* ends a float data block early.

    Shared by :meth:`Read._read_float_block_slow` (line-by-line) and
    :meth:`Read._read_float_block_fast` (bulk, on parse failure) so the two readers
    agree on exactly where a block stops.

    Example:
        >>> _is_float_block_boundary("ZONE T=\\"next\\"")
        True
    """
    upper = line.lstrip().upper()
    if not upper:
        return False
    if upper.split("=")[0].split()[0] == "ZONE":
        return True
    return (
        upper.startswith(("DATASETAUXDATA", "VARAUXDATA"))
        or upper.startswith("TITLE")
        and "=" in upper
        or upper.startswith("VARIABLES")
        and "=" in upper
    )


def _is_int_block_boundary(line: str) -> bool:
    r"""Return ``True`` if a stripped *line* ends an int (connectivity) block.

    Example:
        >>> _is_int_block_boundary("DATASETAUXDATA foo=\\"bar\\"")
        True
    """
    upper = line.lstrip().upper()
    return upper.startswith(("ZONE", "DATASETAUXDATA"))


def _infer_data_type(arr: npt.NDArray) -> DataType:
    """Return the most appropriate :class:`DataType` for *arr*'s dtype.

    Example:
        >>> dt = _infer_data_type(arr)
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


# ======================================================================================
# Parsing helpers (used only by Read)
# ======================================================================================


def _extract_quoted_strings(text: str) -> list[str]:
    """Return all double-quoted strings found in *text* (content unescaped).

    Example:
        >>> names = _extract_quoted_strings('"X" "Y" "Pressure"')
    """
    return [
        m.group(1).replace('\\"', '"')
        for m in re.finditer(r'"((?:[^"\\]|\\.)*)"', text)
    ]


def _next_token_is_value(text: str, i: int) -> bool:
    """Return ``True`` if the token at *i* is a value, not the next key.

    Supports the legacy whitespace-separated ``KEY VALUE`` form in zone headers
    (e.g. ``STRANDID 2``). A token is treated as the *next key* — and therefore not a
    value for the preceding key — when it is immediately followed (after optional
    whitespace) by ``=``. A quoted token is always a value.

    Example:
        >>> _next_token_is_value("STRANDID 2", 9)
        True

    Args:
        text: The full header string being parsed.
        i: Index of the first character of the candidate token.

    Returns:
        ``True`` if the token should be consumed as the current key's value.
    """
    n = len(text)
    if i < n and text[i] == '"':
        return True
    j = i
    while j < n and text[j] not in " \t,=":
        j += 1
    # Skip trailing whitespace and see whether the token is followed by '='.
    while j < n and text[j] in " \t":
        j += 1
    return not (j < n and text[j] == "=")


def _kv_split(text: str) -> dict[str, str]:
    """Parse a loose ``KEY=VALUE`` string into an upper-cased-key dict.

    Handles quoted values, parenthesised blocks (VARLOCATION, VARSHARELIST), and
    bracketed lists (PASSIVEVARLIST).

    Example:
        >>> d = _kv_split("I=3, J=4, K=1")

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

        # Determine the separator between key and value. The modern form is
        # ``KEY = VALUE``; some legacy files (and ``preplot``) also accept a bare
        # whitespace separator, e.g. ``STRANDID 2``.
        while i < n and text[i] in " \t":
            i += 1
        if i < n and text[i] == "=":
            # Standard ``KEY=VALUE``.
            i += 1
            while i < n and text[i] in " \t":
                i += 1
        elif i >= n or not _next_token_is_value(text, i):
            # Bare flag keyword with no value (e.g. at end of header or directly
            # before another ``KEY=`` pair).
            result[key] = ""
            continue
        # else: legacy ``KEY VALUE`` form — ``i`` already points at the value.

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


def _parse_legacy_format(text: str) -> tuple[bool, DataPacking]:
    """Parse a legacy ``F=`` value into ``(is_fe, DataPacking)``.

    The ``F`` keyword predates the modern ``ZONETYPE`` / ``DATAPACKING`` pair and
    encodes both pieces of information at once (``POINT``/``BLOCK`` for ordered zones,
    ``FEPOINT``/``FEBLOCK`` for finite-element zones).

    Example:
        >>> is_fe, packing = _parse_legacy_format("FEPOINT")

    Args:
        text: Raw ``F`` value string, e.g. ``"FEPOINT"`` (case-insensitive, any
            trailing comma is ignored).

    Returns:
        Tuple of ``(is_fe, packing)`` where *is_fe* is ``True`` for ``FEPOINT`` /
        ``FEBLOCK`` and *packing* is the corresponding :class:`DataPacking`.

    Raises:
        ValueError: If *text* is not one of the four recognised legacy formats.
    """
    raw = text.rstrip(",").strip().upper()
    try:
        return _STR_TO_LEGACY_FORMAT[raw]
    except KeyError:
        raise ValueError(
            f"Unrecognised legacy zone format F={text!r}; expected one of "
            "POINT, BLOCK, FEPOINT, FEBLOCK."
        ) from None


def _parse_legacy_element_type(text: str) -> ZoneType:
    """Parse a legacy ``ET=`` (element type) value into a :class:`ZoneType`.

    Recognises the canonical Tecplot element names (``TRIANGLE``,
    ``QUADRILATERAL``, ``TETRAHEDRON``, ``BRICK``, ``LINESEG``). A prefix-based
    fallback provides a little tolerance for the minor spelling variants some
    third-party exporters emit, keeping the reader as permissive as ``preplot``.

    Example:
        >>> zt = _parse_legacy_element_type("QUADRILATERAL")

    Args:
        text: Raw ``ET`` value string (case-insensitive, any trailing comma is
            ignored).

    Returns:
        The matching finite-element :class:`ZoneType`.

    Raises:
        ValueError: If *text* does not resemble any known element type.
    """
    raw = text.rstrip(",").strip().upper()
    if raw in _STR_TO_ELEMENT_TYPE:
        return _STR_TO_ELEMENT_TYPE[raw]
    # Prefix-based fallback for minor spelling variants.
    if raw.startswith("TRI"):
        return ZoneType.FETRIANGLE
    if raw.startswith("QUAD"):
        return ZoneType.FEQUADRILATERAL
    if raw.startswith("TET"):
        return ZoneType.FETETRAHEDRON
    if raw.startswith("BRICK"):
        return ZoneType.FEBRICK
    if raw.startswith("LINE"):
        return ZoneType.FELINESEG
    raise ValueError(f"Unrecognised legacy element type ET={text!r} in ZONE header.")


def _parse_index_list(text: str) -> list[int]:
    """Parse a bracketed index list (with optional range notation) into 0-based ints.

    Handles both comma-separated individuals and inclusive ranges::

        "[1,3,5]"    → [0, 2, 4]
        "[1-3,5]"    → [0, 1, 2, 4]
        "[2-4]"      → [1, 2, 3]

    Args:
        text: Raw value string including the surrounding brackets, e.g.
              ``"[1-3,5]"`` or ``"[2]"``.

    Returns:
        List of 0-based variable indices in the order they were listed.

    """
    text = text.strip().lstrip("[").rstrip("]")
    result: list[int] = []
    for tok in re.split(r"[,\s]+", text):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            # Range notation: "N-M" (both 1-based, inclusive).
            parts = tok.split("-", 1)
            with contextlib.suppress(ValueError):
                start_1 = int(parts[0])
                end_1 = int(parts[1])
                result.extend(range(start_1 - 1, end_1))  # → 0-based
        else:
            with contextlib.suppress(ValueError):
                result.append(int(tok) - 1)  # 1-based → 0-based
    return result


def _parse_varlocation(text: str) -> dict[int, ValueLocation]:
    """Parse ``VARLOCATION=([i-j,...]=CELLCENTERED)`` → ``{0-based: loc}``.

    Handles both individual indices and inclusive range notation::

        "([3,4]=CELLCENTERED)"   → {2: CELL_CENTERED, 3: CELL_CENTERED}
        "([3-5]=CELLCENTERED)"   → {2: CELL_CENTERED, 3: CELL_CENTERED,
                                    4: CELL_CENTERED}

    Args:
        text: Raw VARLOCATION value string, e.g.
              ``"([3-5]=CELLCENTERED)"``.

    Returns:
        Dict mapping 0-based variable index to :class:`ValueLocation`.

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
        for idx_0based in _parse_index_list(f"[{indices_str}]"):
            result[idx_0based] = loc
    return result


def _parse_varsharelist(text: str) -> dict[int, int]:
    """Parse ``VARSHARELIST=([i-j]=z,...)`` → ``{0-based var: 1-based zone}``.

    Handles both individual indices and inclusive range notation::

        "([1]=2,[3]=2)"   → {0: 2, 2: 2}
        "([1-2]=1)"       → {0: 1, 1: 1}
        "([1-3,5]=2)"     → {0: 2, 1: 2, 2: 2, 4: 2}

    Note that the bracketed group may contain either a single integer or a
    hyphen-separated range. Mixed forms such as ``[1-3,5]`` are also produced by some
    writers.

    Args:
        text: Raw VARSHARELIST value string including its outer parentheses,
              e.g. ``"([1-2]=1,[4]=2)"``.

    Returns:
        Dict mapping 0-based variable index to 1-based source zone number.
    """
    result: dict[int, int] = {}
    # Each match captures one bracketed group and the zone it maps to.
    # Group 1: everything inside [...] — may be "N", "N-M", or "N-M,P,..."
    # Group 2: the zone number after "=".
    for m in re.finditer(r"\[([^\]]+)\]\s*=\s*(\d+)", text):
        zone_1based = int(m.group(2))
        indices_str = m.group(1)
        # Reuse _parse_index_list to handle ranges and comma lists uniformly.
        for idx_0based in _parse_index_list(f"[{indices_str}]"):
            result[idx_0based] = zone_1based
    return result


def _parse_dt(text: str) -> list[DataType]:
    """Parse ``DT=(SINGLE SINGLE ...)`` -> one :class:`DataType` per variable.

    Note:
        Tokens may be separated by whitespace, commas, or both.

    Args:
        text: Raw DT value string including its outer parentheses, e.g. ``"(SINGLE,
              SINGLE, DOUBLE)"``.

    Returns:
        One :class:`DataType` per variable, in dataset variable order.

    Raises:
        ValueError: If a token isn't a recognized Tecplot data type keyword.

    Example:
        >>> _parse_dt("(SINGLE SINGLE DOUBLE)")
        [DataType.FLOAT, DataType.FLOAT, DataType.DOUBLE]
    """
    inner = text.strip()
    if inner.startswith("(") and inner.endswith(")"):
        inner = inner[1:-1]
    tokens = [tok for tok in re.split(r"[,\s]+", inner) if tok]
    result: list[DataType] = []
    for tok in tokens:
        dt = _STR_TO_DATATYPE.get(tok.strip().lower())
        if dt is None:
            raise ValueError(
                f"DT={text!r} contains unrecognized data type {tok!r}; expected one "
                "of SINGLE, DOUBLE, LONGINT, SHORTINT, BYTE."
            )
        result.append(dt)
    return result


def _parse_auxdata_line(line: str) -> tuple[str, str]:
    """Parse ``DATASETAUXDATA name="value"`` → ``(name, value)``.

    Example:
        >>> name, val = _parse_auxdata_line('DATASETAUXDATA Solver="MyCFD"')
    """
    m = re.match(r"(?i)DATASETAUXDATA\s+", line)
    rest = line[m.end() :] if m else line
    eq = rest.find("=")
    if eq < 0:
        return "", ""
    return rest[:eq].strip(), _unquote(rest[eq + 1 :].strip())


def _apply_varauxdata(line: str, var_auxdata_list: list) -> None:
    """Parse ``VARAUXDATA 1-based-idx name="value"`` and store in list.

    Example:
        >>> _apply_varauxdata(line, var_auxdata_list)
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

    Also exposes a small *raw* interface (:meth:`take_raw`, :meth:`position`,
    :meth:`seek`) used only by the vectorized numeric-block readers
    (:meth:`Read._read_float_block_fast`/:meth:`Read._read_int_block_fast`).  Those
    readers bypass per-line comment-stripping for speed and instead detect the rare line
    that *isn't* plain numeric data (a comment, a header keyword) by letting NumPy's
    parser fail on it, then falling back to this class's ordinary stripped-line
    interface. Keeping both interfaces on one object means the fast path can hand back
    an exact resume position for that fallback via :meth:`position`/:meth:`seek`.

    Example:
        >>> buf = _LineBuffer(lines)
    """

    __slots__ = ("_lines", "_peeked", "_peeked_pos", "_pos")

    def __init__(self, lines: list[str]) -> None:
        self._lines: list[str] = lines
        self._pos: int = 0
        # One-line lookahead cache so a peek_stripped() immediately followed by
        # next_stripped() strips comments/whitespace only once instead of twice.
        self._peeked: str | None = None
        self._peeked_pos: int = -1

    def has_more(self) -> bool:
        """Return ``True`` if there are unconsumed lines."""
        return self._peeked is not None or self._pos < len(self._lines)

    def peek_stripped(self) -> str:
        """Return the next non-blank stripped line without consuming it."""
        if self._peeked is not None:
            return self._peeked
        pos = self._pos
        n = len(self._lines)
        while pos < n:
            stripped = _strip_comment(self._lines[pos]).strip()
            pos += 1
            if stripped:
                self._peeked = stripped
                self._peeked_pos = pos
                return stripped
        self._peeked = ""
        self._peeked_pos = pos
        return ""

    def next_stripped(self) -> str:
        """Consume and return the next non-blank stripped line."""
        if self._peeked is None:
            self.peek_stripped()
        value = self._peeked or ""
        self._pos = self._peeked_pos
        self._peeked = None
        self._peeked_pos = -1
        return value

    def take_raw(self, count: int) -> list[str]:
        """Consume and return up to *count* **unprocessed** source lines.

        Unlike :meth:`next_stripped`, this does not strip comments, skip blank lines, or
        strip whitespace -- it is a raw slice of the underlying line list, advancing the
        position by exactly how many lines were returned (fewer than *count* at end of
        file). Any pending :meth:`peek_stripped` lookahead is discarded first, since
        this call always resumes from the true unconsumed position.

        Intended only for the numeric fast-path readers, which feed the result straight
        to a whitespace-tolerant NumPy parser and don't need per-line preprocessing.

        Example:
            >>> lines = buf.take_raw(64)
        """
        self._peeked = None
        self._peeked_pos = -1
        end = min(self._pos + count, len(self._lines))
        out = self._lines[self._pos : end]
        self._pos = end
        return out

    def position(self) -> int:
        """Return an opaque marker for the current position.

        Example:
            >>> marker = buf.position()
        """
        return self._pos

    def seek(self, marker: int) -> None:
        """Restore the position to a marker previously returned by :meth:`position`.

        Example:
            >>> buf.seek(marker)
        """
        self._pos = marker
        self._peeked = None
        self._peeked_pos = -1


# ======================================================================================
# ReadAuxData
# ======================================================================================


class ReadAuxData:
    """Dictionary-like container for Tecplot auxiliary data strings.

    Interface matches :class:`szl.ReadAuxData` exactly.

    Example:
        >>> aux = ReadAuxData({"Solver": "MyCFD"})
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

    def keys(self) -> KeysView[str]:
        """Return iterator over auxiliary data names."""
        return self._data.keys()

    def values(self) -> ValuesView[str]:
        """Return iterator over auxiliary data values."""
        return self._data.values()

    def items(self) -> ItemsView[str, str]:
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

        Example:
            >>> t = aux.as_float("TimeValue", default=0.0)
        """
        try:
            return float(self._data[key])
        except (KeyError, ValueError):
            return default

    def as_bool(self, key: str, default: bool | None = None) -> bool | None:
        """Return value for *key* as :class:`bool`, or *default* on failure.

        Recognises ``"true"``/``"false"``, ``"yes"``/``"no"``,
        ``"1"``/``"0"`` (case-insensitive).

        Example:
            >>> flag = aux.as_bool("IsBoundaryZone", default=False)
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


# ======================================================================================
# ReadVariable
# ======================================================================================


class ReadVariable:
    """Variable metadata and data for one zone parsed from an ASCII DAT file.

    Args:
        zone_index: 1-based index of the zone this variable belongs to.
        var_index:  1-based dataset variable index.
    """

    def __init__(
        self,
        zone_index: int,
        var_index: int,
        name: str,
        data: npt.NDArray | None,
        value_location: ValueLocation = ValueLocation.NODAL,
        is_passive: bool = False,
        shared_zone: int | None = None,
    ) -> None:
        self.zone_index: int = zone_index
        self.var_index: int = var_index
        self._name: str = name
        self._data: npt.NDArray | None = data
        self._value_location: ValueLocation = value_location
        self._is_passive: bool = is_passive
        self._shared_zone: int | None = shared_zone

    def __repr__(self) -> str:
        """Set repr string with only relevant metadata."""
        parts = [repr(self.name)]
        if self.is_passive():
            parts.append("passive")
        elif self.shared_zone is not None:
            parts.append(f"shared(zone={self.shared_zone})")
        else:
            parts.append(f"dtype={self.data_type.name}")
            if self.values is not None:
                parts.append(f"shape={self.values.shape}")
            if self.value_location == ValueLocation.CELL_CENTERED:
                parts.append("CELL_CENTERED")
        return f"ReadVariable({', '.join(parts)})"

    @property
    def name(self) -> str:
        """Return variable name string."""
        return self._name

    @property
    def data_type(self) -> DataType:
        """Return :class:`DataType` inferred from the stored NumPy dtype.

        A shared variable reports the source zone's actual dtype, since its data array
        is resolved from that zone (see :attr:`shared_zone`). Only a passive variable
        (which has no data array anywhere in the file) returns :attr:`DataType.FLOAT` as
        a placeholder.
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
        """Return the data array.

        Returns the source zone's array for a shared variable (see
        :attr:`shared_zone`) exactly as if this zone owned the data itself.
        ``None`` only for a passive variable, which has no data anywhere in
        the file.
        """
        return self._data

    def get_values(
        self,
        value_range: tuple[int | None, int | None] = (None, None),
    ) -> npt.NDArray | None:
        """Return a slice of the data array using a 1-based range.

        Example:
            >>> arr = var.get_values((1, 100))

        Args:
            value_range ((None, None) | (start, end)):
                1-based start (inclusive) and end (exclusive).

        Returns:
            The array (or a slice of it), or ``None`` for a passive variable.
            A shared variable resolves to the source zone's array, per
            :attr:`values`.

        Raises:
            ValueError: If only one of start/end is given.
        """
        if self._data is None:
            return None
        start, end = value_range
        if start is None and end is None:
            return self._data
        if start is None or end is None:
            raise ValueError("Both start and end indices must be specified.")
        return self._data[start - 1 : end - 1]


# ======================================================================================
# ReadZone
# ======================================================================================


class ReadZone:
    """Zone data parsed from a Tecplot ASCII DAT file.

    Attributes:
        zone_index: 1-based dataset zone index.
        datapacking: :class:`~tecio.libtecio.DataPacking` member reflecting the
            ``DATAPACKING`` keyword found in the zone header (``BLOCK`` or
            ``POINT``). The data arrays are identical either way; this attribute
            records how the values were laid out on disk.
        shared_connectivity: 1-based source zone index if this zone's connectivity is
            shared via ``CONNECTIVITYSHAREZONE``, else ``None``.  :attr:`node_map`
            automatically resolves to the source zone's array.

    Example:
        >>> zone = ReadZone(zone_index, title, zone_type, I, J, K, ...)
    """

    def __init__(
        self,
        zone_index: int,
        title: str,
        zone_type: ZoneType,
        I: int,  # noqa E741
        J: int,
        K: int,
        solution_time: float,
        strand_id: int,
        variables: list[ReadVariable],
        auxdata: ReadAuxData,
        node_map: npt.NDArray | None = None,
        datapacking: DataPacking = DataPacking.BLOCK,
        shared_connectivity: int | None = None,
    ) -> None:
        self.zone_index: int = zone_index
        self.title: str = title
        self.zone_type: ZoneType = zone_type
        self.I: int = I
        self.J: int = J
        self.K: int = K
        self.solution_time: float = solution_time
        self.strand_id: int = strand_id
        self._variable: VariableList[ReadVariable] = VariableList(variables)
        self.auxdata: ReadAuxData = auxdata
        self.node_map: npt.NDArray | None = node_map
        self.datapacking: DataPacking = datapacking
        self.shared_connectivity: int | None = shared_connectivity

    def __repr__(self) -> str:
        """Set a more descriptive repr string for reading zones."""
        title = self.title
        if len(title) > 30:
            title = title[:29] + "\u2026"
        if self.zone_type == ZoneType.ORDERED:
            size = f"I={self.I}, J={self.J}, K={self.K}"
        else:
            size = f"N={self.num_nodes}, E={self.num_elements}"
        extra = f", aux={len(self.auxdata)}" if len(self.auxdata) else ""
        return f"ReadZone({title!r}, {self.zone_type.name}, {size}{extra})"

    # -- Properties --------------------------------------------------------------------

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

    @property
    def nodes_per_cell(self) -> int:
        """Nodes per cell based on zone type.

        Uses the module-level ``_NODES_PER_ELEM`` table for simple FE types. For
        ORDERED zones the count is inferred from the number of active dimensions (1-D →
        2, 2-D → 4, 3-D → 8).

        Raises:
            ValueError: For zone types without a fixed nodes-per-cell count.

        """
        zt = self.zone_type
        if zt in _NODES_PER_ELEM:
            return _NODES_PER_ELEM[zt]
        if zt == ZoneType.ORDERED:
            dims = sum(1 for x in (self.I, self.J, self.K) if x > 1)
            return 2**dims
        raise ValueError(f"ZoneType {zt} does not have a fixed nodes-per-cell count.")

    @property
    def variable(self) -> VariableList[ReadVariable]:
        """Variables in this zone, by 0-based index or exact name."""
        return self._variable

    # -- Methods -----------------------------------------------------------------------

    @overload
    def get_array(self, key: int | str) -> npt.NDArray | None: ...
    @overload
    def get_array(self, key: list[str]) -> tuple[npt.NDArray | None, ...]: ...

    def get_array(
        self, key: int | str | list[str]
    ) -> npt.NDArray | None | tuple[npt.NDArray | None, ...]:
        """Return variable data array(s) for this zone.

        A single key (0-based index or exact name) returns one array. A list of exact
        names returns a tuple of arrays in the order given, suitable for unpacking::

            p = zone.get_array("p")
            x, y, z = zone.get_array(["x", "y", "z"])

        Returns:
            Array correcsponding to scalar key (or ``None`` only if the variable is
            passive); a tuple of such arrays for a list of names. A single-element list
            yields a 1-tuple, not a bare array. A shared variable resolves to its
            source zone's array, per :attr:`ReadVariable.values`.

        Raises:
            KeyError:   If a name does not exist.
            IndexError: If an index is out of range.

        """
        return select_variable_arrays(self.variable, key)

    def is_enabled(self) -> bool:
        """Always ``True`` for zones successfully parsed from a file."""
        return True


# ======================================================================================
# Public Read API
# ======================================================================================


class Read:
    """Read a Tecplot ASCII DAT file into memory.

    The entire file is parsed on construction. All data is then available through the
    same attributes and methods as :class:`szl.Read`.

    Example:
        >>> dat = Read("Onera.dat")
        >>> dat = tecio.open("Onera.dat", "r")

    Args:
        path (str):
            Path to the ``.dat`` file.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: On unsupported format features.

    """

    def __init__(self, path: str) -> None:
        self._path: str = str(path)
        self._title: str = ""
        self._file_type: FileType = FileType.FULL
        self._variable_names: list[str] = []
        self._zones: list[ReadZone] = []
        self._zone_list: ZoneList[ReadZone] | None = None
        self._auxdata: ReadAuxData = ReadAuxData()
        # Index 0 is a None placeholder so that 1-based indexing works directly
        self._var_auxdata: list[ReadAuxData | None] = [None]
        # Raw VARAUXDATA lines seen before the first zone, buffered by
        # _parse_file_header() and applied once _var_auxdata is allocated with the
        # correct length (num_vars isn't known until the header finishes parsing)
        self._deferred_var_aux_lines: list[str] = []
        self._parse()

    def __repr__(self) -> str:
        """Set a nice repr string for the read object."""
        cls = type(self).__name__
        name = self._path.replace("\\", "/").rsplit("/", 1)[-1]
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

    def __enter__(self) -> Read:
        """Context manager for Read class."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager protocol for read-only file interfaces.

        Read classes hold no open file handle between accesses, so there is nothing to
        release on exit. Provided for API consistency with the Write classes and to
        support the ``with tecio.open(...) as r:`` pattern.
        """

    # -- Properties --------------------------------------------------------------------

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
    def zone(self) -> ZoneList[ReadZone]:
        """Zones in this file, by index or slice."""
        if self._zone_list is None:
            self._zone_list = ZoneList(self._zones)
        return self._zone_list

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
        """Return auxiliary data for variable ``var_index`` (1-based).

        Example:
            >>> aux = dat.get_var_auxdata(1)

        Raises:
            IndexError: If ``var_index`` is out of range.
        """
        if var_index < 1 or var_index > self.num_vars:
            raise IndexError(
                f"Variable index {var_index} out of range [1, {self.num_vars}]"
            )
        result = self._var_auxdata[var_index]
        assert result is not None
        return result

    def get_zone_auxdata(self, zone_index: int) -> ReadAuxData:
        """Return auxiliary data for zone ``zone_index`` (1-based).

        Example:
            >>> aux = dat.get_zone_auxdata(1)

        Raises:
            IndexError: If ``zone_index`` is out of range.
        """
        if zone_index < 1 or zone_index > self.num_zones:
            raise IndexError(
                f"Zone index {zone_index} out of range [1, {self.num_zones}]"
            )
        return self._zones[zone_index - 1].auxdata

    # -- parser ---------------------------------------------------------------

    def _parse(self) -> None:
        """Read and parse the entire DAT file.

        Example:
            >>> self._parse()
        """
        with open(self._path, encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()

        tokens = _LineBuffer(lines)
        self._parse_file_header(tokens)

        # Build per-variable aux data slots now that num_vars is known.
        self._var_auxdata = [None] + [ReadAuxData() for _ in range(self.num_vars)]

        # Now that _var_auxdata exists, apply any VARAUXDATA lines that appeared before
        # the first zone
        for raw in self._deferred_var_aux_lines:
            _apply_varauxdata(raw, self._var_auxdata)
        self._deferred_var_aux_lines.clear()

        while tokens.has_more():
            line = tokens.peek_stripped()
            upper = line.upper()
            # if upper.startswith("ZONE"):
            if upper.lstrip().split("=")[0].split()[0] == "ZONE":
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

        Example:
            >>> self._parse_file_header(tokens)
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

            elif upper_key.startswith("VARAUXDATA"):
                # Can't process this without num_vars, and therefore _var_auxdata, isn't
                # known until this header finishes parsing. Buffer the raw line;
                # _parse() applies these once _var_auxdata is allocated.
                self._deferred_var_aux_lines.append(line)

    def _parse_zone(self, tokens: _LineBuffer) -> None:
        """Parse one ZONE block (header + data blocks + connectivity).

        Example:
            >>> self._parse_zone(tokens)
        """
        # -- Collect header lines ------------------------------------------------------

        header_lines: list[str] = [tokens.next_stripped()]  # ZONE T=...

        while tokens.has_more():
            nxt = tokens.peek_stripped()
            if not nxt:
                tokens.next_stripped()
                continue
            nxt_upper = nxt.lstrip().upper()
            # Stop at a new zone, top-level keyword, or the first data line.
            if nxt_upper.split("=")[0].split()[0] == "ZONE":
                break
            if nxt_upper.startswith(("DATASETAUXDATA", "VARAUXDATA")):
                break
            first_ch = nxt.lstrip()[0] if nxt.lstrip() else ""
            # A data line begins with a numeric token (covers leading-dot values
            # such as ``.5`` and signed values such as ``-1.2e3``).
            if first_ch in "0123456789+-.":
                break
            header_lines.append(tokens.next_stripped())

        header_text = " ".join(header_lines)

        # Strip the leading ZONE keyword from the header text.
        m_zone = re.match(r"(?i)^ZONE\s*", header_text)
        if m_zone:
            header_text = header_text[m_zone.end() :]

        # -- Parse header key=value pairs ----------------------------------------------

        kv = _kv_split(header_text)

        zone_title = _unquote(kv.get("T", ""))
        strand_id = int(kv.get("STRANDID", "0") or "0")
        solution_time = float(kv.get("SOLUTIONTIME", "0.0") or "0.0")

        # -- Determine zone type and data packing --------------------------------------
        #
        # Two header dialects are supported:
        #   * Modern:  ZONETYPE=FEQuadrilateral, DATAPACKING=POINT
        #   * Legacy:  F=FEPOINT, ET=QUADRILATERAL
        #
        # The FE-vs-ordered distinction comes from ``ZONETYPE`` (modern) or ``F``
        # (legacy).  ``ET`` (element type) only ever appears on finite-element zones —
        # ordered/structured data has no elements — so it merely names the element
        # *shape* once a zone is already known to be FE, and is ignored on ordered
        # zones. Modern keywords win when present.
        legacy_is_fe: bool | None = None
        legacy_packing: DataPacking | None = None
        if "F" in kv:
            legacy_is_fe, legacy_packing = _parse_legacy_format(kv["F"])

        if "ZONETYPE" in kv:
            zt_raw = kv["ZONETYPE"].rstrip(",").strip().lower()
            zone_type = _STR_TO_ZONETYPE.get(zt_raw, ZoneType.ORDERED)
        elif legacy_is_fe is False:
            # F=POINT/BLOCK: ordered zone. A stray ET (if any) does not apply.
            zone_type = ZoneType.ORDERED
        elif legacy_is_fe or "ET" in kv:
            # Finite-element zone: F=FEPOINT/FEBLOCK, or an ET keyword with no F
            # (some exporters omit F). The element shape comes from ET, which is
            # then required.
            if "ET" not in kv:
                raise ValueError(
                    "Legacy FE zone header specifies F=FEPOINT/FEBLOCK but is "
                    "missing the required ET (element type) keyword."
                )
            zone_type = _parse_legacy_element_type(kv["ET"])
        else:
            zone_type = ZoneType.ORDERED

        if zone_type in _FE_POLY:
            raise ValueError(
                f"Zone type {zone_type.name!r} is not yet supported by the "
                "ASCII reader."
            )

        if zone_type == ZoneType.ORDERED:
            I = int(kv.get("I", "1") or "1")  # noqa E741
            J = int(kv.get("J", "1") or "1")
            K = int(kv.get("K", "1") or "1")
            num_nodes = I * J * K
            num_cells = max(I - 1, 1) * max(J - 1, 1) * max(K - 1, 1)
        else:
            # FE zones accept both the modern (NODES/ELEMENTS) and legacy (N/E)
            # spellings for the node and element counts.
            num_nodes = int(kv.get("NODES", kv.get("N", "0")) or "0")
            num_cells = int(kv.get("ELEMENTS", kv.get("E", "0")) or "0")
            I, J, K = num_nodes, num_cells, 0  # noqa E741

        # Packing: DATAPACKING wins, then legacy F, else BLOCK (Tecplot default).
        if "DATAPACKING" in kv:
            packing_raw = kv["DATAPACKING"].strip().lower()
            packing = _STR_TO_DATAPACKING.get(packing_raw)
            if packing is None:
                raise ValueError(
                    f"DATAPACKING={packing_raw!r} is not supported; "
                    "only BLOCK and POINT are implemented in the ASCII reader."
                )
        elif legacy_packing is not None:
            packing = legacy_packing
        else:
            packing = DataPacking.BLOCK

        # Variable locations (0-based index → ValueLocation)
        var_locs: dict[int, ValueLocation] = {}
        if "VARLOCATION" in kv:
            var_locs = _parse_varlocation(kv["VARLOCATION"])

        # Per-variable data types. Tecplot default when DT= is omitted is SINGLE for
        # every variable.
        if "DT" in kv:
            var_types = _parse_dt(kv["DT"])
            if len(var_types) != len(self._variable_names):
                raise ValueError(
                    f"DT={kv['DT']!r} declares {len(var_types)} data types, but "
                    f"the dataset has {len(self._variable_names)} variables."
                )
        else:
            var_types = [DataType.FLOAT] * len(self._variable_names)

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

        # -- Read variable data blocks -------------------------------------------------

        if packing == DataPacking.POINT:
            var_arrays = self._read_point_var_data(
                tokens,
                self.num_vars,
                num_nodes,
                num_cells,
                var_locs,
                passive_set,
                share_map,
                var_types,
            )
        else:
            var_arrays = self._read_block_var_data(
                tokens,
                self.num_vars,
                num_nodes,
                num_cells,
                var_locs,
                passive_set,
                share_map,
                var_types,
            )

        # -- Read connectivity (FE zones only) -----------------------------------------

        node_map: npt.NDArray | None = None

        if zone_type != ZoneType.ORDERED:
            if con_share_zone and self._zones:
                node_map = self._zones[con_share_zone - 1].node_map
            else:
                nodes_per_cell = _NODES_PER_ELEM[zone_type]
                flat = self._read_int_block(tokens, num_cells * nodes_per_cell)
                node_map = flat.reshape(num_cells, nodes_per_cell)

        # -- Build ReadVariable and ReadZone objects -----------------------------------

        # For ordered zones, reshape each variable array from flat 1-D to (I, J, K) for
        # nodal variables or (I-1, J-1, K-1) for cell-centered so that zone dimensions
        # can be inferred from array shape downstream.
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

        # Resolve variable data shared from an earlier zone (VARSHARELIST). Sharing is
        # only valid against a zone that has already been parsed (DAT files reference
        # zones by position, forward-only); an out-of-range reference is left as
        # ``None`` rather than raising, since malformed input shouldn't crash a read
        # that's otherwise recoverable.
        for var_idx, src_zone_1based in share_map.items():
            if 1 <= src_zone_1based <= len(self._zones):
                var_arrays[var_idx] = (
                    self._zones[src_zone_1based - 1].variable[var_idx].values
                )

        # This zone's own 1-based index
        zone_index = len(self._zones) + 1

        read_vars = [
            ReadVariable(
                zone_index=zone_index,
                var_index=idx + 1,
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
                zone_index=zone_index,
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
                datapacking=packing,
                shared_connectivity=con_share_zone if con_share_zone else None,
            )
        )

    # -- Block readers -------------------------------------------------------

    @staticmethod
    def _read_float_block(tokens: _LineBuffer, n_values: int) -> npt.NDArray:
        """Read *n_values* floats from *tokens* into an intermediate float64 array.

        This is a pure text-parsing primitive. Callers cast the result to each
        variable's real dtype afterward, once individual per-variable arrays have been
        split out.

        Dispatches to :meth:`_read_float_block_fast`, a vectorized NumPy parser that
        handles the overwhelming majority of (well-formed) input in a handful of C-level
        calls regardless of block size, and falls back to :meth:`_read_float_block_slow`
        -- a tolerant, token-by-token parser -- only if the fast path can't make sense
        of what it read (a stray comment or non-numeric token inside the block, or a
        boundary reached before *n_values* values were collected). The fallback re-reads
        from *tokens* at the position the fast path started from, so the two never
        disagree about what was consumed.

        See Also:
            :meth:`_read_block_var_data`/:meth:`_read_point_var_data`.

        Examples:
            >>> arr = Read._read_float_block(tokens, 100)
        """
        if n_values <= 0:
            return np.empty(0, dtype=np.float64)

        marker = tokens.position()
        try:
            arr = Read._read_float_block_fast(tokens, n_values)
        except ValueError:
            arr = None
        if arr is not None:
            return arr

        tokens.seek(marker)
        return Read._read_float_block_slow(tokens, n_values)

    @staticmethod
    def _read_float_block_fast(
        tokens: _LineBuffer, n_values: int
    ) -> npt.NDArray | None:
        """Vectorized fast path for :meth:`_read_float_block`.

        Reads raw (unprocessed) lines in a handful of growing bulk grabs (sized from a
        one-line width probe, so a multi-million-value block costs a small constant
        number of Python-level calls). Passing ``sep`` here is the (fully supported)
        text-parsing mode of that function, not its deprecated binary-decoding mode.

        Returns ``None`` (never a short array) if a block boundary is hit before
        *n_values* values are collected, so the caller can unambiguously fall back to
        the slow, boundary-aware parser. A ``ValueError`` from ``numpy.fromstring`` is
        allowed to propagate to the caller for the same reason.

        Example:
            >>> arr = Read._read_float_block_fast(tokens, 100)

        """
        pieces: list[npt.NDArray] = []
        have = 0
        per_line: int | None = None
        for _ in range(_MAX_FAST_BLOCK_ITERATIONS):
            remaining = n_values - have
            if remaining <= 0:
                break
            # First grab is a single-line probe to measure the block's (typically
            # constant) values-per-line width
            n_lines = 1 if per_line is None else -(-remaining // per_line)
            batch = tokens.take_raw(n_lines)
            if not batch:
                return None  # ran out of input before reaching n_values
            piece = np.fromstring("".join(batch), dtype=np.float64, sep=" ")
            if per_line is None:
                per_line = max(piece.size, 1)
            pieces.append(piece)
            have += piece.size
        else:
            return None  # didn't converge; let the slow path sort it out
        arr = pieces[0] if len(pieces) == 1 else np.concatenate(pieces)
        return arr[:n_values] if arr.size >= n_values else None

    @staticmethod
    def _read_float_block_slow(tokens: _LineBuffer, n_values: int) -> npt.NDArray:
        """Tolerant, token-by-token fallback for :meth:`_read_float_block`.

        Reads one stripped line at a time, converting tokens one at a time and silently
        skipping any that aren't valid floats, stopping at a block boundary (see
        :func:`_is_float_block_boundary`) even if *n_values* hasn't been reached
        yet. Slow by design, but guarantees a result either way.

        Example:
            >>> arr = Read._read_float_block_slow(tokens, 100)

        """
        values: list[float] = []
        while len(values) < n_values and tokens.has_more():
            nxt = tokens.peek_stripped()
            if _is_float_block_boundary(nxt):
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
    def _read_block_var_data(
        tokens: _LineBuffer,
        num_vars: int,
        num_nodes: int,
        num_cells: int,
        var_locs: dict[int, ValueLocation],
        passive_set: set[int],
        share_map: dict[int, int],
        var_types: list[DataType],
    ) -> list[npt.NDArray | None]:
        """Read ``DATAPACKING=BLOCK`` variable data for one zone.

        One contiguous block of values is read per active variable, in dataset variable
        order. Each block is parsed as text into an intermediate ``float64`` array
        (``_read_float_block`` doesn't know or care about per-variable types), then cast
        to that variable's actual declared dtype (from ``DT=``, or the spec's documented
        SINGLE default) right here, where each variable's array is finalized.

        Reading one variable at a time (rather than all of them in a single combined
        call) is deliberate: :meth:`_read_float_block`'s vectorized fast path already
        costs only a handful of C-level calls per block regardless of size.

        Note:
            Passive and shared variables contribute a ``None`` placeholder to the
            returned list. The caller resolves shared variables against the zones parsed
            so far afterward, since this function only reads bytes off disk and has no
            zone list to resolve against.

        Example:
            >>> arrays = Read._read_block_var_data(
            ...     tokens, 3, 100, 80, {}, set(), {}, [DataType.FLOAT] * 3
            ... )

        """
        var_arrays: list[npt.NDArray | None] = [None] * num_vars
        for var_idx in range(num_vars):
            if var_idx in passive_set or var_idx in share_map:
                continue  # no data block for passive/shared variables
            loc = var_locs.get(var_idx, ValueLocation.NODAL)
            n_vals = num_cells if loc == ValueLocation.CELL_CENTERED else num_nodes
            raw = Read._read_float_block(tokens, n_vals)
            var_arrays[var_idx] = raw.astype(_DT_TO_DTYPE[var_types[var_idx]])
        return var_arrays

    @staticmethod
    def _read_point_var_data(
        tokens: _LineBuffer,
        num_vars: int,
        num_nodes: int,
        num_cells: int,
        var_locs: dict[int, ValueLocation],
        passive_set: set[int],
        share_map: dict[int, int],
        var_types: list[DataType],
    ) -> list[npt.NDArray | None]:
        """Read ``DATAPACKING=POINT`` variable data for one zone.

        The spec writes two interleaved sections:

            * **Nodal section** — ``num_nodes`` rows, one per node. Each row contains
              the values of all active nodal variables in dataset order.
            * **Cell-centered section** — ``num_cells`` rows, one per element. Each row
              contains the values of all active CC variables in dataset order.

        When all variables are nodal the cell-centered section is empty and is skipped
        automatically. Passive and shared variables are excluded from both sections and
        contribute a ``None`` placeholder to the returned list. Each column is cast to
        its variable's actual declared dtype (from ``DT=``, or the spec's documented
        SINGLE default) right where it's extracted from the interleaved rows.

        See Also:
            :meth:`_read_block_var_data`

        Example:
            >>> arrays = Read._read_point_var_data(
            ...     tokens, 3, 100, 80, {}, set(), {}, [DataType.FLOAT] * 3
            ... )

        """
        # Active variable indices, preserving dataset order.
        nodal_active: list[int] = [
            i
            for i in range(num_vars)
            if i not in passive_set
            and i not in share_map
            and var_locs.get(i, ValueLocation.NODAL) == ValueLocation.NODAL
        ]
        cc_active: list[int] = [
            i
            for i in range(num_vars)
            if i not in passive_set
            and i not in share_map
            and var_locs.get(i, ValueLocation.NODAL) == ValueLocation.CELL_CENTERED
        ]

        var_arrays: list[npt.NDArray | None] = [None] * num_vars

        # Nodal section: num_nodes rows × len(nodal_active) columns.
        n_nodal = len(nodal_active)
        if n_nodal > 0 and num_nodes > 0:
            flat = Read._read_float_block(tokens, num_nodes * n_nodal)
            # Row-major: flat[node * n_nodal + col] = value for that variable.
            matrix = flat.reshape(num_nodes, n_nodal)
            for col, var_idx in enumerate(nodal_active):
                var_arrays[var_idx] = np.ascontiguousarray(
                    matrix[:, col], dtype=_DT_TO_DTYPE[var_types[var_idx]]
                )

        # Cell-centred section: num_cells rows × len(cc_active) columns.
        n_cc = len(cc_active)
        if n_cc > 0 and num_cells > 0:
            flat = Read._read_float_block(tokens, num_cells * n_cc)
            matrix = flat.reshape(num_cells, n_cc)
            for col, var_idx in enumerate(cc_active):
                var_arrays[var_idx] = np.ascontiguousarray(
                    matrix[:, col], dtype=_DT_TO_DTYPE[var_types[var_idx]]
                )

        return var_arrays

    @staticmethod
    def _read_int_block(tokens: _LineBuffer, n_values: int) -> npt.NDArray:
        """Read exactly *n_values* integers from *tokens* into an int64 array.

        Used for connectivity (node-map) data, which can run to tens of millions of
        entries for large volume meshes. Follows the same vectorized-fast-path /
        tolerant-slow-path split as :meth:`_read_float_block`; see that method's
        docstring for the rationale.

        Examples:
            >>> arr = Read._read_int_block(tokens, 24)
        """
        if n_values <= 0:
            return np.empty(0, dtype=np.int64)

        marker = tokens.position()
        try:
            arr = Read._read_int_block_fast(tokens, n_values)
        except ValueError:
            arr = None
        if arr is not None:
            return arr

        tokens.seek(marker)
        return Read._read_int_block_slow(tokens, n_values)

    @staticmethod
    def _read_int_block_fast(tokens: _LineBuffer, n_values: int) -> npt.NDArray | None:
        """Vectorized fast path for :meth:`_read_int_block`.

        See :meth:`_read_float_block_fast`, which this mirrors exactly except for the
        target dtype.

        Example:
            >>> arr = Read._read_int_block_fast(tokens, 24)
        """
        pieces: list[npt.NDArray] = []
        have = 0
        per_line: int | None = None
        for _ in range(_MAX_FAST_BLOCK_ITERATIONS):
            remaining = n_values - have
            if remaining <= 0:
                break
            n_lines = 1 if per_line is None else -(-remaining // per_line)
            batch = tokens.take_raw(n_lines)
            if not batch:
                return None
            piece = np.fromstring("".join(batch), dtype=np.int64, sep=" ")
            if per_line is None:
                per_line = max(piece.size, 1)
            pieces.append(piece)
            have += piece.size
        else:
            return None
        arr = pieces[0] if len(pieces) == 1 else np.concatenate(pieces)
        return arr[:n_values] if arr.size >= n_values else None

    @staticmethod
    def _read_int_block_slow(tokens: _LineBuffer, n_values: int) -> npt.NDArray:
        """Tolerant, token-by-token fallback for :meth:`_read_int_block`.

        Example:
            >>> arr = Read._read_int_block_slow(tokens, 24)
        """
        values: list[int] = []
        while len(values) < n_values and tokens.has_more():
            nxt = tokens.peek_stripped()
            if _is_int_block_boundary(nxt):
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
