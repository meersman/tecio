"""Entry point for tecio such that IO capability can be called from a single function.

Notes:
- Could add all modes from builtin open
  - "x" open for exclusive editing -> fail if already exists
  - "a" open for writing, appending to the end of file if it exists
  - "a+" open for reading and append to the end of file. Combine reading/writing
    attributes in output object
  - "A" append to beginning of file (after header) -- idk what is the point of this?

"""

from __future__ import annotations

from pathlib import Path

from . import plt, szl

_HANDLERS = {
    # New SZPLT API
     ".szplt": {
        "r": szl.Read,
        "w": szl.Write,
        "a": None,  # lambda path, **kw: _append_szl(path, **kw)
    },
    # Classic PLT API
    ".plt": {
        "r": plt.Read,
        "w": plt.Write,
        "a": None,  # lambda path, **kw: _append_plt(path, **kw)
    },
    ".bin": {
        "r": plt.Read,
        "w": plt.Write,
        "a": None,  # lambda path, **kw: _append_plt(path, **kw)
    },
    # Ascii format
    ".dat": {
        "r": None,
        "w": None,
        "a": None,  # lambda path, **kw: _append_dat(path, **kw)
    },
    ".tec": {
        "r": None,
        "w": None,
        "a": None,  # lambda path, **kw: _append_dat(path, **kw)
    },
}


def _append_szl(path, **kwargs):
    """Placeholder for future append functionality."""
    raise NotImplementedError("Append functions not implemented yet.")


def _append_plt(path, **kwargs):
    """Placeholder for future append functionality."""
    raise NotImplementedError("Append functions not implemented yet.")


def _append_dat(path, **kwargs):
    """Placeholder for future append functionality."""
    raise NotImplementedError("Append functions not implemented yet.")


def open(path: str, mode: str = "r", **kwargs) -> szl.Read | szl.Write | plt.Write:
    """Open a Tecplot file for reading or writing.

    Args:
        path:        File path.
        mode:        'r' for read, 'w' for write.
        **kwargs:    Passed to writer constructors.

    """
    ext = Path(path).suffix.lower()

    try:
        mode_map = _HANDLERS[ext]
    except KeyError as exc:
        raise ValueError(f"Unsupported file extension: {ext}") from exc

    if mode not in mode_map:
        raise ValueError(f"Mode '{mode}' not supported for {ext}")

    handler = mode_map[mode]

    if handler is None:
        raise NotImplementedError(
            f"Mode '{mode}' for '{ext}' files is not implemented yet"
        )

    return handler(path, **kwargs)
