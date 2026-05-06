"""Python interface for reading and writing Tecplot data files.

Wraps the TecIO C library for ``.szplt``, ``.plt``, and ``.dat`` formats.
Requires Python 3.10+, NumPy, and a Tecplot 360 installation.
"""

from importlib import metadata

try:
    __version__ = metadata.version("tecio")
except metadata.PackageNotFoundError:
    __version__ = "0.0.0"

from . import cli, dat, libtecio, plt, szl, tecutils
from ._io import open

__all__ = [
    "libtecio",
    "open",
    "dat",
    "plt",
    "pltread",
    "szl",
    "tecutils",
    "cli",
    "__version__",
]
