"""Top-level package for tecio.

Exports common submodules and provides package version.
"""

from importlib import metadata

try:
    __version__ = metadata.version("tecio")
except metadata.PackageNotFoundError:
    __version__ = "0.0.0"

from . import libtecio, dat, plt, szl, tecutils
from .io import open

__all__ = [
    "libtecio",
    "open",
    "dat",
    "plt",
    "szl",
    "tecutils",
    "__version__",
]
