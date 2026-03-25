"""Top-level package for tecio.

Exports common submodules and provides package version.
"""

from importlib import metadata

try:
    __version__ = metadata.version("tecio")
except metadata.PackageNotFoundError:
    __version__ = "0.0.0"

from . import libtecio, tecutils
from .szl import Read, ReadAuxData, ReadVariable, ReadZone, Write, write_data

__all__ = [
    "Read",
    "ReadZone",
    "ReadVariable",
    "ReadAuxData",
    "Write",
    "write_data",
    "libtecio",
    "szl",
    "tecutils",
    "__version__",
]
