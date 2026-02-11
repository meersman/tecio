"""Top-level package for tecio.

Exports common submodules and provides package version.
"""

from importlib import metadata

try:
    __version__ = metadata.version("tecio")
except metadata.PackageNotFoundError:
    __version__ = "0.0.0"

from . import libtecio, pltio, tecutils
from .szlfile import Read, ReadAuxData, ReadVariable, ReadZone, Write, WriteZone
from .tecdata import TecData, TecVariable, TecZone

__all__ = [
    "TecData",
    "TecZone",
    "TecVariable",
    "Read",
    "ReadZone",
    "ReadVariable",
    "ReadAuxData",
    "Write",
    "WriteZone",
    "pltio",
    "libtecio",
    "tecutils",
    "__version__",
]
