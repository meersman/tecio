"""Top-level package for tecio.

Exports common submodules and provides package version.
"""
from importlib import metadata

try:
    __version__ = metadata.version("tecio")
except metadata.PackageNotFoundError:
    __version__ = "0.0.0"

from .tecdata import TecData, TecZone, TecVariable
from .szlfile import Read, ReadZone, ReadVariable, ReadAuxData, Write, WriteZone
from . import pltio, szlio, tecutils

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
    "szlio",
    "tecutils",
    "__version__",
]
