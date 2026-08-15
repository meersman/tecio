"""Python interface for reading and writing Tecplot data files.

Wraps the TecIO C library for ``.szplt``, ``.plt``, and ``.dat`` formats.
Requires Python 3.10+, NumPy, and a Tecplot 360 installation.
"""

from importlib import metadata

try:
    __version__ = metadata.version("tecio")
except metadata.PackageNotFoundError:
    __version__ = "0.2.3"

from . import cli, dat, libtecio, plt, szl
from ._containers import VariableList, ZoneList
from ._dat_read import TecplotDatReader
from ._io import AppendReadWrite, AppendWrite, open
from ._plt_read import TecplotPltReader
from ._reader import (
    TecplotAuxDataReader,
    TecplotFEZoneReader,
    TecplotOrderedZoneReader,
    TecplotReader,
    TecplotVariableReader,
    TecplotZoneReader,
)
from ._szl_read import TecplotSzlReader

# Ensure these display as their canonical public name in docs and help(),
# rather than the private module they're actually defined in.
open.__module__ = "tecio"
AppendWrite.__module__ = "tecio"
AppendReadWrite.__module__ = "tecio"
ZoneList.__module__ = "tecio"
VariableList.__module__ = "tecio"
for _cls in (
    TecplotReader,
    TecplotZoneReader,
    TecplotOrderedZoneReader,
    TecplotFEZoneReader,
    TecplotVariableReader,
    TecplotAuxDataReader,
    TecplotSzlReader,
    TecplotPltReader,
    TecplotDatReader,
):
    _cls.__module__ = "tecio"
del _cls

__all__ = [
    "libtecio",
    "open",
    "dat",
    "plt",
    "szl",
    "cli",
    "AppendWrite",
    "AppendReadWrite",
    "ZoneList",
    "VariableList",
    "TecplotReader",
    "TecplotZoneReader",
    "TecplotOrderedZoneReader",
    "TecplotFEZoneReader",
    "TecplotVariableReader",
    "TecplotAuxDataReader",
    "TecplotSzlReader",
    "TecplotPltReader",
    "TecplotDatReader",
    "__version__",
]
