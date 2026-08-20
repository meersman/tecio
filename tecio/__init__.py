"""Python interface for reading and writing Tecplot data files.

Wraps the TecIO C library for ``.szplt``, ``.plt``, and ``.dat`` formats.
Requires Python 3.10+, NumPy, and a Tecplot 360 installation.
"""

from importlib import metadata

try:
    __version__ = metadata.version("tecio")
except metadata.PackageNotFoundError:
    __version__ = "0.2.3"

from . import cli, libtecio
from ._constants import (
    Boolean,
    DataPacking,
    DataType,
    Debug,
    FaceNeighborMode,
    FeCellShape,
    FileFormat,
    FileType,
    ValueLocation,
    VarStatus,
    ZoneType,
)
from ._containers import VariableList, ZoneList
from ._dat_read import TecplotDatReader
from ._dat_write import TecplotDatWriter
from ._io import AppendReadWrite, AppendWrite, open
from ._plt_read import TecplotPltReader
from ._plt_write import TecplotPltWriter
from ._reader import (
    TecplotAuxDataReader,
    TecplotFEZoneReader,
    TecplotOrderedZoneReader,
    TecplotReader,
    TecplotVariableReader,
    TecplotZoneReader,
)
from ._szl_read import TecplotSzlReader
from ._szl_write import TecplotSzlWriter
from ._writer import TecplotWriter

# Ensure these display as their canonical public name in docs and help(),
# rather than the private module they're actually defined in.
open.__module__ = "tecio"
for _cls in (
    AppendWrite,
    AppendReadWrite,
    ZoneList,
    VariableList,
    TecplotReader,
    TecplotZoneReader,
    TecplotOrderedZoneReader,
    TecplotFEZoneReader,
    TecplotVariableReader,
    TecplotAuxDataReader,
    TecplotSzlReader,
    TecplotPltReader,
    TecplotDatReader,
    TecplotWriter,
    TecplotSzlWriter,
    TecplotPltWriter,
    TecplotDatWriter,
    Boolean,
    DataPacking,
    DataType,
    Debug,
    FaceNeighborMode,
    FeCellShape,
    FileFormat,
    FileType,
    ValueLocation,
    VarStatus,
    ZoneType,
):
    _cls.__module__ = "tecio"
del _cls

__all__ = [
    "libtecio",
    "open",
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
    "TecplotWriter",
    "TecplotSzlWriter",
    "TecplotPltWriter",
    "TecplotDatWriter",
    "Boolean",
    "DataPacking",
    "DataType",
    "Debug",
    "FaceNeighborMode",
    "FeCellShape",
    "FileFormat",
    "FileType",
    "ValueLocation",
    "VarStatus",
    "ZoneType",
    "__version__",
]

# Only public API visible to user
def __dir__() -> list:
    return sorted(__all__)
