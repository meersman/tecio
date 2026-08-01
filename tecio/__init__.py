"""Python interface for reading and writing Tecplot data files.

Wraps the TecIO C library for ``.szplt``, ``.plt``, and ``.dat`` formats.
Requires Python 3.10+, NumPy, and a Tecplot 360 installation.
"""
# ruff: noqa: I001

from importlib import metadata

try:
    __version__ = metadata.version("tecio")
except metadata.PackageNotFoundError:
    __version__ = "0.0.0"

from . import cli, dat, libtecio, plt, szl
from ._containers import VariableList, ZoneList
from ._io import AppendReadWrite, AppendWrite, open
from ._dataset import Dataset
from ._variable import Variable
from ._zone import AuxData, Zone

# Ensure tecio.open displays as the canonical public name in docs and help().
open.__module__ = "tecio"
AppendWrite.__module__ = "tecio"
AppendReadWrite.__module__ = "tecio"
Dataset.__module__ = "tecio"
Zone.__module__ = "tecio"
Variable.__module__ = "tecio"
AuxData.__module__ = "tecio"
ZoneList.__module__ = "tecio"
VariableList.__module__ = "tecio"

__all__ = [
    "libtecio",
    "open",
    "dat",
    "plt",
    "szl",
    "cli",
    "AppendWrite",
    "AppendReadWrite",
    "Dataset",
    "Zone",
    "Variable",
    "AuxData",
    "ZoneList",
    "VariableList",
    "__version__",
]
