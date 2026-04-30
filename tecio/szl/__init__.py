"""Higher level API for reading and writing SZPLT files."""

from ._read import Read
from ._write import Write

__all__ = [
    "Read",
    "Write",
]
