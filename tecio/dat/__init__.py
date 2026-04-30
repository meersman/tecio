"""Higher level API for reading and writing DAT files."""

from ._read import Read
from ._write import Write

__all__ = [
    "Read",
    "Write",
]
