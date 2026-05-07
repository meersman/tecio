"""Read and write Tecplot ASCII DAT (``.dat`` / ``.tec``) files.

Tecplot ASCII DAT files are human-readable and portable across platforms.
They are slower to read and write than binary formats and produce larger
files, but are useful for small datasets, debugging, and interoperability
with other tools.

Key behaviours:

- The reader parses the entire file into memory on construction.
- Only ``DATAPACKING=BLOCK`` is supported for reading; ``POINT`` packing
  raises :exc:`ValueError`.
- The writer outputs floating-point data in scientific notation with a
  configurable number of significant digits (default 9; pass
  ``sig_digits=17`` for full ``float64`` round-trip fidelity).
- FEPOLYGON and FEPOLYHEDRON zone types are not supported.
- Dataset-level and variable-level auxiliary data are written via
  :meth:`~tecio.dat.Write.add_auxdataset_dict` before the first zone.

Example:
    >>> with tecio.open("out.dat", "w") as w:
    ...     w.write_ijk_zone(
    ...         data=[x, y, p],
    ...         variables=["x", "y", "pressure"],
    ...     )
"""

from ._read import Read
from ._write import Write

# Set canonical public module paths so docs and help() show tecio.dat.Read,
# not tecio.dat._read.Read.
Read.__module__ = "tecio.dat"
Write.__module__ = "tecio.dat"

__all__ = [
    "Read",
    "Write",
]
