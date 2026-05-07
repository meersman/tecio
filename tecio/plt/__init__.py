"""Read and write Tecplot PLT (``.plt``) binary files.

Tecplot PLT is the classic Tecplot binary format, supported by both
v112 and v191 file versions.

Key behaviours:

- Zone metadata is parsed eagerly at open time; variable data is read
  lazily from disk on first access.
- The writer uses the classic TecIO API, which maintains a single
  implicit global file context — only one PLT file may be open for
  writing at a time.
- Zone data must be written in strict order: zone header, then variable
  data, then connectivity before starting the next zone.
- FEPOLYGON and FEPOLYHEDRON zone types are not supported by the writer.

Example:
    >>> with tecio.open("out.plt", "w") as w:
    ...     w.write_ijk_zone(
    ...         data=[x, y, p],
    ...         variables=["x", "y", "pressure"],
    ...     )
"""

from ._read import Read
from ._write import Write

# Set canonical public module paths so docs and help() show tecio.plt.Read,
# not tecio.plt._read.Read.
Read.__module__ = "tecio.plt"
Write.__module__ = "tecio.plt"

__all__ = [
    "Read",
    "Write",
]
