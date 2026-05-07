"""Read and write Tecplot SZL (``.szplt``) files.

Tecplot SZL is a subzone-loadable binary format. Data is stored in
compressed subzones so Tecplot 360 can load only the portions of the
dataset it needs, making it the preferred format for large files.

Key behaviours:

- Variable data is read lazily from disk on first access via
  :attr:`~tecio.szl.Read.zone`.
- Variable sharing across zones (``var_sharing``) avoids duplicating
  grid coordinates for time-series data, significantly reducing file
  size for transient datasets.
- The writer uses a lazy-open strategy: the file is not created on disk
  until the first :meth:`~tecio.szl.Write.write_ijk_zone` or
  :meth:`~tecio.szl.Write.write_fe_zone` call.
- Auxiliary data is buffered and flushed automatically before the first
  zone is written.
- Multiple SZL files can be open simultaneously since each read/write
  handle is independent.

Example:
    >>> with tecio.open("out.szplt", "w") as w:
    ...     w.write_ijk_zone(
    ...         data=[x, y, p],
    ...         variables=["x", "y", "pressure"],
    ...         strand_id=1,
    ...         solution_time=t,
    ...     )
"""

from ._read import Read
from ._write import Write

# Set canonical public module paths so docs and help() show tecio.szl.Read,
# not tecio.szl._read.Read.
Read.__module__ = "tecio.szl"
Write.__module__ = "tecio.szl"

__all__ = [
    "Read",
    "Write",
]
