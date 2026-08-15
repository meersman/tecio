"""Write Tecplot SZL (``.szplt``) files.

For reading, see :class:`tecio.TecplotSzlReader`. Tecplot SZL is a
subzone-loadable binary format. Data is stored in compressed subzones so
Tecplot 360 can load only the portions of the dataset it needs, making it
the preferred format for large files.

Key behaviours:

- Variable sharing across zones (``var_sharing``) avoids duplicating
  grid coordinates for time-series data, significantly reducing file
  size for transient datasets.
- The writer uses a lazy-open strategy: the file is not created on disk
  until the first :meth:`~tecio.szl.Write.write_ijk_zone` or
  :meth:`~tecio.szl.Write.write_fe_zone` call.
- Auxiliary data is buffered and flushed automatically before the first
  zone is written.

Example:
    >>> with tecio.open("out.szplt", "w") as w:
    ...     w.write_ijk_zone(
    ...         data=[x, y, p],
    ...         variables=["x", "y", "pressure"],
    ...         strand_id=1,
    ...         solution_time=t,
    ...     )
"""

from ._write import Write

# Set canonical public module path so docs and help() show tecio.szl.Write,
# not tecio.szl._write.Write.
Write.__module__ = "tecio.szl"

__all__ = [
    "Write",
]
