"""Write Tecplot ASCII DAT (``.dat`` / ``.tec``) files.

For reading, see :class:`tecio.TecplotDatReader`. Tecplot ASCII DAT files
are human-readable and portable across platforms. They are slower to read
and write than binary formats and produce larger files, but are useful for
small datasets, debugging, and interoperability with other tools.

This module provides :class:`Write` for producing Tecplot 360 ASCII data
files (``.dat`` / ``.tec``). It mirrors the interfaces of :class:`szl.Write`
and :class:`plt.Write` so that downstream code can switch between file
formats by changing only the file extension passed to :func:`tecio.open`.

Writing:
    :class:`Write` is a context-manager writer that supports lazy-open,
    buffered aux data, and atomic (all-or-nothing) zone writes::

        with tecio.open(
            "result.dat", "w", title="Demo", variables=["X", "Y", "P"]
        ) as w:
            w.write_ijk_zone(data=[x, y, p], title="Zone 1")

    All floating-point variable data is written in scientific notation with a
    configurable number of significant digits (default 9, pass ``sig_digits=17`` for
    full ``float64``)

Note:
    FEPOLYGON and FEPOLYHEDRON zone types are not supported.

Note:
    Dataset-level and variable-level auxiliary data are written via
    :meth:`~tecio.dat.Write.add_auxdataset_dict` before the first zone.
"""

from ._write import Write

# Set canonical public module path so docs and help() show tecio.dat.Write,
# not tecio.dat._write.Write.
Write.__module__ = "tecio.dat"

__all__ = [
    "Write",
]
