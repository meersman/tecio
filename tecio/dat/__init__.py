"""Read and write Tecplot ASCII DAT (``.dat`` / ``.tec``) files.

Tecplot ASCII DAT files are human-readable and portable across platforms.
They are slower to read and write than binary formats and produce larger
files, but are useful for small datasets, debugging, and interoperability
with other tools.

This module provides :class:`Read` for parsing and :class:`Write` for
producing Tecplot 360 ASCII data files (``.dat`` / ``.tec``).  Both
classes mirror the interfaces of :class:`szl.Read` / :class:`szl.Write`
and :class:`plt.Read` / :class:`plt.Write` so that downstream code can
switch between file formats by changing only the file extension passed to
:func:`tecio.open`.

Reading:
    :class:`Read` parses the entire file on construction and stores all data
    in memory::

        dat = tecio.open("result.dat", "r")

        print(dat.title)
        print(dat.variables)  # list of variable name strings
        print(dat.num_zones)

        zone = dat.zone[0]
        print(zone.title, zone.zone_type, zone.solution_time)

        var = zone.variable[0]
        print(var.name, var.data_type, var.values)

        if zone.zone_type != ZoneType.ORDERED:
            print(zone.node_map)  # (num_elements, nodes_per_cell) int64 array

Supported read features:
    * ``FULL``, ``GRID``, and ``SOLUTION`` file types
    * Ordered and simple FE zones (FELINESEG through FEBRICK)
    * ``DATAPACKING=BLOCK`` only (POINT packing raises :exc:`ValueError`)
    * ``VARLOCATION`` (cell-centred variables)
    * ``PASSIVEVARLIST`` and ``VARSHARELIST``
    * ``CONNECTIVITYSHAREZONE``
    * Dataset-level ``DATASETAUXDATA`` and variable-level ``VARAUXDATA``
    * Zone-level ``AUXDATA``

Writing:
    :class:`Write` is a context-manager writer that supports lazy-open,
    buffered aux data, and atomic (all-or-nothing) zone writes::

        with tecio.open("result.dat", "w", title="Demo",
                        variables=["X", "Y", "P"]) as w:
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
