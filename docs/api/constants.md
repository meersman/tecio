# Constants

The TecIO library often uses integers with special meanings (zone
types, data types, data locations). The same values are used both for
writing (`tec*142` functions) and for SZL reading and writing (`tec_*`
functions). The classes below provide a more readable format of these
values. Where available, the equivalent keywords used in Tecplot ASCII
files are set as the class property, returning the corresponding int
value.

These are pure Python enums with no dependency on the TecIO C library
itself, importing them never touches `libtecio.so`/`.dylib`. Available
directly from {mod}`tecio` (e.g. `tecio.ZoneType`), and also re-exported
from {mod}`tecio.libtecio` for backward compatibility, since every C wrapper
function there uses them too.

```{eval-rst}
.. currentmodule:: tecio

.. autosummary::
   :nosignatures:

   FileFormat
   FileType
   ZoneType
   DataType
   ValueLocation
   FaceNeighborMode
   FeCellShape
   DataPacking
   VarStatus
   Boolean
   Debug
```

```{eval-rst}
.. autoclass:: tecio.FileFormat
.. autoclass:: tecio.FileType
.. autoclass:: tecio.ZoneType
.. autoclass:: tecio.DataType
.. autoclass:: tecio.ValueLocation
.. autoclass:: tecio.FaceNeighborMode
.. autoclass:: tecio.FeCellShape
.. autoclass:: tecio.DataPacking
.. autoclass:: tecio.VarStatus
.. autoclass:: tecio.Boolean
.. autoclass:: tecio.Debug
```
