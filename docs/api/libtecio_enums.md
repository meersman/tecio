# Enums

The TecIO library often uses integers with special meanings (zone
types, data types, data locations). The same values are used both for
writing (`tec*142` functions) and for SZL reading and writing (`tec_*`
functions). The classes below provide a more readable format of these
values. Where available, the equivalent keywords used in Tecplot ASCII
files are set as the class property, returning the corresponding int
value.


```{eval-rst}
.. currentmodule:: tecio.libtecio

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
```

```{eval-rst}
.. autoclass:: tecio.libtecio.FileFormat
.. autoclass:: tecio.libtecio.FileType
.. autoclass:: tecio.libtecio.ZoneType
.. autoclass:: tecio.libtecio.DataType
.. autoclass:: tecio.libtecio.ValueLocation
.. autoclass:: tecio.libtecio.FaceNeighborMode
.. autoclass:: tecio.libtecio.FeCellShape
.. autoclass:: tecio.libtecio.DataPacking
.. autoclass:: tecio.libtecio.VarStatus
.. autoclass:: tecio.libtecio.Boolean
```