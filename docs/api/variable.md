# tecio.TecplotVariableReader

One variable's metadata and data within a zone. Not split by value location
(nodal vs. cell-centered) or by the owning zone's topology, neither removes
a property, both only change the shape ``get_values()`` returns, which the
method already computes from ``value_location`` and the owning zone's
topology.

```{eval-rst}
.. currentmodule:: tecio

.. autoclass:: TecplotVariableReader
   :members:
   :show-inheritance:
```
