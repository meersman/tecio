# tecio.TecplotAuxDataReader

A read-only, dict-like view of one auxiliary-data mapping. The same class
backs dataset-, zone-, and variable-level auxiliary data for every format;
implements {class}`collections.abc.Mapping`, so ``get``, ``keys``,
``values``, ``items``, ``in``, and ``len()`` all work as expected, plus a
few typed convenience accessors for reading values back as ``int``,
``float``, or ``bool``.

```{eval-rst}
.. currentmodule:: tecio

.. autoclass:: TecplotAuxDataReader
   :members:
   :show-inheritance:
```
