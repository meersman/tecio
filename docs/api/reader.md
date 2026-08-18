# tecio.TecplotReader

The shared interface for every reader ({class}`~tecio.TecplotSzlReader`,
{class}`~tecio.TecplotPltReader`, {class}`~tecio.TecplotDatReader`). Not
constructed directly, {func}`tecio.open` returns the concrete class for you.
Application code can be written against this base without caring which
format produced a given file.

```{eval-rst}
.. currentmodule:: tecio

.. autoclass:: TecplotReader
   :members:
   :show-inheritance:
```
