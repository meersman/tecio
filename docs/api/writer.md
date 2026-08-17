# tecio.TecplotWriter

The shared interface and lifecycle for every writer
({class}`~tecio.TecplotSzlWriter`, {class}`~tecio.TecplotPltWriter`,
{class}`~tecio.TecplotDatWriter`): construction with a path and optional
variable list (eager or lazy open), buffered auxiliary data via
``add_auxdataset_dict``/``add_auxvar_dict``, zone writing via
``write_ijk_zone``/``write_fe_zone``, and closing directly or through the
context manager. How a file is actually opened, closed, and written to is
format-specific and lives on the concrete subclass; everything else is
defined once here.

```{eval-rst}
.. currentmodule:: tecio

.. autoclass:: TecplotWriter
   :members:
   :show-inheritance:
```
