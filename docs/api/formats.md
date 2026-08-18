# Format-Specific Classes

Each of these implements the {doc}`Core Classes <index>` interface for one
of the three supported file formats. {func}`tecio.open` picks the right one
for you based on the file extension (``.szplt``/``.plt``/``.dat``/``.tec``);
most of what's worth knowing about each is on its base class's page, what
follows is each format's constructor and its own quirks, e.g. SZL's
automatic per-variable precision, PLT's single whole-file precision, DAT's
ASCII-specific ``precision``/significant-digit behavior.

## Readers

```{eval-rst}
.. currentmodule:: tecio

.. autosummary::
   :nosignatures:

   TecplotSzlReader
   TecplotPltReader
   TecplotDatReader
```

## Writers

```{eval-rst}
.. currentmodule:: tecio

.. autosummary::
   :nosignatures:

   TecplotSzlWriter
   TecplotPltWriter
   TecplotDatWriter
```

```{toctree}
:hidden:

szl_reader
plt_reader
dat_reader
szl_writer
plt_writer
dat_writer
```
