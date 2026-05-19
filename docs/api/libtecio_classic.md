# Classic API Functions

The classic `tec*142` API uses a single implicit global file context.
Only one file may be active at a time; call :func:`tecini142` to open it
and {func}`tecend142` to close it. Both PLT and SZL formats are supported.

```{eval-rst}
.. currentmodule:: tecio.libtecio

.. autosummary::
   :nosignatures:

   tecini142
   tecend142
   tecflush142
   tecfil142
   tecforeign142
   teczne142
   tecpolyzne142
   tecznefemixed142
   tecdat142
   tecnode142
   tecface142
   tecpolyface142
   tecpolybconn142
   tecauxstr142
   tecvauxstr142
   teczauxstr142
   tecusr142
```

```{eval-rst}
.. autofunction:: tecio.libtecio.tecini142
.. autofunction:: tecio.libtecio.tecend142
.. autofunction:: tecio.libtecio.tecflush142
.. autofunction:: tecio.libtecio.tecfil142
.. autofunction:: tecio.libtecio.tecforeign142
.. autofunction:: tecio.libtecio.teczne142
.. autofunction:: tecio.libtecio.tecpolyzne142
.. autofunction:: tecio.libtecio.tecznefemixed142
.. autofunction:: tecio.libtecio.tecdat142
.. autofunction:: tecio.libtecio.tecnode142
.. autofunction:: tecio.libtecio.tecface142
.. autofunction:: tecio.libtecio.tecpolyface142
.. autofunction:: tecio.libtecio.tecpolybconn142
.. autofunction:: tecio.libtecio.tecauxstr142
.. autofunction:: tecio.libtecio.tecvauxstr142
.. autofunction:: tecio.libtecio.teczauxstr142
.. autofunction:: tecio.libtecio.tecusr142
```
