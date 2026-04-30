"""Command-line interface tools for the :mod:`tecio` package.

Each module in this subpackage implements a standalone console script that is
registered as an entry point in ``pyproject.toml``.  After installation the
scripts are available directly from the shell:

.. code-block:: console

    $ tecdump flow.szplt
    $ tecfix flow.szplt
    $ tecslice -i ::2 -o thinned.szplt flow.szplt
    $ tecmerge -o combined.szplt "step_*.szplt"
    $ teconvert -dat flow.szplt
    $ tecextract -zones 1,2 -o subset.szplt flow.szplt
    $ tecscale -variable Pressure -scale 1e-3 flow.szplt
    $ tecstat flow.szplt

Modules
-------
.. autosummary::
   :toctree: generated

   tecdump
   tecfix
   tecslice
   tecmerge
   teconvert
   tecextract
   tecscale
   tecstat
"""
