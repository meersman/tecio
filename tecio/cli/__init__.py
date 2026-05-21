"""Command-line tools for common Tecplot data file operations.

Binary data formats such as ``.plt`` and ``.szplt`` are opaque to standard shell
utilities, meaning any inspection or manipulation of their contents typically requires
either dedicated post-processing software or a non-trivial amount of scripting.  For
routine tasks encountered in CFD and related workflows, verifying zone extents,
extracting a subset of variables, computing summary statistics, this overhead is
disproportionate to the complexity of the task itself.

The tools collected here address this by exposing the most common file-level operations
as console scripts, executable directly from the terminal and composable with standard
shell pipelines.  Each tool is intentionally narrow in scope: it does one thing and
surfaces its result without requiring the user to open a GUI or write any Python.

All scripts are installed automatically alongside the :mod:`tecio` package and are
available on ``PATH`` immediately after installation.  Every tool accepts ``--help`` for
a full description of its arguments.

"""
