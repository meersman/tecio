# Changelog

Notable changes to this project are documented below.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

- Updaded passive and shared variable specification for the writer
  methods. Previously any passive or shared variables required providing a
  flag for all global variables, while data arrays and value loacations and
  only provided for active variables. Now the passive and shared variables can
  be specified by only referencing those variables. Users may specify passive
  variables with `passive_vars=["x", "y", "z"]` and shared variables with
  `var_sharing={"x": 1, "y": 1, "z": 1}` (specified as varname or index:
  shared zone index).
- Changed entire Python to 0-baed indexing. In all Python functions, classes,
  and methods, any zone or variable index is 0-based. The
  :mod:`~tecio.libtecio` module wrapper functions maintain 1-based indexing
  convention, but all reader and writer classes make this conversion
  automatically.
- Added ability to transform multiple varialbes in single
  :mod:`~tecio.cli.tecscale` call.
- Added new `--strand-offset` flag for :mod:`~tecio.cli.tecmerge` to allow for
  multiple transient data files to be loaded without automerging overlapping
  strand IDs (eg create strand ids 101, 102, 103, ...).
- Added first version of Tecplot data writer plugin for Paraview,
  `TecplotTecioWriter.py`.
- Added MATLAB function to reformat zone structures created by
  :mod:`~tecio.cli.tec2mat` into struct array of zones wth arrays accessible
  by variable name, `tecmat2struct`.

---

```{include} changelog/v0.3.3.md
```

---

```{include} changelog/v0.3.2.md
```

---

```{include} changelog/v0.3.1.md
```

---

```{include} changelog/v0.3.0.md
```

---

```{include} changelog/v0.2.4.md
```

---

```{include} changelog/v0.2.3.md
```

---

```{include} changelog/v0.2.2.md
```

---

```{include} changelog/v0.2.1.md
```

---

```{include} changelog/v0.2.0.md
```

---

```{include} changelog/v0.1.1.md
```

---

```{include} changelog/v0.1.0.md
```

---

[Unreleased]: https://github.com/meersman/tecio/compare/v0.3.3...HEAD

```{toctree}
:hidden:
:maxdepth: 1
:caption: Changelog
:glob:

changelog/*
```
