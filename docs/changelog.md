# Changelog

Notable changes to this project are documented below.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added

- `tecio.ZoneList` / `tecio.VariableList` container types for `Read.zone` and
  `ReadZone.variable`, with slicing and exact-name lookup.
- `ReadZone.get_array(key)` for direct array access by index, name, or list
  of names.
- `tecsplit` CLI tool to split grid and solutions to separate files.
- `tec2mat` command line tool to convert input Tecplot file to MATLAB
  compatible `.mat` file format. Importable as a MATLAB structure with all
  zones, variables, and metadata.

### Changed

- `Read.zone` / `ReadZone.variable` now return `ZoneList` / `VariableList`
  instead of plain `list`.
- Read tests rewritten to match writer-test conventions (fixtures,
  `pytest.raises`, class grouping).

### Fixed
- Legacy style unstructured zone headers in ASCII files use different keywords
  for data packing and zone type resulting in `tecio` interpreting these as
  single point ordered zone. These special cases have been added.

### Removed

- **Breaking:** attribute-style variable access (`zone.x`) — was broken
  (infinite recursion) on SZL and case-insensitive on PLT. Use
  `zone.get_array("x")` instead.

---

```{include} changelog/v0.1.0.md
```

---

[Unreleased]: https://github.com/meersman/tecio/compare/v0.1.0...HEAD

```{toctree}
:hidden:
:maxdepth: 1
:caption: Changelog
:glob:

changelog/*
```
