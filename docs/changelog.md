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
- API docs for the new container types and access patterns.

### Changed

- `Read.zone` / `ReadZone.variable` now return `ZoneList` / `VariableList`
  instead of plain `list`.
- Read tests rewritten to match writer-test conventions (fixtures,
  `pytest.raises`, class grouping).

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
