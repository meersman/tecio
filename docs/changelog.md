# Changelog

Notable changes to this project are documented below.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Changed

- Added face neightbor support for PLT and DAT file formats. Hooks
  are in place for SZL format, but current version of TecIO library
  has a bug that does not read or write face neighbors correctly.
- **Breaking:** `TecplotWriter.write_ijk_zone` renamed to
  `write_ordered_zone`, for symmetry with `write_fe_zone`.
- **Breaking:** Reader container properties renamed for clarity:
  `.zone` → `.zones` and `.variable` → `.variables`, on every reader and
  zone reader class.
- Variable names are now supported as an additional input option for CLI tools
  wherever previously only variable index was accepted.

### Fixed

- The number of values in ordered zones without point counts are now inferred
  instead of defaulting to one.
- `pragma: no cover` added wherever there is environment dependent or
  non-converable code execution so coverage report is more accurate.

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

[Unreleased]: https://github.com/meersman/tecio/compare/v0.3.1...HEAD

```{toctree}
:hidden:
:maxdepth: 1
:caption: Changelog
:glob:

changelog/*
```
