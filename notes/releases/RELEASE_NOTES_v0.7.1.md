# Juniper Data v0.7.1 Release Notes

**Release Date:** 2026-06-19
**Version:** 0.7.1
**Release Type:** PATCH

---

## Overview

Patch release fixing a **packaging defect in 0.7.0**: the equities generators' bundled S&P 500
constituents CSV was not shipped inside the wheel, leaving the `equities` / `equities_seq` extras
non-functional from a pip install. No API change.

> **Status:** STABLE — backward-compatible patch. No migration.

---

## Fixed

- **Ship `sp500_constituents.csv` inside the wheel.** 0.7.0 packaged only `*.py`, so the equities
  generators raised `FileNotFoundError` on the bundled constituents file from a pip install of
  `juniper-data[equities]==0.7.0` (the file is loaded via `Path(__file__).parent / "sp500_constituents.csv"`
  — fine in a source checkout, absent from the built wheel). Adds a `[tool.setuptools.package-data]`
  entry (`juniper_data.generators.equities = ["*.csv"]`) so the constituents list ships in the
  wheel + sdist, plus a CI build-step assertion that the CSV is present in the built wheel (guards
  the actual failure mode against a future regression). (juniper-data#193)

The defect was surfaced by the `juniper-recurrence` benchmark's `equities_seq` row, which could not
load the generator from the published `juniper-data[equities]==0.7.0` wheel.

---

## Upgrade

```bash
pip install --upgrade "juniper-data[equities]"   # 0.7.1 — equities/equities_seq now work from PyPI
```

Backward-compatible; no migration steps. The synthetic generators (`multi_sine`, `mackey_glass`,
`ar_p`, `irregular_sine`) were unaffected by the 0.7.0 defect and continue to work without any extra.

---

## Known Issues

None. All required CI checks pass; the new build-step assertion confirms the CSV ships in the wheel.

---

## Version History

| Version | Date       | Description                                              |
| ------- | ---------- | ------------------------------------------------------- |
| 0.7.1   | 2026-06-19 | Fix: equities constituents CSV now ships in the wheel   |
| 0.7.0   | 2026-06-19 | Synthetic dt-sequence generators + scaling meta channel |
| 0.6.0   | 2026-04-08 | Versioning, batch ops, systemd, PostgreSQL fixes        |

---

## Links

- [Full Changelog](../../CHANGELOG.md)
- [Previous Release](RELEASE_NOTES_v0.7.0.md)
- Fix PR: juniper-data#193
