# Juniper Data v0.4.2 Release Notes

**Release Date:** 2026-02-17
**Version:** 0.4.2
**Codename:** First JuniperData Release
**Release Type:** PATCH

---

## Overview

This is the first official release of JuniperData as a standalone dataset generation microservice. It addresses critical bug fixes discovered during CI/CD validation: 12 failing MNIST generator tests, a Bandit security scan suppression issue, an arc-agi import failure affecting downstream consumers, CI workflow branch coverage, and repository hygiene. All 658 service tests and 41 client tests now pass across Python 3.12-3.14.

> **Status:** ALPHA -- Feature-complete. CI/CD pipeline fully green. Ready for integration testing with JuniperCascor and JuniperCanopy.

---

## Release Summary

- **Release type:** PATCH
- **Primary focus:** Bug Fixes, CI/CD Compliance, Repository Hygiene
- **Breaking changes:** No
- **Priority summary:** All P0 blockers resolved (failing tests, CI scan failures, import errors, branch triggers)

---

## Features Summary

| ID        | Feature                              | Status  | Version | Notes                    |
| --------- | ------------------------------------ | ------- | ------- | ------------------------ |
| MNIST-001 | Fix 12 failing MNIST generator tests | ✅ Done | 0.4.1   | Mock + generator fix     |
| SEC-007   | Bandit B615 nosec placement          | ✅ Done | 0.4.1   | CI compliance            |
| DEP-001   | Make arc-agi optional dependency     | ✅ Done | 0.4.1   | Import fix               |
| CI-003    | CI workflow branch triggers          | ✅ Done | 0.4.2   | All JuniperData branches |
| GIT-001   | Gitignore "\__pycache__\" cleanup    | ✅ Done | 0.4.2   | Repository hygiene       |

---

## What's New

### MNIST Generator n_samples Support (MNIST-001)

Added missing `n_samples` parameter handling in the MNIST generator's `_load_and_preprocess` method. When `n_samples` is specified in `MnistParams`, the generator now calls `ds.select(range(n_samples))` to limit the dataset before preprocessing.

**Files:**

- `juniper_data/generators/mnist/generator.py` (line 94-95)

---

## Bug Fixes

### 12 Failing MNIST Generator Tests (MNIST-001)

**Problem:** All 12 `TestMnistGenerator` tests failed with `ValueError: cannot reshape array of size 0 into shape (0,newaxis)` or `IndexError: arrays used as indices must be of integer (or boolean) type`.

**Root Cause:** The generator's `_load_and_preprocess` method calls `ds.with_format("numpy")` for bulk column access before accessing `ds["image"]` and `ds["label"]`. The test mocks did not configure `with_format()`, so it returned a generic `MagicMock`. Calling `np.array()` on a `MagicMock` produces an empty array, which then failed on reshape. Additionally, the generator was missing `n_samples` support via `ds.select()`.

**Solution:**

- Added `formatted_ds` mock with proper `__getitem__` returning numpy arrays for `"image"` and `"label"` keys
- Set `mock_ds.with_format.return_value = formatted_ds` in `_make_mock_hf_dataset()` and `test_generate_image_without_convert()`
- Added `if params.n_samples is not None: ds = ds.select(range(params.n_samples))` to the generator

**Files:**

- `juniper_data/generators/mnist/generator.py` (lines 94-95)
- `juniper_data/tests/unit/test_mnist_generator.py` (lines 17-38, 255-264)

### Bandit B615 Security Scan Failure (SEC-007)

**Problem:** CI Bandit security scan failed with `[B615:huggingface_unsafe_download]` at medium severity, blocking the pipeline.

**Root Cause:** The `# nosec B615` suppression directive was placed on the comment line above the `hf_load_dataset()` call. Bandit only recognizes `# nosec` directives when placed on the same line as the flagged code. The scan reported `Total lines skipped (#nosec): 0`.

**Solution:** Moved `# nosec B615` from the comment line to inline on the `hf_load_dataset()` call.

**Files:**

- `juniper_data/generators/mnist/generator.py` (line 91)

### arc-agi Optional Dependency (DEP-001)

**Problem:** Importing `juniper_data` in JuniperCascor failed because `arc-agi>=0.9.0` was a hard dependency that wasn't installed in Cascor's environment.

**Root Cause:** The arc-agi package was listed as a required dependency in `pyproject.toml` rather than as an optional dependency, causing `ImportError` when the package wasn't available.

**Solution:** Made `arc-agi` an optional dependency, allowing the module to gracefully degrade when the package is not installed.

**Files:**

- `juniper_data/storage/__init__.py`
- `juniper_data/storage/hf_store.py`

---

## Improvements

### Test Suite Stability

All test failures from v0.4.0 have been resolved:

| Metric | v0.4.0 | v0.4.1 | Change |
| ------ | ------ | ------ | ------ |
| Failing tests | 12 | 0 | -12 |
| Passing tests (service) | 646 | 658 | +12 |
| CI Security Scans | Failing | Passing | Fixed |
| CI Pre-commit | Passing | Passing | Maintained |

### CI/CD Pipeline

- Bandit security scan now passes with 5 `# nosec` suppressions (all justified)
- All CI jobs green: pre-commit (3.12-3.14), unit tests (3.12-3.14), integration tests, security scans, build
- CI workflow now triggers automatically on all `subproject.juniper_data.**` branches (v0.4.2)

### Repository Hygiene (v0.4.2)

- Uncommented `__pycache__` exclusion patterns in `.gitignore`
- Removed 12 tracked `__pycache__/*.pyc` files from the repository

---

## Test Results

### Test Suite

| Metric | Result |
| ------ | ------ |
| **Tests passed** | 658 |
| **Tests skipped** | 0 |
| **Tests failed** | 0 |
| **Runtime** | ~10 seconds |
| **Coverage** | 80%+ (meets threshold) |

### Coverage Details

| Component | Coverage | Target | Status |
| --------- | -------- | ------ | ------ |
| juniper_data (overall) | 80%+ | 80% | ✅ Met |
| juniper-data-client | 96% | N/A | ✅ Exceeded |

### Client Tests

| Metric | Result |
| ------ | ------ |
| **Tests passed** | 41 |
| **Tests failed** | 0 |
| **Coverage** | 96% |

### Environments Tested

- Python 3.12 (CI): ✅ All passing
- Python 3.13 (CI): ✅ All passing
- Python 3.14 (CI): ✅ All passing
- Python 3.14.2 (local development): ✅ All 658 tests pass

---

## Upgrade Notes

This is a backward-compatible patch release. No migration steps required.

```bash
# Update and verify
git pull origin main
pip install -e ".[dev]"

# Run test suite
pytest

# Start API server
python -m juniper_data
```

---

## Known Issues

No outstanding known issues.

### Resolved Since Release

- ~~**GENERATOR_REGISTRY:** 5 of 8 generators registered~~ — All 8 generators now registered (spiral, xor, gaussian, circles, checkerboard, csv_import, mnist, arc_agi).
- ~~**Coverage at ~60% overall:** 6 modules at 0% coverage~~ — Coverage is now 99.40% (659 tests passing). The previously-uncovered modules (arc_agi, mnist, hf_store, kaggle_store, postgres_store, redis_store) all have tests.
- ~~**B008 flake8 warnings:** 9 intentional B008 warnings in `datasets.py`~~ — Migrated to ruff with per-file-ignores for `api/routes/*.py` (RD-012).

---

## What's Next

### Planned for v0.5.0 — Quality + Tooling (first standalone release)

- Add security boundary test suite: path traversal, injection, request size limits, parameter bounds, resource exhaustion (RD-006)
- Migrate linting from flake8 to ruff (RD-012)
- Normalize line length to 120 across all configuration (RD-013)

### Completed Since Release

- ~~Publish juniper-data-client to PyPI~~ — Published as `juniper-data-client` v0.3.0
- ~~Update JuniperCascor to use shared client package~~ — Done (`juniper-cascor-client` v0.1.0 on PyPI)
- ~~Update JuniperCanopy to use shared client package~~ — Done (polyrepo migration Phase 4)
- ~~Register remaining generators~~ — All 8 generators registered in GENERATOR_REGISTRY
- ~~Coverage: add tests for 0% modules~~ — Coverage now at 99.40% (659 tests)

### Roadmap

See [development roadmap](../JUNIPER-DATA_POST-RELEASE_DEVELOPMENT-ROADMAP.md) for current priorities.

---

## Contributors

- Paul Calnon

---

## Version History

| Version | Date | Description |
| ------- | ---- | ----------- |
| 0.4.2 | 2026-02-17 | CI branch triggers, gitignore cleanup |
| 0.4.1 | 2026-02-17 | Bug fixes: MNIST tests, Bandit scan, arc-agi dependency |
| 0.4.0 | 2026-02-17 | Integration infrastructure & extended data sources |
| 0.3.0 | 2026-02-04 | Test suite & CI/CD enhancement |
| 0.2.2 | 2026-02-02 | Code coverage configuration fix |
| 0.2.1 | 2026-02-01 | CI/CD parity across Juniper |
| 0.2.0 | 2026-01-31 | Legacy parity mode for spiral |
| 0.1.2 | 2026-01-31 | Conda environment setup |
| 0.1.1 | 2026-01-30 | CI/CD Pipeline & Pre-commit |
| 0.1.0 | 2026-01-29 | Initial release (Phases 0-2) |

---

## Links

- [Full Changelog](../../CHANGELOG.md)
- [Integration Development Plan](../INTEGRATION_DEVELOPMENT_PLAN.md)
- [API Documentation](../../docs/api/JUNIPER_DATA_API.md)
- [Pull Request Details](../pull_requests/PR_DESCRIPTION_JUNIPER_DATA_EXTRACTION_v0.4.0_2026-02-06.md)
