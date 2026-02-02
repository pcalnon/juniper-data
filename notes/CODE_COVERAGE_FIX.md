# Code Coverage Configuration Fix

**Date**: 2026-02-02  
**Issue**: Code coverage reported 0% despite 141 tests passing  
**Resolution**: Fixed coverage source path configuration

---

## Problem Description

Running the code coverage calculation produced the following errors:

```
CoverageWarning: Module was never imported. (module-not-imported)
CoverageWarning: No data was collected. (no-data-collected)
WARNING: Failed to generate report: No data to report.
ERROR: Coverage failure: total of 0.00 is less than fail-under=80.00
```

All 141 tests passed, but coverage reported 0%.

---

## Root Cause Analysis

### Primary Cause: Invalid Source Path Configuration

The coverage configuration in `pyproject.toml` used **path-based** source targets that did not match the actual project directory structure:

**Original (incorrect) configuration:**
```toml
[tool.coverage.run]
source = ["juniper_data/core", "juniper_data/generators", "juniper_data/storage"]
```

**Actual project structure:**
```
JuniperData/
└── juniper_data/           # Project root (where pyproject.toml lives)
    └── juniper_data/       # Actual Python package
        ├── core/
        ├── generators/
        ├── storage/
        └── tests/
```

### How pytest-cov Handles This

When `--cov=juniper_data/core` is passed:
1. pytest-cov first checks if `juniper_data/core` exists as a **path**
2. If the path doesn't exist, it treats the value as a **module name** (literally `juniper_data/core`)
3. That "module" cannot be imported → "Module was never imported" warning
4. Since all `--cov` sources are invalid → "No data was collected"
5. Coverage reports 0%

### Secondary Issue: CI Workflow

The CI workflow (`ci.yml`) also used the incorrect path-based flags:
```yaml
--cov=juniper_data/core \
--cov=juniper_data/generators \
--cov=juniper_data/storage \
```

---

## Solution Implemented

### 1. Updated pyproject.toml

Changed from path-based `source` to package-based `source_pkgs`:

```toml
[tool.coverage.run]
source_pkgs = ["juniper_data"]
omit = [
    "*/tests/*",
    "*/__pycache__/*",
    "*/data/*",
    "*/logs/*",
    "*/__main__.py",
    "*/api/*",
]
branch = true
```

**Key changes:**
- `source_pkgs = ["juniper_data"]` - Uses the installed package name, not file paths
- Added `*/api/*` to omit list - API module is tested separately in integration tests
- Added `*/__main__.py` to omit list - Entry point not typically unit tested

### 2. Updated CI Workflow (ci.yml)

Simplified coverage flags to use a single package reference:

```yaml
--cov=juniper_data \
```

Instead of the three separate path-based flags.

---

## Verification

After the fix and adding comprehensive tests, coverage is at 100%:

```
Name                                          Stmts   Miss Branch BrPart    Cover
juniper_data/__init__.py                          2      0      0      0  100.00%
juniper_data/__main__.py                         25      0      2      0  100.00%
juniper_data/api/__init__.py                      3      0      0      0  100.00%
juniper_data/api/app.py                          41      0      2      0  100.00%
juniper_data/api/routes/__init__.py               2      0      0      0  100.00%
juniper_data/api/routes/datasets.py              81      0     18      0  100.00%
juniper_data/api/routes/generators.py            19      0      4      0  100.00%
juniper_data/api/routes/health.py                 6      0      0      0  100.00%
juniper_data/api/settings.py                     13      0      0      0  100.00%
juniper_data/core/__init__.py                     5      0      0      0  100.00%
juniper_data/core/artifacts.py                   18      0      0      0  100.00%
juniper_data/core/dataset_id.py                   8      0      0      0  100.00%
juniper_data/core/models.py                      13      0      0      0  100.00%
juniper_data/core/split.py                       27      0     14      0  100.00%
juniper_data/generators/__init__.py               2      0      0      0  100.00%
juniper_data/generators/spiral/__init__.py        4      0      0      0  100.00%
juniper_data/generators/spiral/defaults.py       20      0      0      0  100.00%
juniper_data/generators/spiral/generator.py      54      0      6      0  100.00%
juniper_data/generators/spiral/params.py         23      0      2      0  100.00%
juniper_data/storage/__init__.py                  4      0      0      0  100.00%
juniper_data/storage/base.py                      5      0      0      0  100.00%
juniper_data/storage/local_fs.py                 62      0     14      0  100.00%
juniper_data/storage/memory.py                   36      0      4      0  100.00%
-------------------------------------------------------------------------------------------
TOTAL                                           473      0     66      0  100.00%
Required test coverage of 80% reached. Total coverage: 100.00%
```

---

## Files Modified

1. **pyproject.toml**
   - Changed `source` to `source_pkgs`
   - Added exclusion patterns for abstract methods and pass statements

2. **.github/workflows/ci.yml**
   - Simplified `--cov` flags from three path-based flags to single `--cov=juniper_data`

## New Test Files Created

3. **tests/unit/test_main.py** - Tests for `__main__.py` entry point
4. **tests/unit/test_api_app.py** - Tests for FastAPI app factory and lifespan
5. **tests/unit/test_api_routes.py** - Tests for API route edge cases
6. **tests/unit/test_api_settings.py** - Tests for Settings configuration

## Existing Test Files Extended

7. **tests/unit/test_storage.py** - Added edge case tests for LocalFS and abstract base
8. **tests/unit/test_split.py** - Added test for rounding edge case
9. **tests/unit/test_spiral_generator.py** - Added tests for get_schema()

---

## Lessons Learned

1. **Use `source_pkgs` for package-based coverage** - More robust than path-based `source`, works regardless of install method (editable or not)

2. **Path-based coverage requires exact paths** - If using `source = [...]`, paths must exist relative to the working directory

3. **Nested package structures are tricky** - When the project directory and package have the same name (`juniper_data/juniper_data/`), paths can easily be misconfigured

4. **Separate unit and integration coverage** - Unit tests should measure core/generators/storage; API coverage belongs with integration tests

---

## References

- [Coverage.py Documentation: Source Configuration](https://coverage.readthedocs.io/en/latest/source.html)
- [pytest-cov Documentation](https://pytest-cov.readthedocs.io/)
- [Coverage Warning: module-not-imported](https://coverage.readthedocs.io/en/latest/messages.html#warning-module-not-imported)
