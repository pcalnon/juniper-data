# Testing Reference

## juniper-data Test Configuration, Markers, and Fixtures

**Version:** 0.4.2
**Status:** Active
**Last Updated:** March 3, 2026
**Project:** Juniper - Dataset Generation Service

---

## Table of Contents

- [Pytest Markers](#pytest-markers)
- [Pytest Configuration](#pytest-configuration)
- [Fixtures Reference](#fixtures-reference)
- [Coverage Configuration](#coverage-configuration)
- [Test Dependencies](#test-dependencies)
- [Command Reference](#command-reference)
- [File Structure Reference](#file-structure-reference)
- [Warning Filters](#warning-filters)

---

## Pytest Markers

All markers are defined in `pyproject.toml` under `[tool.pytest.ini_options].markers`. Using `--strict-markers` means any undefined marker will cause a test failure.

### Scope Markers

Every test must have exactly one scope marker:

| Marker | Description | Usage Count | CI Behavior |
|--------|-------------|-------------|-------------|
| `unit` | Unit tests for individual components | ~171 | Every push, all Python versions |
| `integration` | Integration tests for full workflows | ~19 | PRs and main/develop only |
| `performance` | Performance and benchmarking tests | ~6 | Benchmarks disabled by default |

### Behavioral Markers

| Marker | Description | Usage Count | CI Behavior |
|--------|-------------|-------------|-------------|
| `slow` | Tests that take a long time to run | ~5 | Separate schedule (daily 6 AM UTC) or manual dispatch |

### Component Markers

| Marker | Description | Usage Count | Typical Files |
|--------|-------------|-------------|---------------|
| `spiral` | Spiral dataset generator tests | ~8 | `test_spiral_generator.py` |
| `api` | API endpoint tests | varies | `test_api_*.py`, `test_health_*.py` |
| `generators` | Data generator tests | ~24 | `test_*_generator.py` |
| `storage` | Storage operation tests | ~36 | `test_*_store.py`, `test_storage.py` |

### Third-Party Markers

| Marker | Source | Description |
|--------|--------|-------------|
| `asyncio` | pytest-asyncio | Marks async test functions for event loop handling |
| `parametrize` | pytest (built-in) | Parameterized test cases |

---

## Pytest Configuration

All settings from `pyproject.toml` `[tool.pytest.ini_options]`:

### Discovery Settings

| Setting | Value | Description |
|---------|-------|-------------|
| `minversion` | `"6.0"` | Minimum pytest version |
| `testpaths` | `["juniper_data/tests"]` | Root directory for test discovery |
| `pythonpath` | `["."]` | Python path for imports |
| `python_files` | `["test_*.py"]` | File name pattern for test discovery |
| `python_classes` | `["Test*"]` | Class name pattern for test discovery |
| `python_functions` | `["test_*"]` | Function name pattern for test discovery |

### Execution Settings

| Setting | Value | Description |
|---------|-------|-------------|
| `timeout` | `60` | Default per-test timeout in seconds |
| `timeout_method` | `"signal"` | Timeout enforcement method |

### Default Options (`addopts`)

| Flag | Description |
|------|-------------|
| `-ra` | Show summary of all test results (except passed) |
| `-q` | Quiet output mode |
| `--strict-markers` | Fail on undefined markers |
| `--strict-config` | Fail on configuration errors |
| `--tb=short` | Short traceback format |
| `--benchmark-disable` | Disable benchmarks by default |

---

## Fixtures Reference

All shared fixtures defined in `juniper_data/tests/conftest.py`:

### Spiral Parameter Fixtures

| Fixture | Scope | Returns | Parameters |
|---------|-------|---------|------------|
| `default_spiral_params` | function | `SpiralParams` | Default `SpiralParams()` constructor |
| `two_spiral_params` | function | `SpiralParams` | `n_spirals=2, n_points_per_spiral=100, seed=42` |
| `three_spiral_params` | function | `SpiralParams` | `n_spirals=3, n_points_per_spiral=50, seed=42` |
| `minimal_spiral_params` | function | `SpiralParams` | `n_spirals=2, n_points_per_spiral=10, seed=42` |

### Generated Dataset Fixtures

| Fixture | Scope | Returns | Depends On |
|---------|-------|---------|-----------|
| `generated_two_spiral_dataset` | function | `dict[str, np.ndarray]` | `two_spiral_params` |
| `generated_three_spiral_dataset` | function | `dict[str, np.ndarray]` | `three_spiral_params` |
| `generated_minimal_dataset` | function | `dict[str, np.ndarray]` | `minimal_spiral_params` |

Dataset dictionaries contain keys: `X_train`, `y_train`, `X_test`, `y_test`, `X_full`, `y_full` (all `float32`).

### Utility Fixtures

| Fixture | Scope | Returns | Description |
|---------|-------|---------|-------------|
| `sample_arrays` | function | `dict[str, np.ndarray]` | `"X"` shape (10,2) and `"y"` shape (10,2), dtype `float32` |

### Golden Dataset Files

| File | Location | Description |
|------|----------|-------------|
| `2_spiral.npz` | `tests/fixtures/golden_datasets/` | Reference 2-spiral dataset |
| `2_spiral_metadata.json` | `tests/fixtures/golden_datasets/` | Generation parameters |
| `3_spiral.npz` | `tests/fixtures/golden_datasets/` | Reference 3-spiral dataset |
| `3_spiral_metadata.json` | `tests/fixtures/golden_datasets/` | Generation parameters |

---

## Coverage Configuration

### `[tool.coverage.run]`

| Setting | Value | Description |
|---------|-------|-------------|
| `source_pkgs` | `["juniper_data"]` | Only measure `juniper_data` package |
| `branch` | `true` | Enable branch coverage measurement |
| `omit` | `["*/tests/*", "*/__pycache__/*", "*/data/*", "*/logs/*"]` | Excluded paths |

### `[tool.coverage.report]`

| Setting | Value | Description |
|---------|-------|-------------|
| `fail_under` | `80` | Aggregate coverage threshold |
| `show_missing` | `true` | Show uncovered line numbers |
| `precision` | `2` | Decimal places in reports |

### Coverage Exclusion Lines

```python
"pragma: no cover"
"def __repr__"
"raise AssertionError"
"raise NotImplementedError"
"if __name__ == .__main__.:"
"if TYPE_CHECKING:"
"@abstractmethod"
"^\\s*pass\\s*$"
```

### `[tool.coverage.html]`

| Setting | Value |
|---------|-------|
| `directory` | `"htmlcov"` |

### `[tool.coverage.xml]`

| Setting | Value |
|---------|-------|
| `output` | `"coverage.xml"` |

### Coverage Thresholds

| Threshold | Value | Source | Enforcement |
|-----------|-------|--------|-------------|
| Aggregate | 80% | `COVERAGE_FAIL_UNDER` env var / `pyproject.toml` | CI `unit-tests` job, `--cov-fail-under` flag |
| Per-module | 85% | `scripts/check_module_coverage.py` | CI `unit-tests` job, pre-push hook |

---

## Test Dependencies

From `pyproject.toml` `[project.optional-dependencies.test]`:

| Package | Version | Purpose |
|---------|---------|---------|
| `pytest` | `>=7.0.0` | Test framework |
| `pytest-cov` | `>=4.0.0` | Coverage reporting plugin |
| `pytest-timeout` | `>=2.2.0` | Per-test timeout enforcement |
| `pytest-asyncio` | `>=0.21.0` | Async test support |
| `pytest-benchmark` | `>=4.0.0` | Performance benchmarking |
| `httpx` | `>=0.24.0` | HTTP client for API tests |
| `coverage[toml]` | `>=7.0.0` | Coverage measurement |
| `juniper-data-client` | `>=0.3.0` | Client library for integration tests |

Install with: `pip install -e ".[test]"` or `pip install -e ".[all]"`

---

## Command Reference

### Test Execution

| Command | Description |
|---------|-------------|
| `pytest` | Run all tests with default options |
| `pytest -v` | Verbose output |
| `pytest -x` | Stop at first failure |
| `pytest --maxfail=N` | Stop after N failures |
| `pytest -m MARKER` | Run tests matching marker expression |
| `pytest -k PATTERN` | Run tests matching keyword pattern |
| `pytest --timeout=N` | Override default timeout (seconds) |
| `pytest --tb=long` | Full tracebacks |
| `pytest --tb=no` | No tracebacks |
| `pytest -p no:warnings` | Suppress all warnings |

### Coverage Commands

| Command | Description |
|---------|-------------|
| `pytest --cov=juniper_data --cov-report=term-missing` | Terminal report with missing lines |
| `pytest --cov=juniper_data --cov-report=html` | HTML report to `htmlcov/` |
| `pytest --cov=juniper_data --cov-report=xml:coverage.xml` | Cobertura XML report |
| `pytest --cov=juniper_data --cov-report=json:coverage.json` | JSON report |
| `pytest --cov-fail-under=N` | Fail if coverage below N% |
| `python scripts/check_module_coverage.py` | Check from existing `.coverage` file |
| `python scripts/check_module_coverage.py --run-tests` | Run tests then check |

### Benchmark Commands

| Command | Description |
|---------|-------------|
| `pytest --benchmark-enable` | Enable benchmarks (disabled by default) |
| `pytest --benchmark-autosave` | Save results for regression tracking |
| `pytest --benchmark-compare` | Compare against saved baseline |
| `pytest --benchmark-sort=mean` | Sort by mean time |

### Pre-commit Commands

| Command | Description |
|---------|-------------|
| `pre-commit install` | Install pre-commit hooks |
| `pre-commit install --hook-type pre-push` | Install pre-push hooks |
| `pre-commit run --all-files` | Run all pre-commit hooks |
| `pre-commit run coverage-check --hook-stage pre-push` | Run coverage check manually |

---

## File Structure Reference

### Test Discovery Path

```
juniper_data/tests/          # Root test path (testpaths in pyproject.toml)
├── conftest.py              # Shared fixtures
├── fixtures/                # Test data and golden datasets
├── unit/                    # @pytest.mark.unit (29 files)
├── integration/             # @pytest.mark.integration (5 files)
└── performance/             # @pytest.mark.performance (2 files)
```

### Report Output Locations

| Report | Location | Generated By |
|--------|----------|-------------|
| JUnit XML (unit) | `reports/junit-unit.xml` | CI `unit-tests` job |
| JUnit XML (integration) | `reports/junit-integration.xml` | CI `integration-tests` job |
| Coverage HTML | `htmlcov/` | `--cov-report=html` |
| Coverage XML | `coverage.xml` | `--cov-report=xml` |
| Coverage JSON | `reports/coverage.json` | `check_module_coverage.py` |

---

## Warning Filters

Configured in `pyproject.toml` `[tool.pytest.ini_options].filterwarnings`:

| Filter | Pattern | Reason |
|--------|---------|--------|
| `ignore::DeprecationWarning` | `uvicorn.*` | Known uvicorn deprecation warnings |
| `ignore::DeprecationWarning` | `httpx.*` | Known httpx deprecation warnings |
| `ignore::PendingDeprecationWarning` | `pydantic.*` | Known pydantic pending deprecation warnings |

---

**Last Updated:** March 3, 2026
**Version:** 0.4.2
**Maintainer:** Paul Calnon
