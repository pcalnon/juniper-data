# Testing Manual

## Comprehensive Testing Guide for juniper-data

**Version:** 0.4.3
**Status:** Active
**Last Updated:** September 5, 2026
**Project:** Juniper - Dataset Generation Service

---

## Table of Contents

1. [Introduction](#introduction)
2. [Test Architecture](#test-architecture)
   - [Directory Structure](#directory-structure)
   - [Test Categories](#test-categories)
   - [Test File Inventory](#test-file-inventory)
   - [Empty-train shape metadata](#empty-train-shape-metadata)
   - [Postgres model-derived schema](#postgres-model-derived-schema)
   - [Artifact streaming](#artifact-streaming)
3. [Running Tests](#running-tests)
   - [Basic Commands](#basic-commands)
   - [Marker-Based Selection](#marker-based-selection)
   - [Targeted Execution](#targeted-execution)
4. [Fixtures](#fixtures)
   - [Spiral Parameter Fixtures](#spiral-parameter-fixtures)
   - [Generated Dataset Fixtures](#generated-dataset-fixtures)
   - [Utility Fixtures](#utility-fixtures)
5. [Golden Datasets](#golden-datasets)
6. [Coverage](#coverage)
   - [Running Coverage](#running-coverage)
   - [Thresholds](#thresholds)
   - [Coverage Script](#coverage-script)
   - [Exclusion Patterns](#exclusion-patterns)
7. [Performance Benchmarks](#performance-benchmarks)
8. [Writing Tests](#writing-tests)
   - [Naming Conventions](#naming-conventions)
   - [Marker Requirements](#marker-requirements)
   - [Async Tests](#async-tests)
   - [Test Organization](#test-organization)
   - [Import-cycle tests](#import-cycle-tests)

   - [DatasetMeta n_val pins](#datasetmeta-n_val-pins)
   - [Truncation-edge tests](#truncation-edge-tests)
9. [Pre-commit Integration](#pre-commit-integration)
10. [Troubleshooting](#troubleshooting)

---

## Introduction

juniper-data uses pytest as its test framework with a structured three-tier test suite: unit tests for isolated component validation, integration tests for cross-component workflows, and performance benchmarks for throughput measurement.

Key characteristics:

- **8 pytest markers** for granular test selection
- **80% aggregate / 85% per-module** coverage enforcement
- **60-second default timeout** per test (signal-based)
- **Golden datasets** for reproducible generator validation
- **Benchmarks disabled by default** to keep CI fast

---

## Test Architecture

### Directory Structure

```
juniper_data/tests/
├── conftest.py                          # Root fixtures (spiral params, datasets, utilities)
├── __init__.py
├── fixtures/
│   ├── generate_golden_datasets.py      # Golden dataset generation utility
│   └── golden_datasets/
│       ├── 2_spiral_metadata.json       # Metadata for 2-spiral golden set
│       ├── 2_spiral.npz                 # Pre-generated 2-spiral dataset
│       ├── 3_spiral_metadata.json       # Metadata for 3-spiral golden set
│       ├── 3_spiral.npz                 # Pre-generated 3-spiral dataset
│       └── README.md                    # Golden dataset documentation
├── unit/                                # 29 test files
│   ├── __init__.py
│   └── test_*.py
├── integration/                         # 5 test files
│   ├── __init__.py
│   └── test_*.py
└── performance/                         # 2 test files
    ├── __init__.py
    └── test_*.py
```

### Test Categories

| Category | Directory | Purpose | Timeout | CI Behavior |
|----------|-----------|---------|---------|-------------|
| **Unit** | `tests/unit/` | Isolated component validation | 60s | Every push, all Python versions |
| **Integration** | `tests/integration/` | Cross-component workflows | 120s | PRs and main/develop only |
| **Performance** | `tests/performance/` | Throughput benchmarks | 60s | Benchmarks disabled by default |

### Test File Inventory

#### Unit Tests (29 files)

| File | Component | Description |
|------|-----------|-------------|
| `test_api_app.py` | API | FastAPI app factory and creation |
| `test_api_routes.py` | API | Route handler functions and endpoints |
| `test_api_settings.py` | API | Pydantic settings; APD-DATA-033 window reaches the live limiter |
| `test_arc_agi_generator.py` | Generator | ARC-AGI dataset generator |
| `test_artifacts.py` | Core | Artifact class and file handling |
| `test_artifact_streaming.py` | Storage | Chunked `open_artifact_stream` (APD-DATA-016) |
| `test_binary_media_types.py` | API | `BINARY_MEDIA_TYPE` is `application/zip` |
| `test_cached_store.py` | Storage | Cached dataset storage |
| `test_checkerboard_generator.py` | Generator | Checkerboard pattern generator |
| `test_circles_generator.py` | Generator | Concentric circles generator |
| `test_csv_import_generator.py` | Generator | CSV/JSON file import; byte-cap refusal and truncation annotation |
| `test_dataset_id.py` | Core | DatasetID class and validation |
| `test_equities_generator.py` | Generator | Flat equities; APD-DATA-018 symbol cap (`TestUniverseSymbolCap`) |
| `test_equities_seq_generator.py` | Generator | Windowed equities; same `_resolve_symbols` + truncation channel |
| `test_gaussian_generator.py` | Generator | Mixture of Gaussians generator |
| `test_health_enhanced.py` | API | Health check endpoint |
| `test_hf_store.py` | Storage | Hugging Face storage backend |
| `test_init.py` | Core | Package initialization and version |
| `test_kaggle_store.py` | Storage | Kaggle storage backend |
| `test_lifecycle.py` | Core | Dataset lifecycle management |
| `test_main.py` | Core | CLI entry point (`__main__.py`) |
| `test_meta_dispatch.py` | Core | Shape metadata: trailing-axis `n_features`, task_type dispatch, empty-train (#340) |
| `test_middleware.py` | API | FastAPI middleware components |
| `test_mnist_generator.py` | Generator | MNIST/Fashion-MNIST generator |
| `test_no_import_cycles.py` | API / generators | Cold-interpreter standalone import of every generator subpackage (#316 / #333) |
| `test_observability.py` | API | Prometheus metrics and Sentry |
| `test_postgres_schema_derivation.py` | Storage | Model-derived Postgres DDL/statements and round-trip (#343) |
| `test_postgres_store.py` | Storage | PostgreSQL storage backend |
| `test_redis_store.py` | Storage | Redis storage backend |
| `test_security.py` | API | Rate limiter window expiry, 429, constructor window |
| `test_security_boundaries.py` | API | Security boundary tests |
| `test_spiral_generator.py` | Generator | Spiral generator (567 lines, 14 test classes) |
| `test_split.py` | Core | Two-way split plus additive three-way sizing (`partition_row_counts`, `split_three_way`; #353) |
| `test_storage.py` | Storage | Storage interface and abstract classes |
| `test_xor_generator.py` | Generator | XOR classification generator |

#### Integration Tests (5 files)

| File | Description |
|------|-------------|
| `test_api.py` | Full REST API workflow tests |
| `test_e2e_workflow.py` | End-to-end dataset generation and persistence (`@slow`) |
| `test_lifecycle_api.py` | Full lifecycle management through API |
| `test_security_integration.py` | Security integration tests |
| `test_storage_workflow.py` | Storage backend workflow tests |

#### Performance Tests (2 files)

| File | Description |
|------|-------------|
| `test_generator_benchmarks.py` | Generator throughput benchmarks |
| `test_storage_benchmarks.py` | Storage operation benchmarks |

### Empty-train shape metadata

`compute_shape_meta` is on the create-dataset path for every generator. An empty train split still has a defined `shape[-1]`; hardcoding `n_features = 2` when `n_train == 0` lied for F ≠ 2 (including 3-D sequence F). `test_classification_3d_uses_trailing_feature_axis` only covers non-empty train, so the narrower `if n_train > 0 else 2` stayed green before #365 landed.

Pin empty train in `test_meta_dispatch.py`: 2-D F=5, 3-D F=3 (not lookback), and classification `n_classes` from `y_test`. Mutation: putting `else 2` back fails exactly those two n_features tests.

> See: [REFERENCE.md -- Empty-Train Shape Metadata](../REFERENCE.md#empty-train-shape-metadata)

### Postgres model-derived schema

The Postgres store used to carry five independent transcriptions of `DatasetMeta`. No test asserted `_row_to_meta(_meta_to_row(m)) == m`, so the copies drifted: `n_classes` stayed `NOT NULL` after the model allowed None, and seven fields were dropped every round trip. #343 derives DDL, upsert, update, and both mappers from `model_fields`.

Pin that contract in `test_postgres_schema_derivation.py` (no database — the mappers are pure). Mutation: re-introducing a hand-written column list, `ADD COLUMN ... NOT NULL` without DEFAULT, or `json.dumps(None)` is expected to fail those pins.

> See: [REFERENCE.md -- Postgres Model-Derived Schema](../REFERENCE.md#postgres-model-derived-schema)

### Artifact streaming

`test_artifact_streaming.py` pins APD-DATA-016 / #313. A whole-file read still round-trips, so the decisive LocalFS arm is that a small `chunk_size` yields more than one chunk. The base default must yield exactly one chunk (honest whole-read). Absence must be `None` from the *call* — a generator object here becomes 200 with an empty body. `test_binary_media_types.py` pins the published `application/zip` type.

> See: [REFERENCE.md -- Artifact Streaming](../REFERENCE.md#artifact-streaming)

---

## Running Tests

### Basic Commands

```bash
# Run all tests
pytest

# Verbose output
pytest -v

# Stop at first failure
pytest -x

# Stop after 5 failures
pytest --maxfail=5

# Run a specific test file
pytest juniper_data/tests/unit/test_spiral_generator.py -v

# Run a specific test class
pytest juniper_data/tests/unit/test_spiral_generator.py::TestSpiralGeneration -v

# Run a specific test function
pytest juniper_data/tests/unit/test_spiral_generator.py::TestSpiralGeneration::test_basic_spiral -v
```

### Marker-Based Selection

```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Run unit tests excluding slow ones (CI default)
pytest -m "unit and not slow"

# Run generator and storage tests together
pytest -m "generators or storage"

# Run everything except performance
pytest -m "not performance"
```

### Targeted Execution

```bash
# Run all tests for a specific directory
pytest juniper_data/tests/unit/ -v

# Run tests matching a keyword pattern
pytest -k "spiral" -v

# Run tests matching a class pattern
pytest -k "TestSpiralGeneration" -v

# Combine marker and keyword
pytest -m unit -k "generator" -v
```

---

## Fixtures

All shared fixtures are defined in `juniper_data/tests/conftest.py`.

### Spiral Parameter Fixtures

| Fixture | Returns | Configuration |
|---------|---------|---------------|
| `default_spiral_params` | `SpiralParams` | Default `SpiralParams()` |
| `two_spiral_params` | `SpiralParams` | `n_spirals=2, n_points_per_spiral=100, seed=42` |
| `three_spiral_params` | `SpiralParams` | `n_spirals=3, n_points_per_spiral=50, seed=42` |
| `minimal_spiral_params` | `SpiralParams` | `n_spirals=2, n_points_per_spiral=10, seed=42` |

### Generated Dataset Fixtures

These fixtures call the spiral generator and return pre-built datasets:

| Fixture | Depends On | Returns |
|---------|-----------|---------|
| `generated_two_spiral_dataset` | `two_spiral_params` | `dict[str, np.ndarray]` (2 spirals, 100 points each) |
| `generated_three_spiral_dataset` | `three_spiral_params` | `dict[str, np.ndarray]` (3 spirals, 50 points each) |
| `generated_minimal_dataset` | `minimal_spiral_params` | `dict[str, np.ndarray]` (2 spirals, 10 points each) |

### Utility Fixtures

| Fixture | Returns | Purpose |
|---------|---------|---------|
| `sample_arrays` | `dict[str, np.ndarray]` | Simple arrays for split/shuffle testing. Keys: `"X"` (10,2) and `"y"` (10,2), dtype `float32` |

---

## Golden Datasets

Golden datasets are pre-generated reference datasets stored in `tests/fixtures/golden_datasets/`. They provide deterministic baselines for regression testing.

**Available golden datasets:**

| File | Description |
|------|-------------|
| `2_spiral.npz` | Reference 2-spiral dataset with known parameters |
| `2_spiral_metadata.json` | Generation parameters for the 2-spiral dataset |
| `3_spiral.npz` | Reference 3-spiral dataset with known parameters |
| `3_spiral_metadata.json` | Generation parameters for the 3-spiral dataset |

**Regenerating golden datasets:**

```bash
python juniper_data/tests/fixtures/generate_golden_datasets.py
```

Golden datasets should only be regenerated when the generator algorithm intentionally changes. See `tests/fixtures/golden_datasets/README.md` for details.

---

## Coverage

### Running Coverage

```bash
# Terminal report with missing lines
pytest --cov=juniper_data --cov-report=term-missing

# HTML report (detailed, browsable)
pytest --cov=juniper_data --cov-report=html
# Then open: htmlcov/index.html

# XML report (for CI/Codecov integration)
pytest --cov=juniper_data --cov-report=xml:coverage.xml

# JSON report (for check_module_coverage.py)
pytest --cov=juniper_data --cov-report=json:reports/coverage.json

# All reports at once
pytest --cov=juniper_data --cov-report=term-missing --cov-report=html --cov-report=xml:coverage.xml
```

### Thresholds

| Threshold | Value | Enforcement |
|-----------|-------|-------------|
| **Aggregate** | 80% | `pyproject.toml` `fail_under`, CI env var `COVERAGE_FAIL_UNDER` |
| **Per-module** | 85% | `scripts/check_module_coverage.py` |
| **Branch coverage** | Enabled | `pyproject.toml` `branch = true` |

### Coverage Script

The `scripts/check_module_coverage.py` script provides fine-grained per-module coverage enforcement:

```bash
# Check from existing .coverage file (CI mode)
python scripts/check_module_coverage.py

# Run tests first, then check (pre-push mode)
python scripts/check_module_coverage.py --run-tests
```

The script:

1. Runs pytest with coverage (if `--run-tests` is passed)
2. Generates a JSON coverage report
3. Checks each source module against the 85% threshold
4. Checks aggregate coverage against 80% (or `COVERAGE_FAIL_UNDER` env var)
5. Detects test file leakage into the source coverage
6. Reports pass/fail with detailed per-module breakdown

### Exclusion Patterns

Lines matching these patterns are excluded from coverage measurement:

```python
"pragma: no cover"        # Explicit exclusion pragma
"def __repr__"            # Repr methods
"raise AssertionError"    # Assertion errors
"raise NotImplementedError"  # Abstract method stubs
"if __name__ == .__main__.:" # Main guard
"if TYPE_CHECKING:"       # Type checking imports
"@abstractmethod"         # Abstract methods
"^\\s*pass\\s*$"          # Pass statements
```

Source files excluded from coverage:

- `*/tests/*` -- test files themselves
- `*/__pycache__/*` -- cache directories
- `*/data/*` -- data files
- `*/logs/*` -- log files

---

## Performance Benchmarks

Benchmarks use `pytest-benchmark` and are **disabled by default** to keep test runs fast.

```bash
# Run benchmarks with timing
pytest juniper_data/tests/performance/ --benchmark-enable -v

# Save benchmark results for regression tracking
pytest juniper_data/tests/performance/ --benchmark-enable --benchmark-autosave

# Compare against saved baseline
pytest juniper_data/tests/performance/ --benchmark-enable --benchmark-compare
```

Benchmark tests are in:

- `test_generator_benchmarks.py` -- measures generator throughput
- `test_storage_benchmarks.py` -- measures storage operation performance

---

## Writing Tests

### Naming Conventions

| Element | Convention | Example |
|---------|-----------|---------|
| Files | `test_<component>.py` | `test_spiral_generator.py` |
| Classes | `Test<ComponentName>` | `TestSpiralGeneration` |
| Functions | `test_<behavior_under_test>` | `test_basic_spiral_generates_correct_shape` |

### Marker Requirements

Every test function or class **must** have at least one scope marker (`unit`, `integration`, or `performance`). Additional component markers are recommended:

```python
import pytest

@pytest.mark.unit
@pytest.mark.generators
class TestSpiralGeneration:
    def test_basic_spiral_generates_correct_shape(self, two_spiral_params):
        """Verify spiral generator produces expected array shapes."""
        ...

@pytest.mark.integration
@pytest.mark.api
class TestAPIWorkflow:
    @pytest.mark.asyncio
    async def test_generate_and_retrieve(self):
        """Full API round-trip: generate dataset, retrieve via endpoint."""
        ...
```

Using `--strict-markers` in pytest config means unknown markers will fail the test run.

### Async Tests

Use `@pytest.mark.asyncio` for async test functions. pytest-asyncio handles event loop creation:

```python
@pytest.mark.integration
@pytest.mark.api
@pytest.mark.asyncio
async def test_health_endpoint(self):
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/v1/health")
        assert response.status_code == 200
```

### Test Organization

- Place unit tests in `tests/unit/`, one file per source module
- Place integration tests in `tests/integration/`, grouped by workflow
- Place benchmark tests in `tests/performance/`
- Use `conftest.py` for shared fixtures; keep test-specific fixtures local
- Use `@pytest.mark.slow` for tests that take more than a few seconds

### Import-cycle tests

A circular import is a property of a **cold interpreter**. Once `juniper_data` is in `sys.modules`, a same-process `import juniper_data.generators.csv_import` succeeds with the cycle fully present.

`test_no_import_cycles.py` therefore runs every assertion in a subprocess (`python -c ...`). An in-process assertion here cannot fail and is worse than no test. Do not "fix" collection of `test_csv_import_generator.py` by pre-importing `juniper_data.api.routes.generators`.

```bash
# Must pass on a file collected first, not only as part of the full suite
pytest juniper_data/tests/unit/test_csv_import_generator.py
pytest juniper_data/tests/unit/test_no_import_cycles.py
```

> See: [REFERENCE.md -- API Package Import Graph](../REFERENCE.md#api-package-import-graph)

### Equities symbol-cap tests

`TestUniverseSymbolCap` (landed with #354) is the pin for APD-DATA-018's equities half. Default is **refusal**: 40 names and no opt-in must raise `InputTooLargeError` with `unit == "symbols"` and `cap == 14`. An authorised cut must write `DatasetMeta.truncation` (`reason=universe_exceeded_symbol_cap`) and the kept prefix must be `sorted(universe)[:14]`.

`test_generate_puts_the_annotation_on_the_returned_arrays` is load-bearing — resolver-only tests stay green if `generate()` drops the channel key. Do not replace these with a live Yahoo/SEC timing assertion. `equities_seq` must keep calling `EquitiesGenerator._resolve_symbols` and attach the same key; a second resolver would need its own pin.

### DatasetMeta `n_val` pins

`compute_shape_meta` is on the create-dataset path for every generator. #358 makes the store able to carry a validation partition: `n_val` is defaulted to `0` (legacy `.meta.json` cannot load a required field), `X_val` is read presence-conditionally, and `n_samples` spans train + val + test. The `y_full`-less classification fallback must stack `y_val`; the pin puts an entire class there so omitting it drops class `1` from the dict.

These pins live in `test_meta_dispatch.py`, on `main` since #358; the three-way *sizing* pins are in `test_split.py`. Generators emit three partitions.

> See: [REFERENCE.md -- DatasetMeta n_val and Three-Partition Counts](../REFERENCE.md#datasetmeta-n_val-and-three-partition-counts)
### Truncation-edge tests

Authorised `csv_import` truncation has three silent failure modes that an ordinary under-cap / over-cap pair will not catch (#372):

- A minified JSON array (`json.dumps(..., separators=(",", ":"))`, no newline) must import complete elements under a byte cap, not raise `No data found in file`.
- A 2-column CSV whose unclosed `"` swallows later lines has no `None` fields; the quote scan must still drop that last row.
- `bind_deployment_defaults` must put the *effective* cap and opt-in in `params.model_dump()` so `generate_dataset_id` does not hash Field defaults. Pin the API cache: a tight then wide deployment cap must not reuse `dataset_id`.

These are on `test_csv_import_generator.py` and `test_api_routes.py`, landed by #372 and #373.

> See: [REFERENCE.md -- CSV Import Truncation Edges](../REFERENCE.md#csv-import-truncation-edges)

---

## Pre-commit Integration

Testing is integrated into the pre-commit workflow at the **pre-push** stage:

```bash
# Install pre-commit hooks (one-time)
pre-commit install
pre-commit install --hook-type pre-push

# The coverage check runs automatically on git push
# To run it manually:
pre-commit run coverage-check --all-files --hook-stage pre-push
```

The coverage check hook runs `python scripts/check_module_coverage.py --run-tests`, which executes the full test suite and enforces both aggregate and per-module thresholds.

Code quality hooks (ruff, mypy, bandit) run on **pre-commit** stage and validate test code as well.

### Rate-limit window tests

`test_configured_window_reaches_the_live_rate_limiter` is the load-bearing APD-DATA-033 pin. A `Settings` field that parses but is never passed to `create_app` is the original defect — the two earlier arms stay green against that bug. Do not delete the live-limiter arm to "simplify" the settings tests. `test_check_resets_after_window_expiry` pins the expiry comparison (`now - window_start >= window`).

---

## Troubleshooting

**Tests not discovered**: Verify your test files match `test_*.py`, classes match `Test*`, and functions match `test_*`.

**`ModuleNotFoundError`**: Ensure you installed in editable mode: `pip install -e ".[all]"`. The `pythonpath = ["."]` setting in pyproject.toml requires this.

**Timeout failures**: Default is 60 seconds per test. For slow tests, either mark them `@pytest.mark.slow` or override with `pytest --timeout=120`.

**Marker errors with `--strict-markers`**: Only use markers defined in `pyproject.toml`. See [Testing Reference](TESTING_REFERENCE.md#pytest-markers) for the full list.

**Benchmark noise**: Run benchmarks in isolation (`pytest juniper_data/tests/performance/ --benchmark-enable`) to minimize interference from other tests.

**Coverage below threshold**: Run `python scripts/check_module_coverage.py --run-tests` to see per-module breakdown and identify which modules need more tests.

**Deprecation warnings from dependencies**: These are filtered by default via `filterwarnings` in pyproject.toml for uvicorn, httpx, and pydantic.

**`ImportError: cannot import name 'VERSION'` collecting `test_csv_import_generator.py` in isolation**: `api/__init__.py` eagerly imported `create_app`, which loaded every route while `csv_import` was still initialising (#316 / #333). Keep `create_app` lazy. Do not pre-import `api.routes.generators` as a collection workaround. Pin: `test_no_import_cycles.py` (subprocess). See [REFERENCE.md -- API Package Import Graph](../REFERENCE.md#api-package-import-graph).

**`csv_import` over-cap must be 422, not 500, and must not be silent.** Default is refusal (`InputTooLargeError`). An authorised prefix must land `DatasetMeta.truncation` as a dict (`None` means complete). Cuts land on a record boundary; a JSONL corrupt line mid-file still raises. A request `max_bytes` must not raise the deployment ceiling; `stat` must not be the bound. Pins: `TestCsvImportByteCap` in `test_csv_import_generator.py` (including `test_request_cannot_RAISE_the_deployment_cap` and `test_a_lying_stat_does_not_bypass_the_cap`), plus the three `test_api_routes.py` APD-DATA-018 cases. Contract: [CSV Import Byte Cap](../REFERENCE.md#csv-import-byte-cap).

---

## End of Testing Manual
