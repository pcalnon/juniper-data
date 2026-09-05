# Testing Reference

## juniper-data Test Configuration, Markers, and Fixtures

**Version:** 0.4.3
**Status:** Active
**Last Updated:** September 5, 2026
**Project:** Juniper - Dataset Generation Service

---

## Table of Contents

- [Pytest Markers](#pytest-markers)
- [Pytest Configuration](#pytest-configuration)
- [Fixtures Reference](#fixtures-reference)
- [Artifact streaming pins](#artifact-streaming-pins)
- [Coverage Configuration](#coverage-configuration)
- [Empty-train shape-meta pins](#empty-train-shape-meta-pins)
- [Postgres schema-derivation pins](#postgres-schema-derivation-pins)
- [Test Dependencies](#test-dependencies)
- [Command Reference](#command-reference)
- [Import-cycle pins](#import-cycle-pins)
- [File Structure Reference](#file-structure-reference)
- [CSV Import Byte-Cap Tests](#csv-import-byte-cap-tests)

- [Equities symbol-cap pins](#equities-symbol-cap-pins)
- [Warning Filters](#warning-filters)
- [DatasetMeta n_val pins](#datasetmeta-n_val-pins)

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

Dataset dictionaries contain keys: `X_train`, `y_train`, `X_val`, `y_val`, `X_test`, `y_test`, `X_full`, `y_full` (all `float32`).

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

## Artifact streaming pins

`juniper_data/tests/unit/test_artifact_streaming.py` (on `main`; APD-DATA-016 / #313). Bytes round-tripping is not the decisive arm — a whole-file read round-trips too.

| Test | Property |
|------|----------|
| `test_small_chunk_size_yields_many_chunks` | LocalFS is incremental; one chunk means the artifact was materialised whole |
| `test_chunk_size_is_honoured` | every chunk but the last equals `chunk_size` |
| `test_bytes_are_identical_to_the_materialised_read` | streamed bytes match `get_artifact_bytes` |
| `test_inheriting_backend_yields_exactly_one_chunk` | InMemory (base default) is an honest whole read |
| `test_local_fs_returns_none_not_a_generator` / `test_default_returns_none_not_a_generator` | missing dataset is `None` from the call, not a generator |
| `test_binary_media_types.py` | `BINARY_MEDIA_TYPE == "application/zip"`; no inline media-type literals |

Moving the LocalFS existence check into `_chunks` fails the eager-absence arms. Reverting the LocalFS override to a whole read fails the two chunking arms.

> See: [REFERENCE.md -- Artifact Streaming](../REFERENCE.md#artifact-streaming)

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

## Empty-train shape-meta pins

`juniper_data/tests/unit/test_meta_dispatch.py` (juniper-data#340). `n_features` is `X_train.shape[-1]` even when `n_train == 0`. The existing 3-D trailing-axis test only covers non-empty train.

| Test | Property |
|------|----------|
| `test_empty_train_2d_uses_trailing_feature_axis` | `(0, 5)` reports `n_features=5`, not 2 |
| `test_empty_train_3d_uses_trailing_feature_axis` | `(0, 7, 3)` reports `n_features=3`, not lookback 7, not 2 |
| `test_empty_train_classification_reads_n_classes_from_y_test` | empty-train 4-class artifact reports `n_classes=4` from `y_test` |

`main` computes `n_features = int(x_train.shape[-1]) if (n_train > 0 or x_train.ndim >= 2) else 2` (#365). Note the `else 2` SURVIVES, for rank < 2 only -- a 1-D empty array has no meaningful trailing axis, and `shape[-1]` would report `0`. The mutation that must go red is dropping the `or x_train.ndim >= 2` disjunct, not "restoring `else 2`". These pins are on `main`.

> See: [REFERENCE.md -- Empty-Train Shape Metadata](../REFERENCE.md#empty-train-shape-metadata)

---

## Postgres schema-derivation pins

`juniper_data/tests/unit/test_postgres_schema_derivation.py` (juniper-data#343). DDL, upsert, update, and both row mappers derive from `DatasetMeta.model_fields`. The mappers are pure — this file needs no database. These pins are on `main` since #343.

| `test_every_model_field_has_a_column` / `test_upsert_names_every_column` / `test_update_names_every_mutable_column` | every `DatasetMeta` field is in the DDL, INSERT, and UPDATE |
| `test_n_classes_and_class_distribution_are_nullable` | regression INSERT is not rejected |
| `test_meta_survives_a_round_trip` | `_row_to_meta(_meta_to_row(m)) == m` for classification and regression/sequence |
| `test_a_row_from_an_older_table_still_loads` | missing columns fall back to the model default |
| `test_a_not_null_added_column_always_carries_a_default` | migration-safe ALTER |
| `test_a_hostile_table_name_is_refused` | `_safe_identifier` rejects injection-shaped names |
| `test_no_python_comment_leaks_into_generated_sql` | `# nosec` cannot leak into SQL |

Re-introducing a hand-written column list, `ADD COLUMN ... NOT NULL` without DEFAULT, or `json.dumps(None)` is expected to fail these pins.

> See: [REFERENCE.md -- Postgres Model-Derived Schema](../REFERENCE.md#postgres-model-derived-schema)


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

## Import-cycle pins

`juniper_data/tests/unit/test_no_import_cycles.py` (juniper-data#316 / #333). Every assertion is a **subprocess**. A cycle is a property of a cold interpreter; an in-process import succeeds once `sys.modules` is populated.

| Test | Property |
|------|----------|
| `test_subpackage_imports_first_in_a_cold_interpreter` | Each of the 16 registered generator subpackages imports standalone |
| `test_csv_import_public_names_resolve` | `VERSION` / `CsvImportGenerator` / `CsvImportParams` / `get_schema` resolve (the failure was on a **name**, not the module object) |
| `test_importing_settings_does_not_pull_in_the_routes` | `import juniper_data.api.settings` leaves `api.routes.generators` out of `sys.modules` — the property that actually breaks the cycle |
| `test_create_app_is_still_importable_from_the_package` | `from juniper_data.api import create_app` still works (lazy, not removed) |
| `test_settings_are_still_eager` | `Settings` / `get_settings` remain eager |
| `test_unknown_attribute_still_raises_attribute_error` | Module `__getattr__` does not swallow typos |

Reverting `api/__init__.py` to `from .app import create_app` is expected to fail the `csv_import` standalone import, its public-name resolution, and the routes-not-loaded property — and nothing else in this file.

> See: [REFERENCE.md -- API Package Import Graph](../REFERENCE.md#api-package-import-graph)

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

## CSV Import Byte-Cap Tests

On-disk `csv_import` sources are bounded (`APD-DATA-018` / #326). Tests must pin **refusal** and **loud truncation**, not a silent prefix.

| Must | Must not |
|------|----------|
| Over-cap without opt-in raises `InputTooLargeError` (a `ValueError`); the route answers **422** not 500 | Let an oversized source reach the bare `except Exception` and surface as 500 |
| Authorised truncation writes `DatasetMeta.truncation` as a dict; `records_imported` equals `X_full.shape[0]` | Store `{}` or omit the field for a partial dataset — `None` means complete |
| Cut on a record boundary (CSV quoted-newline drop, JSON array `raw_decode`, JSONL **final** line only) | Parse a mid-record byte slice; drop a corrupt JSONL line that is not the last line |
| Under-cap result has **no** `"truncation"` key | Put `{"truncated": false}` on a complete dataset |
| Effective cap is `min(request, deployment)` — a huge request `max_bytes` still refuses | Let `model_fields_set` make the request win outright (the DoS bound would be caller-controlled) |
| The **read** enforces the bound (`cap_bytes + 1`); `stat` is only a cheap pre-check | Trust `stat().st_size` then read to EOF (FIFO `st_size == 0`, TOCTOU growth) |
| Cap is `gt=0`; `_read_capped_bytes` refuses `limit <= 0` | Accept a negative cap — Python `read(-1)` is unbounded |

Pins: `TestCsvImportByteCap` in `tests/unit/test_csv_import_generator.py`; `test_create_dataset_over_byte_cap_returns_422_not_500` (and siblings) in `test_api_routes.py`. Contract: [CSV Import Byte Cap](../REFERENCE.md#csv-import-byte-cap).

## Equities symbol-cap pins

Pins for APD-DATA-018's equities half (land with #354). Live in `juniper_data/tests/unit/test_equities_generator.py` (`TestUniverseSymbolCap`). No live Yahoo/SEC.

| Test | Asserts | Mutation that must go red |
|------|---------|---------------------------|
| `test_oversized_universe_is_refused_by_default` | 40 names, no opt-in → `InputTooLargeError` (cap 14, unit `symbols`) | Restore the silent slice, or default `allow_truncation` on |
| `test_opt_in_truncates_and_annotates` | authorised cut → 14 names + `universe_exceeded_symbol_cap` | Bound without the descriptor |
| `test_the_kept_prefix_is_deterministic` | reversed dict → same `sorted(universe)[:14]` | Iteration-order prefix |
| `test_request_cannot_RAISE_the_deployment_cap` | `max_symbols=9999` / `None` still clamp to 14 | Request wins over the ceiling |
| `test_generate_puts_the_annotation_on_the_returned_arrays` | channel key on `generate()` output; `records_imported` is a real row count | Resolver annotates, `generate()` drops the key |
| `test_generate_refuses_an_oversized_universe` | `generate()` itself raises | Only `_resolve_symbols` refuses |
| `test_generate_omits_the_key_entirely_when_nothing_was_cut` | under-cap has no `"truncation"` key | Write `{}` for complete |
| `test_default_cap_matches_the_measured_budget` | constant is 14; opt-in default is `false` | Move the number without re-measuring |

`equities_seq` reuses `_resolve_symbols` and attaches the same channel key. Do not add a second resolver without a matching pin.

---

## Warning Filters

Configured in `pyproject.toml` `[tool.pytest.ini_options].filterwarnings`:

| Filter | Pattern | Reason |
|--------|---------|--------|
| `ignore::DeprecationWarning` | `uvicorn.*` | Known uvicorn deprecation warnings |
| `ignore::DeprecationWarning` | `httpx.*` | Known httpx deprecation warnings |
| `ignore::PendingDeprecationWarning` | `pydantic.*` | Known pydantic pending deprecation warnings |

---

## Rate-limit window pins

Pins for APD-DATA-033 (`Settings.rate_limit_window_seconds` → live `RateLimiter.window`).

| Test | File | Asserts | Mutation that must go red |
|------|------|---------|---------------------------|
| `test_window_default_matches_the_limiter_constructor_default` | `test_api_settings.py` | Setting default is `DEFAULT_RATE_LIMIT_WINDOW_SECONDS` | A second literal that happens to be 60 |
| `test_window_is_settable_from_the_environment` | same | env `300` parses | Field not bound to `JUNIPER_DATA_RATE_LIMIT_WINDOW_SECONDS` |
| `test_configured_window_reaches_the_live_rate_limiter` | same | `create_app` limiter `.window == 300`, `.limit == 7` | Field exists but `app.py` still omits `window_seconds=` |
| `test_window_property_returns_configured_seconds` | `test_security.py` | Constructor window is readable | Property ignores `self._window` |
| `test_check_resets_after_window_expiry` | same | Count resets after the window | Expiry comparison dropped or inverted |

The failed-auth throttle (`DEFAULT_FAILED_AUTH_WINDOW_SECONDS`) is a different object. Do not retarget these pins at it.

## DatasetMeta `n_val` pins

`juniper_data/tests/unit/test_meta_dispatch.py` (juniper-data#358). The store carries a validation partition and generators emit one. `n_val` must stay defaulted (`0`) or every `.meta.json` written before the change fails to load.

| Test | Property |
|------|----------|
| `test_val_partition_absent_reports_zero` | Two-partition artifact: `n_val=0`, `n_samples=n_train+n_test` |
| `test_val_partition_counted_in_shape_meta` | `n_samples` is train + val + test (`6+3+2=11`) |
| `test_class_distribution_without_y_full_includes_val` | A class that lives only in `y_val` is counted |
| `test_class_distribution_prefers_y_full_when_present` | `y_full` still wins when present |
| `test_dataset_meta_n_val_is_defaulted` | Field is not required; default is `0` |

Reverting the shape-count and classification-fallback fixes is expected to fail the two behavioural tests and leave the other three green. These pins are on `main` since #358; the three-way sizing pins are in `test_split.py`.

> See: [REFERENCE.md -- DatasetMeta n_val and Three-Partition Counts](../REFERENCE.md#datasetmeta-n_val-and-three-partition-counts)

---

**Last Updated:** September 5, 2026
**Version:** 0.4.3
**Maintainer:** Paul Calnon
