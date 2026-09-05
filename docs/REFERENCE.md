# Juniper Data Reference

**Version:** 0.4.3
**Status:** Active
**Last Updated:** September 5, 2026
**Project:** Juniper Data - Dataset Generation Service

---

## Table of Contents

- [API Reference](#api-reference)
- [DatasetMeta n_val and Three-Partition Counts](#datasetmeta-n_val-and-three-partition-counts)
- [Configuration Reference](#configuration-reference)
- [Rate-Limit Window](#rate-limit-window)
- [Storage Backend Notes](#storage-backend-notes)
- [Postgres Model-Derived Schema](#postgres-model-derived-schema)
- [Command Reference](#command-reference)
- [Test Reference](#test-reference)
- [Code Quality Tools](#code-quality-tools)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Error Codes](#error-codes)
- [CSV Import Byte Cap](#csv-import-byte-cap)
- [Project Architecture Reference](#project-architecture-reference)
- [API Package Import Graph](#api-package-import-graph)
- [API Design Reference](#api-design-reference)
- [Empty-Train Shape Metadata](#empty-train-shape-metadata)
- [Storage Backend Reference](#storage-backend-reference)
- [CSV Import Truncation Edges](#csv-import-truncation-edges)
- [Prometheus Collector Reference](#prometheus-collector-reference)
- [Artifact Streaming](#artifact-streaming)
- [Docker Reference](#docker-reference)
- [Equities Symbol Cap](#equities-symbol-cap)
- [CI/CD Pipeline Reference](#cicd-pipeline-reference)
- [Additional Resources](#additional-resources)

---

## API Reference

Full REST API documentation is in [JUNIPER_DATA_API.md](api/JUNIPER_DATA_API.md).

### Quick Endpoint Reference

| Endpoint | Method | Description | Auth Required |
|----------|--------|-------------|---------------|
| `/v1/health` | GET | Health check | No |
| `/v1/health/live` | GET | Liveness probe | No |
| `/v1/health/ready` | GET | Readiness probe | No |
| `/v1/generators` | GET | List generators | Yes* |
| `/v1/generators/{name}/schema` | GET | Generator schema | Yes* |
| `/v1/datasets` | POST | Create dataset | Yes* |
| `/v1/datasets` | GET | List datasets | Yes* |
| `/v1/datasets/filter` | GET | Filter datasets | Yes* |
| `/v1/datasets/versions` | GET | List versions by dataset name | Yes* |
| `/v1/datasets/latest` | GET | Get latest version by name | Yes* |
| `/v1/datasets/stats` | GET | Dataset statistics | Yes* |
| `/v1/datasets/batch-create` | POST | Batch create datasets | Yes* |
| `/v1/datasets/batch-delete` | POST | Batch delete datasets | Yes* |
| `/v1/datasets/batch-tags` | PATCH | Batch tag updates | Yes* |
| `/v1/datasets/batch-export` | POST | Batch export datasets | Yes* |
| `/v1/datasets/cleanup-expired` | POST | Cleanup expired datasets | Yes* |
| `/v1/datasets/{id}` | GET | Dataset metadata | Yes* |
| `/v1/datasets/{id}` | DELETE | Delete dataset | Yes* |
| `/v1/datasets/{id}/artifact` | GET | Download NPZ | Yes* |
| `/v1/datasets/{id}/preview` | GET | Preview JSON | Yes* |
| `/v1/datasets/{id}/tags` | PATCH | Update dataset tags | Yes* |

*Auth required only when `JUNIPER_DATA_API_KEYS` is set.

### NPZ Artifact Keys

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `X_train` | `(n_train, n_features)` | `float32` | Training features |
| `y_train` | `(n_train, n_classes)` | `float32` | Training labels (one-hot) |
| `X_test` | `(n_test, n_features)` | `float32` | Test features |
| `y_test` | `(n_test, n_classes)` | `float32` | Test labels (one-hot) |
| `X_full` | `(n_samples, n_features)` | `float32` | Full dataset features |
| `y_full` | `(n_samples, n_classes)` | `float32` | Full dataset labels (one-hot) |

Live generator artifacts still write those six keys. `X_val` / `y_val` are carried by the store and by `DatasetMeta` (`n_val` defaults to `0` so a legacy two-partition `.meta.json` still loads). Generators emit them as of the three-partition work. See [DatasetMeta n_val and Three-Partition Counts](#datasetmeta-n_val-and-three-partition-counts).

---

## DatasetMeta `n_val` and Three-Partition Counts

The store carries a validation partition, and generators now emit one. `n_val` remains **defaulted** to `0` so an artifact written before the change still loads; a `0` therefore means "two-partition, pre-change", not "no validation rows were asked for".

This is the storage and metadata half of the val-first sequence (juniper-data#358), after the sizing primitives already on `main` (#353) and the client `NPZ_SPLITS` work (juniper-data-client#187). Generator wiring is a later PR; the REST size-knob vocabulary is not settled — do not invent `val_ratio` on generator params.

### Why `n_val` is defaulted (R-3)

`DatasetMeta.n_val: int = 0` in `core/models.py`. Existing `.meta.json` files are loaded with `DatasetMeta(**meta_dict)` from JSON written before the third partition existed (`storage/local_fs.py`; Redis and Postgres do the same). A required field with no default would make every stored artifact unreadable. `0` is the honest count for an artifact with no validation rows — not a placeholder.

### `compute_shape_meta` counts three partitions

`core/meta.py` (called from `POST /v1/datasets` / `batch-create` for every generator):

- `n_val = len(arrays["X_val"]) if "X_val" in arrays else 0` — presence-conditional. A two-partition artifact predating the third partition reports `0` rather than failing.
- `n_samples = n_train + n_val + n_test` — not train + test. A 6 / 3 / 2 artifact reports `n_samples=11`.
- The route passes `n_val=shape_meta["n_val"]` onto the constructed `DatasetMeta`.

This landed with #358: `n_samples` is `n_train + n_val + n_test`, and `DatasetMeta.n_val` is on `main` (`juniper_data/core/models.py`).

### Classification `class_distribution` without `y_full`

`_classification_meta` still prefers `y_full` when the artifact carries it. When `y_full` is absent, the fallback must stack **every partition present** (`y_train`, optional `y_val`, `y_test`). Omitting `y_val` silently drops those rows from the distribution — and only on artifacts without `y_full`, which is the path design decision 11 makes the normal case.

The pin puts an **entire class** in `y_val`, so the buggy stack (`y_train` + `y_test` only) drops class `1` from the dict outright rather than shifting a count. A test that only changed a count would be easier to mis-read as noise.

### Postgres

`n_val` is added to `_SQL_DEFAULTS` as `"0"` because it is non-nullable *with* a default — `ADD COLUMN … NOT NULL` against a populated table fails without one. The column itself is emitted by `build_schema_sql` iterating `DatasetMeta.model_fields` (juniper-data#343). Do not add a sixth hand-written `ALTER`. `test_postgres_schema_derivation.py` already asserts against `model_fields` rather than a hardcoded field count, so the new column is covered without a new test.

### Sizing primitives already on `main` (#353)

`core/split.py` already has additive three-way sizing. Generators do **not** call these yet; they still use two-way `split_data` / `shuffle_and_split`.

| Helper | Contract |
|--------|----------|
| `partition_row_counts(n_train, val_percent=40, test_percent=30)` | Train count is honoured literally. Val and test are *additional* rows, as percentages of train. Default `n_train=1000` yields `1000 / 400 / 300`. |
| `split_three_way` / `shuffle_and_split_three_way` | Contiguous, index-disjoint cuts. Rows beyond `n_train + n_val + n_test` are left unused, not folded into a partition. |

Percentages are absolute **rows** of the realised dataset, never per-spiral / per-quadrant / per-class units. Asking a generator for N+M rows does not reproduce the first N rows it would have produced for N — the train *count* is preserved; the train *content* is not.

### What not to do

- Do not make `n_val` required. Legacy `.meta.json` cannot load.
- Do not assume `X_val` is always in the NPZ. The read is presence-conditional.
- Do not compute `n_samples` as `n_train + n_test` once `X_val` exists.
- Do not stack only train + test for `class_distribution` when `y_full` is absent.
- Do not claim generators emit `X_val` / `y_val`. They do not. Live HTTP artifacts remain two-partition; `n_val` reads `0`.
- Do not invent the public REST size-knob vocabulary. That gates the generator half.
- Do not re-document the Postgres field-list derivation (owned by #344) or the empty-train `n_features` trailing-axis contract (owned by #341).

### Pins

These live in `tests/unit/test_meta_dispatch.py`, on `main` since #358. Reverting both shape-count and classification-fallback fixes is expected to fail `test_val_partition_counted_in_shape_meta` and `test_class_distribution_without_y_full_includes_val`; the other three stay green under that mutation because they do not touch what it breaks.

| Test | Property |
|------|----------|
| `test_val_partition_absent_reports_zero` | Two-partition artifact: `n_val=0`, `n_samples=n_train+n_test` |
| `test_val_partition_counted_in_shape_meta` | `n_samples` is train + val + test (`6+3+2=11`), not train + test |
| `test_class_distribution_without_y_full_includes_val` | A class that lives only in `y_val` is counted (`{"0": 4, "1": 3}`) |
| `test_class_distribution_prefers_y_full_when_present` | `y_full` still wins when the artifact carries it |
| `test_dataset_meta_n_val_is_defaulted` | Field is not required; default is `0` |

Three-way sizing pins are already on `main` in `tests/unit/test_split.py` (`partition_row_counts`, `split_three_way`, `shuffle_and_split_three_way`).

---

## Configuration Reference

### Environment Variables

All environment variables use the `JUNIPER_DATA_` prefix and are managed by Pydantic `BaseSettings` in `juniper_data/api/settings.py`.

#### Service Settings

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `JUNIPER_DATA_HOST` | string | `127.0.0.1` | Listen address |
| `JUNIPER_DATA_PORT` | int | `8100` | Service port |
| `JUNIPER_DATA_STORAGE_PATH` | string | `./data/datasets` | Artifact storage directory |
| `JUNIPER_DATA_LOG_LEVEL` | string | `INFO` | Log level: DEBUG, INFO, WARNING, ERROR, CRITICAL |

#### Security Settings

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `JUNIPER_DATA_API_KEYS` | JSON list | *(none)* | `["key1", "key2"]` -- enables API key auth |
| `JUNIPER_DATA_RATE_LIMIT_ENABLED` | bool | `true` | Enable request rate limiting |
| `JUNIPER_DATA_RATE_LIMIT_REQUESTS_PER_MINUTE` | int | `60` | Max requests **per window** per client (name is historical) |
| `JUNIPER_DATA_RATE_LIMIT_WINDOW_SECONDS` | int | `60` | Length of that window (`Settings.rate_limit_window_seconds`) |
| `JUNIPER_DATA_CORS_ORIGINS` | JSON list | `["*"]` | Allowed CORS origins |

#### Observability Settings

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `JUNIPER_DATA_METRICS_ENABLED` | bool | `false` | Enable Prometheus metrics endpoint |
| `JUNIPER_DATA_SENTRY_DSN` | string | *(none)* | Sentry DSN for error tracking |

#### CSV / JSON import (`csv_import`)

On-disk sources, not the HTTP body. `file_path` is resolved inside `JUNIPER_DATA_IMPORT_DIR`. The 10 MB `RequestBodyLimitMiddleware` cap is a separate limit on the JSON request. See [CSV Import Byte Cap](#csv-import-byte-cap).

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `JUNIPER_DATA_IMPORT_DIR` | string | `/data/imports` | Root directory for `csv_import` files. Paths that resolve outside it are refused (path traversal). |
| `JUNIPER_DATA_CSV_IMPORT_MAX_BYTES` | int | `134217728` (128 MiB) | Deployment **ceiling**. Effective cap is `min(request, this)`. Must be `> 0` (`gt=0`): Python `read(n)` with `n < 0` means "read everything". |
| `JUNIPER_DATA_CSV_IMPORT_ALLOW_TRUNCATION` | bool | `false` | Deployment-wide opt-in to a partial import when the source exceeds the cap. Logical OR with the request `allow_truncation` flag; a request cannot opt *out* of a deployment-wide opt-in. |

#### Integration Variables (Used by Consumers)

| Variable | Used By | Default | Description |
|----------|---------|---------|-------------|
| `JUNIPER_DATA_URL` | juniper-cascor, juniper-canopy | `http://localhost:8100` | URL for this service |
| `JUNIPER_DATA_API_KEY` | juniper-cascor | *(none)* | API key for authentication |

### pyproject.toml Tool Configuration

#### Ruff (Linter + Formatter)

```toml
[tool.ruff]
line-length = 120
target-version = "py311"

[tool.ruff.lint]
select = ["E", "W", "F", "B", "C4", "I", "UP", "SIM", "T20"]
```

| Rule Set | Purpose |
|----------|---------|
| `E`, `W` | pycodestyle errors and warnings |
| `F` | pyflakes |
| `B` | flake8-bugbear |
| `C4` | flake8-comprehensions |
| `I` | isort (import sorting) |
| `UP` | pyupgrade |
| `SIM` | flake8-simplify |
| `T20` | flake8-print (catches print statements) |

#### Pytest

```toml
[tool.pytest.ini_options]
testpaths = ["juniper_data/tests"]
pythonpath = ["."]
addopts = ["-ra", "-q", "--strict-markers", "--strict-config", "--tb=short", "--benchmark-disable"]
timeout = 60
timeout_method = "signal"
```

#### Coverage

```toml
[tool.coverage.run]
source_pkgs = ["juniper_data"]
branch = true

[tool.coverage.report]
fail_under = 80
show_missing = true
```

#### mypy

```toml
[tool.mypy]
python_version = "3.14"
warn_return_any = false
warn_unused_configs = true
ignore_missing_imports = true
```

#### Bandit

```toml
[tool.bandit]
exclude_dirs = ["tests", "reports", "logs", "htmlcov", "data"]
skips = ["B101", "B311"]
```

- `B101`: Skip assert checks (used extensively in tests)
- `B311`: Skip random usage warnings (used for data generation)

---

## Rate-Limit Window

`RateLimiter` is a fixed-window, in-memory, single-process quota. It has always taken three constructor knobs — `enabled`, `requests_per_minute` (the count), and `window_seconds` (the duration) — and uses the window in three places: `self._window`, the `TTLCache` TTL, and the `window` property.

Until APD-DATA-033 / #297, `create_app` passed only the first two, so the window was pinned to `DEFAULT_RATE_LIMIT_WINDOW_SECONDS` (60). That constant was never unwired; the gap was the missing operator-facing setting.

`Settings.rate_limit_window_seconds` is now overridable as `JUNIPER_DATA_RATE_LIMIT_WINDOW_SECONDS`. The setting default is the same object as the `RateLimiter` constructor default, not a second literal.

### What the names actually mean

`rate_limit_requests_per_minute` is the **count per window**, not "per 60 seconds" when the window is not 60. Setting `JUNIPER_DATA_RATE_LIMIT_WINDOW_SECONDS=300` and leaving the count at 60 allows 60 requests every five minutes, not 60/min.

There is no `Field(gt=0)` on the window (unlike `csv_import_max_bytes`). Keep it a positive integer. `0` makes `now - window_start >= self._window` true on every check, so the window never holds.

### How a request is counted

1. Exempt paths (`EXEMPT_PATHS`: `/v1/health`, `/v1/health/live`, `/v1/health/ready`, `/metrics`, `/metrics/`) skip **both** this limiter and the failed-auth throttle. `/docs` / `/redoc` are not mounted when keys are configured; `/openapi.json` is mounted and authenticated like any other route (APD-DATA-024) — it consumes quota.
2. Key: `key:{api_key}` when a key authenticated the request, else `ip:{client_ip}`.
3. First hit in a window: count 1, remaining `limit - 1`, reset `window` seconds.
4. Over the count: HTTP **429** with `X-RateLimit-Limit`, `X-RateLimit-Remaining: 0`, `X-RateLimit-Reset`, and `Retry-After`. Allowed responses get the same three `X-RateLimit-*` headers.
5. Cache: `TTLCache(maxsize=10_000, ttl=window_seconds)`. A one-shot warning fires at 80% occupancy.

Default `rate_limit_enabled` is **true** (`_JUNIPER_DATA_API_RATELIMIT_ENABLED`). Disable with `JUNIPER_DATA_RATE_LIMIT_ENABLED=false`.

### Not this knob

`FailedAuthThrottle` is a separate pre-auth, IP-keyed budget (10 failures / 60 s). It is not wired to `Settings.rate_limit_window_seconds`. It only consumes budget on a **failed** credential. A 429 from the identity-keyed limiter is not recorded as an auth failure.

### Operator usage

```bash
# Default: 60 requests per 60-second window, enabled.
export JUNIPER_DATA_RATE_LIMIT_ENABLED=true
export JUNIPER_DATA_RATE_LIMIT_REQUESTS_PER_MINUTE=60
export JUNIPER_DATA_RATE_LIMIT_WINDOW_SECONDS=60

# Stricter: 7 requests per 5-minute window (the pin in test_configured_window_reaches_the_live_rate_limiter).
export JUNIPER_DATA_RATE_LIMIT_REQUESTS_PER_MINUTE=7
export JUNIPER_DATA_RATE_LIMIT_WINDOW_SECONDS=300
```

In-process only. Multiple uvicorn workers do not share counters. Restarting the process resets every bucket.

### What not to do

- Do not treat `REQUESTS_PER_MINUTE` as a true per-minute rate when the window is not 60 s.
- Do not expect `JUNIPER_DATA_RATE_LIMIT_WINDOW_SECONDS` to change the failed-auth throttle.
- Do not set the window to `0` or a negative value — there is no settings-layer reject.
- Do not assume a second replica sees the first replica's counts.

### Pins (on `main`)

| Test | File | Guards |
|------|------|--------|
| `test_window_default_matches_the_limiter_constructor_default` | `tests/unit/test_api_settings.py` | Setting default is `DEFAULT_RATE_LIMIT_WINDOW_SECONDS`, not a second literal |
| `test_window_is_settable_from_the_environment` | same | `JUNIPER_DATA_RATE_LIMIT_WINDOW_SECONDS=300` parses |
| `test_configured_window_reaches_the_live_rate_limiter` | same | `create_app` passes the window onto the live `RateLimiter` (count 7, window 300) |
| `test_window_property_returns_configured_seconds` | `tests/unit/test_security.py` | Constructor window is readable |
| `test_check_resets_after_window_expiry` | same | Count resets once `now - window_start >= window` |
| `test_call_raises_429_when_over_limit` | same | Over-limit path is 429 |

---

## Storage Backend Notes

### `PostgresDatasetStore` Write Consistency Model

`PostgresDatasetStore` stores metadata in PostgreSQL (`datasets` table) and artifacts as NPZ files on disk (`{artifact_path}/{dataset_id}.npz`).

When saving, it uses a staged write path to keep metadata and artifacts aligned:

1. Write NPZ bytes to a temp file: `{dataset_id}.npz.tmp`.
2. Start DB transaction and acquire advisory lock on `dataset_id`.
3. Resolve versioning metadata:
   - If `dataset_id` already exists, preserve stored `dataset_name` and `dataset_version`.
   - Else, if `dataset_name` is set, acquire advisory lock on `dataset_name` and assign `MAX(dataset_version) + 1`.
4. Upsert metadata row.
5. Atomically replace temp artifact with final `.npz`.
6. Commit transaction.

If artifact finalization fails, the transaction is rolled back and the temp file is removed. This prevents metadata-only commits (metadata/artifact split-brain).

### PostgreSQL Versioning Rules

| Scenario | Behavior |
|----------|----------|
| New persisted dataset with `dataset_name` | Allocates next integer version for that name in DB |
| Existing `dataset_id` upsert | Keeps canonical stored `dataset_name` + `dataset_version` |
| Request without `dataset_name` | Leaves `dataset_version` as `NULL` |
| Non-persisted named create (`persist=false`) | API previews next version but does not reserve or store it |

### Operational Constraints

- `artifact_path` must be writable by the service process.
- Temp and final artifact files must be on the same filesystem for atomic rename semantics.
- Install PostgreSQL backend dependency: `pip install psycopg2-binary`.
- The `datasets` table schema is derived from `DatasetMeta` — see [Postgres Model-Derived Schema](#postgres-model-derived-schema). Do not hand-maintain a column list.

---

## Postgres Model-Derived Schema

`PostgresDatasetStore` (`juniper_data/storage/postgres_store.py`) used to carry **five** hand-maintained copies of `DatasetMeta`'s field list: the DDL, `_meta_to_row`, `_row_to_meta`, the upsert, and the update. Each was transcribed independently, and every one had drifted (#320). #343 derives all five from `DatasetMeta.model_fields`.

`create_app` still constructs `LocalFSDatasetStore`. This store is opt-in via `get_postgres_store()` and is **not** on the serving path — the defects below were latent.

### Single source of truth

| Builder / mapper | What it emits |
|------------------|---------------|
| `build_schema_sql` | `CREATE TABLE IF NOT EXISTS` plus `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` for every model field |
| `build_upsert_sql` | `INSERT ... ON CONFLICT DO UPDATE` naming every column |
| `build_update_sql` | `UPDATE ... WHERE dataset_id = ...` for every mutable column |
| `_meta_to_row` / `_row_to_meta` | write / read of every field |

Adding a field to `DatasetMeta` adds its column, INSERT, UPDATE, and read without a second edit. Do not re-introduce a hand-written column list. Deriving the mappers alone is not enough — the INSERT and UPDATE enumerate their own columns. Coverage assertions in `test_postgres_schema_derivation.py` are written against `DatasetMeta.model_fields`, not a hardcoded field count.

### Nullability and the fields that used to vanish

A column is nullable exactly when the model annotation admits `None`. Before #343:

- `n_classes` and `class_distribution` were `NOT NULL` after WS-1 / #168 made both optional. The first **regression** dataset written would have failed the INSERT.
- Seven fields were dropped on every round trip: `task_type`, `sequence`, `lookback`, `time_unit`, `dt_scaling`, `target_scaling`, `truncation`. Sequence and scaling metadata silently reset to defaults.
- `_row_to_meta` raised `TypeError` on a NULL `class_distribution` (`json.loads(None)`).

JSONB columns: `params`, `class_distribution`, `dt_scaling`, `target_scaling`, `truncation`. TEXT[]: `artifact_formats`, `tags`.

### Migration and code-ahead-of-schema

`SCHEMA_SQL` runs on every init when `auto_create_schema=True` (the default). Every non-PK column is emitted as `ADD COLUMN IF NOT EXISTS`, so a second boot does not error. A `NOT NULL` added column always carries a `DEFAULT` (`task_type`, `sequence`, `access_count`, `artifact_formats`, `tags`) — `ADD COLUMN ... NOT NULL` without one fails against a populated table.

`_row_to_meta` omits a column that is absent or NULL where the model does not admit `None`, so the model's default applies (`task_type="classification"`, `sequence=False`). That is what makes a code-ahead-of-schema deploy survivable in either direction.

`_meta_to_row` keeps Python `None` as SQL NULL. `json.dumps(None)` would store the string `"null"`.

An upsert does **not** overwrite `created_at` (or the PK). A re-save is not a re-creation.

### Identifier safety

The builders interpolate a table name (default `datasets`). Values cannot be parameterised into an identifier position, so `_safe_identifier` admits only `[A-Za-z_][A-Za-z0-9_]*` — no quote, semicolon, comment marker, or whitespace. A hostile name raises `ValueError`. Column names come from `model_fields` and are never request-derived; all values stay driver-bound through `%(name)s`.

### What not to do

- Do not hand-maintain a column list in the DDL, mappers, INSERT, or UPDATE.
- Do not put a `# nosec` comment on the `return f"""` line. That is inside the string and prepends the comment to every generated statement. Bind the string first; annotate the closing line.
- Do not emit `ADD COLUMN ... NOT NULL` without a `DEFAULT`.
- Do not `json.dumps(None)` for a JSONB field.

### Pins

These live in `juniper_data/tests/unit/test_postgres_schema_derivation.py`, on `main` since #343. The mappers are pure functions — the file needs no database.

| Test | Property |
|------|----------|
| `test_every_model_field_has_a_column` / `test_upsert_names_every_column` / `test_update_names_every_mutable_column` | every `DatasetMeta` field is in the DDL, INSERT, and UPDATE |
| `test_n_classes_and_class_distribution_are_nullable` | regression INSERT is not rejected |
| `test_meta_survives_a_round_trip` | `_row_to_meta(_meta_to_row(m)) == m` for classification and regression/sequence |
| `test_a_row_from_an_older_table_still_loads` | missing columns fall back to the model default |
| `test_a_not_null_added_column_always_carries_a_default` | migration-safe ALTER |
| `test_a_hostile_table_name_is_refused` | `_safe_identifier` rejects injection-shaped names |
| `test_no_python_comment_leaks_into_generated_sql` | `# nosec` cannot leak into SQL |

---

## Command Reference

### Service Commands

```bash
# Start development server (with auto-reload)
python -m juniper_data --reload

# Start production server
uvicorn --factory juniper_data.api.app:get_app --host 0.0.0.0 --port 8100

# Start with custom options
python -m juniper_data --host 0.0.0.0 --port 8101 --log-level DEBUG --storage-path /tmp/datasets
```

### Test Commands

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test directory
pytest juniper_data/tests/unit/
pytest juniper_data/tests/integration/
pytest juniper_data/tests/performance/

# Run specific test file
pytest juniper_data/tests/unit/test_spiral_generator.py -v

# Run by marker
pytest -m unit
pytest -m integration
pytest -m performance
pytest -m spiral
pytest -m api
pytest -m generators
pytest -m storage

# Run with coverage
pytest juniper_data/tests/ --cov=juniper_data --cov-report=html --cov-report=term-missing --cov-fail-under=80

# Run performance benchmarks
pytest -m performance --benchmark-enable
```

### Code Quality Commands

```bash
# Lint
ruff check juniper_data

# Lint with auto-fix
ruff check --fix juniper_data

# Format check
ruff format --check juniper_data

# Format (apply)
ruff format juniper_data

# Type checking
mypy juniper_data --ignore-missing-imports

# Security scanning (SAST)
bandit -r juniper_data

# Dependency vulnerability scanning
pip-audit

# Pre-commit hooks (run all)
pre-commit run --all-files
```

### Dependency Management

```bash
# Install development
pip install -e ".[dev]"

# Install with API support
pip install -e ".[api]"

# Install everything
pip install -e ".[all]"

# Regenerate lockfile for Docker
uv pip compile pyproject.toml --extra api --extra observability --extra mnist -o requirements.lock
```

---

## Test Reference

### Test Markers

| Marker | Description | Example |
|--------|-------------|---------|
| `@pytest.mark.unit` | Fast, isolated unit tests | `pytest -m unit` |
| `@pytest.mark.integration` | Full workflow tests with real storage | `pytest -m integration` |
| `@pytest.mark.performance` | Benchmark tests | `pytest -m performance` |
| `@pytest.mark.slow` | Tests > 1 second | `pytest -m slow` |
| `@pytest.mark.spiral` | Spiral generator tests | `pytest -m spiral` |
| `@pytest.mark.api` | API endpoint tests | `pytest -m api` |
| `@pytest.mark.generators` | All generator tests | `pytest -m generators` |
| `@pytest.mark.storage` | Storage backend tests | `pytest -m storage` |

### Test File Map

| Directory | Files | Focus |
|-----------|-------|-------|
| `tests/unit/` | 29 files (~7,000 lines) | Individual component tests |
| `tests/integration/` | 5 files | Full workflow tests |
| `tests/performance/` | 2 files | Benchmark tests |
| `tests/fixtures/` | 1 file | Golden dataset generation |

### Key Test Files

| File | Tests | Description |
|------|-------|-------------|
| `test_spiral_generator.py` | ~40 | Spiral generation, parameters, edge cases |
| `test_storage.py` | ~50 | Storage backend operations |
| `test_api.py` | ~30 | API endpoint integration |
| `test_security.py` | ~20 | Auth, rate limiting, CORS |
| `test_security_boundaries.py` | ~25 | Security boundary tests |
| `test_e2e_workflow.py` | ~15 | End-to-end dataset lifecycle |

### Coverage Configuration

- **Threshold:** 80% fail-under
- **Branch coverage:** Enabled
- **Source:** `juniper_data` package (tests excluded)
- **Report formats:** Terminal (term-missing) + HTML

---

## Code Style Conventions Reference

Relocated verbatim from `AGENTS.md` (P3 of the shared-session-memory plan) so it is read on demand rather than loaded into every session.

### Naming Conventions

**Constants**:

- Uppercase with underscores, prefixed by component: `_DATA_DEFAULT_NOISE`
- Hierarchical naming: `_SPIRAL_GENERATOR_DEFAULT_POINTS`
- Layer-scoped constants live in their layer's `constants.py`:
  - `juniper_data/api/constants.py` — header names, status code defaults, body/rate-limit limits, error message templates
  - `juniper_data/storage/constants.py` — filenames, metadata keys, table/column names, storage size limits
  - `juniper_data/core/constants.py` — encoding strings (`utf-8`), magic numbers, fixed metadata keys
  - `juniper_data/generators/<name>/params.py` — per-generator parameter defaults referenced by Pydantic `Field(default=...)`
- Application code (middleware, security, observability, storage backends, generators, route handlers) imports from these modules; inline literals are reserved for genuinely local one-shot values
- HTTP status codes use `starlette.status` constants instead of magic numbers (`HTTP_404_NOT_FOUND` rather than `404`)

**Classes**:

- PascalCase: `SpiralGenerator`, `DatasetStore`, `LocalFSDatasetStore`

**Methods/Functions**:

- snake_case: `generate_dataset`, `get_configuration`

**Private Members**:

- Single underscore prefix: `_internal_method`, `_private_attribute`

**Dunder Methods**:

- Double underscore: `__init__`, `__repr__`

### Code Formatting

- Line length: 320 characters (configured in `[tool.ruff] line-length` in pyproject.toml)
- Ruff formatter (replaces black) with `ruff>=0.9.0`
- Ruff isort rules for imports (profile: known-first-party = `juniper_data`)
- Quote style: double quotes, LF line endings
- Type hints required for all public methods
- Max cyclomatic complexity: 15

### Documentation

- Docstrings for all public classes and methods
- Google-style docstring format
- Type annotations in signatures, not docstrings

---

---

## Code Quality Tools

### Ruff

Ruff replaces black, isort, flake8, and pyupgrade in a single tool:

| Command | Purpose |
|---------|---------|
| `ruff check juniper_data` | Lint (find issues) |
| `ruff check --fix juniper_data` | Lint with auto-fix |
| `ruff format juniper_data` | Format code |
| `ruff format --check juniper_data` | Check formatting |

### mypy

Static type checking configured for Python 3.14:

```bash
mypy juniper_data --ignore-missing-imports
```

Test code has relaxed settings (`disallow_untyped_defs = false`).

### Bandit

Security-focused static analysis:

```bash
bandit -r juniper_data
```

Excludes test directories. Skips `B101` (assert) and `B311` (random).

### pip-audit

Dependency vulnerability scanning:

```bash
pip-audit
```

### Pre-commit

Git hook manager that runs all quality checks before commits:

```bash
# Install hooks (one-time)
pre-commit install

# Run manually on all files
pre-commit run --all-files
```

---

## Development Workflow Reference

Relocated verbatim from `AGENTS.md` (P3 of the shared-session-memory plan) so it is read on demand rather than loaded into every session.

### Adding New Features

1. Create feature in appropriate module
2. Add Pydantic models for validation
3. Add tests in `tests/unit/` or `tests/integration/`
4. Run security scanning (`bandit -r juniper_data`)
5. Run pre-commit hooks (`pre-commit run --all-files`)
6. Update documentation
7. Run full test suite with coverage

### Adding New Generators

1. Create new subpackage under `generators/` with `__init__.py`, `params.py`, and `generator.py`
2. Implement `params.py` with a Pydantic `GeneratorParams` model
3. Implement `generator.py` with a `@staticmethod generate(params)` method returning `dict[str, np.ndarray]`
4. Register generator in `GENERATOR_REGISTRY` in `api/routes/generators.py`
5. Add unit tests in `tests/unit/test_<generator>_generator.py`
6. Add integration test coverage
7. Run full test suite

---

---

## Project Structure

```
juniper-data/
├── juniper_data/                 # Main package
│   ├── __init__.py               # Package init, version (0.4.2)
│   ├── __main__.py               # CLI entry point (argparse)
│   ├── core/                     # Core functionality
│   │   ├── artifacts.py          # NPZ artifact operations
│   │   ├── dataset_id.py         # Deterministic ID generation
│   │   ├── models.py             # Pydantic data models
│   │   └── split.py              # Train/test splitting
│   ├── generators/               # 8 dataset generators
│   │   ├── spiral/               # Multi-spiral (primary)
│   │   ├── xor/                  # XOR classification
│   │   ├── gaussian/             # Gaussian mixture
│   │   ├── circles/              # Concentric circles
│   │   ├── checkerboard/         # 2D checkerboard
│   │   ├── csv_import/           # CSV/JSON import
│   │   ├── mnist/                # MNIST/Fashion-MNIST
│   │   └── arc_agi/              # ARC-AGI tasks
│   ├── storage/                  # 8 storage backends
│   │   ├── base.py               # Abstract DatasetStore
│   │   ├── local_fs.py           # Local filesystem
│   │   ├── memory.py             # In-memory
│   │   ├── cached.py             # Cached wrapper
│   │   ├── postgres_store.py     # PostgreSQL
│   │   ├── redis_store.py        # Redis
│   │   ├── hf_store.py           # HuggingFace Hub
│   │   └── kaggle_store.py       # Kaggle
│   ├── api/                      # FastAPI REST service
│   │   ├── app.py                # Factory-pattern app
│   │   ├── settings.py           # Pydantic BaseSettings
│   │   ├── middleware.py         # SecurityMiddleware
│   │   ├── security.py           # APIKeyAuth, RateLimiter
│   │   ├── observability.py      # Prometheus, Sentry
│   │   ├── models/               # API response models
│   │   └── routes/               # health, generators, datasets
│   └── tests/                    # Test suite (~9,000 lines)
│       ├── conftest.py           # Shared fixtures
│       ├── unit/                 # 29 test files
│       ├── integration/          # 5 test files
│       ├── performance/          # 2 benchmark files
│       └── fixtures/             # Golden dataset generation
├── docs/                         # Documentation
│   ├── DOCUMENTATION_OVERVIEW.md # This navigation index
│   ├── QUICK_START.md            # 5-minute setup
│   ├── ENVIRONMENT_SETUP.md      # Full environment config
│   ├── USER_MANUAL.md            # Comprehensive usage
│   ├── REFERENCE.md              # This file
│   ├── api/                      # API documentation
│   ├── testing/                  # Testing documentation
│   └── ci_cd/                    # CI/CD documentation
├── pyproject.toml                # Project configuration
├── requirements.lock             # Docker dependency lockfile
├── README.md                     # Project overview
├── AGENTS.md                     # Development guide
├── CHANGELOG.md                  # Version history
└── .pre-commit-config.yaml       # Pre-commit hooks
```

---

## Dependencies

### Core (always installed)

| Package | Version | Purpose |
|---------|---------|---------|
| `numpy` | >= 1.24.0 | Numerical computations, NPZ format |
| `pydantic` | >= 2.0.0 | Data validation, parameter models |
| `python-dotenv` | >= 1.0.0 | .env file loading |

### API (optional: `pip install -e ".[api]"`)

| Package | Version | Purpose |
|---------|---------|---------|
| `fastapi` | >= 0.100.0 | REST framework |
| `uvicorn[standard]` | >= 0.23.0 | ASGI server |
| `pydantic-settings` | >= 2.0.0 | Settings from env vars |

### Test (optional: `pip install -e ".[test]"`)

| Package | Version | Purpose |
|---------|---------|---------|
| `pytest` | >= 7.0.0 | Test framework |
| `pytest-cov` | >= 4.0.0 | Coverage |
| `pytest-timeout` | >= 2.2.0 | Timeout enforcement |
| `pytest-asyncio` | >= 0.21.0 | Async test support |
| `pytest-benchmark` | >= 4.0.0 | Benchmarking |
| `httpx` | >= 0.24.0 | Async HTTP testing |
| `coverage[toml]` | >= 7.0.0 | Coverage reporting |
| `juniper-data-client` | >= 0.3.0 | Client integration tests |

### Dev (optional: `pip install -e ".[dev]"`)

| Package | Version | Purpose |
|---------|---------|---------|
| `ruff` | >= 0.9.0 | Linting + formatting |
| `mypy` | >= 1.0.0 | Type checking |
| `bandit[sarif]` | >= 1.7.9 | Security scanning |
| `pip-audit` | >= 2.7.0 | Vulnerability scanning |
| `pre-commit` | >= 3.0.0 | Git hooks |

### Observability (optional: `pip install -e ".[observability]"`)

| Package | Version | Purpose |
|---------|---------|---------|
| `prometheus-client` | >= 0.20.0 | Metrics export |
| `sentry-sdk[fastapi]` | >= 2.0.0 | Error tracking |

### ARC-AGI (optional: `pip install -e ".[arc-agi]"`)

| Package | Version | Purpose |
|---------|---------|---------|
| `arc-agi` | >= 0.9.0 | ARC-AGI dataset access |

### MNIST (optional: `pip install -e ".[mnist]"`)

| Package | Version | Purpose |
|---------|---------|---------|
| `datasets[vision]` | >= 4.0.0 | MNIST / Fashion-MNIST from the Hugging Face Hub (`[vision]` pulls Pillow for image decode) |

Heavy chain (pyarrow, pandas, pillow, huggingface-hub); deliberately extra-gated. First generation
downloads from the Hub into the HF cache (`HF_HOME`); offline deployments need a seeded cache — see
the README section "MNIST / Fashion-MNIST (optional extra)". The Docker image ships this extra via
`requirements.lock`.

---

## Error Codes

### HTTP Status Codes

| Code | Meaning | Common Cause |
|------|---------|--------------|
| `200 OK` | Success | GET requests |
| `201 Created` | Resource created | POST /v1/datasets |
| `204 No Content` | Deleted | DELETE /v1/datasets/{id} |
| `400 Bad Request` | Invalid parameters | Bad generator params |
| `401 Unauthorized` | Missing/invalid API key | Auth enabled, no key sent |
| `404 Not Found` | Resource not found | Unknown generator or dataset ID |
| `422 Unprocessable Entity` | Validation error, or `csv_import` source over its byte cap | Pydantic schema failure (`detail` is a list), or `InputTooLargeError` (`detail` is a string). See [CSV Import Byte Cap](#csv-import-byte-cap). |
| `429 Too Many Requests` | Rate limited | Exceeded requests/minute |
| `500 Internal Server Error` | Server error | Unexpected exception |

### Error Response Format

```json
{
  "detail": "Error message describing what went wrong"
}
```

Schema `422`s instead carry `detail` as a **list** of per-field objects. The over-cap `csv_import` refusal is a `422` with a **string** `detail` (the `InputTooLargeError` message). Check the type before iterating.

---

## CSV Import Byte Cap

Bound for on-disk `csv_import` sources (`APD-DATA-018`, landed with juniper-data#326). Generation runs **inside the request**, so a source large enough to outlive the client timeout cannot succeed however long the caller waits. The remedy is to bound the input, not to move generation to an async job.

This is **not** the HTTP body limit. `RequestBodyLimitMiddleware` rejects JSON bodies over 10 MB. `csv_import` reads a file under `JUNIPER_DATA_IMPORT_DIR`; `file_path` is relative to that directory (absolute paths outside it fail as path traversal).

### Why 128 MiB

`CSV_IMPORT_DEFAULT_MAX_BYTES` in `juniper_data/core/limits.py` is `128 * 1024 * 1024`. The figure is measured, not round: `util/ad-hoc/2026-09-04_measure_csv_import_throughput.py` timed the whole `generate()` path (parse **and** per-cell float conversion) at a median **14.4 MB/s**. 128 MiB is therefore ~8.9 s of parsing, inside a ~30 s client budget with room for split, checksum, and NPZ persist.

Above that size the binding constraint is memory, not time. `_parse_csv_stream` materialises one Python dict per row before any array exists, so raising the cap without a streaming loader trades a timeout for an OOM.

### Surfaces and precedence

| Surface | Cap (`max_bytes`) | Truncation opt-in (`allow_truncation`) |
|---------|-------------------|----------------------------------------|
| Request params (`CsvImportParams`) | May only **lower** the deployment ceiling: `min(requested, deployment)` | Per-request flag |
| Environment / `.env` | `JUNIPER_DATA_CSV_IMPORT_MAX_BYTES` — **hard ceiling**, `gt=0` | `JUNIPER_DATA_CSV_IMPORT_ALLOW_TRUNCATION` |
| Compiled default | 128 MiB | `false` |

Precedence is asymmetric on purpose, and both sides treat the operator as the privileged party:

- **`max_bytes`:** the effective cap is `min(requested, settings.csv_import_max_bytes)`. A request cannot raise the operator ceiling. Omitting the field uses the deployment value. `model_fields_set` still decides "was this sent", but even an explicit huge value is clamped. That clamp is load-bearing for generated clients: serialising schema defaults sends `max_bytes=134217728` on every request, which would otherwise override a *lower* operator ceiling with nobody intending it. The first design let an explicit request win outright (`max_bytes: 10000000000` skipped the cap); that inverted the privilege model used for `allow_truncation` and made the DoS bound caller-controlled.
- **`allow_truncation`:** logical **OR**. Either the request opts in, or the deployment has opted in for every request. There is no way to opt *out* of a deployment-wide opt-in.

Constants live in `juniper_data/core/limits.py` (not `csv_import/defaults.py`) so `api/settings.py` can import them without a package cycle. `defaults.py` re-exports the names.

### The read is the bound

`stat().st_size` is a **cheap pre-check**, not the bound. It can refuse an obviously-oversized source without reading, but nothing is ingested on its authority. Every path then calls `_read_capped_bytes(path, cap_bytes + 1)` and re-checks `len(raw)`. The extra byte distinguishes "fits exactly" from "there is more".

An earlier draft trusted `stat` and, when it reported a size within the cap, read to EOF. Three bypasses that made the cap decorative:

- **TOCTOU.** `import_dir` is shared; a file can grow or be replaced between `stat` and `open`.
- **FIFOs report `st_size == 0`**, take the under-cap branch, then stream without limit.
- **A negative cap inverts `read()`.** Python treats `read(n)` with `n < 0` as "read everything". `Settings.csv_import_max_bytes` carries `gt=0` so a mistyped env var fails deployment loudly; `_read_capped_bytes` refuses `limit <= 0` as the second line of that defence.

There is no unbounded `_load_csv` / `_load_json` path. Those helpers were removed once the capped read replaced their only call sites. For a FIFO (or any source whose `stat` under-reports), `bytes_total` in the truncation descriptor is `max(stat_bytes, len(raw))` — a true lower bound.

### Refusal vs authorised truncation

Default is **refusal**. An over-cap source with neither opt-in raises `InputTooLargeError` (a `ValueError` subclass). `POST /v1/datasets` maps it to **HTTP 422** with a string `detail` that names the source size, the cap, and the remedy (`allow_truncation=true` or the env var). 422 is already on this API (schema validation); the string shape is the exception — schema 422s carry a list.

Subclassing `ValueError` is load-bearing: a call path that forgets the 422 mapping still lands on the app-level `ValueError` handler's **400**, not a 500. `batch-create` reuses `create_dataset`, so the same 422/`HTTPException` becomes the per-item `error` string.

### Record-boundary cuts

A byte offset almost always lands mid-record. The authorised path:

1. Reads at most `cap_bytes + 1` (`_read_capped_bytes`), then trims the first `cap_bytes` to a record boundary (`_trim_to_record_boundary`). Drops a split multi-byte UTF-8 sequence (`errors="ignore"`). Discards everything after the last newline. No newline inside the cap → empty text → existing "No data found in file".
2. **CSV:** `_parse_csv_stream(..., drop_trailing_partial=True)` also drops a final row whose values include `None`. A newline *inside a quoted field* is legal CSV, so the newline trim alone is not enough. `DictReader` reports a missing column as `None` and an empty field as `""`.
3. **JSON array:** `json.loads` cannot succeed without the closing `]`. `_decode_partial_json_array` uses `raw_decode` and keeps complete elements up to the first incomplete one.
4. **JSONL:** drops only an unparseable **final** line. A corrupt line mid-file still raises — otherwise a truncated import would launder a broken source into a short dataset.

### Permanent annotation

When a partial import is authorised, the generator places a descriptor on the reserved `"truncation"` channel (`TRUNCATION_META_KEY`). The route pops it with `pop_truncation_meta` **before** checksum and NPZ persist, so stored arrays stay array-only — the same discipline as `pop_scaling_meta`.

`DatasetMeta.truncation`:

| Field | Meaning |
|-------|---------|
| `None` | Complete. Absence, not `{"truncated": false}` — a reader must not distinguish "complete" from "the generator forgot to report". |
| dict | Partial. Keys: `truncated` (`true`), `reason` (`source_exceeded_byte_cap`), `bytes_read`, `bytes_total`, `cap_bytes`, `records_imported`. |

This is persisted metadata, not a transient warning. A trainer loading the artifact later, who never saw the HTTP response, still learns the data is a prefix of its source. `records_imported` matches `X_full.shape[0]`.

### What not to do

- Do not default truncation on. Silence is the failure mode this bound was warned about.
- Do not cut at an arbitrary byte and parse. The record-boundary trim is the product.
- Do not leave the reserved `"truncation"` key in the NPZ. Pop it before checksum.
- Do not store `{}` for "complete". `pop_truncation_meta` returns `None`.
- Do not raise the 128 MiB default without a streaming loader. The next bottleneck is peak objects, not parse time.
- Do not treat this cap as the 10 MB HTTP body limit, or as a substitute for `JUNIPER_DATA_IMPORT_DIR` path checks.
- Do not let a request `max_bytes` raise the deployment ceiling. The effective cap is `min(requested, deployment)`.
- Do not trust `stat().st_size` as the bound, or keep an under-cap path that reads to EOF. The read (`cap_bytes + 1`) is what enforces it.
- Do not accept a non-positive cap. `read(-1)` is unbounded.

### Pins

| Test | What it pins |
|------|----------------|
| `tests/unit/test_csv_import_generator.py` (`TestCsvImportByteCap`) | Under-cap has no channel key; over-cap without opt-in raises `InputTooLargeError`; request and deployment opt-in annotate; a request **cannot raise** the deployment cap; a lying `stat` (`st_size == 0`) still refuses; non-positive cap is refused; CSV/JSON/JSONL record boundaries; JSONL mid-file still raises; `pop_truncation_meta` returns `None` when absent |
| `tests/unit/test_api_routes.py` | `POST /v1/datasets` over-cap is **422** not 500; authorised truncation reaches `meta.truncation`; within-cap stores `None` |
| `util/ad-hoc/2026-09-04_apd_data_018_mutation_check.py` | Mutation matrix (8/8), including M7 (request can raise the cap) and M8 (`stat` trusted over the read) |

---

## Project Architecture Reference

Relocated verbatim from `AGENTS.md` (P3 of the shared-session-memory plan) so it is read on demand rather than loaded into every session.

### Directory Structure

```bash
juniper-data/
├── juniper_data/                   # Main Python package
│   ├── __init__.py                 # Package init, version, ARC-AGI env helpers
│   ├── __main__.py                 # CLI entry point (python -m juniper_data)
│   ├── core/                       # Core domain logic
│   │   ├── constants.py            # Core-layer constants (encoding, magic numbers, fixed metadata keys)
│   │   ├── models.py               # Pydantic models (DatasetMeta, request/response types)
│   │   ├── dataset_id.py           # Deterministic SHA-256 dataset ID generation
│   │   ├── split.py                # Train/test data splitting
│   │   ├── artifacts.py            # NPZ artifact handling and checksums
│   │   └── secrets.py              # Docker secrets management
│   ├── generators/                 # Dataset generators (10 types)
│   │   ├── spiral/                 # Multi-spiral classification (configurable arms)
│   │   ├── xor/                    # XOR classification
│   │   ├── gaussian/               # Mixture of Gaussians
│   │   ├── circles/                # Concentric circles
│   │   ├── moon/                   # Two interleaving half-moons
│   │   ├── checkerboard/           # 2D checkerboard pattern
│   │   ├── csv_import/             # CSV/JSON file import
│   │   ├── equities/               # S&P 500 equities time-series (Yahoo Finance + SEC EDGAR)
│   │   ├── mnist/                  # MNIST / Fashion-MNIST (HuggingFace)
│   │   └── arc_agi/                # ARC-AGI visual reasoning (optional)
│   ├── storage/                    # Dataset persistence (7 backends)
│   │   ├── constants.py            # Storage-layer constants (filenames, metadata keys, table/column names)
│   │   ├── base.py                 # Abstract DatasetStore interface
│   │   ├── local_fs.py             # Local filesystem (default)
│   │   ├── memory.py               # In-memory (testing)
│   │   ├── cached.py               # Composable caching wrapper
│   │   ├── redis_store.py          # Redis backend
│   │   ├── postgres_store.py       # PostgreSQL backend
│   │   ├── hf_store.py             # Hugging Face Hub integration
│   │   └── kaggle_store.py         # Kaggle dataset integration
│   ├── api/                        # FastAPI application
│   │   ├── app.py                  # Factory-pattern app creation with lifespan
│   │   ├── constants.py            # API-layer constants (header names, status codes, defaults, error messages)
│   │   ├── settings.py             # Pydantic BaseSettings (JUNIPER_DATA_ prefix)
│   │   ├── middleware.py           # Security headers, body limits, rate limiting
│   │   ├── security.py             # API key auth (APIKeyAuth) and RateLimiter
│   │   ├── observability.py        # Prometheus metrics, JSON logging, Sentry, request IDs
│   │   ├── models/                 # Response models
│   │   │   └── health.py           # DependencyStatus, ReadinessResponse
│   │   └── routes/                 # API route handlers
│   │       ├── health.py           # /v1/health, /v1/health/live, /v1/health/ready
│   │       ├── generators.py       # /v1/generators, /v1/generators/{name}/schema
│   │       └── datasets.py         # /v1/datasets (CRUD, batch, versioning, lifecycle)
│   └── tests/                      # Test suite (835+ tests)
│       ├── conftest.py             # Shared fixtures
│       ├── unit/                   # Unit tests (30+ files)
│       ├── integration/            # Integration tests (5 files)
│       ├── performance/            # Benchmarks via pytest-benchmark (41 tests)
│       ├── api/                    # API-specific tests
│       └── fixtures/               # Golden dataset fixtures (NPZ + metadata)
├── docs/                           # User and developer documentation
│   ├── QUICK_START.md              # 5-minute setup guide
│   ├── USER_MANUAL.md              # Full user documentation
│   ├── REFERENCE.md                # API, config, and command reference
│   ├── DEVELOPER_CHEATSHEET.md     # Quick reference for developers
│   ├── ENVIRONMENT_SETUP.md        # Environment configuration guide
│   ├── DOCUMENTATION_OVERVIEW.md   # Documentation navigation guide
│   ├── api/                        # API documentation
│   ├── testing/                    # Testing documentation
│   └── ci_cd/                      # CI/CD documentation
├── scripts/                        # CI and coverage scripts
│   ├── check_module_coverage.py    # Per-module coverage enforcement (85% min)
│   ├── check_doc_links.py          # Internal markdown link validation
│   └── generate_dep_docs.sh        # Dependency documentation generator
├── notes/                          # Development notes, procedures, roadmaps
├── conf/                           # Shell and logging configuration files
├── util/                           # Bash utility scripts (40+ scripts)
├── .github/                        # GitHub Actions workflows and config
│   ├── workflows/                  # CI, CodeQL, security, publish, lockfile, sequence-safety, main-verify
│   ├── CODEOWNERS                  # Code ownership rules
│   └── dependabot.yml              # Automated dependency updates
├── Dockerfile                      # Multi-stage production build (Python 3.14-slim)
├── pyproject.toml                  # Project configuration (authoritative)
├── .pre-commit-config.yaml         # Pre-commit hooks (ruff, mypy, bandit, yamllint, shellcheck)
├── .env.example                    # Environment variables template
├── requirements.lock               # Pinned dependency versions for Docker builds
├── CHANGELOG.md                    # Version history (0.1.0 to 0.5.0)
├── README.md                       # Project overview and PyPI landing page
├── AGENTS.md                       # This file
└── CLAUDE.md                       # Symlink to AGENTS.md
```

### Component Overview

| Component | Purpose |
|-----------|---------|
| `core/constants.py` | Core-layer constants (encoding strings like `utf-8`, magic numbers, fixed metadata keys) |
| `core/limits.py` | `csv_import` byte-cap defaults, `InputTooLargeError`, reserved `"truncation"` channel (`TRUNCATION_META_KEY`) |
| `core/models.py` | Pydantic models: DatasetMeta, CreateDatasetRequest/Response, batch models, filters, stats |
| `core/dataset_id.py` | Deterministic SHA-256 based dataset ID generation |
| `core/split.py` | Shuffle and split data into train/test sets |
| `core/artifacts.py` | NPZ save/load, array-to-bytes conversion, SHA-256 checksums |
| `core/secrets.py` | Docker secrets and environment variable secret loading |
| `generators/` | 10 dataset generator implementations (each with `generator.py` + `params.py`) |
| `generators/spiral/` | Multi-spiral classification dataset (configurable arms, noise, rotation) |
| `generators/xor/` | XOR 4-quadrant binary classification |
| `generators/gaussian/` | Mixture-of-Gaussians multivariate classification |
| `generators/circles/` | Concentric circles binary classification |
| `generators/moon/` | Two interleaving half-moons binary classification |
| `generators/checkerboard/` | 2D grid pattern with alternating classes |
| `generators/csv_import/` | Import datasets from CSV/JSON files (on-disk sources bounded; over-cap refused unless truncation is opted in) |
| `generators/equities/` | S&P 500 equities daily time-series (OHLCV + SEC fundamentals; dual next-day targets) |
| `generators/mnist/` | MNIST and Fashion-MNIST via HuggingFace Hub |
| `generators/arc_agi/` | ARC-AGI visual reasoning tasks (optional dependency) |
| `storage/constants.py` | Storage-layer constants (filenames, metadata keys, table/column names, default size limits) |
| `storage/base.py` | Abstract `DatasetStore` interface with versioning and lifecycle |
| `storage/local_fs.py` | Local filesystem storage (atomic writes, compressed NPZ) |
| `storage/memory.py` | In-memory storage for testing |
| `storage/cached.py` | Composable caching wrapper (read-through, write-through) |
| `storage/redis_store.py` | Redis storage backend (optional) |
| `storage/postgres_store.py` | PostgreSQL storage with JSONB metadata (optional) |
| `storage/hf_store.py` | Hugging Face Hub read-only integration (optional) |
| `storage/kaggle_store.py` | Kaggle dataset download integration (optional) |
| `api/app.py` | FastAPI application factory with lifespan management |
| `api/constants.py` | API-layer constants (HTTP header names, default body limits, rate-limit defaults, error message templates, exempt paths) |
| `api/settings.py` | Pydantic BaseSettings with `JUNIPER_DATA_` prefix |
| `api/middleware.py` | SecurityHeadersMiddleware, RequestBodyLimitMiddleware |
| `api/security.py` | APIKeyAuth authentication, RateLimiter |
| `api/observability.py` | Prometheus metrics, JSON logging, Sentry, RequestIdMiddleware |
| `api/models/health.py` | DependencyStatus, ReadinessResponse models |
| `api/routes/health.py` | Health, liveness, and readiness endpoints |
| `api/routes/generators.py` | Generator listing and schema endpoints |
| `api/routes/datasets.py` | Dataset CRUD, batch, versioning, lifecycle, filtering |

---

## API Package Import Graph

`create_app` is exposed **lazily** from `juniper_data.api` (PEP 562 `__getattr__` in `api/__init__.py`). Eagerly importing it there made `juniper_data.generators.csv_import` unimportable on its own (juniper-data#316 / #333).

The cycle:

```
csv_import/__init__
  -> csv_import.generator          imports juniper_data.api.settings
  -> juniper_data.api/__init__     eagerly imported .app
  -> api.app                       imports .routes.datasets / .routes.generators
  -> api.routes.generators         imports juniper_data.generators.csv_import  (still initialising)
  -> ImportError: cannot import name 'VERSION'
```

Importing any submodule initialises its parent package first. So `from juniper_data.api.settings import get_settings` — a leaf import — dragged in the FastAPI app and every route.

`csv_import.generator` imports `get_settings` because `_load_and_preprocess` resolves `file_path` against `Settings.import_dir`. That leaf import is legitimate. The defect was the parent package pulling routes.

### Why it was invisible in production

Service startup imports the routes long before anything touches `csv_import`. The cycle fires only when the generator subpackage is imported **first**: a test file collected in isolation, a script, or an external consumer. It broke `pytest juniper_data/tests/unit/test_csv_import_generator.py` as a standalone collection.

### The bargain

| Name | When it loads | Why |
|------|---------------|-----|
| `Settings`, `get_settings` | Eager, from `api/__init__.py` | Leaves. `api.settings` imports only `api.constants` and `core.secrets`. |
| `create_app` | Lazy, via `__getattr__` | Importing `.app` pulls every route, which pull every generator. |
| `from juniper_data.api import create_app` | Still works | Public surface. Callers today import `juniper_data.api.app` directly. `__dir__` lists it for tab-completion. |

Deferring `create_app` closes the **class**, not just the `csv_import` instance: nothing that merely wants settings pulls the routes any more. Today only `csv_import` of the 16 registered generators imports `api.settings`. A later generator that does the same stays importable as long as this bargain holds.

### What not to do

- Do not restore `from .app import create_app` in `api/__init__.py`. That re-opens the cycle for every generator that touches settings.
- Do not import `juniper_data.api.app` or `juniper_data.api.routes` from a generator. `api.settings` is the allowed API surface from generator code.
- Do not "fix" a collection-order `ImportError: cannot import name 'VERSION'` by pre-importing `api.routes.generators` in the test. That workaround landed in `test_normaliser_fit_scope.py` and is gone; bringing it back hides the cycle.
- Do not assert import-graph properties in-process. Once `juniper_data` is in `sys.modules`, a same-process import succeeds with the defect fully present.

### Pins

`tests/unit/test_no_import_cycles.py` runs every assertion in a **subprocess**:

- all 16 registered generator subpackages (`ar_p`, `arc_agi`, `checkerboard`, `circles`, `csv_import`, `delay_product`, `equities`, `equities_seq`, `gaussian`, `irregular_sine`, `mackey_glass`, `mnist`, `moon`, `multi_sine`, `spiral`, `xor`) import standalone
- `csv_import` public names (`VERSION` was the one that failed) resolve
- importing `api.settings` does **not** load `api.routes.generators`
- `create_app` remains importable from the package; `Settings` / `get_settings` stay eager
- a genuine typo still raises `AttributeError` (the module `__getattr__` must not swallow it)

On #333, reverting `api/__init__.py` to the eager import failed exactly three: the `csv_import` standalone import, its public-name resolution, and the routes-not-loaded property. The other 15 subpackages stayed green.

---

## API Design Reference

Relocated verbatim from `AGENTS.md` (P3 of the shared-session-memory plan) so it is read on demand rather than loaded into every session.

### REST Conventions

- Use nouns for resources: `/datasets`, `/generators`
- All endpoints prefixed with `/v1/`
- Use HTTP methods appropriately: GET, POST, PATCH, DELETE
- Return proper status codes (200, 201, 204, 400, 404, 413, 422, 429, 500, 501)
- Include pagination for list endpoints (limit, offset)

### Endpoint Catalog

**Health Endpoints**:

| Method | Path | Description |
|--------|------|-------------|
| GET | `/v1/health` | Health check (status + version) |
| GET | `/v1/health/live` | Liveness probe |
| GET | `/v1/health/ready` | Readiness probe with dependency status |

**Generator Endpoints**:

| Method | Path | Description |
|--------|------|-------------|
| GET | `/v1/generators` | List all generators with info (name, version, description, schema) |
| GET | `/v1/generators/{name}/schema` | Get JSON schema for generator parameters |

**Dataset Endpoints**:

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/datasets` | Create dataset (returns 201) |
| GET | `/v1/datasets` | List dataset IDs (paginated) |
| GET | `/v1/datasets/{dataset_id}` | Get dataset metadata |
| DELETE | `/v1/datasets/{dataset_id}` | Delete dataset (returns 204) |
| GET | `/v1/datasets/{dataset_id}/artifact` | Download NPZ artifact |
| GET | `/v1/datasets/{dataset_id}/preview` | Preview first N samples as JSON |
| GET | `/v1/datasets/filter` | Advanced filtering (generator, tags, dates, sample count) |
| GET | `/v1/datasets/stats` | Aggregate statistics |
| GET | `/v1/datasets/versions` | List all versions of a named dataset |
| GET | `/v1/datasets/latest` | Get latest version of a named dataset |
| PATCH | `/v1/datasets/{dataset_id}/tags` | Update tags on a dataset |
| POST | `/v1/datasets/cleanup-expired` | Delete all expired datasets |

**Batch Endpoints**:

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/datasets/batch-create` | Create multiple datasets (max 50; 201 when at least one was created, 200 when none was) |
| POST | `/v1/datasets/batch-delete` | Delete multiple datasets (max 100) |
| POST | `/v1/datasets/batch-export` | Export multiple datasets as ZIP (max 50) |
| PATCH | `/v1/datasets/batch-tags` | Add/remove tags from multiple datasets |

### Middleware Stack

Middleware executes in LIFO order (last added = first to execute):

| Order | Middleware | Purpose |
|-------|-----------|---------|
| 1 | `CORSMiddleware` | Cross-origin resource sharing (if configured) |
| 2 | `RequestIdMiddleware` | Inject/propagate X-Request-ID header |
| 3 | `PrometheusMiddleware` | HTTP request metrics (if enabled) |
| 4 | `SecurityMiddleware` | API key auth + rate limiting |
| 5 | `SecurityHeadersMiddleware` | Security response headers (CSP, HSTS, etc.) |
| 6 | `RequestBodyLimitMiddleware` | Reject bodies > 10 MB |

**`CORSMiddleware` must stay outermost.** A browser preflight carries no
`X-API-Key` -- the browser generates it, and author-defined headers ride only on
the actual request that follows -- so any layer that puts `SecurityMiddleware`
outside CORS answers every preflight to a non-exempt path with 401, and no
browser client can reach a protected endpoint. Running outermost also attaches
the CORS headers to error responses (401/429), so a browser surfaces the real
status instead of an opaque CORS failure.

Note this is a stronger requirement than an `OPTIONS` bypass in
`SecurityMiddleware._is_exempt`: CORS short-circuits only a *genuine* preflight
(one carrying `Access-Control-Request-Method`), so a plain `OPTIONS` request
still authenticates. Pinned by `TestCorsPreflight` in
`juniper_data/tests/unit/test_api_app.py`.

### Response Models

Responses use typed Pydantic models (defined in `core/models.py` and `api/models/`):

- `CreateDatasetResponse` -- dataset_id, generator, meta, artifact_url
- `DatasetListResponse` -- datasets (list of DatasetMeta), total, limit, offset
- `DatasetVersionListResponse` -- dataset_name, versions, total, latest_version
- `BatchCreateResponse` -- results, total_created, total_failed
- `BatchDeleteResponse` -- deleted, not_found, total_deleted
- `DatasetStats` -- total_datasets, total_samples, by_generator, by_tag
- `ReadinessResponse` -- status, version, service, timestamp, dependencies

---

## Empty-Train Shape Metadata

`compute_shape_meta` (`juniper_data/core/meta.py`) derives the shape fields `POST /v1/datasets` writes onto `DatasetMeta` for **every** generator. `n_features` is the **trailing** axis of `X_train`: `shape[-1]` for both 2-D tabular `(N, F)` and 3-D sequence `(W, L, F)`. That matches `derive_sequence_meta`, which already reads rank from an empty `X_train` because an empty split still has a defined shape.

Before #365, empty train (`n_train == 0` — a `train_ratio` that rounds to zero rows, or a generator that emits an empty train split) abandoned that contract and hardcoded `n_features = 2`:

- a 2-D import with F=5 persisted `n_features=2`
- a 3-D sequence with lookback 7 and F=3 persisted `n_features=2` (neither lookback nor F)

That wrong count is stored in metadata and served to every consumer. The route uses this helper for every artifact it creates, so the empty-train arm is not a niche path.

#340 uses `int(x_train.shape[-1])` unconditionally. Empty `(0, 5)` reports 5; empty `(0, 7, 3)` reports 3.

### Classification `n_classes` on empty train

`_classification_meta` already falls back to `y_test.shape[1]` when `n_train == 0`. That is existing behaviour; #340 pins it so a later "simplify" cannot turn a 4-class empty-train artifact into `n_classes=2`. The remaining last-resort `n_classes = 2` applies only when **both** splits are empty — #340 does not change that arm.

Regression artifacts still leave `n_classes` / `class_distribution` as `None`.

### What not to do

- Do not restore `n_features = … if n_train > 0 else 2`. Empty arrays still have `shape[-1]`. Mutation: putting `else 2` back fails the two new n_features tests (`2 == 5` and `2 == 3`) and leaves the rest of `test_meta_dispatch.py` green.
- Do not use `shape[1]` (lookback) as `n_features` for 3-D artifacts. The trailing axis is F.
- Do not special-case empty train in the route. The helper is the single derivation; `create_dataset` already calls it for every generator, including `batch-create`.

### Pins

These live in `tests/unit/test_meta_dispatch.py`, on `main` since #365 (which superseded the still-open #340). The existing `test_classification_3d_uses_trailing_feature_axis` only covers **non-empty** train, which is why the narrower `if n_train > 0 else 2` stayed green before these pins. `main` keeps an `else 2` arm for rank < 2 -- a 1-D empty array has no meaningful trailing axis -- so the mutation to guard against is dropping the `or x_train.ndim >= 2` disjunct, not the `else 2` itself.

| Test | Property |
|------|----------|
| `test_empty_train_2d_uses_trailing_feature_axis` | `(0, 5)` reports `n_features=5`, not 2 |
| `test_empty_train_3d_uses_trailing_feature_axis` | `(0, 7, 3)` reports `n_features=3`, not lookback 7, not 2 |
| `test_empty_train_classification_reads_n_classes_from_y_test` | empty-train 4-class artifact reports `n_classes=4` from `y_test` |

---

---

## Storage Backend Reference

Relocated verbatim from `AGENTS.md` (P3 of the shared-session-memory plan) so it is read on demand rather than loaded into every session.

JuniperData supports 7 storage backend implementations with a composable architecture.

### Abstract Interface

`DatasetStore` (in `storage/base.py`) defines the standard interface:

- **Core**: `save()`, `get_meta()`, `get_artifact_bytes()`, `exists()`, `delete()`, `list_datasets()`
- **Streaming**: `open_artifact_stream()` — not abstract; see [Artifact Streaming](#artifact-streaming)
- **Versioning**: `list_versions()`, `get_latest_version()`, `next_version_number()`, `save_versioned()`
- **Lifecycle**: `record_access()`, `is_expired()`, `delete_expired()`, `filter_datasets()`
- **Batch**: `batch_delete()`, `get_stats()`

### Implementations

| Backend | Module | Use Case | Dependencies |
|---------|--------|----------|--------------|
| **LocalFS** | `storage/local_fs.py` | Default production storage | None (stdlib) |
| **InMemory** | `storage/memory.py` | Testing and development | None (stdlib) |
| **Cached** | `storage/cached.py` | Composable cache wrapper | None (wraps another store) |
| **Redis** | `storage/redis_store.py` | Distributed caching | `redis` |
| **PostgreSQL** | `storage/postgres_store.py` | Persistent metadata with JSONB | `psycopg2` |
| **HuggingFace** | `storage/hf_store.py` | Read-only HF Hub integration | `datasets` |
| **Kaggle** | `storage/kaggle_store.py` | Kaggle dataset downloads | `kaggle` |

### Composable Caching

`CachedDatasetStore` wraps any primary store with a cache store:

```python
primary = LocalFSDatasetStore(path="./data")
cache = InMemoryDatasetStore()
store = CachedDatasetStore(primary=primary, cache=cache)
```

---

## CSV Import Truncation Edges

Authorised `csv_import` truncation (`allow_truncation` / `JUNIPER_DATA_CSV_IMPORT_ALLOW_TRUNCATION`) still had three silent failure modes on the #326 path. #372 closes them (harvested from the never-merged #336). This section is only those edges — not the 128 MiB ceiling, the refusal default, or the read-enforced bound.

### Newline trim is CSV-only

On the #326 path, `_load_and_preprocess` always ran `_trim_to_record_boundary` on the capped prefix, then parsed CSV or JSON. `json.dumps` / `JSON.stringify` emit **one line**. `rfind("\n")` then returns `-1` and the helper returns empty text, so `_convert_to_arrays` raises `No data found in file`. The decoder never sees the complete objects that fit.

#372 keeps the byte prefix for JSON / JSONL and lets `_parse_json_text(..., tolerate_truncated=True)` decode complete elements (`raw_decode` for arrays; drop only a final unparseable JSONL line). CSV still trims to the last newline. A CSV source with no newline inside the cap still yields empty text — that remains the correct CSV outcome, because not one whole record fits.

### Unclosed quotes are not a short row

`drop_trailing_partial` dropped a last CSV row only when some value was `None` (`DictReader`'s missing-column sentinel; an empty field is `""`). A 2-column file whose unclosed `"` swallows later lines still fills every column — the short-row guard cannot see it — and the fabricated `value` is later coerced to `0.0`.

#372 scans `source_text` with `_has_unclosed_quote`. Doubled quotes (`""`) are the RFC 4180 escape and do not toggle state. The capped path passes `source_text`; the whole-file path does not (`drop_trailing_partial=False`).

### Bind the effective policy before the cache key

`POST /v1/datasets` hashes `params.model_dump()` via `generate_dataset_id`. Dump fills Field defaults, so an omitted `max_bytes` is stored as 128 MiB (`CSV_IMPORT_DEFAULT_MAX_BYTES`) even when generation used a tighter deployment ceiling. The same dump stores `allow_truncation=false` when the operator opted in globally.

Raising the cap — or turning truncation off — then reuses the truncated artifact for the "same" request.

#372 adds `CsvImportGenerator.bind_deployment_defaults`, which copies `_resolve_bounds` onto the params object. `create_dataset` calls it via `getattr(generator_class, "bind_deployment_defaults", None)` **before** hashing. Generators without the method are unchanged. `batch-create` reuses `create_dataset`.

### What not to do

- Do not run `_trim_to_record_boundary` on JSON / JSONL. One-line minified arrays become empty text.
- Do not treat `DictReader` `None` as sufficient to drop a capped CSV row. An unclosed quote can populate every column.
- Do not hash unbound params. Omitted `max_bytes` becomes 128 MiB in the dump.
- Do not make `bind_deployment_defaults` a required generator ABC method. The route uses `getattr`.

### Pins

| Test | Property |
|------|----------|
| `test_truncated_minified_json_array_keeps_complete_elements` | A one-line `json.dumps` prefix imports complete elements, not empty text |
| `test_unclosed_quote_drops_last_row_even_when_all_fields_are_present` | A 2-column dangling `"` is dropped even when every field is populated |
| `test_bind_deployment_defaults_puts_effective_policy_in_dump` | Omitted `max_bytes` and the schema default both dump as the deployment ceiling; global `allow_truncation` appears in the dump |
| `test_create_dataset_cache_does_not_reuse_tight_cap_after_operator_raises_it` | Tight then wide deployment caps produce different `dataset_id`s; the wide call is complete |

These live in `tests/unit/test_csv_import_generator.py` and `tests/unit/test_api_routes.py` on main, landed by #372 and #373. They are not on `main` until that PR lands.

---

---

## Prometheus Collector Reference

Relocated verbatim from `AGENTS.md` (P3 of the shared-session-memory plan) so it is read on demand rather than loaded into every session.

For any new `prometheus_client` `Counter` / `Gauge` / `Histogram` / `Summary` / `Info` / `Enum` registration, use the canonical helpers from `juniper-observability` (`>=0.2.0`):

- `register_or_reuse(factory, name, *args, **kwargs)` — adopt-existing on duplicate (the default for almost every call site; preserves accumulated samples across in-process re-init).
- `register_fresh(...)` — drop-and-recreate on duplicate (only when args genuinely differ).
- `register_info_or_update(name, description, **labels)` — sugar for the `Info` two-step register-then-`.info({...})` pattern.
- `lazy_register_or_reuse(...)` — for the lazy-init-with-`None`-sentinel pattern.

Tests touching these collectors should use `juniper_observability.testing.reset_prometheus_registry`. Existing examples in this repo: `juniper_data/api/observability.py:_ensure_dataset_metrics`. See [the design doc in juniper-ml](https://github.com/pcalnon/juniper-ml/blob/main/notes/observability/JUNIPER_2026-05-05_JUNIPER-ML_REGISTER-OR-REUSE-HELPER-DESIGN.md) for the rationale.

---

---

## Artifact Streaming

`GET /v1/datasets/{id}/artifact` (`download_artifact` in `api/routes/datasets.py`) used to wrap `get_artifact_bytes(...)` in `io.BytesIO` and return a `StreamingResponse`. That bounds the **socket buffer**, not process memory: the whole NPZ sat in RAM before the response existed, once per concurrent download. Peak RSS scaled with artifact size × concurrency while the name "streaming" invited the opposite assumption (APD-DATA-016 / #313).

The route now calls `DatasetStore.open_artifact_stream`. The memory bound is a property of the **store**, not of the route.

### Interface

`open_artifact_stream(dataset_id, chunk_size=ARTIFACT_STREAM_CHUNK_SIZE)` is **deliberately not** `@abstractmethod`. The base implementation reads via `get_artifact_bytes` and yields `iter((payload,))` — one chunk, the whole blob. All seven backends keep working without a flag day. Only a backend that can read incrementally should override it.

| Backend | Streaming behaviour |
|---------|---------------------|
| `LocalFSDatasetStore` | Overrides: reads the NPZ file in `chunk_size` blocks (default **1 MiB**, `storage/constants.py`) |
| InMemory, Cached, Redis, Postgres, HuggingFace, Kaggle | Inherit the base whole-read |

`create_app` constructs `LocalFSDatasetStore` directly, so the default serving path is incremental. Wrapping LocalFS in `CachedDatasetStore` silently reverts to a whole read: Cached does not override, so the call hits `Cached.get_artifact_bytes`.

### Absence is decided eagerly

A generator body does not run until first iteration. An `exists()` check placed *inside* the generator defers the `None`/404 until after the route has already committed to 200 and sent headers — the client sees **200 with an empty body**. LocalFS checks `npz_path.exists()` *before* returning `_chunks()`. The route branches on `is None` after `asyncio.to_thread(store.open_artifact_stream, ...)` — only the open (existence + handle) is off-thread; later chunk reads are pulled by the ASGI server.

The LocalFS handle is opened inside the generator and closed by its `with` block, so a client disconnect mid-stream releases the file when the generator is finalised.

### Wire

- **Content-Type:** `application/zip` (`BINARY_MEDIA_TYPE` in `api/constants.py`). Both binary routes derive from that name. Do not spell `application/octet-stream` inline — that is the RFC 9110 §8.3 fallback, not this service's published type. Changing the value is a wire change (`test_binary_media_types.py`).
- **Content-Disposition:** `attachment; filename={dataset_id}.npz`.
- Bytes are identical to `get_artifact_bytes`. The change is a memory profile, not a payload change.
- ETag / conditional GET is APD-DATA-017 and is not implemented here.

### What not to do

- Do not wrap `get_artifact_bytes` in `io.BytesIO` and call it streaming.
- Do not move the existence check inside the generator.
- Do not make `open_artifact_stream` abstract — that is a flag day across seven stores.
- Do not ignore `chunk_size` in an override. Overriders must honour it.
- Do not assume wrapping LocalFS in Cached (or any inheriting store) keeps incremental reads.

### Pins

These live in `juniper_data/tests/unit/test_artifact_streaming.py` (on `main`). A whole-file read still round-trips, so the decisive LocalFS arm is that a small `chunk_size` yields **more than one** chunk.

| Test | Property |
|------|----------|
| `test_small_chunk_size_yields_many_chunks` | LocalFS is incremental; one chunk means a whole read |
| `test_chunk_size_is_honoured` | every chunk but the last is exactly `chunk_size` |
| `test_bytes_are_identical_to_the_materialised_read` | memory-profile change is not a wire change |
| `test_inheriting_backend_yields_exactly_one_chunk` | base default is an honest whole read |
| `test_local_fs_returns_none_not_a_generator` / `test_default_returns_none_not_a_generator` | absence is `None` from the *call*, not from first iteration |
| `test_binary_media_types.py` | published type is `application/zip`; no inline media-type literals |

---

---

## Docker Reference

Relocated verbatim from `AGENTS.md` (P3 of the shared-session-memory plan) so it is read on demand rather than loaded into every session.

### Dockerfile

- **Build**: Multi-stage (builder -> runtime) using `python:3.14-slim`
- **User**: Non-root `juniper:juniper` (UID 1000, GID 1000)
- **Port**: 8100 (exposed)
- **Health Check**: `curl -f http://localhost:8100/v1/health` (30s interval, 10s timeout, 3 retries)
- **Entry Point**: `python -m juniper_data`

### Environment Variables

All `JUNIPER_DATA_*` variables (see [Configuration](../AGENTS.md#configuration)) are supported in the container.

### Docker Compose

Full-stack orchestration is in the `juniper-deploy` repository. JuniperData runs as a service alongside juniper-cascor and JuniperCanopy.

---

## Equities Symbol Cap

Bound for `equities` / `equities_seq` (`APD-DATA-018` second half, landed with juniper-data#354). Generation runs **inside the request**. The previous default was `EQUITIES_DEFAULT_MAX_SYMBOLS = None` — all **503** bundled S&P 500 names, 18–34 minutes against a ~30 s client budget — and the only cut was a bare `ordered[: params.max_symbols]` that recorded nothing. #354 deletes that silent slice.

`equities_seq` reuses `EquitiesGenerator._resolve_symbols` and must therefore carry the same annotation. Inheriting the bound while dropping the record is the worse half to skip.

### Why symbols, not bytes

Call count is `O(symbols)`, not `O(rows)`. Horizon grows the Yahoo payload; it does not add requests. Measurement (`util/ad-hoc/2026-09-04_measure_equities_payloads.py`, shipped in #348):

| Request | Wire bytes | Wall time |
|---------|-----------:|----------:|
| 1 symbol × 26 years | 210 KB | ~2 s |
| Russell 3000 × 1 day | 92 KB | 1.7–3.2 h |

163× the payload cost 1.16× the time. A byte cap would admit the expensive request and refuse the cheap one.

**14 = 30 s ÷ 2.1 s per symbol** (`EQUITIES_DEFAULT_MAX_SYMBOLS` in `juniper_data/core/limits.py`). A second measurement implied ~7; the owner took the optimistic end on 2026-09-04. `defaults.py` re-exports the constant so `api/settings.py` can import it without a generator-package cycle.

### Surfaces and precedence

Same privilege model as `csv_import`'s byte cap:

| Surface | Cap (`max_symbols`) | Truncation opt-in (`allow_truncation`) |
|---------|---------------------|----------------------------------------|
| Request params (`EquitiesParams`) | May only **lower** the deployment ceiling: `min(requested, deployment)` | Per-request flag |
| Environment / `.env` | `JUNIPER_DATA_EQUITIES_MAX_SYMBOLS` — hard ceiling, `gt=0` | `JUNIPER_DATA_EQUITIES_ALLOW_TRUNCATION` |
| Compiled default | **14** | `false` |

- **`max_symbols`:** omit the field (or send `None`) to inherit the deployment ceiling. That is **not** unbounded — a caller cannot raise the operator cap, including via `max_symbols=9999` or `max_symbols=None`. `Settings.equities_max_symbols` rejects non-positive values at boot (`gt=0`).
- **`allow_truncation`:** logical **OR**. Either the request opts in, or the deployment has opted in for every request. A client cannot opt *out* of the operator's choice.

`_resolve_bounds` imports `get_settings` **inside the method**, not at module scope, so this generator does not join `csv_import`'s settings cycle.

### What `_resolve_symbols` does

1. If `params.symbols` is set: strip, uppercase, keep **caller order**. Unknown tickers get a CIK from SEC `company_tickers.json` (cached) or `cik=None`.
2. Else: bundled `generators/equities/sp500_constituents.csv` (**503** names), ordered by `sorted(constituents)` — alphabetical ticker, not market cap.
3. If `len(ordered) <= cap`: return the list, `truncation=None`.
4. Else if truncation is not allowed: raise `InputTooLargeError` (a `ValueError` subclass).
5. Else: keep the leading `cap` symbols and return a `build_truncation_meta(...)` descriptor (`records_imported=-1` until `generate()` fills the real row count).

The silent prefix slice is gone. Default `EquitiesParams()` against the bundled 503 names **refuses**.

`POST /v1/datasets` maps `InputTooLargeError` to **HTTP 422** with a string `detail` that names the actual count, the cap, and the remedy (`allow_truncation=true` or `JUNIPER_DATA_EQUITIES_ALLOW_TRUNCATION`). Schema 422s still carry a list; check the type before iterating. Subclassing `ValueError` is load-bearing: a missed catch lands on the app-level **400**, not a 500. `batch-create` reuses `create_dataset`, so the same 422 becomes the per-item `error` string.

### Permanent annotation

When a cut is authorised, the generator places the descriptor on the reserved `"truncation"` channel (`TRUNCATION_META_KEY`). The route pops it with `pop_truncation_meta` **before** checksum and NPZ persist — same discipline as `pop_scaling_meta`. `generate()` fills `records_imported` after conditioning (`len(full)` for `equities`; `X_full.shape[0]` for `equities_seq`), so it counts rows that survived, not `symbols × sessions`.

`DatasetMeta.truncation` (one shape for every generator):

| Field | Meaning |
|-------|---------|
| `None` | Complete. Absence, not `{}` — a reader tests the field's presence alone. |
| dict | Partial. Keys: `truncated` (`true`), `reason` (`universe_exceeded_symbol_cap`), `unit` (`symbols`), `cap`, `requested`, `imported`, `records_imported`. |

#354 also replaced the earlier byte-specific `cap_bytes` / `bytes_total` / `bytes_read` keys with this shared shape (`unit` is `bytes` or `symbols`). Unreleased-to-unreleased; no published artifact carries the old keys.

14 of 14 is complete (`truncation is None`). Which symbols survive is deterministic: sorted constituents, or the caller's own sequence. Dict-iteration / download-completion order must not change the prefix.

### Per-symbol work (cold cache)

For each remaining ticker, `_condition_one` does one `yf.download` (class shares mapped `BRK.B` → `BRK-B`) plus, if a CIK is known, 1–2 SEC GETs (`dei` then `us-gaap` shares-outstanding), spaced by `_SEC_MIN_INTERVAL = 0.12` s.

`use_cache` defaults `True` (`~/.cache/juniper_data/equities`, override `JUNIPER_DATA_EQUITIES_CACHE_DIR`). Yahoo is `download` only — not `Ticker.info`. Missing SEC facts + default `fundamentals_fill="zero"` writes `0.0`. A later download failure still skips with a warning; only an empty conditioned set raises `ValueError`. Extra: `pip install "juniper-data[equities]"`; missing extra → `501`.

### Operator usage

```python
from juniper_data.generators.equities.params import EquitiesParams
from juniper_data.generators.equities.generator import EquitiesGenerator

# Default universe is 503 names — this 422s / raises InputTooLargeError.
# EquitiesParams()

# Explicit list under the cap: complete, no annotation.
EquitiesGenerator.generate(EquitiesParams(symbols=["AAPL", "MSFT"], start_date="2024-01-01", end_date="2024-06-01"))

# Authorised cut of the bundled snapshot: first 14 alphabetical tickers + meta.truncation.
EquitiesParams(allow_truncation=True)

# Tighten further (caller order, then prefix): AAPL, MSFT — not AMZN.
EquitiesParams(symbols=["AAPL", "MSFT", "AMZN"], max_symbols=2, allow_truncation=True)
```

Via `POST /v1/datasets`: `"generator": "equities"` / `"equities_seq"` with `"params": {"symbols": ["AAPL", "MSFT"]}` or `"params": {"allow_truncation": true}`. Deployment-wide: `JUNIPER_DATA_EQUITIES_MAX_SYMBOLS` / `JUNIPER_DATA_EQUITIES_ALLOW_TRUNCATION`.

### What not to do

- Do not add a byte cap to this generator. Wire size and wall time do not scale together.
- Do not treat omitted `symbols` / omitted `max_symbols` as an unbounded S&P 500 pull. The default universe is 503 names and **refuses**.
- Do not treat `max_symbols=None` as unbounded. It means "no request-side limit"; the deployment ceiling still applies.
- Do not let a request raise the deployment ceiling. The effective cap is `min(requested, deployment)`.
- Do not restore `ordered[: params.max_symbols]` without the refusal / annotation pair. Silence is the failure mode this bound was warned about.
- Do not default `allow_truncation` on.
- Do not leave `"truncation"` in the NPZ. Pop it before checksum.
- Do not store `{}` for "complete". `pop_truncation_meta` returns `None`.
- Do not add a second resolver in `equities_seq`. Inherit both the bound and the record.
- Do not assume the default-universe prefix is the largest names — it is alphabetical.
- Do not call `Ticker.info`. The Yahoo path is `yf.download` only.
- Do not take `total_shares == 0` as "the company has no shares" under default fill.

### Pins (land with #354)

| Test | File | Guards |
|------|------|--------|
| `TestUniverseSymbolCap.test_oversized_universe_is_refused_by_default` | `tests/unit/test_equities_generator.py` | 40 names, no opt-in → `InputTooLargeError` (unit `symbols`, cap 14) |
| `test_opt_in_truncates_and_annotates` | same | Authorised cut writes `universe_exceeded_symbol_cap` + `imported == 14` |
| `test_the_kept_prefix_is_deterministic` | same | Reversed dict order still yields `sorted(universe)[:14]` |
| `test_request_cannot_RAISE_the_deployment_cap` | same | `max_symbols=9999` and `None` still clamp to 14 |
| `test_generate_puts_the_annotation_on_the_returned_arrays` | same | Channel key reaches `generate()` output; `records_imported` is a real row count |
| `test_generate_refuses_an_oversized_universe` | same | Refusal survives the full `generate()` path |
| `test_generate_omits_the_key_entirely_when_nothing_was_cut` | same | Under-cap has no `"truncation"` key |
| `test_default_cap_matches_the_measured_budget` | same | Constant stays 14; truncation default stays `false` |
| `util/ad-hoc/2026-09-04_apd_data_018_mutation_check.py` | ad-hoc | Mutation matrix spanning both generators |

`equities_seq` is covered by calling `EquitiesGenerator._resolve_symbols` and then attaching the same channel key. Re-measure with `util/ad-hoc/2026-09-04_measure_equities_payloads.py` before moving the constant. Keep SEC spacing at `_SEC_MIN_INTERVAL`. Full analysis: juniper-ml `notes/JUNIPER_2026-09-04_JUNIPER-DATA_EQUITIES-INGEST-SIZING-AND-FIELD-AVAILABILITY.md`.

---

---

## CI/CD Pipeline Reference

Relocated verbatim from `AGENTS.md` (P3 of the shared-session-memory plan) so it is read on demand rather than loaded into every session.

### GitHub Actions Workflows

| Workflow | File | Triggers | Purpose |
|----------|------|----------|---------|
| **CI** | `ci.yml` | Push, PR, daily schedule | Pre-commit, tests (3.12-3.14), coverage, security, type checking, docs |
| **CodeQL** | `codeql.yml` | Push, PR, schedule | GitHub code scanning |
| **Security Scan** | `security-scan.yml` | Push, PR | Gitleaks + Bandit SARIF |
| **Publish** | `publish.yml` | GitHub release | TestPyPI -> PyPI (Trusted Publishing/OIDC) |
| **Lockfile Update** | `lockfile-update.yml` | Schedule, manual | Update `requirements.lock` |
| **Sequence Safety** | `sequence-safety.yml` | PR | Advisory per-PR symbol-loss + docs-deletion screens via `juniper-ci-tools` (`--scope 'juniper_data/**'`); never required, never blocks a merge |
| **Main Verify** | `main-verify.yml` | Push (main) | Bypass-proof post-merge compositional-loss net (screens-only, advisory); stable-title failure-issue upsert + catch-up base |

### Pre-Commit Hooks

Configured in `.pre-commit-config.yaml`:

| Hook | Purpose |
|------|---------|
| Ruff (`ruff check --fix`) | Linting with auto-fix |
| Ruff (`ruff format`) | Code formatting |
| MyPy | Type checking |
| Bandit | Security scanning |
| yamllint | YAML validation |
| shellcheck | Shell script analysis |
| SOPS check | Block unencrypted `.env` files |
| General checks | Trailing whitespace, merge conflicts, YAML/TOML/JSON syntax |

### Coverage Gates

- **Aggregate**: 80% minimum (pyproject.toml `fail_under`)
- **Per-module**: 85% minimum (enforced by `scripts/check_module_coverage.py` in CI)
- **Branch coverage**: Enabled

---

### PR base-branch guard (required check)

`.github/workflows/pr-base-branch-guard.yml` fails any PR whose base branch is not the
default branch. Its job name -- **`Guard PR base branch`** -- is a **required status check**
in this repo's ruleset, so renaming the job or deleting the file makes `main` unmergeable
until the context is un-required first.

**What it protects against.** A PR based on another feature branch can squash-merge into
that branch, stranding its content off `main` behind a green **MERGED** badge. It has
happened three times in this ecosystem (`juniper-recurrence#7`/`#8`, `juniper-canopy#365`).

**Why it matters more than it looks.** Both rulesets here are scoped to `~DEFAULT_BRANCH`, so
a PR whose base is a feature branch is governed by **no ruleset at all** -- it has zero
required status checks and merges clean with nothing having run:

```bash
gh api repos/pcalnon/<repo>/rules/branches/feature%2Fanything --jq length   # -> 0
gh api repos/pcalnon/<repo>/rules/branches/main               --jq length   # -> 9
```

This workflow carries no `branches:` filter, so it is the **only** check that runs on such a
PR. It cannot block the merge there -- no ruleset applies -- but it turns a silent merge into
a visibly red one.

**If it fails.** Re-open the work against the default branch. The house practice is
**close and re-open** a fresh PR titled `[retarget #NNN]`. Retargeting in place is *not*
sufficient on its own: every `ci*.yml` here uses the default `pull_request` types
`[opened, synchronize, reopened]`, which exclude `edited`, so a retarget re-runs this guard
and nothing else -- the PR stays blocked on its other required contexts until a push or a
close/re-open.

**`stacked-pr` label.** Silences this guard for a deliberate stack. It does **not** make the
PR mergeable into `main`, and it does **not** re-land the stack -- do that separately.

Rollout and rationale: [juniper-ml#434](https://github.com/pcalnon/juniper-ml/issues/434).

---

## Additional Resources

### Internal Documentation

- [README.md](../README.md) -- Project overview
- [AGENTS.md](../AGENTS.md) -- Development guide
- [JUNIPER_DATA_API.md](api/JUNIPER_DATA_API.md) -- Full API documentation
- [CHANGELOG.md](../CHANGELOG.md) -- Version history

### External Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [NumPy NPZ Format](https://numpy.org/doc/stable/reference/generated/numpy.savez.html)
- [Ruff Documentation](https://docs.astral.sh/ruff/)

### Ecosystem Links

- [juniper-cascor](https://github.com/pcalnon/juniper-cascor) -- CasCor training service
- [juniper-canopy](https://github.com/pcalnon/juniper-canopy) -- Monitoring dashboard
- [juniper-data-client](https://github.com/pcalnon/juniper-data-client) -- Python client library
- [juniper-deploy](https://github.com/pcalnon/juniper-deploy) -- Docker orchestration
- [juniper-ml](https://github.com/pcalnon/juniper-ml) -- Meta-package (`pip install juniper-ml`)

---

## Version Compatibility

| juniper-data | Python | FastAPI | Pydantic | juniper-data-client |
|-------------|--------|---------|----------|---------------------|
| 0.5.x | >=3.12 | >=0.100.0 | >=2.0.0 | >=0.3.0 |

---

**Last Updated:** September 5, 2026
**Version:** 0.4.3
**Maintainer:** Paul Calnon
