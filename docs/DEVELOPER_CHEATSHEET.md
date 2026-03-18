# Developer Cheatsheet -- juniper-data

**Version**: 0.4.2 | **Date**: 2026-03-15 | **Project**: juniper-data -- Dataset Generation REST Service (FastAPI)

---

## Common Commands

| Command                                                                                | Description                                                                     |
|----------------------------------------------------------------------------------------|---------------------------------------------------------------------------------|
| `conda activate JuniperData`                                                           | Activate conda environment (Python 3.14)                                        |
| `pip install -e ".[all]"`                                                              | Install all extras (api, dev, test, observability, arc-agi)                     |
| `python -m juniper_data`                                                               | Start dev server on port 8100                                                   |
| `uvicorn juniper_data.api.app:app --host 0.0.0.0 --port 8100`                          | Production server                                                               |
| `PYTHON_GIL=0 uvicorn juniper_data.api.app:app --host 0.0.0.0 --port 8100`             | Production server, GIL disabled (httptools.parser.parser compatability unknown) |
| `uvicorn -Xgil=0 juniper_data.api.app:app --host 0.0.0.0 --port 8100`                  | Production server, GIL disabled (httptools.parser.parser compatability unknown) |
| `pytest`                                                                               | Run all tests                                                                   |
| `pytest -m unit` / `integration` / `generators` / `storage` / `api`                    | Run by marker                                                                   |
| `pytest --cov=juniper_data --cov-report=term-missing --cov-fail-under=80`              | Coverage                                                                        |
| `ruff check juniper_data`                                                              | Lint (replaces flake8)                                                          |
| `ruff check --fix juniper_data`                                                        | Lint with auto-fix                                                              |
| `ruff format juniper_data`                                                             | Format (replaces black)                                                         |
| `mypy juniper_data --ignore-missing-imports`                                           | Type check                                                                      |
| `pre-commit run --all-files`                                                           | Run all pre-commit hooks                                                        |
| `uv pip compile pyproject.toml --extra api --extra observability -o requirements.lock` | Regenerate lockfile                                                             |

---

## API Endpoints

| Endpoint                                            | Method                    | Auth* |
|-----------------------------------------------------|---------------------------|-------|
| `/v1/health`, `/v1/health/live`, `/v1/health/ready` | GET                       | No    |
| `/v1/generators`                                    | GET                       | Yes   |
| `/v1/generators/{name}/schema`                      | GET                       | Yes   |
| `/v1/datasets`                                      | POST (create), GET (list) | Yes   |
| `/v1/datasets/{id}`                                 | GET (meta), DELETE        | Yes   |
| `/v1/datasets/{id}/artifact`                        | GET (NPZ download)        | Yes   |
| `/v1/datasets/{id}/preview`                         | GET (JSON preview)        | Yes   |
| `/v1/datasets/filter`                               | GET (filter datasets)     | Yes   |
| `/v1/datasets/stats`                                | GET (dataset statistics)  | Yes   |
| `/v1/datasets/batch-delete`                         | POST (bulk delete)        | Yes   |
| `/v1/datasets/cleanup-expired`                      | POST (cleanup expired)    | Yes   |
| `/v1/datasets/{id}/tags`                            | PATCH (update tags)       | Yes   |

*Auth required only when `JUNIPER_DATA_API_KEYS` is set.

### Add/Modify an Endpoint

1. Create route: `juniper_data/api/routes/my_resource.py` with `APIRouter(prefix="/my-resource")`
2. Add Pydantic v2 request/response models
3. Register in `app.py`: `app.include_router(my_resource_router, prefix="/v1")`
4. Add tests, run `pytest -v`

### Add Middleware

Stack order in `create_app()`: CORS -> SecurityMiddleware -> PrometheusMiddleware -> RequestIdMiddleware. Add new class in `juniper_data/api/`, register in `app.py` (outermost runs first).

> See: [JUNIPER_DATA_API.md](api/JUNIPER_DATA_API.md) | [API_SCHEMAS.md](api/API_SCHEMAS.md)

---

## Environment Variables

All use `JUNIPER_DATA_` prefix (pydantic-settings in `juniper_data/api/settings.py`).

| Variable                                      | Default           | Description                                   |
|-----------------------------------------------|-------------------|-----------------------------------------------|
| `JUNIPER_DATA_HOST`                           | `127.0.0.1`       | Listen address                                |
| `JUNIPER_DATA_PORT`                           | `8100`            | Service port                                  |
| `JUNIPER_DATA_STORAGE_PATH`                   | `./data/datasets` | Artifact storage directory                    |
| `JUNIPER_DATA_LOG_LEVEL`                      | `INFO`            | DEBUG, INFO, WARNING, ERROR, CRITICAL         |
| `JUNIPER_DATA_LOG_FORMAT`                     | `text`            | `text` or `json` (structured JSON logging)    |
| `JUNIPER_DATA_API_KEYS`                       | *(none)*          | Comma-separated API keys; unset = open access |
| `JUNIPER_DATA_RATE_LIMIT_ENABLED`             | `true`            | Enable rate limiting                          |
| `JUNIPER_DATA_RATE_LIMIT_REQUESTS_PER_MINUTE` | `60`              | Max requests/min per client                   |
| `JUNIPER_DATA_CORS_ORIGINS`                   | `[]`              | Allowed CORS origins                          |
| `JUNIPER_DATA_METRICS_ENABLED`                | `false`           | Prometheus `/metrics` endpoint                |
| `JUNIPER_DATA_SENTRY_DSN`                     | *(none)*          | Sentry DSN for error tracking                 |

**Add a setting:** Add field to `Settings` in `settings.py`, define `_JUNIPER_DATA_*` default constant. Auto-maps to `JUNIPER_DATA_<FIELD>` env var.

> See: [REFERENCE.md -- Configuration](REFERENCE.md#configuration-reference)

---

## Storage Backends

Seven backends extend `DatasetStore` (`juniper_data/storage/base.py`):

| Backend                    | Class                     | Requires   |
|----------------------------|---------------------------|------------|
| Local filesystem (default) | `LocalFSDatasetStore`     | *(core)*   |
| In-memory                  | `InMemoryDatasetStore`    | *(core)*   |
| Cached wrapper             | `CachedDatasetStore`      | *(core)*   |
| PostgreSQL                 | `PostgresDatasetStore`    | `psycopg2` |
| Redis                      | `RedisDatasetStore`       | `redis`    |
| HuggingFace Hub            | `HuggingFaceDatasetStore` | `datasets` |
| Kaggle                     | `KaggleDatasetStore`      | `kaggle`   |

Default filesystem layout: `{JUNIPER_DATA_STORAGE_PATH}/{dataset_id}.meta.json` + `.npz`. Optional backends use lazy imports; missing packages degrade gracefully. Factory helpers: `get_redis_store()`, `get_hf_store()`, `get_postgres_store()`, `get_kaggle_store()`.

> See: `juniper_data/storage/__init__.py` | `juniper_data/storage/base.py`

---

## Generator Registration

Generators: `spiral`, `xor`, `gaussian`, `circles`, `checkerboard`, `csv_import`, `mnist`, `arc_agi`.

**Add a new generator:**

1. Create subpackage `juniper_data/generators/<name>/`
2. Implement class following `SpiralGenerator` pattern
3. Register in `generators/__init__.py`
4. Add routes or use existing `/v1/datasets` with new generator name
5. Output must conform to NPZ contract below
6. Add tests and schema

> See: `juniper_data/generators/__init__.py` | [AGENTS.md -- Adding New Generators](../AGENTS.md#adding-new-generators)

---

## Data Contract (NPZ)

All arrays `float32`. Keys: `X_train`, `y_train`, `X_test`, `y_test`, `X_full`, `y_full`. Shapes: features `(n, n_features)`, labels `(n, n_classes)` one-hot encoded.

> See: [REFERENCE.md -- NPZ Artifact Keys](REFERENCE.md#npz-artifact-keys)

---

## Testing

| Marker                                      | Scope                             |
|---------------------------------------------|-----------------------------------|
| `unit`                                      | Fast isolated tests               |
| `integration`                               | Full workflow with real storage   |
| `performance`                               | Benchmarks (`--benchmark-enable`) |
| `slow`                                      | Tests > 1 second                  |
| `spiral` / `api` / `generators` / `storage` | Component-specific                |

Coverage thresholds: 80% fail-under (`pyproject.toml`), 95% aggregate + 85% per-module (pre-push hook).

> See: [REFERENCE.md -- Test Reference](REFERENCE.md#test-reference)

---

## Code Quality (Ruff)

juniper-data uses **ruff** (NOT black/isort/flake8). Config in `pyproject.toml`: line-length 320, target Python 3.12+ (py312), rule sets E, W, F, B, C4, I, UP, SIM, T20.

> See: [REFERENCE.md -- Code Quality Tools](REFERENCE.md#code-quality-tools)

---

## Logging and Observability

Metrics use `juniper_data_` namespace. Pattern: `juniper_data_<subsystem>_<name>_<unit>`. Add custom metrics in `juniper_data/api/observability.py` using `prometheus_client` (Counter, Gauge, Histogram).

> See: `juniper_data/api/observability.py` | [Observability Guide](../../juniper-deploy/docs/OBSERVABILITY_GUIDE.md)

---

## CI/CD and Pre-commit

| Hook                 | Stage        | Tool                                                                 |
|----------------------|--------------|----------------------------------------------------------------------|
| Lint                 | pre-commit   | `ruff check --fix`                                                   |
| Format               | pre-commit   | `ruff-format`                                                        |
| Type check (prod)    | pre-commit   | `mypy` (`--ignore-missing-imports --no-strict-optional`)             |
| Type check (tests)   | pre-commit   | `mypy` (relaxed)                                                     |
| Security scan        | pre-commit   | `bandit` (skips B101, B311)                                          |
| YAML/TOML validation | pre-commit   | `pre-commit-hooks` (check-yaml, check-toml, end-of-file-fixer, etc.) |
| YAML lint            | pre-commit   | `yamllint`                                                           |
| Shell lint           | pre-commit   | `shellcheck`                                                         |
| Coverage gate        | **pre-push** | 95% aggregate, 85% per-module                                        |
| SOPS guard           | pre-commit   | Block unencrypted `.env` files                                       |

GitHub Actions: `ci.yml`, `publish.yml`, `security-scan.yml`, `codeql.yml`, `lockfile-update.yml`.

```bash
pre-commit install                       # install hooks (one-time)
pre-commit install --hook-type pre-push  # coverage gate (one-time)
```

---

## Troubleshooting

| Symptom                 | Cause              | Fix                                                      |
|-------------------------|--------------------|----------------------------------------------------------|
| `ruff` not found        | Dev extras missing | `pip install -e ".[dev]"`                                |
| 401 Unauthorized        | API keys set       | Pass `X-API-Key` header or unset `JUNIPER_DATA_API_KEYS` |
| 429 Too Many Requests   | Rate limiter       | Wait, or `JUNIPER_DATA_RATE_LIMIT_ENABLED=false`         |
| Storage path error      | Dir missing        | Set `JUNIPER_DATA_STORAGE_PATH` to writable path         |
| `ImportError: redis`    | Optional backend   | `pip install redis`                                      |
| Coverage pre-push fails | Below threshold    | Add tests; see `scripts/check_module_coverage.py`        |

---

## Cross-References

- [API Reference](api/JUNIPER_DATA_API.md) | [Full Reference](REFERENCE.md) | [Quick Start](QUICK_START.md) | [AGENTS.md](../AGENTS.md) | [Ecosystem Cheatsheet](../../juniper-ml/notes/DEVELOPER_CHEATSHEET.md) | [Juniper CLAUDE.md](../../CLAUDE.md)
