# Juniper Data Reference

**Version:** 0.4.2
**Status:** Active
**Last Updated:** March 3, 2026
**Project:** Juniper Data - Dataset Generation Service

---

## Table of Contents

- [API Reference](#api-reference)
- [Configuration Reference](#configuration-reference)
- [Command Reference](#command-reference)
- [Test Reference](#test-reference)
- [Code Quality Tools](#code-quality-tools)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Error Codes](#error-codes)
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
| `/v1/datasets/{id}` | GET | Dataset metadata | Yes* |
| `/v1/datasets/{id}` | DELETE | Delete dataset | Yes* |
| `/v1/datasets/{id}/artifact` | GET | Download NPZ | Yes* |
| `/v1/datasets/{id}/preview` | GET | Preview JSON | Yes* |

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
| `JUNIPER_DATA_RATE_LIMIT_ENABLED` | bool | `false` | Enable request rate limiting |
| `JUNIPER_DATA_RATE_LIMIT_REQUESTS_PER_MINUTE` | int | `60` | Requests per minute per client |
| `JUNIPER_DATA_CORS_ORIGINS` | JSON list | `["*"]` | Allowed CORS origins |

#### Observability Settings

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `JUNIPER_DATA_METRICS_ENABLED` | bool | `false` | Enable Prometheus metrics endpoint |
| `JUNIPER_DATA_SENTRY_DSN` | string | *(none)* | Sentry DSN for error tracking |

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

## Command Reference

### Service Commands

```bash
# Start development server (with auto-reload)
python -m juniper_data --reload

# Start production server
uvicorn juniper_data.api.app:app --host 0.0.0.0 --port 8100

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
uv pip compile pyproject.toml --extra api --extra observability -o requirements.lock
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
| `422 Unprocessable Entity` | Validation error | Pydantic validation failure |
| `429 Too Many Requests` | Rate limited | Exceeded requests/minute |
| `500 Internal Server Error` | Server error | Unexpected exception |

### Error Response Format

```json
{
  "detail": "Error message describing what went wrong"
}
```

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

**Last Updated:** March 3, 2026
**Version:** 0.4.2
**Maintainer:** Paul Calnon
