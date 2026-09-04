# Juniper Data Reference

**Version:** 0.4.2
**Status:** Active
**Last Updated:** March 3, 2026
**Project:** Juniper Data - Dataset Generation Service

---

## Table of Contents

- [API Reference](#api-reference)
- [Configuration Reference](#configuration-reference)
- [Storage Backend Notes](#storage-backend-notes)
- [Command Reference](#command-reference)
- [Test Reference](#test-reference)
- [Code Quality Tools](#code-quality-tools)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Error Codes](#error-codes)
- [Project Architecture Reference](#project-architecture-reference)
- [API Design Reference](#api-design-reference)
- [Storage Backend Reference](#storage-backend-reference)
- [Prometheus Collector Reference](#prometheus-collector-reference)
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
| `generators/csv_import/` | Import datasets from CSV/JSON files |
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

---

## API Design Reference

Relocated verbatim from `AGENTS.md` (P3 of the shared-session-memory plan) so it is read on demand rather than loaded into every session.

### REST Conventions

- Use nouns for resources: `/datasets`, `/generators`
- All endpoints prefixed with `/v1/`
- Use HTTP methods appropriately: GET, POST, PATCH, DELETE
- Return proper status codes (200, 201, 204, 400, 404, 413, 429, 500)
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

---

## Storage Backend Reference

Relocated verbatim from `AGENTS.md` (P3 of the shared-session-memory plan) so it is read on demand rather than loaded into every session.

JuniperData supports 7 storage backend implementations with a composable architecture.

### Abstract Interface

`DatasetStore` (in `storage/base.py`) defines the standard interface:

- **Core**: `save()`, `get_meta()`, `get_artifact_bytes()`, `exists()`, `delete()`, `list_datasets()`
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

---

## Equities Symbol Cap

`equities` / `equities_seq` have no input file. `generate()` fans out HTTP, **one ticker at a time**, to the two upstreams the generator actually uses: Yahoo daily OHLCV (`yfinance.download`, chart) and SEC EDGAR XBRL `companyconcept` (`_fetch_shares`). APD-DATA-018 therefore bounds this half on **symbol count**, not bytes. The csv_import half is a file and took a byte cap (`CSV_IMPORT_DEFAULT_MAX_BYTES` in `juniper_data/core/limits.py`).

**Why not bytes.** Call count is `O(symbols)`, not `O(rows)`. Horizon (`start_date` → `end_date`) grows the Yahoo payload; it does not add requests. A 1-day × large-universe request is small on the wire and long on the wall; a 1-symbol × 26-year request is the opposite. A byte cap would accept the slow request and refuse the cheap one.

**The knob is off by default.** `EquitiesParams.max_symbols` defaults to `EQUITIES_DEFAULT_MAX_SYMBOLS`, which is `None` (no cap). There is no `Settings` / `JUNIPER_DATA_*` ceiling — unlike csv_import, a request is the only way to bound the fan-out. `equities_seq` inherits the same field (`EquitiesSeqParams` subclasses `EquitiesParams`) and the same resolver.

### What `_resolve_symbols` does

1. If `params.symbols` is set: strip, uppercase, keep **caller order**. Unknown tickers get a CIK from SEC `company_tickers.json` (cached) or `cik=None`.
2. Else: the bundled snapshot `generators/equities/sp500_constituents.csv` (**503** names), ordered by `sorted(constituents)` — alphabetical ticker, not market cap.
3. If `max_symbols` is an int (`ge=1`): `ordered = ordered[: params.max_symbols]`. Prefix only.

The slice is **silent**. It does not raise `InputTooLargeError`, does not return HTTP 422, and does not write `DatasetMeta.truncation`. Dropped tickers never appear in `ticker_vocab`. A later download failure is also silent (`_logger.warning` + skip); only an empty conditioned set raises `ValueError`.

### Per-symbol work (cold cache)

For each remaining ticker, `_condition_one` does:

- one `yf.download(...)` (class shares mapped `BRK.B` → `BRK-B`)
- if a CIK is known: 1–2 SEC GETs (`dei.EntityCommonStockSharesOutstanding`, then `us-gaap.CommonStockSharesOutstanding`), spaced by `_SEC_MIN_INTERVAL = 0.12` s (SEC's published 10 req/s)
- 52-week rolling window, cost basis, next-day targets

`use_cache` defaults `True` (`~/.cache/juniper_data/equities`, override `JUNIPER_DATA_EQUITIES_CACHE_DIR`). The timeout risk is the **uncached** path. Generation still runs inside the HTTP request; `core/limits.py` sized the csv_import half against a ~30 s client budget. Uncached full-universe S&P 500 does not fit that budget.

Yahoo is `download` only — the generator does not call `Ticker.info` / `quoteSummary`. Missing SEC facts return `None` (404 is not an error); `total_shares` is NaN and default `fundamentals_fill="zero"` writes `0.0`. The artifact looks complete.

Requires `pip install "juniper-data[equities]"`. Unavailable extra → `501` from the route.

### Operator usage

```python
from juniper_data.generators.equities.params import EquitiesParams
from juniper_data.generators.equities.generator import EquitiesGenerator

# Cold path: bound the fan-out. Two alphabetical S&P names if symbols is omitted.
params = EquitiesParams(max_symbols=2, start_date="2024-01-01", end_date="2024-06-01")
arrays = EquitiesGenerator.generate(params)

# Explicit list, then cap (keeps caller order): AAPL, MSFT — not AMZN.
EquitiesParams(symbols=["AAPL", "MSFT", "AMZN"], max_symbols=2)
```

Via `POST /v1/datasets`: `"generator": "equities"` / `"equities_seq"` and `"params": {"max_symbols": 8}`. Omitting the field is the full resolved universe.

### What not to do

- Do not add a byte cap to this generator. Wire size and wall time do not scale together.
- Do not treat omitted `max_symbols` as safe on a cold cache against the bundled 503 names.
- Do not assume the default-universe slice is the largest names — it is alphabetical.
- Do not expect 422 or `meta.truncation` when the universe is shortened. That channel is csv_import's byte-cap annotation.
- Do not raise `InputTooLargeError` from `_resolve_symbols`. The route's 422 mapping is for csv_import.
- Do not call `Ticker.info` to "enrich" a row. The generator's Yahoo path is `yf.download` only.
- Do not take `total_shares == 0` as "the company has no shares" under default fill — it is also "SEC returned no facts".

### Pins (on `main`)

| Test | File | Guards |
|------|------|--------|
| `test_resolve_symbols_respects_max_symbols` | `tests/unit/test_equities_generator.py` | Prefix slice of `sorted(constituents)` |
| `test_resolve_symbols_defaults_to_full_universe` | same | `max_symbols=None` keeps every name, sorted |
| `test_resolve_symbols_uses_sec_map_for_unknown` | same | Caller-supplied tickers not in the snapshot still resolve |
| `test_generate_skips_ticker_whose_download_raises` | same | One failed download does not abort the batch |

`equities_seq` is covered by calling `EquitiesGenerator._resolve_symbols` — there is no second resolver.

### Re-measuring the unit

PR #348 adds two ad-hoc scripts (not on `main` until that PR merges). They separate wire bytes from wall time on a handful of tickers and project across index universes. Re-run them before changing `EQUITIES_DEFAULT_MAX_SYMBOLS` off `None`. They must stay gentle: few tickers, and SEC spacing must keep using `_SEC_MIN_INTERVAL`. Full analysis: juniper-ml `notes/JUNIPER_2026-09-04_JUNIPER-DATA_EQUITIES-INGEST-SIZING-AND-FIELD-AVAILABILITY.md`.

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

**Last Updated:** April 1, 2026
**Version:** 0.4.2
**Maintainer:** Paul Calnon
