---
applyTo: '**'
---

# Copilot Instructions for juniper-data

Dataset generation and management REST service for the Juniper ecosystem. Generates, stores, and serves datasets as NPZ artifacts consumed by juniper-cascor and juniper-canopy.

## Project Overview

- **Framework**: FastAPI + uvicorn
- **Core libraries**: numpy, pydantic, pydantic-settings
- **Python**: >=3.12
- **Port**: 8100 (default)
- **API prefix**: `/v1`
- **Conda environment**: `JuniperData`

## Architecture

```
juniper_data/
├── core/           # Base classes, exceptions, configuration
├── generators/     # 8 dataset generators (spiral, xor, gaussian, circles,
│                   #   checkerboard, csv_import, mnist, arc_agi)
├── storage/        # NPZ artifact persistence and retrieval
├── api/
│   ├── app.py      # Factory-pattern app creation (create_app)
│   ├── settings.py # Pydantic BaseSettings with JUNIPER_DATA_ prefix
│   └── routes/     # health, generators, datasets endpoint modules
├── __init__.py     # Package version (0.4.2)
└── __main__.py     # CLI entry point (python -m juniper_data)
```

## Key Patterns

### Generator Registration

Each generator lives in its own subpackage under `generators/` with:
- `generator.py` — Class with `generate(params) -> dict` class method returning float32 arrays
- `params.py` — Pydantic model for parameter validation
- `__init__.py` — Exports `VERSION`, generator class, and params class

All 8 generators are registered in `api/routes/generators.py`.

### App Factory

`juniper_data.api.app` uses a lifespan handler for startup/shutdown. Middleware stack: RequestId, Prometheus (optional), Security (API keys + rate limiting), CORS.

### Configuration

Pydantic BaseSettings with `JUNIPER_DATA_` env var prefix. Key settings: `host`, `port`, `storage_path`, `log_level`, `api_keys`, `cors_origins`, `rate_limit_enabled`, `metrics_enabled`, `sentry_dsn`.

### Data Contract

NPZ artifacts with keys: `X_train`, `y_train`, `X_test`, `y_test`, `X_full`, `y_full` (all `float32`).

## Code Style

- **Formatter/linter**: ruff (replaces black, isort, flake8, pyupgrade)
- **Line length**: 120 characters
- **Docstrings**: Google-style format
- **Type hints**: Required on all public methods
- **Naming**: PascalCase classes, snake_case functions, UPPER_SNAKE constants

## Testing

- **Framework**: pytest with markers: `unit`, `integration`, `api`, `generators`, `storage`, `spiral`
- **Coverage**: 80% threshold (`--cov-fail-under=80`)
- **Test location**: `juniper_data/tests/unit/` and `juniper_data/tests/integration/`

```bash
pytest                          # All tests
pytest -m unit                  # Unit tests only
pytest -m api                   # API endpoint tests
pytest --cov=juniper_data       # With coverage
```

## API Conventions

- REST resource nouns: `/v1/datasets`, `/v1/generators`
- Proper HTTP methods and status codes (201 for creation, 204 for deletion)
- Health probes: `/v1/health`, `/v1/health/live`, `/v1/health/ready`
- Dataset generation cached by parameter hash
- Optional TTL expiration and tagging per dataset

## Running

```bash
python -m juniper_data                                          # Development
uvicorn juniper_data.api.app:app --host 0.0.0.0 --port 8100    # Production
```
