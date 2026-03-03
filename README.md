# Juniper Data

Dataset generation and management service for the Juniper ecosystem.

## Overview

Juniper Data provides a centralized service for generating, storing, and serving datasets used by the Juniper neural network projects. It supports various dataset types including the classic two-spiral classification problem.

## Ecosystem Compatibility

This service is part of the [Juniper](https://github.com/pcalnon/juniper-ml) ecosystem.
Verified compatible versions:

| juniper-data | juniper-cascor | juniper-canopy | data-client | cascor-client | cascor-worker |
|---|---|---|---|---|---|
| 0.4.x | 0.3.x | 0.2.x | >=0.3.1 | >=0.1.0 | >=0.1.0 |

For full-stack Docker deployment and integration tests, see `juniper-deploy`.

## Architecture

JuniperData is the **foundational data layer** of the Juniper ecosystem. JuniperCascor and juniper-canopy both call JuniperData to generate and retrieve datasets.

```
┌─────────────────────┐     REST+WS      ┌──────────────────────┐
│   juniper-canopy     │ ◄──────────────► │    JuniperCascor     │
│   Dashboard         │                  │    Training Svc      │
│   Port 8050         │                  │    Port 8200         │
└──────────┬──────────┘                  └──────────┬───────────┘
           │ REST                                    │ REST
           ▼                                         ▼
┌──────────────────────────────────────────────────────────────┐
│                      JuniperData  ◄── (this service)          │
│                   Dataset Service  ·  Port 8100               │
└──────────────────────────────────────────────────────────────┘
```

**Data contract**: datasets are served as NPZ archives with keys `X_train`, `y_train`, `X_test`, `y_test`, `X_full`, `y_full` (all `float32`).

## Related Services

| Service | Relationship | Environment Variable |
|---------|-------------|---------------------|
| [juniper-cascor](https://github.com/pcalnon/juniper-cascor) | Consumes JuniperData for training datasets | `JUNIPER_DATA_URL=http://localhost:8100` |
| [juniper-canopy](https://github.com/pcalnon/juniper-canopy) | Consumes JuniperData for visualization data | `JUNIPER_DATA_URL=http://localhost:8100` |
| [juniper-data-client](https://github.com/pcalnon/juniper-data-client) | PyPI client library for this service | `pip install juniper-data-client` |

### Service Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `JUNIPER_DATA_HOST` | `0.0.0.0` | Listen address |
| `JUNIPER_DATA_PORT` | `8100` | Service port |
| `JUNIPER_DATA_LOG_LEVEL` | `INFO` | Log verbosity |

### Docker Deployment

```bash
# Full stack with all three services:
git clone https://github.com/pcalnon/juniper-deploy.git  # (private repository)
cd juniper-deploy && docker compose up --build
```

## Dependency Lockfile

The `requirements.lock` file pins exact dependency versions for reproducible Docker builds. The `pyproject.toml` retains flexible `>=` ranges for local development.

**Regenerate after changing dependencies in `pyproject.toml`:**

```bash
uv pip compile pyproject.toml --extra api --extra observability -o requirements.lock
```

## Installation

### Basic Installation

```bash
pip install -e .
```

### With API Support

```bash
pip install -e ".[api]"
```

### Development Installation

```bash
pip install -e ".[dev]"
```

### Full Installation

```bash
pip install -e ".[all]"
```

## Quick Start

### Generate a Spiral Dataset

```python
from juniper_data.generators.spiral import SpiralGenerator

generator = SpiralGenerator()
dataset = generator.generate(n_points=100, n_spirals=2, noise=0.1)
```

### Start the API Server

```bash
uvicorn juniper_data.api.app:app --reload
```

## API Endpoints

| Endpoint                        | Method | Description                        |
| ------------------------------- | ------ | ---------------------------------- |
| `/v1/health`                    | GET    | Health check endpoint              |
| `/v1/datasets`                  | GET    | List available datasets            |
| `/v1/datasets/{id}`             | GET    | Get a specific dataset             |
| `/v1/generators/spiral`         | POST   | Generate a new spiral dataset      |
| `/v1/generators/spiral/config`  | GET    | Get spiral generator configuration |

## Project Structure

```bash
juniper-data/
├── juniper_data/
│   ├── core/           # Core functionality and base classes
│   ├── generators/     # Dataset generators
│   │   └── spiral/     # Spiral dataset generator
│   ├── storage/        # Dataset persistence layer
│   └── api/            # FastAPI application
│       └── routes/     # API route handlers
├── tests/
│   ├── unit/           # Unit tests
│   └── integration/    # Integration tests
├── pyproject.toml      # Project configuration
└── README.md           # This file
```

## Development

### Running Tests

```bash
pytest
```

### Running Tests with Coverage

```bash
pytest --cov=juniper_data --cov-report=html
```

### Code Formatting

```bash
ruff format juniper_data tests
ruff check --fix juniper_data tests
```

### Type Checking

```bash
mypy juniper_data
```

## Juniper Ecosystem

| Repository | Description |
|-----------|-------------|
| [juniper-data](https://github.com/pcalnon/juniper-data) | Dataset generation service (this repo) |
| [juniper-cascor](https://github.com/pcalnon/juniper-cascor) | CasCor neural network training service |
| [juniper-canopy](https://github.com/pcalnon/juniper-canopy) | Real-time monitoring dashboard |
| [juniper-data-client](https://github.com/pcalnon/juniper-data-client) | PyPI: `juniper-data-client` |
| [juniper-cascor-client](https://github.com/pcalnon/juniper-cascor-client) | PyPI: `juniper-cascor-client` |
| [juniper-cascor-worker](https://github.com/pcalnon/juniper-cascor-worker) | PyPI: `juniper-cascor-worker` |

## License

MIT License - Copyright (c) 2024-2026 Paul Calnon

## Git Leaks

![gitleaks badge](https://img.shields.io/badge/protected%20by-gitleaks-blue)
