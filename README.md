<!-- markdownlint-disable MD013 MD033 MD041 -->
<!--
  MD013 (line-length): README contains prose paragraphs that intentionally
                       exceed the 512-char ecosystem limit. Disabled file-wide
                       since wrapping mid-sentence harms PyPI rendering.
  MD033 (no-inline-html): The right-aligned logo + spacing rely on HTML.
  MD041 (first-line-heading): The HTML logo is the first line by design.
-->
<div align="right" width="150px" height="150px" align="right" valign="top"> <img src="images/Juniper_Logo_150px.png" alt="Juniper" align="right" valign="top" width="150px" /></div>
<br /> <br /> <br /> <br />

# Juniper: Dynamic Neural Network Research Platform

Juniper is an AI/ML research platform for investigating dynamic neural network architectures and novel learning paradigms.  The project emphasizes ground-up implementations from primary literature, enabling a more transparent exploration of fundamental algorithms.

## Juniper Data

`juniper-data` is the **dataset-generation service** of the Juniper platform. It is a FastAPI service that produces NPZ-formatted datasets from a catalogue of generators — including the classic two-spiral and concentric-circles problems, XOR and Gaussian mixtures, a CSV/JSON import path, MNIST/Fashion-MNIST, and the ARC-AGI visual-reasoning task families — and serves them through a REST surface that supports a named-version registry, batch creation and export, tag-based filtering, and per-dataset preview. `juniper-data` is the upstream of both `juniper-cascor` (training) and `juniper-canopy` (visualisation): the dataset identifiers it returns are the substrate on which the rest of the platform conducts comparative work.

## Distribution

`juniper-data` is published on PyPI as **[`juniper-data`](https://pypi.org/project/juniper-data/)**.
The package is also surfaced through the platform meta-distribution
**[`juniper-ml`](https://pypi.org/project/juniper-ml/)**, which installs
the full client stack via `pip install juniper-ml[all]`.

```bash
pip install juniper-data
```

## Ecosystem Compatibility

This service is part of the [Juniper](https://github.com/pcalnon/juniper-ml) ecosystem.
Verified compatible versions:

| juniper-data | juniper-cascor | juniper-canopy | data-client | cascor-client | cascor-worker |
|--------------|----------------|----------------|-------------|---------------|---------------|
| 0.6.x        | 0.4.x          | 0.4.x          | >=0.4.1     | >=0.4.0       | >=0.3.0       |

For full-stack Docker deployment and integration tests, see [`juniper-deploy`](https://github.com/pcalnon/juniper-deploy).

## Architecture

`juniper-data` is the **foundational data layer** of the Juniper ecosystem. Both `juniper-cascor` and `juniper-canopy` call `juniper-data` to generate, version, and retrieve datasets.

```text
┌─────────────────────┐     REST+WS      ┌──────────────────────┐
│   juniper-canopy    │ ◄──────────────► │    juniper-cascor    │
│   Dashboard         │                  │    Training Svc      │
│   Port 8050         │                  │    Port 8200         │
└──────────┬──────────┘                  └──────────┬───────────┘
           │ REST                                   │ REST
           ▼                                        ▼
┌──────────────────────────────────────────────────────────────┐
│                  juniper-data  ◄── (this service)            │
│                  Dataset Service · Port 8100                 │
└──────────────────────────────────────────────────────────────┘
```

**Data contract**: datasets are served as NPZ archives with the keys `X_train`, `y_train`, `X_test`, `y_test`, `X_full`, `y_full`, all of dtype `float32`.

## Related Services

| Service | Relationship | Notes |
|---------|-------------|-------|
| [juniper-cascor](https://github.com/pcalnon/juniper-cascor) | Consumes `juniper-data` for training datasets | Set `JUNIPER_DATA_URL` |
| [juniper-canopy](https://github.com/pcalnon/juniper-canopy) | Consumes `juniper-data` for visualisation data | Set `JUNIPER_DATA_URL` |
| [juniper-data-client](https://github.com/pcalnon/juniper-data-client) | Python HTTP client for this service | `pip install juniper-data-client` |

## Service Configuration

Configuration is sourced from `juniper_data/api/settings.py` (Pydantic `BaseSettings`, `env_prefix="JUNIPER_DATA_"`). The complete env-var surface is listed below.

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `JUNIPER_DATA_HOST` | No | `127.0.0.1` | Bind address (override to `0.0.0.0` for Docker) |
| `JUNIPER_DATA_PORT` | No | `8100` | Service port |
| `JUNIPER_DATA_STORAGE_PATH` | No | `./data/datasets` | Filesystem path for persisted dataset artifacts |
| `JUNIPER_DATA_IMPORT_DIR` | No | `/data/imports` | Filesystem path for CSV/JSON imports |
| `JUNIPER_DATA_LOG_LEVEL` | No | `INFO` | Log verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `JUNIPER_DATA_LOG_FORMAT` | No | `text` | `text` or `json` (structured logging) |
| `JUNIPER_DATA_CORS_ORIGINS` | No | `[]` | Allowed CORS origins |
| `JUNIPER_DATA_API_KEYS` | No | `None` | Comma-separated or JSON-array API keys; authentication disabled when unset; Docker-secrets file path supported via the implicit `*_FILE` convention |
| `JUNIPER_DATA_RATE_LIMIT_ENABLED` | No | `true` | Enforce per-IP request rate limiting |
| `JUNIPER_DATA_RATE_LIMIT_REQUESTS_PER_MINUTE` | No | `60` | Per-IP rate limit |
| `JUNIPER_DATA_SENTRY_DSN` | No | `None` | Sentry DSN for error tracking |
| `JUNIPER_DATA_SENTRY_SEND_PII` | No | `false` | Whether Sentry should send personally identifiable information |
| `JUNIPER_DATA_SENTRY_TRACES_SAMPLE_RATE` | No | `0.1` | Sentry tracing sample rate |
| `JUNIPER_DATA_METRICS_ENABLED` | No | `false` | Expose `/metrics` for Prometheus scraping |
| `JUNIPER_DATA_METRICS_TRUSTED_IPS` | No | `["127.0.0.1", "::1"]` | IPs allowed to scrape `/metrics` |

## Docker Deployment

```bash
# Full stack (recommended) — see juniper-deploy:
git clone https://github.com/pcalnon/juniper-deploy.git  # (private repository)
cd juniper-deploy && docker compose up --build

# Standalone:
docker build -t juniper-data:latest .
docker run --rm -p 8100:8100 -e JUNIPER_DATA_HOST=0.0.0.0 juniper-data:latest
```

The Dockerfile is multi-stage (Python 3.14-slim builder + runtime). Container health is probed against `/v1/health/ready`.

## Dependency Lockfile

The `requirements.lock` file pins exact dependency versions for reproducible Docker builds. The `pyproject.toml` retains flexible `>=` ranges for local development.

Regenerate after changing dependencies in `pyproject.toml`:

```bash
uv pip compile pyproject.toml --extra api --extra observability -o requirements.lock
```

The ecosystem-wide lockfile-freshness gate enforces regeneration on every PR that touches `pyproject.toml`; if regeneration triggers the self-pin trap of `uv pip compile -o requirements.lock` reading the existing file, compile to `/tmp/requirements.lock` and `mv` into place.

## Active Research Components

`juniper-data` contributes three research components to the Juniper platform: the **ARC-AGI dataset families** (ARC-AGI-1 and ARC-AGI-2), loadable from the Hugging Face Hub or from local copies and exposed through the same NPZ-artifact contract as the simpler generators, which makes them directly usable as the substrate for comparative architecture-growth experiments; the **named-version dataset registry** (`POST /v1/datasets` with a `name` parameter auto-increments `meta.dataset_version`; `GET /v1/datasets/versions` and `/v1/datasets/latest` resolve the history), which gives experiments reproducible dataset references rather than opaque UUIDs; and the **dataset-API surface** itself — preview, filtering by tags, batch operations, and tag-based metadata queries — which together comprise the operational interface through which platform users compose and curate dataset corpora. The implementation of these surfaces is engineering rather than research; the **availability** of curated datasets and stable versioned references is itself the research artifact.

## Quick Start Guide

### Prerequisites

- Python ≥ 3.12 (Docker image uses 3.14)
- Conda environment `JuniperData`
- For ARC-AGI loading from the Hub: internet access at first load; subsequent loads are cached

### Installation

```bash
git clone https://github.com/pcalnon/juniper-data.git
cd juniper-data
conda activate JuniperData
pip install -e ".[all]"
```

The PyPI release is installable via `pip install juniper-data`; the editable-clone form above is the standard for active development. The optional-dependency extras are `api`, `arc-agi`, `observability`, `test`, `dev`, and `all`.

### Verification

Start the service:

```bash
uvicorn juniper_data.api.app:app --reload
```

Confirm the service responds:

```bash
curl http://localhost:8100/v1/health
curl http://localhost:8100/v1/health/ready
curl http://localhost:8100/v1/generators
```

Generate a small dataset directly from Python:

```python
from juniper_data.generators.spiral import SpiralGenerator

generator = SpiralGenerator()
dataset = generator.generate(n_points=100, n_spirals=2, noise=0.1)
```

### Next Steps

- [`docs/QUICK_START.md`](docs/QUICK_START.md) — complete installation guide
- [`docs/USER_MANUAL.md`](docs/USER_MANUAL.md) — comprehensive usage guide
- [`docs/api/JUNIPER_DATA_API.md`](docs/api/JUNIPER_DATA_API.md) — full REST endpoint reference (filtering, batch operations, tagging, versioning)
- [`juniper-deploy`](https://github.com/pcalnon/juniper-deploy) — Docker Compose orchestration for the full-stack platform
- [`juniper-ml`](https://pypi.org/project/juniper-ml/) — platform meta-package on PyPI

## Research Philosophy

The Juniper platform exists to study learning algorithms whose network architecture is not fixed in advance. Its initial anchor is the Cascade-Correlation algorithm of Fahlman and Lebiere (1990), implemented from the primary literature without recourse to higher-level abstractions that elide the algorithm's operational detail. The organising commitment is that algorithm implementations remain inspectable at the level at which they were originally specified: candidate units, correlation objectives, weight-freezing semantics, and the structural events that grow the network are first-class artifacts of the codebase rather than internal details of a library wrapper. This permits comparative work — across algorithms, datasets, and hyperparameter regimes — to be conducted on a known and reproducible substrate.

The current platform comprises a Cascade-Correlation training service exposing a REST and WebSocket interface, a dataset-generation service with a named-version registry that includes the ARC-AGI families, a real-time monitoring dashboard for inspecting training dynamics as they occur, and a distributed worker that parallelises candidate-unit training across hosts. Near-term work extends the architectural-growth catalogue beyond Cascade-Correlation, introduces multi-network orchestration for comparative experiments at the level of network populations rather than individual runs, and tightens the dataset–training–monitoring loop into a reproducible research workbench. The longer-term direction is the systematic empirical study of constructive and architecture-growing learning algorithms, with first-class infrastructure for the ablation, comparison, and replication that such a study requires.

## Documentation

| Document | Purpose |
|----------|---------|
| [`docs/DOCUMENTATION_OVERVIEW.md`](docs/DOCUMENTATION_OVERVIEW.md) | Navigation index for all `juniper-data` documentation |
| [`docs/QUICK_START.md`](docs/QUICK_START.md) | Get running in five minutes |
| [`docs/USER_MANUAL.md`](docs/USER_MANUAL.md) | Comprehensive usage guide |
| [`docs/REFERENCE.md`](docs/REFERENCE.md) | Configuration, environment variables, and operational reference |
| [`docs/ENVIRONMENT_SETUP.md`](docs/ENVIRONMENT_SETUP.md) | Conda environment and editable-install setup |
| [`docs/DEVELOPER_CHEATSHEET.md`](docs/DEVELOPER_CHEATSHEET.md) | Quick-reference card for development tasks |
| [`docs/api/JUNIPER_DATA_API.md`](docs/api/JUNIPER_DATA_API.md) | Complete REST endpoint reference |
| [`CHANGELOG.md`](CHANGELOG.md) | Version history |

## License

MIT License — Copyright (c) 2024-2026 Paul Calnon
