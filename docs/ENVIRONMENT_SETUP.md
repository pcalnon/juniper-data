# Environment Setup Guide

## Complete Environment Configuration for Juniper Data

**Version:** 0.4.2
**Status:** Active
**Last Updated:** May 4, 2026
**Project:** Juniper Data - Dataset Generation Service

---

## Table of Contents

- [Overview](#overview)
- [System Requirements](#system-requirements)
- [Conda Environment Setup](#conda-environment-setup)
- [Python Dependencies](#python-dependencies)
- [Configuration](#configuration)
- [Environment Variables](#environment-variables)
- [Verification Steps](#verification-steps)
- [Troubleshooting](#troubleshooting)
- [Advanced Configuration](#advanced-configuration)

---

## Overview

This guide covers complete environment setup for Juniper Data development and deployment.

**What you'll set up:**

- Conda environment (JuniperData)
- Python dependencies (core, API, dev, test, observability)
- Environment variables
- Dependency files and lockfile workflow
- Storage directory for dataset artifacts

**Time required:** 10-15 minutes (first-time setup)

---

## System Requirements

### Hardware

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| **RAM** | 2 GB | 4 GB |
| **Disk** | 1 GB free | 5 GB free (for datasets) |
| **CPU** | Any modern CPU | 2+ cores |

### Software

| Requirement | Version | Check Command |
|-------------|---------|---------------|
| **Python** | >= 3.12 | `python --version` |
| **Conda** | Any (Miniforge3 recommended) | `conda --version` |
| **Git** | Any recent version | `git --version` |
| **pip** | >= 21.0 | `pip --version` |

### Ecosystem Compatibility

| juniper-data | juniper-cascor | juniper-canopy | data-client | cascor-client | cascor-worker |
|---|---|---|---|---|---|
| 0.4.x | 0.3.x | 0.2.x | >=0.3.1 | >=0.1.0 | >=0.1.0 |

---

## Conda Environment Setup

### Create the JuniperData Environment

```bash
# Create environment with Python 3.14
conda create -n JuniperData python=3.14 -y

# Activate the environment
conda activate JuniperData

# Verify
python --version
# Python 3.14.x
```

### Environment Location

```
/opt/miniforge3/envs/JuniperData
```

### Activate for Development

Always activate the JuniperData conda environment before working on this project:

```bash
conda activate JuniperData
```

---

## Python Dependencies

### Dependency Groups

Juniper Data uses optional dependency groups defined in `pyproject.toml`:

| Group | Purpose | Install Command |
|-------|---------|-----------------|
| *(core)* | numpy, pydantic, python-dotenv | `pip install -e .` |
| `api` | FastAPI, uvicorn, pydantic-settings | `pip install -e ".[api]"` |
| `arc-agi` | ARC-AGI dataset support | `pip install -e ".[arc-agi]"` |
| `test` | pytest, coverage, httpx, benchmarks | `pip install -e ".[test]"` |
| `observability` | Prometheus, Sentry | `pip install -e ".[observability]"` |
| `dev` | ruff, mypy, bandit, pip-audit, pre-commit | `pip install -e ".[dev]"` |
| `all` | Everything above | `pip install -e ".[all]"` |

### Dependency Artifact Map

Juniper Data keeps several dependency files for different workflows:

| File | Source of Truth? | Purpose | Update Path |
|------|------------------|---------|-------------|
| `pyproject.toml` | Yes | Runtime, optional, test, and development dependency ranges | Edit directly for project dependency changes |
| `requirements.lock` | Yes for Docker pins | Reproducible Docker/API build pins for `api` and `observability` extras | Regenerate with `uv pip compile` after `pyproject.toml` changes |
| `conf/requirements.txt` | No | pip environment snapshot used for developer and CI diagnostics | Updated by dependency snapshot maintenance PRs |
| `conf/requirements-ORIG.txt` | No | historical baseline snapshot kept beside `conf/requirements.txt` | Keep in sync with `conf/requirements.txt` when snapshot floors change |
| `conf/conda_environment.yaml` | No | Conda environment snapshot for reconstructing the JuniperData environment | Regenerate from the active Conda environment when refreshing snapshots |

Snapshot-only packages can appear in `conf/requirements*.txt` without being imported by `juniper_data` or listed in `pyproject.toml`. Treat those updates as environment hygiene unless they also change `pyproject.toml` or `requirements.lock`.

### Dependency Files

Use the project dependency files for different workflows:

| File | Purpose | Update When |
|------|---------|-------------|
| `pyproject.toml` | Authoritative package metadata and optional dependency groups for editable installs | Adding or changing dependencies used by the package, tests, API, observability, or developer tooling |
| `requirements.lock` | Exact pinned dependency set for reproducible Docker builds | Regenerate after `pyproject.toml` dependency changes |
| `conf/requirements.txt` | Supplemental pip requirements installed by `util/setup_environment.bash` after the `JuniperData` conda environment is activated | Dependabot bumps a package in this file, or the bootstrap script needs an additional pip package |
| `conf/requirements-ORIG.txt` | Baseline copy of `conf/requirements.txt` kept in sync with the supplemental requirements list | Make the same package-bound updates as `conf/requirements.txt` |
| `conf/requirements_ci.txt` | Generated CI dependency snapshot from `pip list --format=freeze` | Do not hand-edit; generated by `scripts/generate_dep_docs.sh` in CI |
| `conf/conda_environment_ci.yaml` | Generated Conda environment snapshot from `conda list --explicit` | Do not hand-edit; generated by `scripts/generate_dep_docs.sh` in CI |

Prefer `pip install -e ".[all]"` for local development. Use `util/setup_environment.bash` when you need the project bootstrap workflow that creates or activates the Conda environment and then installs `conf/requirements.txt`.

### Recommended: Full Development Installation

```bash
cd /home/pcalnon/Development/python/Juniper/juniper-data
conda activate JuniperData
pip install -e ".[all]"
```

### Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `numpy` | >= 1.24.0 | Numerical computations, NPZ artifacts |
| `pydantic` | >= 2.0.0 | Data validation, parameter models |
| `python-dotenv` | >= 1.0.0 | Environment file loading |

### API Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `fastapi` | >= 0.100.0 | REST API framework |
| `uvicorn[standard]` | >= 0.23.0 | ASGI server |
| `pydantic-settings` | >= 2.0.0 | Settings management |

### Development Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `ruff` | >= 0.9.0 | Linting + formatting (replaces black, isort, flake8) |
| `mypy` | >= 1.0.0 | Static type checking |
| `bandit[sarif]` | >= 1.7.9 | Security SAST scanning |
| `pip-audit` | >= 2.7.0 | Dependency vulnerability scanning |
| `pre-commit` | >= 3.0.0 | Git hook management |

### Test Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `pytest` | >= 7.0.0 | Test framework |
| `pytest-cov` | >= 4.0.0 | Coverage reporting |
| `pytest-timeout` | >= 2.2.0 | Test timeout enforcement |
| `pytest-asyncio` | >= 0.21.0 | Async test support |
| `pytest-benchmark` | >= 4.0.0 | Performance benchmarking |
| `httpx` | >= 0.24.0 | Async HTTP testing client |
| `coverage[toml]` | >= 7.0.0 | Coverage with pyproject.toml support |
| `juniper-data-client` | >= 0.3.0 | Client library integration tests |

### Dependency Lockfile

For reproducible Docker builds, the project maintains a `requirements.lock` file:

```bash
# Regenerate after changing dependencies in pyproject.toml
uv pip compile pyproject.toml --extra api --extra observability --extra mnist --upgrade -o requirements.lock
```

The `pyproject.toml` retains flexible `>=` ranges for local development, while the lockfile pins exact versions. CI checks lockfile freshness by resolving `pyproject.toml` under the committed `requirements.lock` constraints, so routine `conf/requirements*.txt` snapshot updates do not require a lockfile change.

### Dependency File Roles

| File | Purpose | Update Rule |
|------|---------|-------------|
| `pyproject.toml` | Authoritative package metadata and optional dependency groups | Edit when changing supported install extras or runtime/test/dev requirements |
| `requirements.lock` | Reproducible Docker/API runtime lockfile generated from `pyproject.toml` with `api` and `observability` extras | Regenerate with `uv pip compile` after changing `pyproject.toml` dependency ranges |
| `conf/requirements.txt` | Legacy pip install list used by `util/setup_environment.bash` for the full local Conda environment | Keep aligned with the local environment needs; dependency-only Dependabot PRs may update this directly |
| `conf/requirements-ORIG.txt` | Baseline copy of the legacy pip install list | Update with `conf/requirements.txt` when maintaining the legacy snapshot pair |
| `conf/requirements_ci.txt` | CI artifact generated by `scripts/generate_dep_docs.sh` from `pip list --format=freeze` | Do not edit by hand; generated in the `dependency-docs` workflow |
| `conf/conda_environment_ci.yaml` | CI artifact generated by `scripts/generate_dep_docs.sh` from `conda list --explicit` | Do not edit by hand; generated in the `dependency-docs` workflow |

If a PR only changes the legacy `conf/requirements*.txt` snapshots, do not regenerate `requirements.lock` unless `pyproject.toml` changed too. The lockfile freshness check compares `requirements.lock` to `pyproject.toml`, not to the legacy `conf/` snapshots.

### Dependency File Roles

Use the right dependency file for the job:

| File | Role | Update When |
|------|------|-------------|
| `pyproject.toml` | Authoritative package metadata, install extras, and direct runtime/test/dev dependency ranges | Code starts requiring a new package or a higher minimum version |
| `requirements.lock` | Pinned Docker-build input resolved from `pyproject.toml` | `pyproject.toml` dependency ranges change |
| `conf/requirements.txt` | Committed pip environment snapshot for lightweight/no-CUDA environment replication and dependency review | Dependabot or manual environment snapshot maintenance updates a listed package floor |
| `conf/requirements-ORIG.txt` | Baseline copy of the pip environment snapshot kept alongside `conf/requirements.txt` | Keep in sync with `conf/requirements.txt` for the same package-floor update |
| `conf/requirements_ci.txt` | Generated CI dependency artifact from `scripts/generate_dep_docs.sh` | Do not edit by hand; CI regenerates it as an artifact |

If a Dependabot PR changes only `conf/requirements.txt` and `conf/requirements-ORIG.txt`, review it as a pip snapshot maintenance change. Do not mirror that package into `pyproject.toml` unless the source code or tests import it directly or the package is part of an install extra contract.

The committed `conf/requirements.txt` and `conf/requirements-ORIG.txt` files are environment-oriented pip requirement snapshots used by legacy setup tooling. They are not the authoritative package dependency source; update `pyproject.toml` for project metadata and regenerate `requirements.lock` only when `pyproject.toml` dependency ranges change.

---

## Configuration

### Application Settings

Juniper Data uses Pydantic `BaseSettings` with the `JUNIPER_DATA_` prefix. Settings are loaded from environment variables, with defaults defined in `juniper_data/api/settings.py`.

### Storage Directory

Datasets are stored as NPZ files in a configurable directory:

```bash
# Default location (relative to working directory)
./data/datasets/

# The directory is created automatically on first use
# To use a custom location:
export JUNIPER_DATA_STORAGE_PATH=/path/to/datasets
```

---

## Environment Variables

### Core Service Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `JUNIPER_DATA_HOST` | string | `127.0.0.1` | Listen address |
| `JUNIPER_DATA_PORT` | int | `8100` | Service port |
| `JUNIPER_DATA_STORAGE_PATH` | string | `./data/datasets` | Dataset artifact storage directory |
| `JUNIPER_DATA_LOG_LEVEL` | string | `INFO` | Log verbosity (DEBUG, INFO, WARNING, ERROR) |

### Security Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `JUNIPER_DATA_API_KEYS` | JSON list | *(none)* | API key list for authentication (e.g., `["key1","key2"]`) |
| `JUNIPER_DATA_RATE_LIMIT_ENABLED` | bool | `false` | Enable request rate limiting |
| `JUNIPER_DATA_RATE_LIMIT_REQUESTS_PER_MINUTE` | int | `60` | Rate limit threshold |
| `JUNIPER_DATA_CORS_ORIGINS` | JSON list | `["*"]` | Allowed CORS origins |

### Observability Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `JUNIPER_DATA_METRICS_ENABLED` | bool | `false` | Enable Prometheus metrics |
| `JUNIPER_DATA_SENTRY_DSN` | string | *(none)* | Sentry error tracking DSN |

### CSV / JSON import (`csv_import`)

On-disk sources under `JUNIPER_DATA_IMPORT_DIR`. Over-cap imports are refused unless truncation is opted in. This is not the 10 MB HTTP body limit. See [CSV Import Byte Cap](REFERENCE.md#csv-import-byte-cap).

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `JUNIPER_DATA_IMPORT_DIR` | string | `/data/imports` | Root directory for `csv_import` files |
| `JUNIPER_DATA_CSV_IMPORT_MAX_BYTES` | int | `134217728` | Byte-cap **ceiling** (128 MiB). Request `max_bytes` may only lower it (`gt=0`) |
| `JUNIPER_DATA_CSV_IMPORT_ALLOW_TRUNCATION` | bool | `false` | Deployment-wide opt-in to a partial import |

### Integration Configuration

| Variable | Used By | Description |
|----------|---------|-------------|
| `JUNIPER_DATA_URL` | juniper-cascor, juniper-canopy | URL for this service (default: `http://localhost:8100`) |
| `JUNIPER_DATA_API_KEY` | juniper-cascor | API key for authentication |

### Example: Development Environment

```bash
# Minimal development setup (all defaults are fine)
conda activate JuniperData
python -m juniper_data
```

### Example: Production Environment

```bash
export JUNIPER_DATA_HOST=0.0.0.0
export JUNIPER_DATA_PORT=8100
export JUNIPER_DATA_STORAGE_PATH=/var/data/juniper/datasets
export JUNIPER_DATA_LOG_LEVEL=WARNING
export JUNIPER_DATA_API_KEYS='["your-secret-api-key"]'
export JUNIPER_DATA_RATE_LIMIT_ENABLED=true
export JUNIPER_DATA_METRICS_ENABLED=true

uvicorn juniper_data.api.app:app --host 0.0.0.0 --port 8100
```

---

## Verification Steps

After completing setup, verify each component:

### 1. Python Environment

```bash
conda activate JuniperData
python --version
# Python 3.14.x

pip list | grep juniper
# juniper-data    0.4.2    /path/to/juniper-data
```

### 2. Core Dependencies

```bash
python -c "import numpy; print(f'numpy {numpy.__version__}')"
python -c "import pydantic; print(f'pydantic {pydantic.__version__}')"
```

### 3. API Dependencies

```bash
python -c "import fastapi; print(f'fastapi {fastapi.__version__}')"
python -c "import uvicorn; print(f'uvicorn {uvicorn.__version__}')"
```

### 4. Service Starts

```bash
python -m juniper_data &
sleep 2
curl http://localhost:8100/v1/health
# {"status": "ok", "version": "0.4.2"}
kill %1
```

### 5. Tests Pass

```bash
pytest -x -q
# XX passed in XX.XXs
```

### 6. Linting Clean

```bash
ruff check juniper_data
ruff format --check juniper_data
mypy juniper_data --ignore-missing-imports
```

### Verification Checklist

- [ ] Conda environment `JuniperData` active
- [ ] `pip install -e ".[all]"` completed without errors
- [ ] `python -c "import juniper_data"` succeeds
- [ ] Service starts on port 8100 and responds to health check
- [ ] Tests pass (`pytest -x -q`)
- [ ] Linters clean (`ruff check`, `ruff format --check`)

---

## Troubleshooting

### ModuleNotFoundError: No module named 'juniper_data'

**Cause:** Package not installed in editable mode.

```bash
conda activate JuniperData
pip install -e ".[all]"
```

### ModuleNotFoundError: No module named 'fastapi'

**Cause:** API optional dependency group not installed.

```bash
pip install -e ".[api]"
# or install everything:
pip install -e ".[all]"
```

### Port 8100 Already in Use

**Cause:** Another service or previous instance running on port 8100.

```bash
# Find the process
lsof -i :8100

# Kill it
kill <PID>

# Or use a different port
python -m juniper_data --port 8101
```

### Permission Denied on Storage Path

**Cause:** Storage directory not writable.

```bash
# Check current storage path
echo $JUNIPER_DATA_STORAGE_PATH
# Default: ./data/datasets

# Create and set permissions
mkdir -p ./data/datasets
chmod 755 ./data/datasets
```

### Tests Fail with Import Errors

**Cause:** Test dependencies not installed.

```bash
pip install -e ".[test]"
```

---

## Advanced Configuration

### Using a Custom Storage Backend

By default, Juniper Data uses the local filesystem. The `storage/` module provides additional backends:

| Backend | Module | Use Case |
|---------|--------|----------|
| Local FS | `storage.local_fs` | Development, standalone deployment |
| In-Memory | `storage.memory` | Testing |
| Cached | `storage.cached` | Production (wraps any backend) |
| PostgreSQL | `storage.postgres_store` | Shared/persistent storage |
| Redis | `storage.redis_store` | Fast caching layer |
| HuggingFace | `storage.hf_store` | Public dataset sharing |
| Kaggle | `storage.kaggle_store` | Kaggle dataset integration |

### Docker Deployment

For containerized deployment, see [juniper-deploy](https://github.com/pcalnon/juniper-deploy):

```bash
git clone https://github.com/pcalnon/juniper-deploy.git
cd juniper-deploy && docker compose up --build
```

### Pre-commit Hooks

Install pre-commit hooks for automated code quality checks:

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

---

## Next Steps

- **[QUICK_START.md](QUICK_START.md)** -- Start the service and generate datasets
- **[USER_MANUAL.md](USER_MANUAL.md)** -- Comprehensive usage guide
- **[REFERENCE.md](REFERENCE.md)** -- Configuration and command reference
- **[AGENTS.md](../AGENTS.md)** -- Development conventions

---

**Last Updated:** March 3, 2026
**Version:** 0.4.2
**Maintainer:** Paul Calnon
