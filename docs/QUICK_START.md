# Quick Start Guide

Get Juniper Data running in 5 minutes.

**Version:** 0.4.2
**Last Updated:** March 2026

---

## Prerequisites

- **Python 3.12+** (`python --version`)
- **Conda** (Miniforge3 or Miniconda) (`conda --version`)
- **Git** (`git --version`)

---

## 1. Clone and Install

```bash
git clone https://github.com/pcalnon/juniper-data.git
cd juniper-data

conda activate JuniperData
pip install -e ".[dev]"
```

For API support only: `pip install -e ".[api]"`
For everything: `pip install -e ".[all]"`

---

## 2. Start the Service

```bash
python -m juniper_data
```

The API server starts on **port 8100**. Verify it's running:

```bash
curl http://localhost:8100/v1/health
# {"status": "ok", "version": "0.4.2"}
```

Optional CLI flags: `--host`, `--port`, `--storage-path`, `--log-level`, `--reload`.

For production deployments:

```bash
uvicorn juniper_data.api.app:app --host 0.0.0.0 --port 8100
```

---

## 3. Generate a Dataset

Create a two-spiral classification dataset:

```bash
curl -X POST http://localhost:8100/v1/datasets \
  -H "Content-Type: application/json" \
  -d '{
    "generator": "spiral",
    "params": {"n_points": 200, "n_spirals": 2, "noise": 0.1},
    "persist": true
  }'
```

The response includes a `dataset_id` and `artifact_url`.

---

## 4. Retrieve Data

```bash
# Get dataset metadata
curl http://localhost:8100/v1/datasets/{dataset_id}

# Preview first samples as JSON
curl http://localhost:8100/v1/datasets/{dataset_id}/preview

# Download the NPZ artifact
curl -O http://localhost:8100/v1/datasets/{dataset_id}/artifact
```

The NPZ artifact contains keys: `X_train`, `y_train`, `X_test`, `y_test`, `X_full`, `y_full` (all `float32`).

---

## 5. Available Generators

| Generator | Description |
|-----------|-------------|
| `spiral` | Multi-spiral classification |
| `xor` | XOR classification (4 quadrants) |
| `gaussian` | Mixture of Gaussians |
| `circles` | Concentric circles |
| `checkerboard` | 2D checkerboard pattern |
| `csv_import` | CSV/JSON file import |
| `mnist` | MNIST / Fashion-MNIST |
| `arc_agi` | ARC-AGI visual reasoning tasks |

List all generators and their parameter schemas:

```bash
curl http://localhost:8100/v1/generators
curl http://localhost:8100/v1/generators/spiral/schema
```

---

## 6. Configuration

Settings use Pydantic BaseSettings with the `JUNIPER_DATA_` prefix:

| Variable | Default | Description |
|----------|---------|-------------|
| `JUNIPER_DATA_HOST` | `127.0.0.1` | Listen address |
| `JUNIPER_DATA_PORT` | `8100` | Service port |
| `JUNIPER_DATA_STORAGE_PATH` | `./data/datasets` | Dataset artifact storage |
| `JUNIPER_DATA_LOG_LEVEL` | `INFO` | Log verbosity |
| `JUNIPER_DATA_API_KEYS` | *(none)* | Optional API key list for auth |
| `JUNIPER_DATA_CORS_ORIGINS` | `["*"]` | Allowed CORS origins |

---

## 7. Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest juniper_data/tests/ --cov=juniper_data --cov-report=html --cov-report=term-missing --cov-fail-under=80

# Run by marker
pytest -m unit
pytest -m integration
pytest -m api
pytest -m generators
```

---

## 8. Next Steps

- **[API Reference](api/JUNIPER_DATA_API.md)** — Full endpoint documentation with schemas
- **[README.md](../README.md)** — Project overview and ecosystem context
- **[AGENTS.md](../AGENTS.md)** — Development guide, conventions, and architecture
