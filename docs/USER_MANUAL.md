# Juniper Data User Manual

**Version:** 0.4.2
**Status:** Active
**Last Updated:** March 3, 2026
**Project:** Juniper Data - Dataset Generation Service

---

## Table of Contents

1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Dataset Generators](#dataset-generators)
4. [REST API](#rest-api)
5. [Storage Backends](#storage-backends)
6. [Data Contract](#data-contract)
7. [Security](#security)
8. [Configuration](#configuration)
9. [Integration with Juniper Ecosystem](#integration-with-juniper-ecosystem)
10. [Troubleshooting](#troubleshooting)

---

## Introduction

### What is Juniper Data?

Juniper Data is the foundational data layer of the Juniper ecosystem. It generates, stores, and serves datasets used by JuniperCascor (neural network training backend) and JuniperCanopy (real-time monitoring dashboard).

### Key Features

- **8 Dataset Generators** -- Spiral, XOR, Gaussian, Circles, Checkerboard, CSV Import, MNIST, ARC-AGI
- **REST API** -- Full CRUD operations for dataset management via FastAPI on port 8100
- **Multiple Storage Backends** -- Local filesystem, PostgreSQL, Redis, HuggingFace, Kaggle, in-memory, cached
- **Reproducible Datasets** -- Deterministic ID generation from generator + version + parameter hash
- **NPZ Data Contract** -- Standardized binary format consumed by all Juniper services
- **Security** -- Optional API key authentication, rate limiting, CORS configuration
- **Observability** -- Optional Prometheus metrics and Sentry error tracking

### Architecture

```
juniper-canopy (8050) ──REST──> JuniperData (8100) <──REST── juniper-cascor (8200)
                                     │
                              8 Generators
                              8 Storage Backends
                              FastAPI REST Service
```

---

## Getting Started

### Prerequisites

- Python 3.12+ with Conda (JuniperData environment)
- `pip install -e ".[all]"` completed

If you haven't set up your environment yet, see [ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md).

### Starting the Service

```bash
conda activate JuniperData

# Development mode (with auto-reload)
python -m juniper_data --reload

# Production mode
uvicorn juniper_data.api.app:app --host 0.0.0.0 --port 8100
```

### CLI Options

```bash
python -m juniper_data --help
```

| Flag | Default | Description |
|------|---------|-------------|
| `--host` | `127.0.0.1` | Listen address |
| `--port` | `8100` | Service port |
| `--storage-path` | `./data/datasets` | Dataset storage directory |
| `--log-level` | `INFO` | Log verbosity |
| `--reload` | *(off)* | Auto-reload on code changes |

### Quick Verification

```bash
curl http://localhost:8100/v1/health
# {"status": "ok", "version": "0.4.2"}
```

---

## Dataset Generators

### Available Generators

| Generator | Description | Features | Classes |
|-----------|-------------|----------|---------|
| `spiral` | Multi-spiral classification | 2 (x, y) | n_spirals (default 2) |
| `xor` | XOR classification (4 quadrants) | 2 | 2 |
| `gaussian` | Mixture of Gaussians | 2 | configurable |
| `circles` | Concentric circles | 2 | 2 |
| `checkerboard` | 2D checkerboard pattern | 2 | 2 |
| `csv_import` | Import from CSV/JSON files | varies | varies |
| `mnist` | MNIST / Fashion-MNIST | 784 (28x28) | 10 |
| `arc_agi` | ARC-AGI visual reasoning tasks | varies | varies |

### Using Generators via API

**List available generators:**

```bash
curl http://localhost:8100/v1/generators
```

**Get parameter schema for a generator:**

```bash
curl http://localhost:8100/v1/generators/spiral/schema
```

**Generate a dataset:**

```bash
curl -X POST http://localhost:8100/v1/datasets \
  -H "Content-Type: application/json" \
  -d '{
    "generator": "spiral",
    "params": {
      "n_spirals": 2,
      "n_points_per_spiral": 100,
      "noise": 0.1,
      "seed": 42
    },
    "persist": true
  }'
```

### Using Generators via Python

```python
from juniper_data.generators.spiral import SpiralGenerator

generator = SpiralGenerator()
dataset = generator.generate(n_points=100, n_spirals=2, noise=0.1)
```

### Spiral Generator Parameters

The spiral generator is the primary generator used by the Juniper ecosystem:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_spirals` | int | 2 | Number of spiral classes |
| `n_points_per_spiral` | int | 100 | Points per spiral arm |
| `seed` | int | None | Random seed for reproducibility |
| `algorithm` | string | `"modern"` | `"modern"` or `"legacy_cascor"` |
| `noise` | float | 0.1 | Noise level |
| `radius` | float | 10.0 | Maximum radius (legacy mode) |
| `origin` | [float, float] | [0.0, 0.0] | Center offset |
| `n_rotations` | float | 1.5 | Number of full rotations |
| `clockwise` | bool | true | Spiral direction |
| `train_ratio` | float | 0.8 | Training set ratio |
| `test_ratio` | float | 0.2 | Test set ratio |
| `shuffle` | bool | true | Shuffle before splitting |

### Other Generator Examples

**XOR dataset:**

```bash
curl -X POST http://localhost:8100/v1/datasets \
  -H "Content-Type: application/json" \
  -d '{"generator": "xor", "params": {"n_points": 200, "noise": 0.1}}'
```

**Gaussian mixture:**

```bash
curl -X POST http://localhost:8100/v1/datasets \
  -H "Content-Type: application/json" \
  -d '{"generator": "gaussian", "params": {"n_clusters": 3, "n_points": 300}}'
```

**CSV import:**

```bash
curl -X POST http://localhost:8100/v1/datasets \
  -H "Content-Type: application/json" \
  -d '{"generator": "csv_import", "params": {"file_path": "/path/to/data.csv"}}'
```

---

## REST API

### Endpoint Summary

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/health` | GET | Health check |
| `/v1/health/live` | GET | Liveness probe (Kubernetes) |
| `/v1/health/ready` | GET | Readiness probe (Kubernetes) |
| `/v1/generators` | GET | List available generators |
| `/v1/generators/{name}/schema` | GET | Get parameter schema |
| `/v1/datasets` | POST | Create dataset |
| `/v1/datasets` | GET | List all datasets |
| `/v1/datasets/{id}` | GET | Get dataset metadata |
| `/v1/datasets/{id}` | DELETE | Delete dataset |
| `/v1/datasets/{id}/artifact` | GET | Download NPZ artifact |
| `/v1/datasets/{id}/preview` | GET | Preview samples as JSON |

For complete API documentation with request/response schemas, see [JUNIPER_DATA_API.md](api/JUNIPER_DATA_API.md).

### Dataset Lifecycle

1. **Create** -- `POST /v1/datasets` with generator name and parameters
2. **Inspect** -- `GET /v1/datasets/{id}` for metadata, `/preview` for samples
3. **Download** -- `GET /v1/datasets/{id}/artifact` for the NPZ binary
4. **Delete** -- `DELETE /v1/datasets/{id}` to remove

### Caching Behavior

Datasets are cached by their deterministic ID (SHA-256 hash of generator name + version + parameters). Requesting the same parameters returns the existing dataset without regeneration:

```bash
# First call: generates and stores
curl -X POST http://localhost:8100/v1/datasets \
  -d '{"generator": "spiral", "params": {"seed": 42}}'
# Returns: dataset_id = "spiral-1.0.0-a1b2c3d4..."

# Second call with same params: returns existing
curl -X POST http://localhost:8100/v1/datasets \
  -d '{"generator": "spiral", "params": {"seed": 42}}'
# Returns: same dataset_id = "spiral-1.0.0-a1b2c3d4..."
```

---

## Storage Backends

### Available Backends

| Backend | Module | Best For |
|---------|--------|----------|
| **LocalFSDatasetStore** | `storage.local_fs` | Development, standalone deployment |
| **InMemoryDatasetStore** | `storage.memory` | Testing (no persistence) |
| **CachedDatasetStore** | `storage.cached` | Production (wraps any backend with in-memory cache) |
| **PostgresDatasetStore** | `storage.postgres_store` | Shared multi-service storage |
| **RedisDatasetStore** | `storage.redis_store` | Fast caching layer |
| **HFDatasetStore** | `storage.hf_store` | Public dataset sharing (HuggingFace Hub) |
| **KaggleDatasetStore** | `storage.kaggle_store` | Kaggle dataset integration |

### Common Interface

All storage backends implement the `DatasetStore` abstract base class:

```python
class DatasetStore:
    def save(self, dataset_id, meta, arrays) -> None: ...
    def get_meta(self, dataset_id) -> DatasetMeta: ...
    def get_artifact_bytes(self, dataset_id) -> bytes: ...
    def exists(self, dataset_id) -> bool: ...
    def delete(self, dataset_id) -> None: ...
    def list_ids(self) -> list[str]: ...
```

### Default Configuration

The API service uses `LocalFSDatasetStore` by default, storing NPZ artifacts in `JUNIPER_DATA_STORAGE_PATH` (default: `./data/datasets`).

---

## Data Contract

### NPZ Artifact Schema

All datasets conform to this standardized format, which serves as the primary data contract between Juniper Data and its consumers:

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `X_train` | `(n_train, n_features)` | `float32` | Training features |
| `y_train` | `(n_train, n_classes)` | `float32` | Training labels (one-hot) |
| `X_test` | `(n_test, n_features)` | `float32` | Test features |
| `y_test` | `(n_test, n_classes)` | `float32` | Test labels (one-hot) |
| `X_full` | `(n_samples, n_features)` | `float32` | Full dataset features |
| `y_full` | `(n_samples, n_classes)` | `float32` | Full dataset labels (one-hot) |

### Label Encoding

Labels are **one-hot encoded**:

```python
# 2-class problem:
# Class 0: [1.0, 0.0]
# Class 1: [0.0, 1.0]

# 3-class problem:
# Class 0: [1.0, 0.0, 0.0]
# Class 1: [0.0, 1.0, 0.0]
# Class 2: [0.0, 0.0, 1.0]
```

Each row sums to 1.0 and contains exactly one 1.0 value.

### Schema Guarantees

Juniper Data guarantees:

1. All arrays are `float32` dtype
2. All arrays are 2-dimensional
3. `X_*` arrays have shape `(n, n_features)`
4. `y_*` arrays have shape `(n, n_classes)`
5. `y_*` arrays are valid one-hot encodings (each row sums to 1.0)
6. `len(X_train) + len(X_test) == len(X_full)`

### Loading Artifacts

**From file:**

```python
import numpy as np

with np.load("dataset.npz") as data:
    X_train = data["X_train"]  # (160, 2) float32
    y_train = data["y_train"]  # (160, 2) float32
    X_test = data["X_test"]    # (40, 2) float32
    y_test = data["y_test"]    # (40, 2) float32
```

**From API response:**

```python
import io
import numpy as np
import requests

response = requests.get(f"{BASE_URL}/v1/datasets/{dataset_id}/artifact")
with np.load(io.BytesIO(response.content)) as data:
    X_train = data["X_train"]
    y_train = data["y_train"]
```

**Converting to PyTorch:**

```python
import torch
import numpy as np

with np.load("dataset.npz") as data:
    X_train = torch.from_numpy(data["X_train"])  # torch.float32
    y_train = torch.from_numpy(data["y_train"])  # torch.float32
```

---

## Security

### API Key Authentication

When `JUNIPER_DATA_API_KEYS` is set, all non-health endpoints require an API key:

```bash
# Set API keys (JSON list)
export JUNIPER_DATA_API_KEYS='["my-secret-key-1", "my-secret-key-2"]'

# Make authenticated requests
curl -H "X-API-Key: my-secret-key-1" http://localhost:8100/v1/datasets
```

Health endpoints (`/v1/health`, `/v1/health/live`, `/v1/health/ready`) are always accessible without authentication.

### Rate Limiting

```bash
export JUNIPER_DATA_RATE_LIMIT_ENABLED=true
export JUNIPER_DATA_RATE_LIMIT_REQUESTS_PER_MINUTE=60
```

When enabled, clients exceeding the rate limit receive `429 Too Many Requests`.

### CORS

```bash
# Allow specific origins
export JUNIPER_DATA_CORS_ORIGINS='["http://localhost:8050", "http://localhost:3000"]'

# Allow all (default, development only)
export JUNIPER_DATA_CORS_ORIGINS='["*"]'
```

### Input Validation

All request parameters are validated using Pydantic models. Invalid inputs return `400 Bad Request` or `422 Unprocessable Entity` with descriptive error messages.

---

## Configuration

### Settings Source

Settings are managed by Pydantic `BaseSettings` in `juniper_data/api/settings.py`. All settings use the `JUNIPER_DATA_` prefix.

### Complete Configuration Table

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `JUNIPER_DATA_HOST` | string | `127.0.0.1` | Listen address |
| `JUNIPER_DATA_PORT` | int | `8100` | Service port |
| `JUNIPER_DATA_STORAGE_PATH` | string | `./data/datasets` | Dataset artifact storage |
| `JUNIPER_DATA_LOG_LEVEL` | string | `INFO` | Log verbosity |
| `JUNIPER_DATA_API_KEYS` | JSON list | *(none)* | API key list for auth |
| `JUNIPER_DATA_RATE_LIMIT_ENABLED` | bool | `false` | Enable rate limiting |
| `JUNIPER_DATA_RATE_LIMIT_REQUESTS_PER_MINUTE` | int | `60` | Rate limit threshold |
| `JUNIPER_DATA_CORS_ORIGINS` | JSON list | `["*"]` | CORS allowed origins |
| `JUNIPER_DATA_METRICS_ENABLED` | bool | `false` | Prometheus metrics |
| `JUNIPER_DATA_SENTRY_DSN` | string | *(none)* | Sentry error tracking |

---

## Integration with Juniper Ecosystem

### Consumers

| Service | Relationship | Connection |
|---------|-------------|------------|
| **juniper-cascor** | Requests training datasets | `JUNIPER_DATA_URL=http://localhost:8100` |
| **juniper-canopy** | Requests visualization datasets | `JUNIPER_DATA_URL=http://localhost:8100` |
| **juniper-data-client** | Python client library | `pip install juniper-data-client` |

### Data Flow

```
juniper-data generates datasets
    -> Stores as NPZ artifacts
    -> Serves via REST API
        -> juniper-cascor downloads for training
        -> juniper-canopy downloads for visualization
```

### Docker Deployment

For full-stack deployment with all Juniper services:

```bash
git clone https://github.com/pcalnon/juniper-deploy.git
cd juniper-deploy && docker compose up --build
```

In Docker, services communicate using internal DNS (`http://juniper-data:8100`).

---

## Troubleshooting

### Service Won't Start

**Port already in use:**

```bash
lsof -i :8100
# Kill the conflicting process, or use --port flag
python -m juniper_data --port 8101
```

**Missing API dependencies:**

```bash
pip install -e ".[api]"
```

### Dataset Generation Fails

**Unknown generator name:**

```bash
# List available generators
curl http://localhost:8100/v1/generators
```

**Invalid parameters:**

Check the generator's parameter schema:

```bash
curl http://localhost:8100/v1/generators/spiral/schema
```

### Storage Issues

**Disk full:**

```bash
# Check storage usage
du -sh ./data/datasets/

# Clean old datasets
curl -X DELETE http://localhost:8100/v1/datasets/{old_dataset_id}
```

**Permission denied:**

```bash
mkdir -p ./data/datasets
chmod 755 ./data/datasets
```

### Connection Issues from Other Services

**juniper-cascor or juniper-canopy can't connect:**

1. Verify the service is running: `curl http://localhost:8100/v1/health`
2. Check the `JUNIPER_DATA_URL` environment variable in the consumer
3. Verify firewall/network rules allow port 8100
4. In Docker, use `http://juniper-data:8100` instead of `localhost`

---

## End of User Manual
