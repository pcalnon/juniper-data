# JuniperData API Reference

**Version:** 0.4.2
**Last Updated:** 2026-04-01
**Base URL:** `http://localhost:8100`  
**API Prefix:** `/v1`

---

## Table of Contents

1. [Overview](#overview)
2. [API Versioning Strategy](#api-versioning-strategy)
3. [Health Endpoints](#health-endpoints)
4. [Generators Endpoints](#generators-endpoints)
5. [Datasets Endpoints](#datasets-endpoints)
6. [NPZ Artifact Schema](#npz-artifact-schema)
7. [Error Handling](#error-handling)
8. [Client Examples](#client-examples)

---

## Overview

JuniperData is a dataset generation and management service for the Juniper ecosystem. It provides a REST API for generating, storing, and serving datasets used by JuniperCascor (neural network backend) and JuniperCanopy (web dashboard).

### API Characteristics

- **Protocol:** HTTP/1.1
- **Data Format:** JSON (metadata), NPZ (binary artifacts)
- **Encoding:** UTF-8
- **CORS:** Enabled (configurable)
- **Authentication:** None (internal service)

### Configuration

| Environment Variable        | Default           | Description               |
| --------------------------- | ----------------- | ------------------------- |
| `JUNIPER_DATA_HOST`         | `0.0.0.0`         | Host to bind              |
| `JUNIPER_DATA_PORT`         | `8100`            | Port to bind              |
| `JUNIPER_DATA_STORAGE_PATH` | `./data/datasets` | Dataset storage directory |
| `JUNIPER_DATA_LOG_LEVEL`    | `INFO`            | Logging level             |
| `JUNIPER_DATA_CORS_ORIGINS` | `["*"]`           | Allowed CORS origins      |

---

## API Versioning Strategy

### Current Version

**v1** - All endpoints are prefixed with `/v1/`

### Versioning Policy

1. **Semantic Versioning**: The API follows [SemVer](https://semver.org/):
   - **MAJOR** version for incompatible API changes
   - **MINOR** version for backward-compatible functionality additions
   - **PATCH** version for backward-compatible bug fixes

2. **URL Versioning**: Major versions are indicated in the URL path (`/v1/`, `/v2/`, etc.)

3. **Backward Compatibility Guarantees**:
   - Response fields will NOT be removed within a major version
   - New optional fields MAY be added to responses
   - New optional parameters MAY be added to requests
   - Existing endpoints will NOT change behavior within a major version

4. **Deprecation Policy**:
   - Deprecated features will be announced at least 2 minor versions in advance
   - Deprecated endpoints will return a `Deprecation` header
   - Old API versions will be supported for at least 6 months after a new major version

5. **Breaking Changes** (require major version bump):
   - Removing an endpoint
   - Removing a response field
   - Changing the type of a response field
   - Changing the NPZ artifact schema
   - Changing default behavior of existing parameters

---

## Health Endpoints

### GET /v1/health

Combined health check endpoint (backward compatible).

**Response:**

```json
{
  "status": "ok",
  "version": "0.4.0"
}
```

**Status Codes:**

- `200 OK` - Service is healthy

---

### GET /v1/health/live

Liveness probe for container orchestration.

Used by Kubernetes/Docker to determine if the container should be restarted.

**Response:**

```json
{
  "status": "alive"
}
```

**Status Codes:**

- `200 OK` - Process is running

**Docker Configuration:**

```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8100/v1/health/live"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 5s
```

---

### GET /v1/health/ready

Readiness probe for container orchestration.

Used by Kubernetes/Docker to determine if the container can accept traffic.

**Response:**

```json
{
  "status": "ready",
  "version": "0.4.0"
}
```

**Status Codes:**

- `200 OK` - Service is ready to accept requests

**Kubernetes Configuration:**

```yaml
readinessProbe:
  httpGet:
    path: /v1/health/ready
    port: 8100
  initialDelaySeconds: 5
  periodSeconds: 10
livenessProbe:
  httpGet:
    path: /v1/health/live
    port: 8100
  initialDelaySeconds: 5
  periodSeconds: 30
```

---

## Generators Endpoints

### GET /v1/generators

List available dataset generators.

**Response:**

```json
[
  {
    "name": "spiral",
    "version": "1.0.0",
    "description": "Multi-spiral classification dataset generator"
  }
]
```

---

### GET /v1/generators/{name}/schema

Get the JSON schema for a generator's parameters.

**Path Parameters:**

- `name` (string): Generator name (e.g., `spiral`)

**Response:**

```json
{
  "properties": {
    "n_spirals": {
      "default": 2,
      "description": "Number of spiral arms",
      "minimum": 2,
      "title": "N Spirals",
      "type": "integer"
    },
    "n_points_per_spiral": {
      "default": 100,
      "description": "Points per spiral arm",
      "minimum": 1,
      "title": "N Points Per Spiral",
      "type": "integer"
    },
    "algorithm": {
      "default": "modern",
      "enum": ["modern", "legacy_cascor"],
      "title": "Algorithm",
      "type": "string"
    }
  },
  "title": "SpiralParams",
  "type": "object"
}
```

**Status Codes:**

- `200 OK` - Schema returned
- `404 Not Found` - Unknown generator name

---

## Datasets Endpoints

### POST /v1/datasets

Create a new dataset or retrieve an existing one with matching parameters.

**Request Body:**

```json
{
  "generator": "spiral",
  "params": {
    "n_spirals": 2,
    "n_points_per_spiral": 100,
    "seed": 42,
    "algorithm": "modern",
    "noise": 0.1,
    "train_ratio": 0.8,
    "test_ratio": 0.2
  },
  "persist": true,
  "tags": ["benchmark", "candidate"],
  "ttl_seconds": 3600,
  "name": "spiral-baseline",
  "description": "Baseline spiral run for CAN-DEF-005",
  "created_by": "ci-bot",
  "parent_dataset_id": null
}
```

**Request Fields:**

| Field               | Type             | Required | Description                                                      |
| ------------------- | ---------------- | -------- | ---------------------------------------------------------------- |
| `generator`         | string           | Yes      | Generator name (for example, `spiral`)                          |
| `params`            | object           | No       | Generator-specific parameters                                    |
| `persist`           | boolean          | No       | Whether to persist to storage (default: `true`)                 |
| `tags`              | array of strings | No       | Optional metadata tags (stored when persisted)                  |
| `ttl_seconds`       | integer          | No       | Optional TTL in seconds (`>=1`) used to compute `expires_at`    |
| `name`              | string           | No       | Logical dataset name used for dataset versioning                |
| `description`       | string           | No       | Optional description (`max_length=500`)                         |
| `created_by`        | string           | No       | Optional creator identifier (`max_length=100`)                  |
| `parent_dataset_id` | string           | No       | Optional lineage pointer to a parent dataset                    |

**Spiral Generator Parameters:**

| Parameter             | Type           | Default    | Description                     |
| --------------------- | -------------- | ---------- | ------------------------------- |
| `n_spirals`           | int            | 2          | Number of spiral classes        |
| `n_points_per_spiral` | int            | 100        | Points per spiral               |
| `seed`                | int            | None       | Random seed for reproducibility |
| `algorithm`           | string         | `"modern"` | `"modern"` or `"legacy_cascor"` |
| `noise`               | float          | 0.1        | Noise level                     |
| `radius`              | float          | 10.0       | Maximum radius (legacy mode)    |
| `origin`              | [float, float] | [0.0, 0.0] | Center offset                   |
| `n_rotations`         | float          | 1.5        | Number of full rotations        |
| `clockwise`           | bool           | true       | Spiral direction                |
| `train_ratio`         | float          | 0.8        | Training set ratio              |
| `test_ratio`          | float          | 0.2        | Test set ratio                  |
| `shuffle`             | bool           | true       | Shuffle before splitting        |

**Response:**

```json
{
  "dataset_id": "spiral-1.0.0-a1b2c3d4e5f6...",
  "generator": "spiral",
  "meta": {
    "dataset_id": "spiral-1.0.0-a1b2c3d4e5f6...",
    "generator": "spiral",
    "generator_version": "1.0.0",
    "params": {
      "n_spirals": 2,
      "n_points_per_spiral": 100,
      "seed": 42
    },
    "n_samples": 200,
    "n_features": 2,
    "n_classes": 2,
    "n_train": 160,
    "n_test": 40,
    "class_distribution": {"0": 100, "1": 100},
    "artifact_formats": ["npz"],
    "created_at": "2026-02-05T12:00:00.000000",
    "checksum": "f95ad2200996f29c4f9f48f2e7f1844f36f31472f17032cae78363493ee4f4b3",
    "dataset_name": "spiral-baseline",
    "dataset_version": 1,
    "parent_dataset_id": null,
    "description": "Baseline spiral run for CAN-DEF-005",
    "created_by": "ci-bot",
    "tags": ["benchmark", "candidate"],
    "ttl_seconds": 3600,
    "expires_at": "2026-02-05T13:00:00.000000",
    "last_accessed_at": null,
    "access_count": 0
  },
  "artifact_url": "/v1/datasets/spiral-1.0.0-a1b2c3d4e5f6.../artifact"
}
```

**Status Codes:**

- `201 Created` - Dataset created or retrieved
- `400 Bad Request` - Invalid parameters or unknown generator

**Caching Behavior:**

Datasets are cached by their deterministic ID (hash of generator + version + params). Requesting the same parameters returns the existing dataset.

### Versioning Behavior for Named Datasets

- If `name` is set and `persist=true`, the store assigns `meta.dataset_version` atomically during save.
- If `name` is set and `persist=false`, the response includes a preview version number, but no dataset is saved.
- If the request resolves to an existing cached dataset ID, the existing metadata is returned and no new version is allocated.
- Cache identity is based on `generator + generator_version + params`; metadata fields such as `name`, `description`, and `created_by` do not change `dataset_id`.

### Checksum Format

- `meta.checksum` is a SHA-256 hex digest of the NPZ artifact bytes (64 lowercase hex characters, no `sha256:` prefix).

---

### GET /v1/datasets

List all stored datasets.

**Query Parameters:**

- `limit` (int, optional): Maximum IDs to return (default: `100`, range: `1..1000`)
- `offset` (int, optional): Number of IDs to skip (default: `0`)

**Response:**

```json
[
  "spiral-1.0.0-a1b2c3d4e5f6...",
  "spiral-1.0.0-b2c3d4e5f6a7..."
]
```

The endpoint returns dataset IDs only. Use `GET /v1/datasets/{id}` for metadata.

---

### GET /v1/datasets/filter

Filter dataset metadata by generator, tags, creation time, size, and version fields.

**Query Parameters:**

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `generator` | string | `null` | Filter by generator name |
| `tags` | string | `null` | Comma-separated tags (for example, `prod,baseline`) |
| `tags_match` | string | `"any"` | Tag matching mode: `any` or `all` |
| `created_after` | datetime | `null` | Include datasets created at/after timestamp |
| `created_before` | datetime | `null` | Include datasets created at/before timestamp |
| `min_samples` | integer | `null` | Minimum `n_samples` |
| `max_samples` | integer | `null` | Maximum `n_samples` |
| `include_expired` | boolean | `false` | Include TTL-expired datasets |
| `dataset_name` | string | `null` | Filter by logical dataset name |
| `dataset_version` | integer | `null` | Filter by version number |
| `limit` | integer | `100` | Page size (`1..1000`) |
| `offset` | integer | `0` | Page offset |

**Response:**

```json
{
  "datasets": [
    {
      "dataset_id": "spiral-1.0.0-a1b2c3d4e5f6...",
      "generator": "spiral",
      "generator_version": "1.0.0",
      "params": {"n_spirals": 2, "seed": 42},
      "n_samples": 200,
      "n_features": 2,
      "n_classes": 2,
      "n_train": 160,
      "n_test": 40,
      "class_distribution": {"0": 100, "1": 100},
      "artifact_formats": ["npz"],
      "created_at": "2026-02-05T12:00:00.000000",
      "checksum": "f95ad2200996f29c4f9f48f2e7f1844f36f31472f17032cae78363493ee4f4b3",
      "dataset_name": "spiral-baseline",
      "dataset_version": 1
    }
  ],
  "total": 1,
  "limit": 100,
  "offset": 0
}
```

---

### GET /v1/datasets/versions

List all stored versions for a logical dataset name.

**Query Parameters:**

- `name` (string, required): Logical dataset name

**Response:**

```json
{
  "dataset_name": "spiral-baseline",
  "versions": [
    {"dataset_id": "spiral-...111", "dataset_name": "spiral-baseline", "dataset_version": 1},
    {"dataset_id": "spiral-...222", "dataset_name": "spiral-baseline", "dataset_version": 2}
  ],
  "total": 2,
  "latest_version": 2
}
```

`versions` are sorted by `dataset_version` ascending.

---

### GET /v1/datasets/latest

Get metadata for the latest stored version of a logical dataset name.

**Query Parameters:**

- `name` (string, required): Logical dataset name

**Response:**

Returns a full `DatasetMeta` object (same schema as `GET /v1/datasets/{id}`).

```json
{
  "dataset_id": "spiral-...222",
  "dataset_name": "spiral-baseline",
  "dataset_version": 2
}
```

**Status Codes:**

- `200 OK` - Latest version metadata returned
- `404 Not Found` - No versions exist for the requested name

---

### Additional Dataset Management Endpoints

The dataset router also includes operational endpoints:

| Endpoint | Method | Purpose |
| -------- | ------ | ------- |
| `/v1/datasets/stats` | GET | Aggregate dataset statistics |
| `/v1/datasets/batch-delete` | POST | Delete multiple datasets by ID |
| `/v1/datasets/batch-create` | POST | Create multiple datasets in one request |
| `/v1/datasets/batch-tags` | PATCH | Add/remove tags across multiple datasets |
| `/v1/datasets/batch-export` | POST | Export multiple NPZ artifacts as a ZIP |
| `/v1/datasets/cleanup-expired` | POST | Delete all expired datasets |
| `/v1/datasets/{id}/tags` | PATCH | Add/remove tags for a single dataset |

See endpoint models in `juniper_data/core/models.py` and route behavior in `juniper_data/api/routes/datasets.py`.

---

### GET /v1/datasets/{id}

Get metadata for a specific dataset.

**Path Parameters:**

- `id` (string): Dataset ID

**Response:**

```json
{
  "dataset_id": "spiral-1.0.0-a1b2c3d4e5f6...",
  "generator": "spiral",
  "generator_version": "1.0.0",
  "params": {...},
  "n_samples": 200,
  "n_features": 2,
  "n_classes": 2,
  "n_train": 160,
  "n_test": 40,
  "class_distribution": {"0": 100, "1": 100},
  "artifact_formats": ["npz"],
  "created_at": "2026-02-05T12:00:00.000000",
  "checksum": "f95ad2200996f29c4f9f48f2e7f1844f36f31472f17032cae78363493ee4f4b3",
  "dataset_name": "spiral-baseline",
  "dataset_version": 2,
  "parent_dataset_id": "spiral-1.0.0-a1b2c3d4e5f6...",
  "description": "Second run with tuning tweaks",
  "created_by": "ci-bot"
}
```

**Status Codes:**

- `200 OK` - Metadata returned
- `404 Not Found` - Dataset not found

---

### GET /v1/datasets/{id}/artifact

Download the dataset as an NPZ file.

**Path Parameters:**

- `id` (string): Dataset ID

**Response:**

- **Content-Type:** `application/octet-stream`
- **Body:** Binary NPZ file

**Status Codes:**

- `200 OK` - Artifact returned
- `404 Not Found` - Dataset not found

---

### GET /v1/datasets/{id}/preview

Get a JSON preview of dataset samples.

**Path Parameters:**

- `id` (string): Dataset ID

**Query Parameters:**

- `n` (int, optional): Number of samples to return (default: 100, max: 1000)

**Response:**

```json
{
  "n_samples": 10,
  "X_sample": [[0.5, 0.3], [0.2, -0.4], ...],
  "y_sample": [[1.0, 0.0], [0.0, 1.0], ...]
}
```

**Status Codes:**

- `200 OK` - Preview returned
- `404 Not Found` - Dataset not found

---

### DELETE /v1/datasets/{id}

Delete a dataset.

**Path Parameters:**

- `id` (string): Dataset ID

**Status Codes:**

- `204 No Content` - Dataset deleted
- `404 Not Found` - Dataset not found

---

## NPZ Artifact Schema

The NPZ artifact is the primary data contract between JuniperData and its consumers (JuniperCascor, JuniperCanopy).

### Keys and Shapes

| Key       | Shape                     | Dtype     | Description                   |
| --------- | ------------------------- | --------- | ----------------------------- |
| `X_train` | `(n_train, n_features)`   | `float32` | Training features             |
| `y_train` | `(n_train, n_classes)`    | `float32` | Training labels (one-hot)     |
| `X_test`  | `(n_test, n_features)`    | `float32` | Test features                 |
| `y_test`  | `(n_test, n_classes)`     | `float32` | Test labels (one-hot)         |
| `X_full`  | `(n_samples, n_features)` | `float32` | Full dataset features         |
| `y_full`  | `(n_samples, n_classes)`  | `float32` | Full dataset labels (one-hot) |

### Spiral Dataset Specifics

For spiral datasets:

- `n_features = 2` (x, y coordinates)
- `n_classes = n_spirals` (typically 2)
- `n_samples = n_spirals × n_points_per_spiral`

### Label Encoding

Labels are **one-hot encoded**:

```python
# Class 0: [1.0, 0.0]
# Class 1: [0.0, 1.0]
```

Each row sums to 1.0 and contains exactly one 1.0 value.

### Loading Example

```python
import numpy as np

# Load from file
with np.load("dataset.npz") as data:
    X_train = data["X_train"]  # (160, 2) float32
    y_train = data["y_train"]  # (160, 2) float32
    X_test = data["X_test"]    # (40, 2) float32
    y_test = data["y_test"]    # (40, 2) float32

# Load from API response
import io
response = requests.get(f"{BASE_URL}/v1/datasets/{dataset_id}/artifact")
with np.load(io.BytesIO(response.content)) as data:
    X_train = data["X_train"]
    y_train = data["y_train"]
```

### PyTorch Conversion

```python
import torch

with np.load("dataset.npz") as data:
    X_train = torch.from_numpy(data["X_train"])  # torch.float32
    y_train = torch.from_numpy(data["y_train"])  # torch.float32
```

### Schema Validation

JuniperData guarantees:

1. All arrays are `float32` dtype
2. All arrays are 2-dimensional
3. `X_*` arrays have shape `(n, n_features)`
4. `y_*` arrays have shape `(n, n_classes)`
5. `y_*` arrays are valid one-hot encodings (each row sums to 1.0)
6. `len(X_train) + len(X_test) == len(X_full)`

---

## Error Handling

### Error Response Format

```json
{
  "detail": "Error message describing what went wrong"
}
```

### Common Status Codes

| Code                        | Description                |
| --------------------------- | -------------------------- |
| `200 OK`                    | Request succeeded          |
| `201 Created`               | Resource created           |
| `204 No Content`            | Resource deleted           |
| `400 Bad Request`           | Invalid request parameters |
| `404 Not Found`             | Resource not found         |
| `422 Unprocessable Entity`  | Validation error           |
| `500 Internal Server Error` | Server error               |

---

## Client Examples

### Python (requests)

```python
import requests
import numpy as np
import io

BASE_URL = "http://localhost:8100"

# Create dataset
response = requests.post(f"{BASE_URL}/v1/datasets", json={
    "generator": "spiral",
    "params": {
        "n_spirals": 2,
        "n_points_per_spiral": 100,
        "seed": 42
    }
})
result = response.json()
dataset_id = result["dataset_id"]

# Download artifact
response = requests.get(f"{BASE_URL}/v1/datasets/{dataset_id}/artifact")
with np.load(io.BytesIO(response.content)) as data:
    X_train = data["X_train"]
    y_train = data["y_train"]
    print(f"Training set: {X_train.shape}")
```

### curl

```bash
# Health check
curl http://localhost:8100/v1/health

# Create dataset
curl -X POST http://localhost:8100/v1/datasets \
  -H "Content-Type: application/json" \
  -d '{"generator": "spiral", "params": {"n_spirals": 2, "seed": 42}}'

# Download artifact
curl -O http://localhost:8100/v1/datasets/{dataset_id}/artifact
```

### Docker Compose Integration

```yaml
services:
  juniper-data:
    build: ./JuniperData
    ports:
      - "8100:8100"
    volumes:
      - juniper-data:/app/data/datasets
    environment:
      - JUNIPER_DATA_LOG_LEVEL=INFO
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8100/v1/health')"]
      interval: 30s
      timeout: 10s
      retries: 3

volumes:
  juniper-data:
```

---

## See Also

- [INTEGRATION_DEVELOPMENT_PLAN.md](../../notes/history/INTEGRATION_DEVELOPMENT_PLAN.md) - Integration roadmap
- [CHANGELOG.md](../../CHANGELOG.md) - Version history
- [README.md](../../README.md) - Quick start guide
