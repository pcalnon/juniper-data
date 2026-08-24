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

JuniperData is a dataset generation and management service for the Juniper ecosystem. It provides a REST API for generating, storing, and serving datasets used by juniper-cascor (neural network backend) and JuniperCanopy (web dashboard).

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

List registered dataset generators.

**Response:**

```json
[
  {
    "name": "spiral",
    "version": "1.0.0",
    "description": "Multi-spiral classification dataset generator",
    "available": true
  }
]
```

`available` reports whether the generator's optional dependencies are present in the running deployment (e.g. `mnist` requires the Hugging Face `datasets` package; `equities` / `equities_seq` require the `equities` extra). Creating a dataset with an unavailable generator returns `501 Not Implemented` with an install hint.

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
  "type": "object",
  "available": true
}
```

`available` is an additive top-level key (JSON Schema consumers ignore unknown keywords) reporting whether the generator's optional dependencies are present in the running deployment.

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
  "tags": ["baseline", "can-def-005"],
  "ttl_seconds": 86400,
  "name": "spiral-baseline",
  "description": "Reference dataset for model comparisons",
  "created_by": "ml-platform",
  "parent_dataset_id": "spiral-1.0.0-previous..."
}
```

**Request Fields:**

| Field | Type | Required | Description |
| ----- | ---- | -------- | ----------- |
| `generator` | string | Yes | Generator name (e.g., `spiral`) |
| `params` | object | No | Generator-specific parameters |
| `persist` | boolean | No | Whether to persist to storage (default: `true`) |
| `tags` | array[string] | No | Dataset tags for filtering and organization |
| `ttl_seconds` | integer | No | Dataset time-to-live in seconds (minimum `1`) |
| `name` | string | No | Logical dataset name used for version tracking |
| `description` | string | No | Free-text description (max 500 chars) |
| `created_by` | string | No | Creator identifier (max 100 chars) |
| `parent_dataset_id` | string | No | Parent dataset ID for lineage tracking |

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
    "checksum": "4bf28dcf4f5eb0866b7f2e4d3d4a2d4d4bbf2e8ab6f6fb4112d59bd95af3f412",
    "dataset_name": "spiral-baseline",
    "dataset_version": 3,
    "parent_dataset_id": "spiral-1.0.0-previous...",
    "description": "Reference dataset for model comparisons",
    "created_by": "ml-platform",
    "tags": ["baseline", "can-def-005"],
    "ttl_seconds": 86400,
    "expires_at": "2026-02-06T12:00:00.000000",
    "last_accessed_at": null,
    "access_count": 0
  },
  "artifact_url": "/v1/datasets/spiral-1.0.0-a1b2c3d4e5f6.../artifact"
}
```

**Status Codes:**

- `201 Created` - Dataset created or retrieved
- `400 Bad Request` - Unknown generator or invalid parameters
- `501 Not Implemented` - Generator's optional dependencies are missing in this deployment (the `detail` carries an actionable install hint, e.g. `pip install datasets` for `mnist`)

**Caching Behavior:**

Datasets are cached by their deterministic ID (hash of generator + version + params). Requesting the same parameters returns the existing dataset.

**Versioning Behavior (named datasets):**

- If `name` is provided, the service assigns `dataset_version` using stored datasets with the same `dataset_name`.
- Only persisted datasets (`persist=true`) advance the stored version sequence.
- The first stored version is `1`; subsequent stored versions increment by 1.
- Repeating an identical create request returns the cached dataset and preserves its existing version (no new version is assigned).

---

### GET /v1/datasets

List stored dataset IDs.

**Query Parameters:**

- `limit` (int, optional): Maximum IDs to return (default: `100`, max: `1000`)
- `offset` (int, optional): Number of IDs to skip (default: `0`)

**Query Parameters:**

- `limit` (int, optional): Maximum IDs to return (default: `100`, range: `1..1000`)
- `offset` (int, optional): Number of IDs to skip (default: `0`)

**Response:**

```json
[
  "spiral-1.0.0-a1b2c3d4e5f6...",
  "spiral-1.0.0-f6e5d4c3b2a1..."
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
| `offset` | integer | `0` | Page offset. Mutually exclusive with `cursor`. |
| `cursor` | string | `null` | Opaque token from a previous response's `next_cursor`. Stable under concurrent writes. |

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

### GET /v1/datasets/filter

Filter datasets and return full metadata results with pagination.

**Query Parameters (all optional):**

- `generator` - Exact generator name
- `tags` - Comma-separated tags (example: `baseline,prod`)
- `tags_match` - `any` (OR) or `all` (AND), default `any`
- `created_after` - ISO datetime lower bound
- `created_before` - ISO datetime upper bound
- `min_samples` - Minimum sample count
- `max_samples` - Maximum sample count
- `include_expired` - Include expired datasets (`false` by default)
- `dataset_name` - Logical dataset name filter
- `dataset_version` - Exact dataset version filter
- `limit` - Page size (default `100`, max `1000`)
- `offset` - Pagination offset (default `0`); mutually exclusive with `cursor`
- `cursor` - Opaque token from a previous response's `next_cursor` (see **Ordering and pagination** below)

**Response:**

```json
{
  "datasets": [
    {
      "dataset_id": "spiral-1.0.0-a1b2c3...",
      "generator": "spiral",
      "dataset_name": "spiral-baseline",
      "dataset_version": 3,
      "created_at": "2026-02-05T12:00:00.000000"
    }
  ],
  "total": 1,
  "limit": 100,
  "offset": 0
}
```

---

### GET /v1/datasets/versions

List all versions for a logical dataset name.

**Query Parameters:**

- `name` (string, required): Dataset name to list versions for

**Response:**

```json
{
  "dataset_name": "spiral-baseline",
  "versions": [
    {"dataset_id": "spiral-1.0.0-v1...", "dataset_version": 1},
    {"dataset_id": "spiral-1.0.0-v2...", "dataset_version": 2},
    {"dataset_id": "spiral-1.0.0-v3...", "dataset_version": 3}
  ],
  "total": 3,
  "latest_version": 3
}
```

---

### GET /v1/datasets/latest

Get the latest stored version for a logical dataset name.

**Query Parameters:**

- `name` (string, required): Dataset name

**Status Codes:**

- `200 OK` - Latest version metadata returned
- `404 Not Found` - No versions found for the provided name

---

#### Ordering and pagination

Results are ordered **newest first**, ties broken by `dataset_id` ascending. That second
key matters: sorting on `created_at` alone is not a total order, so datasets sharing a
timestamp used to come back in whatever sequence the storage layer happened to enumerate.

Every response carries `next_cursor` — the position of the last returned row in that
order. There are two ways to page:

| | Behaviour |
|---|---|
| `offset` | Re-slices the current result set. A dataset created or deleted before the offset shifts every later page, so a row can be **returned twice or skipped**. Fine for a one-shot page; unsafe for a full walk of a live collection. |
| `cursor` | Asks for the rows strictly after a named position. Inserts and deletes ahead of the cursor cannot shift it, so a full walk neither repeats nor skips. |

Pass `cursor` **or** `offset`, never both — a request carrying both is rejected with `400`,
because a cursor already determines where the page starts. A cursor the service did not
issue is also a `400`. Treat the token as opaque: it is not a stable identifier, and its
encoding may change.

```bash
# Stable walk of the whole collection.
curl "$BASE/v1/datasets/filter?limit=100" | jq -r '.next_cursor'
curl "$BASE/v1/datasets/filter?limit=100&cursor=<next_cursor>"
```

### GET /v1/datasets/stats

Get aggregate statistics across stored datasets.

**Response:**

```json
{
  "total_datasets": 42,
  "total_samples": 8400,
  "by_generator": {"spiral": 30, "xor": 12},
  "by_tag": {"baseline": 10, "prod": 8},
  "oldest_created_at": "2026-02-01T10:00:00.000000",
  "newest_created_at": "2026-02-06T17:30:00.000000",
  "expired_count": 3
}
```

---

### POST /v1/datasets/batch-create

Create multiple datasets in one request.

Each item is processed independently. A failure in one item does not fail the whole batch.

**Status codes:**

| Status | When |
|--------|------|
| `201 Created` | At least one dataset was created (`total_created > 0`), whether or not other items failed. |
| `200 OK` | No dataset was created (`total_created == 0`). The batch was processed; read `results` for the per-item reason. |
| `422 Unprocessable Entity` | The request itself is invalid — e.g. an empty `datasets` list, or more than 50 items. |

`200` is not an error status here. Because every item reports its own outcome, the body is
the authority for what happened; the status line only distinguishes "something was created"
from "nothing was". A caller that checks only the status must not read `201` as a guarantee
that *every* item succeeded — inspect `total_failed`.

**Request Body:**

```json
{
  "datasets": [
    {
      "generator": "spiral",
      "params": {"n_spirals": 2, "seed": 42},
      "persist": true,
      "name": "batch-exp",
      "tags": ["baseline"]
    },
    {
      "generator": "unknown-generator",
      "params": {},
      "persist": true
    }
  ]
}
```

**Response:**

```json
{
  "results": [
    {
      "index": 0,
      "dataset_id": "spiral-1.0.0-a1b2c3d4e5f6a7b8",
      "generator": "spiral",
      "success": true,
      "error": null,
      "artifact_url": "/v1/datasets/spiral-1.0.0-a1b2c3d4e5f6a7b8/artifact"
    },
    {
      "index": 1,
      "dataset_id": null,
      "generator": "unknown-generator",
      "success": false,
      "error": "Unknown generator 'unknown-generator'. Available: ['spiral', ...]",
      "artifact_url": null
    }
  ],
  "total_created": 1,
  "total_failed": 1
}
```

---

### POST /v1/datasets/batch-delete

Delete multiple datasets by ID.

**Request Body:**

```json
{
  "dataset_ids": ["spiral-1.0.0-a1...", "spiral-1.0.0-b2..."]
}
```

**Response:**

```json
{
  "deleted": ["spiral-1.0.0-a1..."],
  "not_found": ["spiral-1.0.0-b2..."],
  "total_deleted": 1
}
```

---

### PATCH /v1/datasets/batch-tags

Add/remove tags on multiple datasets.

**Request Body:**

```json
{
  "dataset_ids": ["spiral-1.0.0-a1...", "spiral-1.0.0-b2..."],
  "add_tags": ["prod"],
  "remove_tags": ["stale"]
}
```

---

### POST /v1/datasets/batch-export

Export multiple artifacts as a ZIP archive of `*.npz` files.

**Request Body:**

```json
{
  "dataset_ids": ["spiral-1.0.0-a1...", "spiral-1.0.0-b2..."]
}
```

**Status Codes:**

- `200 OK` - ZIP archive returned (`application/zip`)
- `404 Not Found` - None of the requested dataset IDs exist

**Partial exports:** a `200` does **not** guarantee every requested dataset is in the
archive. An id can be absent because it did not exist when the request was received, or
because it was deleted while the archive was being streamed.

When anything is missing, the archive carries an extra member, `manifest.json`:

```json
{
  "requested": ["spiral-1.0.0-a1...", "spiral-1.0.0-b2...", "spiral-1.0.0-gone..."],
  "exported": ["spiral-1.0.0-a1...", "spiral-1.0.0-b2..."],
  "missing": {"spiral-1.0.0-gone...": "not_found"}
}
```

`missing` maps each absent id to a reason: `not_found` (absent when the request was
received) or `vanished_during_export` (deleted mid-stream). `requested` always equals
`exported` plus the keys of `missing`, so a caller can reconcile without guessing.

**A complete export contains no `manifest.json`** — its presence *is* the signal that
something is missing, and an archive with every requested dataset is byte-for-byte what
this endpoint has always returned. Callers that only read `*.npz` members are unaffected
either way.

The manifest lives inside the archive rather than in a header because the response is
streamed: the status line and headers are sent before the first artifact is read, so
neither can report a dataset that disappears later.

---

### POST /v1/datasets/cleanup-expired

Delete all datasets currently past their `expires_at` timestamp.

**Response:**

```json
["spiral-1.0.0-expired1...", "spiral-1.0.0-expired2..."]
```

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
  "checksum": "4bf28dcf4f5eb0866b7f2e4d3d4a2d4d4bbf2e8ab6f6fb4112d59bd95af3f412",
  "dataset_name": "spiral-baseline",
  "dataset_version": 3
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

### PATCH /v1/datasets/{id}/tags

Add/remove tags on a single dataset.

**Request Body:**

```json
{
  "add_tags": ["golden", "prod"],
  "remove_tags": ["stale"]
}
```

**Status Codes:**

- `200 OK` - Updated metadata returned
- `404 Not Found` - Dataset not found

---

## NPZ Artifact Schema

The NPZ artifact is the primary data contract between JuniperData and its consumers (juniper-cascor, JuniperCanopy).

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

Every error response is a JSON object with a `detail` key. **`detail` has two shapes, and
which one you get depends on the status code** — check the type before consuming it.

Most errors carry a human-readable **string**:

```json
{
  "detail": "Unknown generator 'nope'. Available: ['spiral', ...]"
}
```

A `422` carries a **list of per-field error objects**, so a caller can report which field
failed and why:

```json
{
  "detail": [
    {"type": "missing", "loc": ["body", "generator"], "msg": "Field required"}
  ]
}
```

This split is a known limitation, not an accident: unifying the two shapes requires a
response envelope (RFC 9457 problem details), which is tracked separately. `juniper-data-client`
already handles both — it renders the list as `body.generator: Field required` for the
exception message while leaving the structure intact on `exc.detail`.

### Common Status Codes

| Code                        | Description                                      | `detail` shape |
| --------------------------- | ------------------------------------------------ | -------------- |
| `200 OK`                    | Request succeeded                                | —              |
| `201 Created`               | Resource created                                 | —              |
| `204 No Content`            | Resource deleted                                 | —              |
| `400 Bad Request`           | Schema-valid, but semantically wrong             | `str`          |
| `404 Not Found`             | Resource not found                               | `str`          |
| `422 Unprocessable Content` | Violates the declared request schema             | `list[object]` |
| `500 Internal Server Error` | Server error                                     | `str`          |
| `501 Not Implemented`       | Generator's optional dependency is not installed | `str`          |

#### 400 vs 422 — the rule

Both mean "the caller sent something wrong", and the boundary between them is deliberate:

- **`422`** — the request violates the **declared schema**, and is rejected before the
  handler runs: a missing required field, `ttl_seconds: 0`, or `params` that is not an object.
- **`400`** — the request is schema-valid but **semantically wrong for the generator it
  names**: an unknown generator, or params that the named generator rejects. `params` is
  typed as a free-form object, so only the resolved generator can validate its contents.

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
