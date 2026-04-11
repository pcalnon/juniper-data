# Hardcoded Values Analysis — juniper-data

**Version**: 0.6.0
**Analysis Date**: 2026-04-08
**Analyst**: Claude Code (Automated Code Review)
**Status**: PLANNING ONLY — No source code modifications

---

## Executive Summary

The juniper-data codebase contains approximately **118 identified hardcoded values** across its source files. The application has a well-structured `api/settings.py` (Pydantic BaseSettings) and `generators/spiral/defaults.py` that cover many infrastructure and spiral-specific values. However, significant gaps exist in security headers, observability config, storage backends, non-spiral generator defaults, and HTTP status codes.

---

## 1. Existing Constants Infrastructure

| File | Purpose | Coverage |
|------|---------|----------|
| `api/settings.py` | API host, port, log level, rate limits, storage path, import dir | Good — environment-configurable |
| `generators/spiral/defaults.py` | Spiral generator defaults and validation bounds | Excellent — comprehensive |

**Gap**: No dedicated constants file for middleware, security, observability, storage, or non-spiral generators.

---

## 2. Hardcoded Values Inventory

### 2.1 Security Headers & Middleware (`api/middleware.py`) — NOT COVERED

| Line | Value | Type | Context | Proposed Constant Name |
|------|-------|------|---------|----------------------|
| 46 | `"max-age=31536000"` | str | HSTS max-age header | `HSTS_MAX_AGE_VALUE` |
| 38 | `"nosniff"` | str | X-Content-Type-Options | `HEADER_NOSNIFF` |
| 39 | `"DENY"` | str | X-Frame-Options | `HEADER_FRAME_DENY` |
| 40 | `"strict-origin-when-cross-origin"` | str | Referrer-Policy | `HEADER_REFERRER_POLICY` |
| 41 | `"camera=(), microphone=(), geolocation=()"` | str | Permissions-Policy | `HEADER_PERMISSIONS_POLICY` |
| 51 | `10 * 1024 * 1024` | int | Max request body (10 MB) | `MAX_REQUEST_BODY_BYTES` |

**Target location**: New `api/constants.py`

### 2.2 Security & Authentication (`api/security.py`) — NOT COVERED

| Line | Value | Type | Context | Proposed Constant Name |
|------|-------|------|---------|----------------------|
| 12 | `"X-API-Key"` | str | API key header name | `API_KEY_HEADER_NAME` |
| 92 | `60` | int | Rate limit window (seconds) | `RATE_LIMIT_WINDOW_SECONDS` |

**Target location**: `api/constants.py`

### 2.3 Observability (`api/observability.py`) — NOT COVERED

| Line | Value | Type | Context | Proposed Constant Name |
|------|-------|------|---------|----------------------|
| 29-34 | `"timestamp"`, `"level"`, etc. | str | JSON log field keys (6 values) | `LOG_KEY_TIMESTAMP`, `LOG_KEY_LEVEL`, etc. |
| 116 | `"json"` | str | JSON format identifier | `LOG_FORMAT_JSON` |
| 119 | `"%(asctime)s - ..."` | str | Plain text log format | `LOG_FORMAT_PLAIN` |
| 141 | `1.0` | float | Sentry traces sample rate | `SENTRY_TRACES_SAMPLE_RATE` |
| 195 | `(0.01, 0.05, ...)` | tuple | Prometheus histogram buckets | `HISTOGRAM_BUCKETS_SECONDS` |

**Target location**: `api/constants.py`

### 2.4 HTTP Status Codes (Multiple Files) — NOT COVERED

| File | Line | Value | Context | Proposed Constant Name |
|------|------|-------|---------|----------------------|
| `api/app.py` | 126 | `400` | Bad request | `HTTP_400_BAD_REQUEST` |
| `api/app.py` | 134 | `500` | Internal server error | `HTTP_500_INTERNAL_ERROR` |
| `api/routes/datasets.py` | 77, 89 | `400` | Bad request | (same) |
| `api/routes/datasets.py` | 45 | `500` | Storage not initialized | (same) |
| `api/routes/generators.py` | 114 | `404` | Generator not found | `HTTP_404_NOT_FOUND` |
| `api/middleware.py` | 64 | `413` | Payload too large | `HTTP_413_PAYLOAD_TOO_LARGE` |

**Note**: FastAPI provides `starlette.status` constants. Recommend using those directly rather than creating custom HTTP status constants.

### 2.5 Storage Backends — NOT COVERED

| File | Line | Value | Type | Context | Proposed Constant Name |
|------|------|-------|------|---------|----------------------|
| `storage/redis_store.py` | 34 | `"localhost"` | str | Redis default host | `REDIS_HOST_DEFAULT` |
| `storage/redis_store.py` | 35 | `6379` | int | Redis default port | `REDIS_PORT_DEFAULT` |
| `storage/redis_store.py` | 36 | `0` | int | Redis default DB | `REDIS_DB_DEFAULT` |
| `storage/postgres_store.py` | 76 | `"localhost"` | str | PostgreSQL default host | `POSTGRES_HOST_DEFAULT` |
| `storage/postgres_store.py` | 77 | `5432` | int | PostgreSQL default port | `POSTGRES_PORT_DEFAULT` |
| `storage/base.py` | 93-94 | `100`, `0` | int | Default list limit, offset | `DEFAULT_LIST_LIMIT`, `DEFAULT_LIST_OFFSET` |
| `storage/local_fs.py` | 45 | `".meta.json"` | str | Metadata file suffix | `META_FILE_SUFFIX` |
| `storage/local_fs.py` | 49 | `".npz"` | str | NPZ file suffix | `NPZ_FILE_SUFFIX` |
| `storage/local_fs.py` | 71 | `".tmp"` | str | Temp file suffix | `TEMP_FILE_SUFFIX` |
| `storage/local_fs.py` | 77 | `2` | int | JSON indent spaces | `JSON_INDENT` |

**Target location**: New `storage/constants.py`

### 2.6 Encoding Strings (Multiple Files) — NOT COVERED

| File | Line | Value | Context |
|------|------|-------|---------|
| `core/dataset_id.py` | 36 | `"utf-8"` | JSON encoding for SHA256 |
| `storage/redis_store.py` | 84, 88 | `"utf-8"` | Redis encode/decode |
| `generators/csv_import/generator.py` | 109, 128 | `"utf-8"` | File encoding |
| `generators/arc_agi/generator.py` | 155 | `"utf-8"` | JSON file encoding |

**Proposed constant**: `CHARSET_UTF8 = "utf-8"` in `api/constants.py` or a shared `core/constants.py`

### 2.7 Pydantic Field Constraints (`core/models.py`) — NOT COVERED

| Line | Value | Type | Context | Proposed Constant Name |
|------|-------|------|---------|----------------------|
| 63, 157 | `500` | int | Description max_length | `DESCRIPTION_MAX_LENGTH` |
| 64, 158 | `100` | int | Created_by max_length | `CREATED_BY_MAX_LENGTH` |

**Target location**: `core/constants.py`

### 2.8 Non-Spiral Generator Defaults — NOT COVERED

Each generator has its own `params.py` with hardcoded Pydantic Field defaults:

**XOR** (`generators/xor/params.py`):
- `n_points_per_quadrant=50`, `x_range=1.0`, `y_range=1.0`, `margin=0.1`, `noise=0.0`, `train_ratio=0.8`, `test_ratio=0.2`

**Checkerboard** (`generators/checkerboard/params.py`):
- `n_samples=200`, `n_squares=4`, `x_range=(0.0, 1.0)`, `y_range=(0.0, 1.0)`, `noise=0.0`, `train_ratio=0.8`, `test_ratio=0.2`

**Gaussian** (`generators/gaussian/params.py`):
- `n_classes=2`, `n_samples_per_class=50`, `n_features=2`, `class_std=1.0`, `center_radius=3.0`, `noise=0.0`, `train_ratio=0.8`, `test_ratio=0.2`

**Circles** (`generators/circles/params.py`):
- `n_samples=100`, `outer_radius=1.0`, `factor=0.5`, `noise=0.0`, `inner_ratio=0.5`, `train_ratio=0.8`, `test_ratio=0.2`

**MNIST** (`generators/mnist/params.py`):
- `dataset="mnist"`, `flatten=True`, `normalize=True`, `one_hot_labels=True`, `train_ratio=0.8`, `test_ratio=0.2`

**CSV Import** (`generators/csv_import/params.py`):
- `file_format="auto"`, `label_column="label"`, `delimiter=","`, `header=True`, `one_hot_labels=True`, `normalize_features=False`, `train_ratio=0.8`, `test_ratio=0.2`

**ARC-AGI** (`generators/arc_agi/params.py`):
- `source="huggingface"`, `subset="training"`, `pad_to=30`, `pad_value=-1`, `include_test=True`, `flatten_pairs=True`, `train_ratio=0.8`, `test_ratio=0.2`

**Target location**: Create `generators/defaults.py` (consolidated) or individual `defaults.py` per generator (following the spiral pattern)

### 2.9 Generator Algorithm Constants — NOT COVERED

| File | Line | Value | Context | Proposed Constant Name |
|------|------|-------|---------|----------------------|
| `generators/spiral/generator.py` | 80 | `2 * np.pi` | Full rotation (radians) | `TWO_PI` (mathematical constant) |
| `generators/gaussian/generator.py` | 119 | `2 * np.pi` | Full rotation | (same) |
| `generators/xor/generator.py` | 121 | `4` | Number of quadrants | `XOR_QUADRANT_COUNT` |

---

## 3. Coverage Summary

| Category | Total | Covered | Not Covered | Priority |
|----------|-------|---------|-------------|----------|
| API Settings | 12 | 12 | 0 | — |
| Spiral Defaults | 18 | 18 | 0 | — |
| Security Headers | 6 | 0 | 6 | **HIGH** |
| Observability | 11 | 0 | 11 | **HIGH** |
| HTTP Status Codes | 7 | 0 | 7 | **MEDIUM** |
| Storage Backends | 10 | 0 | 10 | **MEDIUM** |
| Generator Defaults | 40+ | 0 | 40+ | **MEDIUM** |
| Encoding Strings | 7 | 0 | 7 | **LOW** |
| Field Constraints | 4 | 0 | 4 | **LOW** |
| Algorithm Constants | 3 | 0 | 3 | **LOW** |
| **TOTAL** | **~118** | **30** | **~88** | — |

---

## 4. Remediation Approaches

### Approach A: Per-Layer Constants Modules (RECOMMENDED)

Create constants files organized by architectural layer:

1. **`api/constants.py`** — Security headers, observability config, HTTP status codes, rate limit settings
2. **`storage/constants.py`** — Storage backend defaults (Redis, PostgreSQL, local FS file extensions)
3. **`core/constants.py`** — Shared values (charset, field constraints)
4. **`generators/defaults.py`** — Consolidated generator defaults (or per-generator `defaults.py` mirroring spiral pattern)

**Strengths**:
- Follows existing patterns (spiral `defaults.py` is the template)
- Layer-appropriate organization
- Minimal import distance
- Easy to discover

**Weaknesses**:
- Multiple constants files to maintain
- Cross-layer values may need duplication or shared module

### Approach B: Single Centralized Constants Module

Create a single `juniper_data/constants.py` with all constants organized by section.

**Strengths**:
- Single source of truth
- Easy to audit

**Weaknesses**:
- File grows large (100+ constants)
- Not aligned with per-generator `defaults.py` pattern
- Layer boundaries are lost

### Recommended Approach: **A** (Per-Layer Constants)

Follow the established spiral `defaults.py` pattern. Each generator gets its own defaults file. Infrastructure constants go in layer-appropriate files.

---

## 5. Files Requiring Modification (Implementation Phase)

| File | Action | Constants Count |
|------|--------|-----------------|
| `api/constants.py` | **NEW** | ~25 |
| `storage/constants.py` | **NEW** | ~10 |
| `core/constants.py` | **NEW** | ~6 |
| `generators/xor/defaults.py` | **NEW** (following spiral pattern) | 7 |
| `generators/checkerboard/defaults.py` | **NEW** | 7 |
| `generators/gaussian/defaults.py` | **NEW** | 8 |
| `generators/circles/defaults.py` | **NEW** | 7 |
| `generators/mnist/defaults.py` | **NEW** | 6 |
| `generators/csv_import/defaults.py` | **NEW** | 8 |
| `generators/arc_agi/defaults.py` | **NEW** | 8 |
| `api/middleware.py` | **MODIFY** — import and use constants | 6 replacements |
| `api/security.py` | **MODIFY** | 2 replacements |
| `api/observability.py` | **MODIFY** | 11 replacements |
| `api/app.py` | **MODIFY** | 2 replacements |
| `api/routes/datasets.py` | **MODIFY** | 3 replacements |
| `api/routes/generators.py` | **MODIFY** | 1 replacement |
| `storage/redis_store.py` | **MODIFY** | 3 replacements |
| `storage/postgres_store.py` | **MODIFY** | 2 replacements |
| `storage/local_fs.py` | **MODIFY** | 4 replacements |
| `storage/base.py` | **MODIFY** | 2 replacements |
| `core/dataset_id.py` | **MODIFY** | 1 replacement |
| `core/models.py` | **MODIFY** | 4 replacements |
| Multiple generator `params.py` files | **MODIFY** — import from new defaults | ~45 replacements |

---

## 6. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Generator behavior changes | Very Low | High | Constants preserve exact literal values |
| Storage connection failures from wrong defaults | Very Low | Medium | Constants match current hardcoded values |
| Import errors from new module structure | Low | Low | Run full test suite after each change |
| HTTP status code misuse | Very Low | Low | Use `starlette.status` for HTTP codes |
| Test failures | Low | Low | Tests assert behavior, not specific literal values |
