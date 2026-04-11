# Hardcoded Values Refactor Plan — juniper-data

**Version**: 0.6.0
**Created**: 2026-04-08
**Status**: PLANNING — No source code modifications
**Companion Document**: `HARDCODED_VALUES_ANALYSIS.md`

---

## Phase 1: Constants Infrastructure (Priority: HIGH)

### Step 1.1: Create API Constants Module

**Task**: Create `juniper_data/api/constants.py`
**Constants** (~25):
- Security headers: CSP, HSTS, nosniff, frame deny, referrer policy, permissions policy
- Max request body size
- API key header name
- Rate limit window
- Log format strings and field keys (11 values)
- Sentry traces sample rate
- Prometheus histogram buckets
- HTTP status codes (or adopt `starlette.status`)

### Step 1.2: Create Storage Constants Module

**Task**: Create `juniper_data/storage/constants.py`
**Constants** (~10):
- Redis: host, port, DB defaults
- PostgreSQL: host, port defaults
- Local FS: file suffixes (`.meta.json`, `.npz`, `.tmp`), JSON indent
- List defaults: limit, offset

### Step 1.3: Create Core Constants Module

**Task**: Create `juniper_data/core/constants.py`
**Constants** (~6):
- `CHARSET_UTF8 = "utf-8"` (used in 7+ locations)
- Pydantic field constraints: `DESCRIPTION_MAX_LENGTH`, `CREATED_BY_MAX_LENGTH`
- Algorithm constants

### Step 1.4: Create Per-Generator Defaults (Following Spiral Pattern)

**Task**: Create `defaults.py` for each generator without one:
- `generators/xor/defaults.py` (7 constants)
- `generators/checkerboard/defaults.py` (7 constants)
- `generators/gaussian/defaults.py` (8 constants)
- `generators/circles/defaults.py` (7 constants)
- `generators/mnist/defaults.py` (6 constants)
- `generators/csv_import/defaults.py` (8 constants)
- `generators/arc_agi/defaults.py` (8 constants)

**Pattern**: Follow `generators/spiral/defaults.py` structure exactly

---

## Phase 2: Source File Refactor (Priority: HIGH)

### Step 2.1: Refactor API Layer

**Files**: `api/middleware.py`, `api/security.py`, `api/observability.py`, `api/app.py`
**Changes**: Import from `api/constants.py` — ~22 replacements

### Step 2.2: Refactor Routes

**Files**: `api/routes/datasets.py`, `api/routes/generators.py`
**Changes**: Replace HTTP status code literals (or use `starlette.status`)

### Step 2.3: Refactor Storage Layer

**Files**: `storage/redis_store.py`, `storage/postgres_store.py`, `storage/local_fs.py`, `storage/base.py`
**Changes**: Import from `storage/constants.py` — ~11 replacements

### Step 2.4: Refactor Core Layer

**Files**: `core/dataset_id.py`, `core/models.py`
**Changes**: Import from `core/constants.py` — ~5 replacements

### Step 2.5: Refactor Generator Params

**Files**: All generator `params.py` files (7 generators)
**Changes**: Replace Field defaults with imported constants from each `defaults.py` — ~45 replacements

### Step 2.6: Refactor Encoding Strings

**Files**: `core/dataset_id.py`, `storage/redis_store.py`, `generators/csv_import/generator.py`, `generators/arc_agi/generator.py`
**Changes**: Replace `"utf-8"` with `CHARSET_UTF8` — ~7 replacements

---

## Phase 3: HTTP Status Code Strategy Decision

### Option A: Use `starlette.status` Constants (RECOMMENDED)

Replace `400`, `404`, `413`, `500` with `status.HTTP_400_BAD_REQUEST`, etc.
- Already available as FastAPI dependency
- Industry standard
- No custom constants needed

### Option B: Define Custom HTTP Constants

Create project-specific HTTP status code constants.
- More verbose
- Duplicates framework functionality

**Recommended**: Option A

---

## Phase 4: Validation (Priority: HIGH)

### Step 4.1: Run Full Test Suite

```bash
cd /home/pcalnon/Development/python/Juniper/juniper-data
conda activate JuniperData
pytest juniper_data/tests/ -v
```

### Step 4.2: Run Pre-commit Hooks

```bash
pre-commit run --all-files
```

### Step 4.3: Run Type Checking

```bash
mypy juniper_data/ --strict
```

### Step 4.4: Verify All Generator Outputs

**Task**: Run each generator with default parameters before and after refactor. Verify output arrays are identical.

---

## Phase 5: Documentation & Release (Priority: MEDIUM)

### Step 5.1: Update AGENTS.md

Document new constants files and their locations

### Step 5.2: Update CHANGELOG.md

### Step 5.3: Create Release Description

---

## Execution Order

```
Phase 1 (Infrastructure) → Phase 2 (Refactor) → Phase 3 (HTTP Strategy) → Phase 4 (Validation) → Phase 5 (Documentation)
```
