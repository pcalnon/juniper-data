# AGENTS.md Drift Analysis

**Project**: Juniper Data - Dataset Generation Service
**Application**: juniper-data
**Analysis Date**: 2026-04-02
**Analyst**: Claude Code (Opus 4.6)
**AGENTS.md Version Audited**: 0.4.2 (Last Updated: 2026-02-06)
**Current Application Version**: 0.5.0 (pyproject.toml)

---

## Executive Summary

A comprehensive audit of the juniper-data `AGENTS.md` file reveals **significant drift** between the documented state and the actual application. The file has not been substantively updated since 2026-02-06, while the application has undergone major changes including a security hardening release (v0.5.0), polyrepo migration, ruff migration, performance test infrastructure, observability features, and expanded storage/API capabilities.

**28 discrete drift items** were identified across 11 categories. Of these:

- **5 are Critical** (version, directory structure, security, dependencies, line length)
- **9 are High** (missing sections, outdated references, incomplete component documentation)
- **14 are Medium/Low** (incomplete details, minor inaccuracies)

---

## Methodology

This analysis was performed by:

1. Reading the complete `AGENTS.md` file (406 lines)
2. Exploring the full directory tree of the juniper-data repository
3. Reading all Python source files, configuration files, and documentation
4. Cross-referencing against `pyproject.toml`, CI workflows, and the development roadmap
5. Analyzing git history for changes since the last AGENTS.md update
6. Comparing documented state against actual codebase state

---

## Drift Items

### Category 1: Version and Metadata

| ID | Field | AGENTS.md Value | Actual Value | Severity |
|----|-------|-----------------|--------------|----------|
| D-001 | Version | 0.4.2 | 0.5.0 (pyproject.toml) | **CRITICAL** |
| D-002 | Last Updated | 2026-02-06 | Needs update to 2026-04-02 | MEDIUM |
| D-003 | `__init__.py` version | 0.4.2 (referenced in directory tree comment) | 0.4.2 (stale, should match pyproject.toml 0.5.0) | HIGH |

**Analysis**: The application shipped v0.5.0 as a security hardening release (SecurityHeadersMiddleware, RequestBodyLimitMiddleware, CORS defaults, rate limiting). The AGENTS.md still references 0.4.2. The `__init__.py` version also lags at 0.4.2, creating a three-way inconsistency between pyproject.toml (0.5.0), `__init__.py` (0.4.2), and AGENTS.md (0.4.2).

---

### Category 2: Directory Structure

| ID | Item | Status | Severity |
|----|------|--------|----------|
| D-004 | `docs/` directory tree | **Missing entirely** | **CRITICAL** |
| D-005 | `scripts/` directory tree | **Missing entirely** | HIGH |
| D-006 | `notes/` directory tree | **Missing entirely** | HIGH |
| D-007 | `conf/` directory tree | **Missing entirely** | MEDIUM |
| D-008 | `util/` directory tree | **Missing entirely** | MEDIUM |
| D-009 | `.github/` directory tree | **Missing entirely** | HIGH |
| D-010 | `Dockerfile` | **Missing from tree** | HIGH |
| D-011 | `.pre-commit-config.yaml` | **Missing from tree** | MEDIUM |
| D-012 | `.env.example` | **Missing from tree** | LOW |
| D-013 | `requirements.lock` | **Missing from tree** | LOW |
| D-014 | `CHANGELOG.md` | **Missing from tree** | MEDIUM |
| D-015 | `api/` subdirectory incomplete | Missing `middleware.py`, `security.py`, `observability.py`, `models/` | **CRITICAL** |
| D-016 | `storage/` not expanded | Does not show 7 backend implementations | HIGH |
| D-017 | `core/` not expanded | Does not show `dataset_id.py`, `split.py`, `artifacts.py`, `secrets.py` | HIGH |
| D-018 | `tests/` subdirectories incomplete | Missing `performance/`, `api/`, `fixtures/` | HIGH |

**Analysis**: The directory structure section shows only 15 entries when the actual repository contains 100+ significant files across 20+ directories. The tree omits entire top-level directories (`docs/`, `scripts/`, `notes/`, `conf/`, `util/`, `.github/`) and critical application files (`Dockerfile`, middleware, security, observability). This renders the directory structure section inadequate as a codebase navigation aid.

**Actual Top-Level Structure** (for reference):
```
juniper-data/
├── juniper_data/          # Main Python package (104 .py files)
│   ├── __init__.py
│   ├── __main__.py
│   ├── api/               # FastAPI application (8 files)
│   ├── core/              # Core domain logic (6 files)
│   ├── generators/        # 8 generator types (32 files)
│   ├── storage/           # 7 storage backends (9 files)
│   └── tests/             # Test suite (45 files)
├── docs/                  # User and developer documentation
├── scripts/               # CI/coverage scripts
├── notes/                 # Development notes, procedures, roadmaps
├── conf/                  # Shell/logging configuration
├── util/                  # Bash utility scripts (40+ scripts)
├── .github/               # Workflows, CODEOWNERS, dependabot
├── Dockerfile             # Multi-stage production build
├── pyproject.toml         # Project configuration
├── .pre-commit-config.yaml
├── .env.example
├── requirements.lock
├── CHANGELOG.md
├── README.md
└── AGENTS.md / CLAUDE.md
```

---

### Category 3: Component Overview Table

| ID | Item | AGENTS.md State | Actual State | Severity |
|----|------|-----------------|--------------|----------|
| D-019 | `generators/spiral/` description | "Two-spiral classification dataset" | Multi-spiral (configurable arms) | LOW |
| D-020 | Missing component entries | 6 entries | Should have 15+ entries | HIGH |

**Missing component entries**:
- `api/middleware.py` -- Request processing middleware (security headers, body limits, rate limiting)
- `api/security.py` -- API key authentication and rate limiting
- `api/observability.py` -- Prometheus metrics, structured logging, Sentry integration
- `api/models/` -- Pydantic response models (health, readiness)
- `core/dataset_id.py` -- Deterministic SHA-256 dataset ID generation
- `core/split.py` -- Train/test data splitting
- `core/artifacts.py` -- NPZ artifact handling and checksums
- `core/secrets.py` -- Docker secrets management
- `storage/local_fs.py` -- Local filesystem storage backend
- `storage/memory.py` -- In-memory storage (testing)
- `storage/cached.py` -- Composable caching wrapper
- `storage/redis_store.py` -- Redis storage backend
- `storage/postgres_store.py` -- PostgreSQL storage backend
- `storage/hf_store.py` -- Hugging Face Hub integration
- `storage/kaggle_store.py` -- Kaggle dataset integration

---

### Category 4: Code Style and Formatting

| ID | Item | AGENTS.md Value | Actual Value | Severity |
|----|------|-----------------|--------------|----------|
| D-021 | Line length | 120 characters | **320 characters** (`pyproject.toml` `[tool.ruff] line-length = 320`) | **CRITICAL** |

**Analysis**: The line length has changed multiple times (120 -> 512 -> 320) per roadmap audit notes. The AGENTS.md documents 120, while the actual configured value is 320. This is a critical discrepancy because developers following AGENTS.md guidance would format code incorrectly.

---

### Category 5: Dependencies

| ID | Item | AGENTS.md State | Actual State | Severity |
|----|------|-----------------|--------------|----------|
| D-022 | Core dependencies missing `python-dotenv` | Not listed | `python-dotenv>=1.0.0` in pyproject.toml | HIGH |
| D-023 | API dependencies missing `pydantic-settings` | Not listed | `pydantic-settings>=2.0.0` in pyproject.toml | MEDIUM |
| D-024 | Missing dependency groups | Only core/API/dev listed | 6 groups: core, api, arc-agi, test, observability, dev | HIGH |

**Missing dependency groups**:
- `[arc-agi]`: `arc-agi>=0.9.0`
- `[test]`: pytest, pytest-cov, pytest-timeout, pytest-asyncio, pytest-benchmark, httpx, coverage, juniper-data-client
- `[observability]`: prometheus-client, sentry-sdk[fastapi]

**Dev dependencies incomplete**: Missing `bandit[sarif]>=1.9.4`, `pip-audit>=2.7.0`, `pre-commit>=3.0.0`

---

### Category 6: Testing Section

| ID | Item | AGENTS.md State | Actual State | Severity |
|----|------|-----------------|--------------|----------|
| D-025 | Test directories | 2 (unit, integration) | 4 (unit, integration, performance, api) + fixtures | HIGH |
| D-026 | Test markers | 6 markers listed | 8 markers configured in pyproject.toml | MEDIUM |
| D-027 | Test count | Not mentioned | 766+ tests (per PR notes) | MEDIUM |
| D-028 | Performance testing | Not mentioned | 41 benchmarks via pytest-benchmark | MEDIUM |

**Missing markers**: `performance`, `slow`

**Missing test infrastructure**:
- `tests/performance/` -- Generator and storage benchmarks (41 tests)
- `tests/api/` -- Batch operations tests
- `tests/fixtures/` -- Golden dataset fixtures
- `conftest.py` -- 1,909 LOC of shared fixtures

---

### Category 7: API Design Section

| ID | Item | AGENTS.md State | Actual State | Severity |
|----|------|-----------------|--------------|----------|
| D-029 | Response format | Generic `{status, data, meta}` | Typed Pydantic models (CreateDatasetResponse, DatasetListResponse, etc.) | MEDIUM |
| D-030 | Endpoint catalog | Not documented | 20+ endpoints across health, generators, datasets, batch ops | HIGH |
| D-031 | Middleware stack | Not mentioned | 6-layer middleware (CORS, body limit, security headers, auth, metrics, request ID) | **CRITICAL** |
| D-032 | Batch operations | Not mentioned | 4 batch endpoints (create, delete, export, tags) | MEDIUM |
| D-033 | Dataset versioning | Not mentioned | Full versioning API (versions, latest, named datasets) | MEDIUM |
| D-034 | Dataset lifecycle | Not mentioned | TTL, expiration, cleanup, access tracking | MEDIUM |

---

### Category 8: Security Section

| ID | Item | AGENTS.md State | Actual State | Severity |
|----|------|-----------------|--------------|----------|
| D-035 | Security documentation | 3 bullet points (no secrets, Pydantic, .gitignore) | Comprehensive security: API key auth, rate limiting, security headers, body limits, CORS, Docker secrets, Sentry | **CRITICAL** |

**Actual security features not documented**:
- `SecurityMiddleware` -- API key authentication via X-API-Key header
- `RateLimiter` -- Per-client fixed-window rate limiting (configurable per-minute)
- `SecurityHeadersMiddleware` -- X-Content-Type-Options, X-Frame-Options, Referrer-Policy, Permissions-Policy, CSP, HSTS
- `RequestBodyLimitMiddleware` -- 10 MB body size limit
- CORS configuration (restricted by default)
- Docker secrets support for API keys (`JUNIPER_DATA_API_KEYS_FILE`)
- Conditional `/docs` and `/redoc` endpoints (disabled by default in production)
- Security scanning CI: Bandit SAST, pip-audit, gitleaks, CodeQL

---

### Category 9: Missing Sections

The following sections are entirely absent from AGENTS.md but represent significant application capabilities:

| ID | Missing Section | Importance | Description |
|----|----------------|------------|-------------|
| D-036 | Storage Backends | HIGH | 7 implementations (local_fs, memory, cached, redis, postgres, hf, kaggle) with composable caching |
| D-037 | Observability & Metrics | HIGH | Prometheus metrics, structured JSON logging, Sentry integration, request ID propagation |
| D-038 | Docker & Containerization | MEDIUM | Multi-stage Dockerfile, .dockerignore, health checks, non-root user, environment configuration |
| D-039 | CI/CD Pipeline | HIGH | 5 GitHub Actions workflows, pre-commit hooks, coverage gates, security scanning, doc validation |
| D-040 | Configuration Reference | MEDIUM | Full environment variable catalog (JUNIPER_DATA_ prefix), Pydantic Settings, defaults |
| D-041 | Documentation Structure | LOW | 20+ documentation files across docs/, notes/, with testing/CI/CD/API subdirectories |
| D-042 | Performance Testing | MEDIUM | pytest-benchmark infrastructure, 41 benchmarks, generator/storage throughput tracking |
| D-043 | Batch Operations | MEDIUM | 4 batch API endpoints for create, delete, export, and tag management |
| D-044 | Dataset Versioning | MEDIUM | Named datasets, version tracking, latest-version queries |
| D-045 | Dataset Lifecycle | MEDIUM | TTL, expiration, cleanup, access tracking, filtering |

---

### Category 10: Integration Context

| ID | Item | AGENTS.md State | Actual State | Severity |
|----|------|-----------------|--------------|----------|
| D-046 | Key Documentation table | References `INTEGRATION_DEVELOPMENT_PLAN.md` | File does not exist; replaced by `JUNIPER-DATA_POST-RELEASE_DEVELOPMENT-ROADMAP.md` | HIGH |
| D-047 | Environment variables | Only `JUNIPER_DATA_URL` mentioned | 10+ env vars: HOST, PORT, STORAGE_PATH, LOG_LEVEL, CORS_ORIGINS, API_KEYS, RATE_LIMIT_*, LOG_FORMAT, SENTRY_DSN, METRICS_ENABLED | HIGH |
| D-048 | Consumer list | JuniperCascor (SpiralDataProvider), JuniperCanopy (DemoMode, CascorIntegration) | Consumers now use `juniper-data-client` PyPI package; consumer class names may have changed | MEDIUM |
| D-049 | Ecosystem context | Pre-polyrepo | Post-polyrepo: standalone repo, PyPI client package, independent CI | MEDIUM |

---

### Category 11: Production Entry Point

| ID | Item | AGENTS.md State | Actual State | Severity |
|----|------|-----------------|--------------|----------|
| D-050 | Production startup command | `uvicorn juniper_data.api.app:app` | Should use `create_app()` factory: `uvicorn juniper_data.api.app:create_app --factory` or `python -m juniper_data` | MEDIUM |

---

## Impact Assessment

### Risk Matrix

| Severity | Count | Impact |
|----------|-------|--------|
| CRITICAL | 5 | Developers will receive incorrect guidance on version, security posture, line length, directory layout, and middleware |
| HIGH | 9 | Missing documentation of major features (storage, CI/CD, observability, testing infrastructure) |
| MEDIUM | 14 | Incomplete details that may cause confusion but won't break workflows |
| LOW | 2 | Minor inaccuracies with minimal practical impact |
| **Total** | **30** | |

### Sections Requiring Complete Rewrite

1. **Directory Structure** -- Must be expanded to show actual layout
2. **Component Overview** -- Must include all major components
3. **Security Notes** -- Must document actual security implementation
4. **Dependencies** -- Must include all dependency groups

### Sections Requiring New Content

1. Storage Backends
2. Observability & Metrics
3. Docker & Containerization
4. CI/CD Pipeline
5. Configuration Reference
6. Performance Testing

### Sections Requiring Updates

1. Version and metadata (header)
2. Code Style (line length)
3. Testing (directories, markers, infrastructure)
4. API Design (endpoints, middleware, batch, versioning, lifecycle)
5. Integration Context (references, env vars, consumers)
6. Development Workflow (security scanning, pre-commit)

---

## Recommendations

1. **Immediate**: Update version to 0.5.0, fix line length to 320, correct directory structure
2. **High Priority**: Add missing sections (storage, security, observability, CI/CD)
3. **Medium Priority**: Expand testing, API, and integration documentation
4. **Low Priority**: Add Docker, configuration reference, documentation structure sections

---

## Cross-References

| Document | Location | Relevance |
|----------|----------|-----------|
| Development Roadmap | `notes/JUNIPER-DATA_POST-RELEASE_DEVELOPMENT-ROADMAP.md` | Contains audit notes on line length and coverage changes |
| Security PR | `notes/pull_requests/PR_SECURITY_HARDENING_2026-03-03.md` | Documents v0.5.0 security hardening scope |
| CHANGELOG | `CHANGELOG.md` | Version history from 0.1.0 to 0.5.0 |
| pyproject.toml | Root | Authoritative source for version, dependencies, tool config |
| Parent CLAUDE.md | `/home/pcalnon/Development/python/Juniper/CLAUDE.md` | Ecosystem-level conventions and integration points |

---

## Document History

| Date | Author | Changes |
|------|--------|---------|
| 2026-04-02 | Claude Code (Opus 4.6) | Initial drift analysis -- 30 items across 11 categories |
