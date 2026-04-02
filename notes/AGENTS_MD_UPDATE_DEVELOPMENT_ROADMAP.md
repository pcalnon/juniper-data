# AGENTS.md Update Development Roadmap

**Project**: Juniper Data - Dataset Generation Service
**Application**: juniper-data
**Roadmap Date**: 2026-04-02
**Author**: Claude Code (Opus 4.6)
**Source Documents**:
- `notes/AGENTS_MD_DRIFT_ANALYSIS.md` (30 drift items)
- `notes/AGENTS_MD_UPDATE_PLAN.md` (11 phases)

---

## Overview

This roadmap details all required work to resolve the AGENTS.md drift identified in the analysis document. Work is organized into prioritized phases with specific, actionable tasks. Each task includes acceptance criteria for validation.

---

## Phase 1: Critical Fixes (Priority: IMMEDIATE)

**Rationale**: These items cause incorrect guidance if left unresolved. Developers and AI agents following the current AGENTS.md will receive wrong information about version, security, formatting, and project structure.

### Task 1.1: Update Header Metadata

**Drift Items**: D-001, D-002
**Effort**: Trivial

| Action | Detail |
|--------|--------|
| Update `Version` | 0.4.2 -> 0.5.0 |
| Update `Last Updated` | 2026-02-06 -> 2026-04-02 |

**Acceptance**: Header fields match pyproject.toml version and current date.

---

### Task 1.2: Fix Line Length Documentation

**Drift Items**: D-021
**Effort**: Trivial

| Action | Detail |
|--------|--------|
| Update line length in Code Formatting section | 120 -> 320 |

**Acceptance**: Documented line length matches `[tool.ruff] line-length` in pyproject.toml.

---

### Task 1.3: Rewrite Directory Structure

**Drift Items**: D-004 through D-018
**Effort**: Medium

**Actions**:
1. Expand top-level tree to include all significant directories and files
2. Expand `api/` to show `middleware.py`, `security.py`, `observability.py`, `models/`
3. Expand `storage/` to show all 7 backend files plus `base.py`
4. Expand `core/` to show all 5 module files
5. Expand `tests/` to show `performance/`, `api/`, `fixtures/` directories
6. Add `docs/`, `scripts/`, `notes/`, `conf/`, `util/`, `.github/`
7. Add `Dockerfile`, `.pre-commit-config.yaml`, `.env.example`, `requirements.lock`, `CHANGELOG.md`

**Acceptance**: Every directory and significant file visible in `ls -R` has a corresponding entry in the tree.

---

### Task 1.4: Rewrite Security Section

**Drift Items**: D-035
**Effort**: Medium

**Actions**:
1. Replace 3 bullet points with structured security documentation
2. Document API key authentication mechanism
3. Document rate limiting (per-client, configurable)
4. Document security headers middleware (full header list)
5. Document request body size limits
6. Document CORS configuration
7. Document Docker secrets support
8. Document conditional API docs
9. Document security scanning tools and CI integration

**Acceptance**: Every security feature in `api/security.py`, `api/middleware.py`, and CI workflows is documented.

---

### Task 1.5: Expand API Middleware Documentation

**Drift Items**: D-031
**Effort**: Small

**Actions**:
1. Add middleware stack section showing all 6 layers
2. Document execution order (LIFO)

**Acceptance**: Middleware stack matches `create_app()` in `api/app.py`.

---

## Phase 2: High Priority Updates (Priority: HIGH)

**Rationale**: Major application capabilities are undocumented. Developers will not discover these features through AGENTS.md.

### Task 2.1: Expand Component Overview Table

**Drift Items**: D-019, D-020
**Effort**: Small

**Actions**:
1. Fix spiral description ("Multi-spiral classification")
2. Add 15+ missing component entries for api, core, and storage modules

**Acceptance**: Every Python module under `juniper_data/` has a component overview entry.

---

### Task 2.2: Add Storage Backends Section

**Drift Items**: D-036
**Effort**: Medium

**Actions**:
1. Create new "Storage Backends" section
2. Document the abstract base interface (`DatasetStore`)
3. Document each backend: LocalFS, InMemory, Cached, Redis, PostgreSQL, HuggingFace, Kaggle
4. Document the composable caching pattern
5. Document storage-related environment variables

**Acceptance**: All storage classes in `storage/__init__.py` exports are documented.

---

### Task 2.3: Add Observability Section

**Drift Items**: D-037
**Effort**: Small

**Actions**:
1. Create new "Observability" section
2. Document Prometheus metrics (counters, histograms, info)
3. Document structured JSON logging
4. Document Sentry integration
5. Document request ID propagation

**Acceptance**: All metrics in `api/observability.py` are documented.

---

### Task 2.4: Add CI/CD Pipeline Section

**Drift Items**: D-039
**Effort**: Medium

**Actions**:
1. Create new "CI/CD Pipeline" section
2. Document all 5 GitHub Actions workflows
3. Document pre-commit hook configuration
4. Document coverage gates (aggregate and per-module)
5. Document security scanning integration
6. Document documentation validation CI

**Acceptance**: All workflows in `.github/workflows/` are referenced.

---

### Task 2.5: Update Dependencies Section

**Drift Items**: D-022, D-023, D-024
**Effort**: Small

**Actions**:
1. Add `python-dotenv` to core dependencies
2. Add `pydantic-settings` to API dependencies
3. Add `[arc-agi]`, `[test]`, `[observability]` dependency groups
4. Update `[dev]` dependencies (add bandit, pip-audit, pre-commit)

**Acceptance**: Every dependency group in pyproject.toml `[project.optional-dependencies]` is documented.

---

### Task 2.6: Expand API Endpoint Catalog

**Drift Items**: D-030, D-032, D-033, D-034
**Effort**: Medium

**Actions**:
1. Add complete endpoint catalog organized by resource
2. Document health endpoints (health, live, ready)
3. Document generator endpoints (list, schema)
4. Document dataset CRUD endpoints
5. Document batch operation endpoints (create, delete, export, tags)
6. Document versioning endpoints (versions, latest)
7. Document lifecycle endpoints (cleanup-expired, tags)
8. Document filtering and stats endpoints

**Acceptance**: Every route in `api/routes/*.py` is documented.

---

### Task 2.7: Update Testing Section

**Drift Items**: D-025, D-026, D-027, D-028
**Effort**: Small

**Actions**:
1. Add `performance/`, `api/`, `fixtures/` to test organization
2. Add `performance` and `slow` markers
3. Document test count and coverage threshold
4. Add performance testing subsection with benchmark commands

**Acceptance**: Test organization matches actual `tests/` directory structure.

---

### Task 2.8: Fix Integration Context References

**Drift Items**: D-046, D-047, D-048, D-049
**Effort**: Small

**Actions**:
1. Replace `INTEGRATION_DEVELOPMENT_PLAN.md` reference with `JUNIPER-DATA_POST-RELEASE_DEVELOPMENT-ROADMAP.md`
2. Expand environment variables catalog
3. Update consumer references for polyrepo architecture
4. Note `juniper-data-client` PyPI package

**Acceptance**: All file references point to existing files. All environment variables in `api/settings.py` are documented.

---

## Phase 3: Medium Priority Updates (Priority: MEDIUM)

**Rationale**: Useful context that improves developer experience but does not cause incorrect behavior if missing.

### Task 3.1: Add Configuration Reference

**Drift Items**: D-040, D-047
**Effort**: Medium

**Actions**:
1. Create new "Configuration" section
2. Document all `JUNIPER_DATA_*` environment variables
3. Document defaults, types, and descriptions
4. Document Docker secrets support
5. Document `.env.example` as reference

**Acceptance**: Every field in `api/settings.py` Settings class is documented.

---

### Task 3.2: Add Docker Section

**Drift Items**: D-038
**Effort**: Small

**Actions**:
1. Create new "Docker" section
2. Document multi-stage build
3. Document environment variables
4. Document health check configuration
5. Document port exposure and non-root user

**Acceptance**: Key Dockerfile decisions are documented.

---

### Task 3.3: Update API Response Format

**Drift Items**: D-029
**Effort**: Small

**Actions**:
1. Replace generic response format with actual Pydantic model references
2. Document key response types (CreateDatasetResponse, DatasetListResponse, etc.)

**Acceptance**: Response format section references actual model classes.

---

### Task 3.4: Update Development Workflow

**Drift Items**: Related to D-035, D-039
**Effort**: Small

**Actions**:
1. Add security scanning step to feature workflow
2. Add pre-commit hook step
3. Update generator development workflow with registration and storage considerations

**Acceptance**: Workflow steps include all quality gates that CI enforces.

---

### Task 3.5: Fix Production Startup Command

**Drift Items**: D-050
**Effort**: Trivial

**Actions**:
1. Update production command to use factory pattern or module entry point

**Acceptance**: Command works when executed.

---

## Phase 4: Low Priority Updates (Priority: LOW)

### Task 4.1: Add Documentation Structure Reference

**Drift Items**: D-041
**Effort**: Trivial

**Actions**:
1. Add brief reference to documentation locations (docs/, notes/)
2. Reference `docs/DOCUMENTATION_OVERVIEW.md` for navigation

**Acceptance**: Developers know where to find documentation.

---

## Implementation Summary

| Phase | Tasks | Priority | Total Effort |
|-------|-------|----------|--------------|
| Phase 1: Critical Fixes | 5 tasks | IMMEDIATE | ~2 hours |
| Phase 2: High Priority | 8 tasks | HIGH | ~3 hours |
| Phase 3: Medium Priority | 5 tasks | MEDIUM | ~1.5 hours |
| Phase 4: Low Priority | 1 task | LOW | ~15 min |
| **Total** | **19 tasks** | | **~7 hours** |

---

## Drift Item Cross-Reference

| Drift Item | Task | Phase |
|------------|------|-------|
| D-001 | 1.1 | Phase 1 |
| D-002 | 1.1 | Phase 1 |
| D-003 | Out of scope (code change) | N/A |
| D-004 to D-014 | 1.3 | Phase 1 |
| D-015 | 1.3 | Phase 1 |
| D-016 | 1.3 | Phase 1 |
| D-017 | 1.3 | Phase 1 |
| D-018 | 1.3 | Phase 1 |
| D-019 | 2.1 | Phase 2 |
| D-020 | 2.1 | Phase 2 |
| D-021 | 1.2 | Phase 1 |
| D-022 | 2.5 | Phase 2 |
| D-023 | 2.5 | Phase 2 |
| D-024 | 2.5 | Phase 2 |
| D-025 | 2.7 | Phase 2 |
| D-026 | 2.7 | Phase 2 |
| D-027 | 2.7 | Phase 2 |
| D-028 | 2.7 | Phase 2 |
| D-029 | 3.3 | Phase 3 |
| D-030 | 2.6 | Phase 2 |
| D-031 | 1.5 | Phase 1 |
| D-032 | 2.6 | Phase 2 |
| D-033 | 2.6 | Phase 2 |
| D-034 | 2.6 | Phase 2 |
| D-035 | 1.4 | Phase 1 |
| D-036 | 2.2 | Phase 2 |
| D-037 | 2.3 | Phase 2 |
| D-038 | 3.2 | Phase 3 |
| D-039 | 2.4 | Phase 2 |
| D-040 | 3.1 | Phase 3 |
| D-041 | 4.1 | Phase 4 |
| D-042 | 2.7 | Phase 2 |
| D-043 | 2.6 | Phase 2 |
| D-044 | 2.6 | Phase 2 |
| D-045 | 2.6 | Phase 2 |
| D-046 | 2.8 | Phase 2 |
| D-047 | 3.1 | Phase 3 |
| D-048 | 2.8 | Phase 2 |
| D-049 | 2.8 | Phase 2 |
| D-050 | 3.5 | Phase 3 |

---

## Document History

| Date | Author | Changes |
|------|--------|---------|
| 2026-04-02 | Claude Code (Opus 4.6) | Initial roadmap -- 4 phases, 19 tasks, 30+ drift items mapped |
