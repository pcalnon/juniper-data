# AGENTS.md Update Plan

**Project**: Juniper Data - Dataset Generation Service
**Application**: juniper-data
**Plan Date**: 2026-04-02
**Author**: Claude Code (Opus 4.6)
**Source**: AGENTS_MD_DRIFT_ANALYSIS.md (30 drift items, 11 categories)

---

## Objective

Bring the juniper-data `AGENTS.md` file into full alignment with the current application state (v0.5.0), ensuring it serves as an accurate, comprehensive, and actionable guide for both human developers and AI coding agents working on the project.

---

## Scope

### In Scope

- All 30 drift items identified in `AGENTS_MD_DRIFT_ANALYSIS.md`
- Structural reorganization of AGENTS.md sections
- Addition of missing sections for undocumented capabilities
- Correction of all factual inaccuracies
- Update of all version references and metadata

### Out of Scope

- Changes to application source code (this is a documentation-only update)
- Updates to other documentation files (docs/, notes/) unless directly referenced by AGENTS.md
- Resolution of the `__init__.py` version mismatch (0.4.2 vs 0.5.0) -- that is a code change requiring a separate task
- Line length decision (320 is the current value; documenting it accurately is in scope, changing it is not)

---

## Phase 1: Metadata and Header Updates

**Effort**: Small (15 min) | **Risk**: None | **Dependencies**: None

### Step 1.1: Update Header Block

| Field | Old Value | New Value |
|-------|-----------|-----------|
| Version | 0.4.2 | 0.5.0 |
| Last Updated | 2026-02-06 | 2026-04-02 |

**Addresses**: D-001, D-002

---

## Phase 2: Directory Structure Overhaul

**Effort**: Medium (1 hour) | **Risk**: Low | **Dependencies**: None

### Step 2.1: Expand Top-Level Directory Tree

Replace the existing 15-entry tree with a comprehensive tree showing all significant top-level entries and key subdirectories.

**Addresses**: D-004, D-005, D-006, D-007, D-008, D-009, D-010, D-011, D-012, D-013, D-014

### Step 2.2: Expand `api/` Subdirectory

Add `middleware.py`, `security.py`, `observability.py`, `models/` to the tree.

**Addresses**: D-015

### Step 2.3: Expand `storage/` Subdirectory

Show all 7 backend implementations plus `base.py` and `__init__.py`.

**Addresses**: D-016

### Step 2.4: Expand `core/` Subdirectory

Show `dataset_id.py`, `split.py`, `artifacts.py`, `secrets.py`, `models.py`.

**Addresses**: D-017

### Step 2.5: Expand `tests/` Subdirectory

Add `performance/`, `api/`, `fixtures/` directories and `conftest.py`.

**Addresses**: D-018

---

## Phase 3: Component Overview Update

**Effort**: Small (30 min) | **Risk**: None | **Dependencies**: Phase 2

### Step 3.1: Expand Component Overview Table

Add all missing component entries (15+ new rows). Fix `generators/spiral/` description from "Two-spiral" to "Multi-spiral".

**Addresses**: D-019, D-020

---

## Phase 4: Code Style Corrections

**Effort**: Small (15 min) | **Risk**: None | **Dependencies**: None

### Step 4.1: Update Line Length

Change documented line length from 120 to 320 (matching `pyproject.toml [tool.ruff] line-length = 320`).

**Addresses**: D-021

### Step 4.2: Update Formatter References

Ensure all references to "Ruff formatter" are accurate and include the ruff version constraint (`>=0.9.0`).

---

## Phase 5: Dependencies Overhaul

**Effort**: Medium (30 min) | **Risk**: None | **Dependencies**: None

### Step 5.1: Update Core Dependencies Table

Add `python-dotenv` to core dependencies.

**Addresses**: D-022

### Step 5.2: Update API Dependencies Table

Add `pydantic-settings` to API dependencies.

**Addresses**: D-023

### Step 5.3: Add Missing Dependency Groups

Add sections for: `[arc-agi]`, `[test]`, `[observability]`. Update `[dev]` with bandit, pip-audit, pre-commit.

**Addresses**: D-024

---

## Phase 6: Testing Section Expansion

**Effort**: Medium (30 min) | **Risk**: None | **Dependencies**: None

### Step 6.1: Update Test Organization

Add `performance/`, `api/`, `fixtures/` directories. Mention `conftest.py` and test count.

**Addresses**: D-025, D-027

### Step 6.2: Update Test Markers

Add `performance` and `slow` markers.

**Addresses**: D-026

### Step 6.3: Add Performance Testing Subsection

Document pytest-benchmark infrastructure, benchmark commands, and `--benchmark-disable/enable` flags.

**Addresses**: D-028, D-042

---

## Phase 7: Security Section Rewrite

**Effort**: Medium (45 min) | **Risk**: None | **Dependencies**: None

### Step 7.1: Rewrite Security Notes

Replace 3 bullet points with comprehensive security documentation covering:
- API key authentication (X-API-Key header)
- Rate limiting (per-client, configurable)
- Security headers middleware (CSP, HSTS, X-Frame-Options, etc.)
- Request body size limits (10 MB)
- CORS configuration
- Docker secrets support
- Conditional API docs (disabled by default)
- Security scanning tools (Bandit, pip-audit, gitleaks, CodeQL)

**Addresses**: D-035

---

## Phase 8: API Design Section Expansion

**Effort**: Medium (45 min) | **Risk**: None | **Dependencies**: None

### Step 8.1: Update Response Format

Replace generic response format with actual Pydantic model references.

**Addresses**: D-029

### Step 8.2: Add Endpoint Catalog

Document all REST endpoints organized by resource (health, generators, datasets, batch).

**Addresses**: D-030

### Step 8.3: Document Middleware Stack

Add middleware stack documentation showing execution order.

**Addresses**: D-031

### Step 8.4: Add Batch Operations

Document batch create, delete, export, and tag endpoints.

**Addresses**: D-032, D-043

### Step 8.5: Add Dataset Versioning

Document named datasets, version tracking, and latest-version queries.

**Addresses**: D-033, D-044

### Step 8.6: Add Dataset Lifecycle

Document TTL, expiration, cleanup, access tracking, and filtering.

**Addresses**: D-034, D-045

---

## Phase 9: New Sections

**Effort**: Large (2 hours) | **Risk**: Low | **Dependencies**: None

### Step 9.1: Add Storage Backends Section

Document the 7 storage implementations, their configuration, and the composable caching pattern.

**Addresses**: D-036

### Step 9.2: Add Observability Section

Document Prometheus metrics, structured logging, Sentry integration, and request ID propagation.

**Addresses**: D-037

### Step 9.3: Add Docker Section

Document Dockerfile, environment variables, health checks, and container configuration.

**Addresses**: D-038

### Step 9.4: Add CI/CD Section

Document GitHub Actions workflows, pre-commit hooks, coverage gates, and security scanning.

**Addresses**: D-039

### Step 9.5: Add Configuration Reference

Document all JUNIPER_DATA_ environment variables with types and defaults.

**Addresses**: D-040, D-047

---

## Phase 10: Integration Context Updates

**Effort**: Small (30 min) | **Risk**: None | **Dependencies**: None

### Step 10.1: Fix Key Documentation Table

Replace reference to non-existent `INTEGRATION_DEVELOPMENT_PLAN.md` with `JUNIPER-DATA_POST-RELEASE_DEVELOPMENT-ROADMAP.md`.

**Addresses**: D-046

### Step 10.2: Update Environment Variables

Expand from 1 variable to the full catalog.

**Addresses**: D-047

### Step 10.3: Update Consumer References

Update to reflect polyrepo architecture and `juniper-data-client` PyPI package.

**Addresses**: D-048, D-049

### Step 10.4: Fix Production Startup Command

Update to use `create_app()` factory or `python -m juniper_data`.

**Addresses**: D-050

---

## Phase 11: Development Workflow Updates

**Effort**: Small (15 min) | **Risk**: None | **Dependencies**: None

### Step 11.1: Update Feature Development Workflow

Add security scanning and pre-commit hook steps.

### Step 11.2: Update Generator Development Workflow

Add Pydantic parameter model requirements, storage backend considerations, and registration in `GENERATOR_REGISTRY`.

---

## Execution Order

| Order | Phase | Effort | Cumulative |
|-------|-------|--------|------------|
| 1 | Phase 1: Metadata | 15 min | 15 min |
| 2 | Phase 4: Code Style | 15 min | 30 min |
| 3 | Phase 2: Directory Structure | 60 min | 1h 30m |
| 4 | Phase 3: Component Overview | 30 min | 2h 00m |
| 5 | Phase 5: Dependencies | 30 min | 2h 30m |
| 6 | Phase 6: Testing | 30 min | 3h 00m |
| 7 | Phase 7: Security | 45 min | 3h 45m |
| 8 | Phase 8: API Design | 45 min | 4h 30m |
| 9 | Phase 9: New Sections | 120 min | 6h 30m |
| 10 | Phase 10: Integration Context | 30 min | 7h 00m |
| 11 | Phase 11: Development Workflow | 15 min | 7h 15m |

**Total Estimated Effort**: ~7 hours (manual) / ~30 minutes (automated via Claude Code)

---

## Validation Criteria

After all phases are complete, the updated AGENTS.md must:

1. **Version accuracy**: Header version matches `pyproject.toml`
2. **Directory completeness**: Every significant file and directory is represented
3. **Component coverage**: All major modules have component overview entries
4. **Dependency accuracy**: All dependency groups from pyproject.toml are documented
5. **Test accuracy**: Test directories, markers, and infrastructure match reality
6. **Security completeness**: All security features are documented
7. **API completeness**: All endpoints are cataloged
8. **Configuration accuracy**: All environment variables are documented with defaults
9. **Reference integrity**: All cross-references point to existing files
10. **Style accuracy**: Line length and formatter configuration match pyproject.toml

---

## Document History

| Date | Author | Changes |
|------|--------|---------|
| 2026-04-02 | Claude Code (Opus 4.6) | Initial plan -- 11 phases, 30 drift items addressed |
