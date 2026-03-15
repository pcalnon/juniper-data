# JuniperData Cross-References from JuniperCascor Audit

**Source**: JuniperCascor exhaustive notes audit (2026-02-18)
**Created**: 2026-02-18
**Status**: Active

---

## Items Identified in JuniperCascor Audit That Affect JuniperData

### INT-P1-001: Duplicated JuniperDataClient

**Priority**: HIGH
**Source**: INTEGRATION_ROADMAP-01.md (JuniperCascor)

**Description**: `JuniperDataClient` is duplicated in both JuniperCascor (`src/juniper_data_client/client.py`) and JuniperCanopy. Changes to the client API must be synchronized manually.

**Action for JuniperData**: Consider publishing a shared client package (e.g., `juniper-common`) that both consuming applications can depend on.

---

### INT-P1-002: `requests` as Undeclared Dependency

**Priority**: HIGH
**Source**: INTEGRATION_ROADMAP-01.md (JuniperCascor)

**Description**: The `requests` library is used by `JuniperDataClient` in JuniperCascor but is not declared in the project's dependency files.

**Action for JuniperData**: If a shared client package is created, ensure `requests` is declared as its dependency.

---

### INT-P1-003: No Shared Protocol Package

**Priority**: HIGH
**Source**: INTEGRATION_ROADMAP-01.md (JuniperCascor)

**Description**: Three Juniper applications share API contracts, data formats, and client code but have no shared protocol/interface package.

**Action for JuniperData**: Define and publish API schemas/contracts that consuming applications can validate against.

---

### INT-P3-002: E2E Live Service Integration Tests

**Priority**: MEDIUM
**Source**: INTEGRATION_ROADMAP-01.md (JuniperCascor)

**Description**: No automated JuniperCascor tests currently spin up a live JuniperData service. All E2E tests use in-process `TestClient`.

**Action for JuniperData**: Consider providing a test fixture or Docker image that CasCor CI can use for live service E2E testing.

---

### INT-P3-003: Docker Compose Validation

**Priority**: MEDIUM
**Source**: INTEGRATION_ROADMAP-01.md (JuniperCascor)

**Description**: The docker-compose configuration shows a 3-service deployment (JuniperData, JuniperCascor, JuniperCanopy) but has not been tested end-to-end.

**Action for JuniperData**: Validate JuniperData's Docker container works correctly in the multi-service compose configuration.

---

### Phase 5: Extended Data Sources

**Priority**: LOW (DEFERRED)
**Source**: JUNIPER_CASCOR_SPIRAL_DATA_GEN_REFACTOR_PLAN.md (JuniperCascor)

**Description**: Extended data source support (S3, database, HuggingFace). Deferred until JuniperData core is stable.

**Action for JuniperData**: This is a JuniperData feature request — track in JuniperData's own roadmap.

---

### CAS-REF-004: Legacy Spiral Code Removal

**Priority**: MEDIUM
**Source**: PRE-DEPLOYMENT_ROADMAP-2.md (JuniperCascor)

**Description**: JuniperCascor plans to remove 16 deprecated local spiral generation methods once JuniperData deployment and integration testing is confirmed stable.

**Action for JuniperData**: Ensure stability and SLA guarantees for the spiral generation API before CasCor removes its fallback code.

---

## Summary

| Item | Priority | CasCor ID | JuniperData Action |
| --- | --- | --- | --- |
| Shared client package | HIGH | INT-P1-001 | Publish or coordinate shared package |
| Declare `requests` dependency | HIGH | INT-P1-002 | Include in shared package deps |
| Shared protocol/contracts | HIGH | INT-P1-003 | Define API schemas |
| Live service test fixtures | MEDIUM | INT-P3-002 | Provide Docker test image |
| Docker Compose validation | MEDIUM | INT-P3-003 | Validate container in compose |
| Extended data sources | LOW | Phase 5 | Track in JuniperData roadmap |
| Stability for legacy removal | MEDIUM | CAS-REF-004 | Ensure API stability/SLA |
