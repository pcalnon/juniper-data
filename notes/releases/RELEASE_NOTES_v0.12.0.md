# Juniper Data v0.12.0 Release Notes

**Release Date:** 2026-08-30
**Version:** 0.12.0
**Codename:** Contract Honesty
**Release Type:** MINOR

---

## Overview

The first juniper-data release since 2026-07-29. It closes a cluster of API-contract defects found by the APD register — routes that answered with the wrong status, a list endpoint with no total order, a documented field that did not exist, and an OpenAPI document that a secured deployment could not fetch at all. The unifying theme is that the service's *declared* contract and its *actual* behaviour had drifted apart in a dozen small places.

> **Status:** STABLE – no known regressions; 25 merged PRs since v0.11.0.

---

## Release Summary

- **Release type:** MINOR
- **Primary focus:** API contract correctness (APD register) + one security fix
- **Breaking changes:** NO for HTTP callers; **see Upgrade Notes** if you generate an SDK from the OpenAPI document
- **Priority summary:** unblocks juniper-ml's experiment driver (plan W-4) and the juniper-data-client generator-parity CI lane

---

## Features Summary

| Area | Change | Reference |
|---|---|---|
| Generators | `install_hint` is returned by `GET /v1/generators` | #277 (plan W-4) |
| Rate limiting | window is operator-configurable | APD-DATA-033 / #297 |
| Filter | total ordering + keyset pagination | APD-DATA-011/012 / #283 |
| Security | OpenAPI document served behind the API key, scheme declared | APD-DATA-005/024 / #295 |

---

## What's New

**`GET /v1/generators` finally carries `install_hint`.** juniper-ml's experiment driver refuses an unavailable generator with *"see `GET /v1/generators` for the install hint"* — and that endpoint carried no hint. The message pointed at a field that did not exist. It does now.

This is the change that motivates cutting the release: the field has been on `main` since 2026-08-21 but absent from every published artifact, so the named downstream consumer — juniper-data-client's generator-parity CI lane, which installs `juniper-data[api]` from PyPI — could not see it.

**The rate-limit window is settable.** `Settings.rate_limit_window_seconds` / `JUNIPER_DATA_RATE_LIMIT_WINDOW_SECONDS`. Worth restating what the defect was *not*: `DEFAULT_RATE_LIMIT_WINDOW_SECONDS` was never an unwired constant — it is load-bearing at all three sites that read the window. The gap was only the absent operator-facing setting.

**`/v1/datasets/filter` has a total order.** Sorting on `created_at` alone plus a stable sort meant ties resolved to whatever `Path.glob` happened to enumerate. Measured: the same six datasets inserted in two orders produced exactly reversed pages. Now `created_at` descending, `dataset_id` ascending as tie-break — reproducible across calls, processes, and both store implementations, whose enumeration orders disagree by construction.

---

## Bug Fixes

- A malformed `Content-Length` is a **400**, not a 500 (`APD-DATA-036`).
- Serialisation faults report **500**, not 400 (`APD-DATA-034`).
- A `GET` can no longer silently undo a tag edit — `record_access` made the metadata read-modify-write lossy (`APD-DATA-006`).
- Batch-create no longer answers **201** when it created nothing (`APD-DATA-009`).
- Batch-export accounts for datasets it could not include (`APD-DATA-010`).
- Metadata read-modify-write is atomic across processes (`APD-DATA-007`).
- The 501 detail no longer echoes an undeclared `ImportError` (#275).

---

## Improvements

- Every route declares an explicit `operation_id`, pinned by a contract test (`APD-DATA-023`).
- Both binary routes declare `application/zip`, spelled once as `BINARY_MEDIA_TYPE` (`APD-DATA-025`).
- The `/v1` prefix is spelled once as `API_PREFIX` and pinned (`APD-DATA-020`).
- Route declaration order is pinned against the `/{dataset_id}` catch-all (`APD-DATA-015`).
- The never-wired `DatasetListFilter` model is gone; the `/filter` contract is pinned (`APD-DATA-021`).
- `juniper-service-core` ceiling raised to `<0.7.0` and 0.6.0 adopted in the lockfile (#299, #300).

---

## API Changes

**Security — the OpenAPI document (`APD-DATA-005` + `APD-DATA-024`).** One defect wearing two faces: `openapi_url` was `None` whenever any key was configured, so a **secured deployment served no schema at all**; and the declared `api_key_header` was referenced nowhere, so what it did serve declared no `securitySchemes` — an SDK generated from it never sent `X-API-Key`.

The trap worth carrying forward: `EXEMPT_PATHS` listed `/docs`, `/openapi.json` and `/redoc`, and `_is_exempt()` is a bare membership test evaluated **regardless of whether a key was supplied**. So simply re-enabling `openapi_url` would not have put the document behind the key — it would have served it to *everyone*, while looking exactly like the intended fix. All three paths were removed from the exempt set instead.

The interactive explorers stay unmounted when keys are configured: they fetch `/openapi.json` by XHR with no `X-API-Key`, so serving them behind the key could only 401. Local development, with no keys set, is unchanged.

---

## Test Results

Full suite green on `main` at the release commit across the supported Python matrix. Each APD entry above ships with the contract test that pins it — the register's working rule is that a fix without a pinning test is not closed.

---

## Upgrade Notes

**If you generate a client SDK from the OpenAPI document, regenerate it.** Two changes affect generated output:

1. Every route now declares an explicit `operation_id` (`APD-DATA-023`). FastAPI previously derived them from function names, so **generated method names will change**. This is the intended fix — derived ids were unstable under refactoring — but it is the one change in this release that can break a downstream build.
2. The document now declares `securitySchemes` per protected router, so a regenerated SDK will send `X-API-Key` where before it sent nothing.

**If you consume `/v1/datasets/filter` pages**, the order is now specified and may differ from what you observed before. The previous order was not stable, so no correct client can have depended on it — but a snapshot test might have.

**No action needed** for local/unsecured deployments.

---

## Known Issues

- `juniper-recurrence` pins `juniper-data>=0.9.0,<0.12.0`, so its bench harness will **not** pick this release up until that cap is raised. Nothing in the bench needs 0.12.0; recorded so the omission is deliberate rather than surprising.

---

## What's Next

- Raise the `<0.12.0` cap in juniper-recurrence's `[bench]` extra.
- The remaining APD-DATA register entries.

---

## Version History

| Version | Date | Focus |
|---|---|---|
| 0.12.0 | 2026-08-30 | API contract correctness + OpenAPI security |
| 0.11.0 | 2026-07-28 | SEC-F01 auth-posture self-check + fail-closed flag |
| 0.10.0 | 2026-07-17 | D1 generator availability + D2 MNIST extra |

---

## Links

- Changelog: [`CHANGELOG.md`](../../CHANGELOG.md)
- Requirements register: `JR-DATA-*` in the juniper-ml requirements corpus
