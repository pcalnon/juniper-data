# Pull Request: Security Hardening — Middleware, Defaults, and Scanning

**Date:** 2026-03-03
**Version(s):** 0.4.2 → 0.5.0
**Author:** Paul Calnon
**Status:** READY_FOR_MERGE

---

## Summary

Comprehensive security hardening for juniper-data as part of the cross-ecosystem security audit. Adds security headers middleware, request body size limits, error response sanitization, restrictive CORS and rate limiting defaults, /metrics authentication, conditional API docs, and scheduled security scanning.

---

## Context / Motivation

A full security audit of the Juniper ecosystem identified 24 findings across 7 repositories. This PR addresses the juniper-data portion:

- No security response headers (X-Content-Type-Options, X-Frame-Options, etc.)
- Error handlers exposing internal details via `str(exc)` to clients
- CORS wildcard default (`["*"]`) allowing any origin
- Rate limiting disabled by default
- `/metrics` endpoint accessible without authentication
- API documentation exposed in production
- No scheduled dependency vulnerability scanning

---

## Changes

### Security

- Added `SecurityHeadersMiddleware` with X-Content-Type-Options, X-Frame-Options, Referrer-Policy, Permissions-Policy, and conditional HSTS
- Added `RequestBodyLimitMiddleware` with configurable max body size (default 10MB)
- Sanitized `ValueError` handler — generic error messages returned to clients; internal details logged at DEBUG
- Changed CORS origins default from `["*"]` to `[]` (restrictive by default)
- Changed rate limiting default from disabled to enabled
- Removed `/metrics` from authentication-exempt paths
- Added conditional API docs — disabled when API keys are configured

### Added

- `.github/workflows/security-scan.yml` — Weekly Bandit SAST and pip-audit dependency scanning

### Changed

- Updated test fixtures to accommodate new security defaults

---

## Impact & SemVer

- **SemVer impact:** MINOR (0.4.2 → 0.5.0)
- **User-visible behavior change:** YES — CORS now blocks all origins by default; rate limiting enabled; API docs hidden when keys configured
- **Breaking changes:** NO — All changes configurable via environment variables; set `CORS_ORIGINS=*` and `RATE_LIMIT_ENABLED=false` to restore previous behavior
- **Performance impact:** NONE
- **Security/privacy impact:** HIGH — Addresses 7 of 24 ecosystem-wide security findings
- **Guarded by feature flag:** YES — All security features configurable via environment variables

---

## Testing & Results

### Test Summary

| Test Type   | Passed | Failed | Skipped | Notes              |
| ----------- | ------ | ------ | ------- | ------------------ |
| Unit        | 766    | 0      | 0       | All tests passing  |

### Environments Tested

- JuniperData conda environment: All tests pass
- Python 3.14: Compatible

---

## Verification Checklist

- [x] Security headers present on all responses
- [x] Error responses do not leak internal details
- [x] CORS rejects unknown origins by default
- [x] Rate limiting active by default
- [x] `/metrics` requires authentication
- [x] API docs hidden when API keys configured
- [x] All existing tests pass with new defaults

---

## Files Changed

### New Components

- `.github/workflows/security-scan.yml` — Scheduled security scanning workflow

### Modified Components

**Backend:**

- `juniper_data/api/middleware.py` — Added SecurityHeadersMiddleware, RequestBodyLimitMiddleware
- `juniper_data/api/app.py` — Registered new middleware, sanitized error handlers, conditional docs
- `juniper_data/api/settings.py` — Changed CORS and rate limiting defaults

**Tests:**

- `juniper_data/tests/unit/test_api_app.py` — Updated for new security defaults
- `juniper_data/tests/unit/test_api_settings.py` — Updated for new setting defaults

---

## Risks & Rollback Plan

- **Key risks:** Existing deployments relying on `CORS_ORIGINS=*` default will need to set the env var explicitly
- **Rollback plan:** Set `CORS_ORIGINS=*`, `RATE_LIMIT_ENABLED=false`, `DOCS_ENABLED=true` to restore previous behavior

---

## Related Issues / Tickets

- Related PRs: Security hardening PRs in juniper-cascor, juniper-canopy, juniper-deploy, juniper-cascor-worker, juniper-data-client, juniper-ml
- Phase Documentation: `juniper-ml/notes/SECURITY_AUDIT_PLAN.md`

---

## Notes for Release

**v0.5.0** — Security hardening release. Adds security response headers, request body limits, error sanitization, and restrictive CORS/rate limiting defaults. All security features configurable via environment variables. Part of cross-ecosystem security audit addressing 24 findings across 7 repositories.
