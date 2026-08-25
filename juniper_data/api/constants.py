"""Constants for the JuniperData API layer.

Centralizes hardcoded literals used by ``middleware.py``, ``security.py``,
``observability.py``, and ``app.py``. Includes security header values,
request limits, observability defaults, and the exempt-paths set.

HTTP status code constants are intentionally NOT defined here — Wave 2
will adopt ``starlette.status`` (e.g., ``status.HTTP_413_REQUEST_ENTITY_TOO_LARGE``)
rather than custom integers, per the data plan doc Phase 3 decision.

Project: Juniper
Sub-Project: juniper-data
Application: JuniperData API
Author: Paul Calnon
Version: 0.6.0
License: MIT License
"""


# ─── API Versioning ──────────────────────────────────────────────────────────

# The public wire prefix every router is mounted under. It used to be an independent
# ``/v1`` literal in eight places -- the three ``include_router`` calls, the two
# self-referential ``artifact_url`` f-strings and the three health entries of
# ``EXEMPT_PATHS`` below -- with nothing tying them together (APD-DATA-020). Spell it
# once here; every site derives from it. Changing the value is a BREAKING change for
# every client (``juniper-data-client`` hard-codes ``/v1/...``), so
# ``tests/unit/test_api_prefix.py`` pins the published value and fails on any
# ``/v1`` literal that creeps back into the package.
API_VERSION: str = "v1"
API_PREFIX: str = f"/{API_VERSION}"

# ─── Security: Exempt Paths ──────────────────────────────────────────────────

# Paths exempt from API key auth and rate limiting (health checks + docs +
# the Prometheus /metrics scrape endpoint). The /metrics surface remains
# gated by SEC-16's MetricsAuthMiddleware (IP allowlist) — see
# juniper_data.api.observability.MetricsAuthMiddleware. The two layers
# compose: SecurityMiddleware skips /metrics, MetricsAuthMiddleware then
# enforces the trusted-IP gate at the ASGI mount.
#
# Both `/metrics` and `/metrics/` are listed because FastAPI's mount of the
# Prometheus ASGI sub-app redirects 307 from the bare path to the trailing-
# slash form; without the trailing-slash entry the redirect target trips
# SecurityMiddleware again and returns 401.
EXEMPT_PATHS: frozenset[str] = frozenset(
    {
        f"{API_PREFIX}/health",
        f"{API_PREFIX}/health/live",
        f"{API_PREFIX}/health/ready",
        "/docs",
        "/openapi.json",
        "/redoc",
        "/metrics",
        "/metrics/",
    }
)

# ─── Security: Header Names ──────────────────────────────────────────────────

HEADER_X_CONTENT_TYPE_OPTIONS: str = "X-Content-Type-Options"
HEADER_X_FRAME_OPTIONS: str = "X-Frame-Options"
HEADER_REFERRER_POLICY: str = "Referrer-Policy"
HEADER_PERMISSIONS_POLICY: str = "Permissions-Policy"
HEADER_CONTENT_SECURITY_POLICY: str = "Content-Security-Policy"
HEADER_STRICT_TRANSPORT_SECURITY: str = "Strict-Transport-Security"
HEADER_X_FORWARDED_PROTO: str = "X-Forwarded-Proto"
HEADER_X_REQUEST_ID: str = "X-Request-ID"
HEADER_X_API_KEY: str = "X-API-Key"
HEADER_X_RATELIMIT_LIMIT: str = "X-RateLimit-Limit"
HEADER_X_RATELIMIT_REMAINING: str = "X-RateLimit-Remaining"
HEADER_X_RATELIMIT_RESET: str = "X-RateLimit-Reset"
HEADER_RETRY_AFTER: str = "Retry-After"

# ─── Security: Header Values ─────────────────────────────────────────────────

DEFAULT_CONTENT_SECURITY_POLICY: str = "default-src 'none'; frame-ancestors 'none'"
X_CONTENT_TYPE_OPTIONS_NOSNIFF: str = "nosniff"
X_FRAME_OPTIONS_DENY: str = "DENY"
REFERRER_POLICY_STRICT_ORIGIN: str = "strict-origin-when-cross-origin"
PERMISSIONS_POLICY_RESTRICTED: str = "camera=(), microphone=(), geolocation=()"
HSTS_MAX_AGE_VALUE: str = "max-age=31536000; includeSubDomains"
PROXY_PROTOCOL_HTTPS: str = "https"

# ─── Request Body Limits ─────────────────────────────────────────────────────

MAX_REQUEST_BODY_BYTES: int = 10 * 1024 * 1024  # 10 MB

# ─── Rate Limiting Defaults ──────────────────────────────────────────────────

DEFAULT_RATE_LIMIT_REQUESTS_PER_MINUTE: int = 60
DEFAULT_RATE_LIMIT_WINDOW_SECONDS: int = 60

# ─── Failed-Authentication Throttle Defaults ─────────────────────────────────
#
# Budget for the *pre-authentication* throttle (APD-DATA-001). Distinct from the
# rate-limit defaults above: those are an identity-keyed fairness quota applied
# after auth succeeds, these bound how many *failed* credential attempts a single
# source IP may make. Only a failed attempt consumes budget, so a caller with a
# valid key is never counted.

DEFAULT_FAILED_AUTH_MAX_FAILURES: int = 10
DEFAULT_FAILED_AUTH_WINDOW_SECONDS: int = 60

# ─── Observability ───────────────────────────────────────────────────────────

DEFAULT_SERVICE_NAME: str = "juniper-data"
DEFAULT_NAMESPACE: str = "juniper_data"

# Plain-text log format (used when log_format != "json").
DEFAULT_LOG_FORMAT_PLAIN: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FORMAT_JSON: str = "json"

# Sentry tracing.
DEFAULT_SENTRY_TRACES_SAMPLE_RATE: float = 0.1

# Prometheus histogram buckets for dataset generation duration (seconds).
DATASET_GENERATION_DURATION_BUCKETS: tuple[float, ...] = (
    0.01,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
    float("inf"),
)

# Status outcome strings recorded by ``record_dataset_generation``.
GENERATION_STATUS_SUCCESS: str = "success"
GENERATION_STATUS_ERROR: str = "error"

# METRICS-MON R4.5 / R3.1 follow-up: ``cache`` label values for
# ``juniper_data_dataset_post_total``. Keep this a closed set (two values
# only) so cardinality stays bounded — a typo like ``"miss "`` would
# create a spurious bucket and undermine the R1.1 cardinality discipline.
POST_CACHE_HIT: str = "hit"
POST_CACHE_MISS: str = "miss"

# APD-DATA-010: name of the accounting member added to a batch-export archive when at
# least one requested dataset could not be included. Absent from a complete export, so
# the archive a caller already knows how to read is unchanged; its presence is the
# signal that something is missing. Cannot collide with a dataset member, which is
# always ``{dataset_id}.npz``.
BATCH_EXPORT_MANIFEST_NAME: str = "manifest.json"
