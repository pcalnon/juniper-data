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


# ─── Security: Exempt Paths ──────────────────────────────────────────────────

# Paths exempt from API key auth and rate limiting (health checks + docs).
EXEMPT_PATHS: frozenset[str] = frozenset(
    {
        "/v1/health",
        "/v1/health/live",
        "/v1/health/ready",
        "/docs",
        "/openapi.json",
        "/redoc",
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
