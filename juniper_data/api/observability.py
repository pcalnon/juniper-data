"""Observability surface for juniper-data.

METRICS-MON R2.1.2 / seed-06: the cross-cutting machinery
(:class:`JuniperJsonFormatter`, :class:`RequestIdMiddleware`,
:class:`PrometheusMiddleware`, :func:`configure_logging`,
:func:`configure_sentry`, :func:`get_prometheus_app`,
:func:`set_build_info`, :data:`request_id_var`, and the
``UNMATCHED_ENDPOINT_LABEL`` constant) lives in the shared
:mod:`juniper_observability` package and is re-exported here for
backwards compatibility. Existing call sites in
``juniper_data.api.app``, route handlers, and tests continue to import
from this module unchanged.

What stays in this module:

- :class:`MetricsAuthMiddleware` — juniper-data-specific SEC-16 IP
  allowlist for the ``/metrics`` mount. Promotion to the shared lib
  is a roadmap §R5 gating issue.
- Dataset-generation Prometheus metrics
  (:func:`record_dataset_generation`, :func:`set_datasets_cached`,
  and the lazy-init helper :func:`_ensure_dataset_metrics`).

New code should prefer ``from juniper_observability import …`` for the
re-exported symbols to make the dependency on the shared lib explicit.
"""

# Cross-service primitives — re-exported from juniper-observability.
from juniper_observability import (  # noqa: F401 — re-exported for backwards compat
    DEFAULT_LOG_FORMAT_PLAIN,
    DEFAULT_SENTRY_TRACES_SAMPLE_RATE,
    LOG_FORMAT_JSON,
    UNMATCHED_ENDPOINT_LABEL,
    JuniperJsonFormatter,
    PrometheusMiddleware,
    RequestIdMiddleware,
    configure_logging,
    configure_sentry,
    get_prometheus_app,
    request_id_var,
    set_build_info,
)

# Re-export the private SEC-10 hook so existing tests
# (test_phase1d_security.py) that exercise it via the juniper-data path
# continue to work without per-test refactor. The hook lives in
# juniper_observability.sentry; it's the same object.
from juniper_observability.sentry import _strip_sensitive_headers  # noqa: F401 — re-exported for backwards compat

from juniper_data.api.constants import (
    DATASET_GENERATION_DURATION_BUCKETS,
    GENERATION_STATUS_SUCCESS,
)

# ---------------------------------------------------------------------------
# SEC-16: MetricsAuthMiddleware — juniper-data-specific IP allowlist for the
# ``/metrics`` mount. The Prometheus ASGI app is mounted as a sub-app and
# therefore bypasses ``SecurityMiddleware``; this wrapper inspects the raw
# ASGI scope so it can reject untrusted scrapers without depending on
# FastAPI. An empty allowlist blocks everything — operators who don't want
# ``/metrics`` exposed should flip ``metrics_enabled=False`` instead.
#
# Promotion to juniper-observability is tracked as a roadmap §R5 gating
# issue (see notes/code-review/METRICS_MONITORING_ROADMAP_2026-04-25.md).
# ---------------------------------------------------------------------------

# Default allowlist matches ``Settings.metrics_trusted_ips``. Duplicated
# at module level so ``MetricsAuthMiddleware`` can be constructed without
# passing trusted IPs explicitly (useful in tests).
METRICS_DEFAULT_TRUSTED_IPS = ("127.0.0.1", "::1")


class MetricsAuthMiddleware:
    """ASGI wrapper that restricts ``/metrics`` to a trusted IP allowlist."""

    def __init__(self, app, trusted_ips: list[str] | tuple[str, ...] | None = None) -> None:
        self.app = app
        self.trusted_ips: frozenset[str] = frozenset(trusted_ips if trusted_ips is not None else METRICS_DEFAULT_TRUSTED_IPS)

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            client = scope.get("client")
            client_ip = client[0] if client else None
            if not client_ip or client_ip not in self.trusted_ips:
                await send(
                    {
                        "type": "http.response.start",
                        "status": 403,
                        "headers": [(b"content-type", b"text/plain; charset=utf-8")],
                    }
                )
                await send({"type": "http.response.body", "body": b"Forbidden"})
                return
        await self.app(scope, receive, send)


# ---------------------------------------------------------------------------
# Dataset-generation Prometheus metrics — juniper-data specific. Lazily
# initialized to avoid requiring ``prometheus_client`` at import time (it is
# an optional dependency on the ``[api]`` extra path).
# ---------------------------------------------------------------------------

_dataset_metrics: dict | None = None


def _ensure_dataset_metrics() -> dict:
    """Create dataset-related Prometheus metrics on first access."""
    global _dataset_metrics
    if _dataset_metrics is None:
        from prometheus_client import Counter, Gauge, Histogram

        _dataset_metrics = {
            "generations_total": Counter(
                "juniper_data_dataset_generations_total",
                "Total dataset generation requests",
                ["generator", "status"],
            ),
            "generation_duration_seconds": Histogram(
                "juniper_data_dataset_generation_duration_seconds",
                "Dataset generation duration in seconds",
                ["generator"],
                buckets=DATASET_GENERATION_DURATION_BUCKETS,
            ),
            "datasets_cached": Gauge(
                "juniper_data_datasets_cached",
                "Number of datasets currently cached in storage",
            ),
            # METRICS-MON R4.5 / R3.1 follow-up: per-POST request volume,
            # split by cache outcome. ``generations_total`` only counts
            # actual generation work (cache misses); this counts every
            # incoming POST so capacity-planning queries don't undercount
            # deterministic re-POSTs (see roadmap §7 R4.5).
            "post_total": Counter(
                "juniper_data_dataset_post_total",
                "Total POST /v1/datasets requests, split by cache outcome",
                ["generator", "status", "cache"],
            ),
        }
    return _dataset_metrics


def record_dataset_generation(generator: str, status: str, duration: float) -> None:
    """Record a dataset generation event in Prometheus metrics.

    Args:
        generator: Generator type name (e.g. "spiral").
        status: Outcome — "success" or "error".
        duration: Generation duration in seconds.
    """
    m = _ensure_dataset_metrics()
    m["generations_total"].labels(generator=generator, status=status).inc()
    if status == GENERATION_STATUS_SUCCESS:
        m["generation_duration_seconds"].labels(generator=generator).observe(duration)


def record_dataset_post(generator: str, status: str, cache: str) -> None:
    """Record a POST /v1/datasets request, split by cache outcome.

    Bumped on every POST regardless of whether the route short-circuited
    on a cached ``dataset_id`` or executed the generator. ``cache`` must
    be one of the closed-set values ``POST_CACHE_HIT`` / ``POST_CACHE_MISS``
    in :mod:`juniper_data.api.constants` — typos would create spurious
    label buckets and undermine the R1.1 cardinality discipline.

    Args:
        generator: Generator type name (e.g. ``"spiral"``).
        status: Outcome — ``"success"`` or ``"error"``.
        cache: Cache outcome — ``"hit"`` (route returned cached meta) or
            ``"miss"`` (route ran the generator).
    """
    m = _ensure_dataset_metrics()
    m["post_total"].labels(generator=generator, status=status, cache=cache).inc()


def set_datasets_cached(count: int) -> None:
    """Update the cached datasets gauge.

    Args:
        count: Current number of datasets in cache/storage.
    """
    _ensure_dataset_metrics()["datasets_cached"].set(count)
