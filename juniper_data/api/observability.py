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

- Dataset-generation Prometheus metrics
  (:func:`record_dataset_generation`, :func:`set_datasets_cached`,
  and the lazy-init helper :func:`_ensure_dataset_metrics`).

What moved out:

- :class:`MetricsAuthMiddleware`, :data:`METRICS_DEFAULT_TRUSTED_IPS`,
  :func:`parse_trusted_networks`, :func:`normalize_client_ip`, and
  :class:`TrustedNetwork` — promoted to ``juniper-observability``
  0.3.0 (the §6 promotion in POC_REMEDIATION_PLAN_2026-05-27;
  ``juniper-observability`` 0.3.1 then aligned the logging behaviour
  cascor had carried inline since #313). Re-exported here so the
  historical ``from juniper_data.api.observability import …`` import
  shape stays valid.

New code should prefer ``from juniper_observability import …`` for the
re-exported symbols to make the dependency on the shared lib explicit.
"""

# Cross-service primitives — re-exported from juniper-observability.
from juniper_observability import (  # noqa: F401 — re-exported for backwards compat
    DEFAULT_LOG_FORMAT_PLAIN,
    DEFAULT_SENTRY_TRACES_SAMPLE_RATE,
    LOG_FORMAT_JSON,
    METRICS_DEFAULT_TRUSTED_IPS,
    UNMATCHED_ENDPOINT_LABEL,
    JuniperJsonFormatter,
    MetricsAuthMiddleware,
    PrometheusMiddleware,
    RequestIdMiddleware,
    TrustedNetwork,
    configure_logging,
    configure_sentry,
    get_prometheus_app,
    normalize_client_ip,
    parse_trusted_networks,
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
# SEC-16: ``MetricsAuthMiddleware`` — juniper-data-side IP allowlist for the
# ``/metrics`` mount. The Prometheus ASGI app is mounted as a sub-app and
# therefore bypasses ``SecurityMiddleware``; the wrapper inspects the raw
# ASGI scope so it can reject untrusted scrapers without depending on
# FastAPI. An empty allowlist blocks everything — operators who don't want
# ``/metrics`` exposed should flip ``metrics_enabled=False`` instead.
#
# Implementation lives in ``juniper-observability`` (promoted in 0.3.0;
# 0.3.1 aligned the deny-reason logging cascor had inline since #313).
# Re-exported above for backwards compatibility with existing
# ``from juniper_data.api.observability import MetricsAuthMiddleware``
# call sites in ``juniper_data.api.app`` and the test suite.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Dataset-generation Prometheus metrics — juniper-data specific. Lazily
# initialized to avoid requiring ``prometheus_client`` at import time (it is
# an optional dependency on the ``[api]`` extra path).
# ---------------------------------------------------------------------------

_dataset_metrics: dict | None = None


def _ensure_dataset_metrics() -> dict:
    """Create dataset-related Prometheus metrics on first access.

    Idempotent against the global ``prometheus_client.REGISTRY`` via
    :func:`juniper_observability.register_or_reuse`: if the module-level
    cache has been cleared (e.g. by a test fixture resetting
    ``_dataset_metrics = None``) but the underlying counters / histogram
    / gauge are still registered, the helper re-fetches the existing
    collectors instead of raising ``ValueError: Duplicated timeseries``.
    Production behaviour unchanged on the happy path.
    """
    global _dataset_metrics
    if _dataset_metrics is None:
        from juniper_observability import register_or_reuse
        from prometheus_client import Counter, Gauge, Histogram

        _dataset_metrics = {
            "generations_total": register_or_reuse(
                Counter,
                "juniper_data_dataset_generations_total",
                "Total dataset generation requests",
                ["generator", "status"],
            ),
            "generation_duration_seconds": register_or_reuse(
                Histogram,
                "juniper_data_dataset_generation_duration_seconds",
                # METRICS-MON R4.1: bucket layout is **tentative pending
                # R5.1**. Per-boundary SLO rationale lives in
                # ``notes/observability/HISTOGRAM_BUCKETS_RATIONALE_2026-05-02.md``.
                # R5.1's SLO catalog will ratify or reshape; re-bucketing
                # is a metric-version event but not a public-API break
                # (Prometheus rediscovers buckets on every scrape).
                "Dataset generation duration in seconds (R4.1 buckets tentative pending R5.1)",
                ["generator"],
                buckets=DATASET_GENERATION_DURATION_BUCKETS,
            ),
            "datasets_cached": register_or_reuse(
                Gauge,
                "juniper_data_datasets_cached",
                "Number of datasets currently cached in storage",
            ),
            # METRICS-MON R4.5 / R3.1 follow-up: per-POST request volume,
            # split by cache outcome. ``generations_total`` only counts
            # actual generation work (cache misses); this counts every
            # incoming POST so capacity-planning queries don't undercount
            # deterministic re-POSTs (see roadmap §7 R4.5).
            "post_total": register_or_reuse(
                Counter,
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
