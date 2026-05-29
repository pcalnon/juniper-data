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

import ipaddress
from collections.abc import Sequence

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


def _parse_trusted_networks(
    raw: Sequence[str],
) -> tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...]:
    """Compile bare IPs / CIDR strings to ``ipaddress`` network objects.

    Bare-IP entries are widened to host networks (``/32`` for IPv4,
    ``/128`` for IPv6) by ``ip_network(entry, strict=False)``. Unparseable
    entries fail loud at init time so operator typos surface as a clear
    ``ValueError`` instead of a silently-empty allowlist that 403s
    everything.
    """
    nets: list[ipaddress.IPv4Network | ipaddress.IPv6Network] = []
    for entry in raw:
        try:
            nets.append(ipaddress.ip_network(entry, strict=False))
        except ValueError as exc:
            raise ValueError(f"metrics_trusted_ips entry {entry!r} is not a valid IP or CIDR: {exc}") from exc
    return tuple(nets)


def _normalize_client_ip(client_ip: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address:
    """Strip IPv6 zone id and unwrap IPv4-mapped IPv6 to its IPv4 form.

    Uvicorn can surface zone-scoped link-local addresses like
    ``fe80::1%eth0`` which ``ip_address`` rejects. Docker on some kernels
    surfaces ``::ffff:172.18.0.5`` for IPv4 clients; without unwrapping,
    membership in an IPv4 network like ``172.18.0.0/16`` returns
    ``False`` — silent rejection in the exact docker scenario the
    allowlist exists to support.
    """
    if "%" in client_ip:
        client_ip = client_ip.split("%", 1)[0]
    addr = ipaddress.ip_address(client_ip)
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped is not None:
        addr = addr.ipv4_mapped
    return addr


class MetricsAuthMiddleware:
    """ASGI wrapper that restricts ``/metrics`` to a trusted IP allowlist.

    Accepts bare IPs (``"127.0.0.1"``, ``"::1"``) and CIDR ranges
    (``"172.18.0.0/16"``, ``"fd00::/8"``). Bad entries raise ``ValueError``
    at construction time, not silently at scrape time.
    """

    def __init__(
        self,
        app,
        trusted_ips: Sequence[str] | None = None,
    ) -> None:
        self.app = app
        raw = trusted_ips if trusted_ips is not None else METRICS_DEFAULT_TRUSTED_IPS
        self.networks = _parse_trusted_networks(raw)

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            allowed = False
            client = scope.get("client")
            client_ip = client[0] if client else None
            if client_ip:
                try:
                    addr = _normalize_client_ip(client_ip)
                    allowed = any(addr in net for net in self.networks)
                except ValueError:
                    pass  # Unparseable client IP — treat as untrusted.
            if not allowed:
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
