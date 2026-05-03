# juniper-data — Histogram Bucket Rationale

**Date:** 2026-05-02
**METRICS-MON sub-track:** R4.1 / seed-14
**Status:** Initial draft — bucket layouts marked **tentative pending R5.1**.
**Related:** [`METRICS_MONITORING_R4_ENTRY_PLAN_2026-05-01.md`](https://github.com/pcalnon/juniper-ml/blob/main/notes/code-review/METRICS_MONITORING_R4_ENTRY_PLAN_2026-05-01.md) §3 Q1 (resolution: hybrid — document current rationale now, mark tentative, R5.1 ratifies or reshapes against SLOs).

---

## 1. Inventory

juniper-data exposes **one** Prometheus histogram on the production
surface (other shapes are Counters / Gauges):

| Metric | Labels | Bucket constant | Purpose |
|---|---|---|---|
| `juniper_data_dataset_generation_duration_seconds` | `generator` | `DATASET_GENERATION_DURATION_BUCKETS` (in `juniper_data/api/constants.py`) | Wall-clock seconds spent in `generator_class.generate(params)` for a successful POST `/v1/datasets`. Suppressed for `status="error"` to avoid mixing distributions. |

---

## 2. `juniper_data_dataset_generation_duration_seconds`

### 2.1 Current bucket layout

```python
DATASET_GENERATION_DURATION_BUCKETS: tuple[float, ...] = (
    0.01,    # 10 ms
    0.05,    # 50 ms
    0.1,     # 100 ms
    0.25,    # 250 ms
    0.5,     # 500 ms
    1.0,     # 1 s
    2.5,     # 2.5 s
    5.0,     # 5 s
    10.0,    # 10 s
    30.0,    # 30 s
    float("inf"),
)
```

11 buckets including `+inf`. Bucket boundaries span 4.5 orders of
magnitude (10 ms → 30 s).

### 2.2 Rationale per boundary

| Boundary | What it discriminates | SLO target served | R5.1 status |
|---|---|---|---|
| **0.01 s (10 ms)** | "Trivial" generations served from a warm in-process state — observed only on the deterministic-spiral path with very small `n_points_per_spiral`. | None (tentative). | **Tentative.** R5.1 may collapse this with 0.05 if no SLO references "p99 < 10 ms". |
| **0.05 s (50 ms)** | Typical small-spiral generation. Useful as the "fast path" lower bound for capacity-planning queries (`rate(...)` over the 0–50 ms histogram_quantile). | **Candidate** for "p50 generation latency < 50 ms" SLO if R5.1 defines one. | **Tentative.** |
| **0.1 s (100 ms)** | Crosses the human-perceptible boundary. If juniper-cascor (the primary consumer) waits synchronously on POST, p99 above 100 ms degrades cascor's training loop responsiveness. | **Strong candidate** for "p95 generation latency < 100 ms" SLO. | **Tentative — high confidence.** |
| **0.25 s (250 ms)** | Filler between 100 ms and 500 ms. Useful for histogram_quantile precision in the p95–p99 range. | None directly, but supports SLO quantile estimation around the 100 ms boundary. | **Tentative.** |
| **0.5 s (500 ms)** | Default-spiral-params generation (n_spirals=2, n_points_per_spiral=50). Observed median in dev. | Operationally useful as "warm cache miss baseline". | **Tentative.** |
| **1.0 s (1 s)** | Generation crossing into "noticeable lag" territory. Above this, calling code should consider async handoff. | **Candidate** for "p99 generation latency < 1 s" SLO. | **Tentative — high confidence.** |
| **2.5 s (2.5 s)** | Filler for histogram_quantile resolution between 1 s and 5 s. | None directly. | **Tentative.** |
| **5.0 s (5 s)** | Outer bound for "complex" spiral params (n_spirals×n_points_per_spiral large). Above this, capacity planning needs to assume request-queue backpressure. | **Candidate** for capacity-headroom SLO (e.g. "rate of >5 s generations < 0.1/s"). | **Tentative.** |
| **10.0 s (10 s)** | Pathological — likely a cache miss against a recently-evicted large dataset. Should be rare in steady state. | Useful for alerting on slowdown trends. | **Tentative.** |
| **30.0 s (30 s)** | Extreme — likely indicates upstream resource starvation (CPU saturation, disk I/O contention) rather than dataset complexity. | Alert threshold for "generator-worker degraded". | **Tentative.** |
| **+inf** | Mandatory upper bound; `histogram_quantile` requires it. | — | Required. |

### 2.3 Trade-off

The current layout was chosen to bracket the observed dev-time
distribution (median ~500 ms, tail to 5 s) with reasonable resolution
in the SLO-relevant 100 ms / 1 s region. The 4.5-decade spread is
generous for a single generator; if R5.1 SLOs target only sub-second
generations, the 10 s and 30 s buckets become low-information and
could be collapsed.

---

## 3. R5.1 ratification queue

When R5.1 designs the SLO catalog:

- [ ] Decide whether "p95 generation latency < 100 ms" is the primary
      SLO. If yes, the 0.1 s boundary stays as the load-bearing
      assertion and the test added by R3.1 should grow a quantile
      assertion.
- [ ] Decide whether "p99 generation latency < 1 s" is a separate SLO
      or implied by the p95 < 100 ms target. If separate, retain 1 s
      and 2.5 s buckets.
- [ ] If neither sub-second SLO is adopted, consider re-bucketing
      with a logarithmic spread (e.g. powers-of-2 from 10 ms to 30 s)
      for uniform quantile precision.
- [ ] Re-evaluate 0.01 s and 30 s — currently tentative; both may be
      removable if no SLO references them.

---

## 4. Process notes

- HELP-string marker: `juniper_data_dataset_generation_duration_seconds`
  carries a "tentative pending R5.1" suffix on its HELP line as a
  forward-pointer to this doc. Operators reading `/metrics` directly
  see the marker.
- Re-bucketing is a metric-version event but **not** a public-API
  break — Prometheus consumers re-discover buckets on every scrape.
  No SemVer-major beat is required when R5.1 ratifies or reshapes.
