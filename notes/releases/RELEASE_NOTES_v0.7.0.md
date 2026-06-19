# Juniper Data v0.7.0 Release Notes

**Release Date:** 2026-06-19
**Version:** 0.7.0
**Codename:** Δt Sequence Data Foundation
**Release Type:** MINOR

---

## Overview

This release completes the **Δt-native sequence data foundation** for the Juniper
recurrence workstream. JuniperData can now generate irregular- and regular-Δt
time-series datasets — both synthetic (closed-form, zero-dependency) and real
(S&P 500 equities) — that emit the additive 3-D NPZ sequence contract (WS-1),
plus an advisory scaling-meta channel and build provenance on the health surface.

> **Status:** STABLE — backward-compatible, additive contract. No breaking changes.

---

## Release Summary

- **Release type:** MINOR
- **Primary focus:** New features — irregular/regular-Δt sequence generators, the 3-D sequence contract, scaling meta, build provenance
- **Breaking changes:** NO (every existing classification generator and NPZ invariant is unchanged; all new fields are optional/additive)
- **Headline:** ships the generators that were merged to `main` after v0.6.0 but were absent from the published 0.6.0 wheel — closing the publish-first gap that blocked the `juniper-recurrence` benchmark and recurrence-model evaluation

---

## What's New

### Δt sequence generators (the recurrence "hello-world" datasets)

#### Synthetic regression generators — `multi_sine`, `mackey_glass`, `ar_p` (#187)

Three numpy-only, deterministic, offline generators emitting the additive 3-D
sequence NPZ contract (WS-1) as `task_type="regression"`. Each samples a process
at a regular Δt and windows it into `(W, L, 1)` sequences with a per-step `dt`, a
fixed `target_dt` forecast horizon, an all-ones `observed_mask`, and the target
carried directly in `y_*`. `multi_sine` is a superposition of K sinusoids
(closed-form known answer when noise-free); `mackey_glass` integrates the chaotic
delay-differential equation (β=0.2, γ=0.1, n=10, τ=17); `ar_p` is a stable
autoregressive process. **No optional extra required** — pure numpy.

#### Irregular-Δt synthetic generator — `irregular_sine` (#188)

A fourth numpy-only regression generator that samples a continuous-time sinusoid
superposition at **non-uniform** times (`sample_dt · U[1−jitter, 1+jitter]`), so
the windowed artifact carries a genuinely non-uniform per-step `dt` and a variable
`target_dt`. The synthetic, known-answer counterpart to `equities_seq`'s
calendar-gap irregularity. Backed by a new `window_timed_series(values, times, …)`
helper.

#### Real irregular-Δt sequences — `equities_seq` (#171) and `equities` (#164)

`equities` produces daily per-(ticker, day) records for S&P 500 constituents
(Yahoo Finance OHLCV + SEC EDGAR shares/market-cap, 52-week high/low, cost basis)
with dual targets (one-hot next-day direction + auxiliary next-day-close
regression). `equities_seq` is its windowed 3-D sequence variant carrying genuine
calendar-gap irregular Δt. Both require the `[equities]` extra (`yfinance`,
`pandas`).

### Advisory `dt` / `target` scaling-meta channel (#189)

A generator may now report *how* its per-step `dt` and regression target should be
standardized, via a reserved `"scaling"` key that the dataset route pops into two
new optional `DatasetMeta` fields — `dt_scaling` and `target_scaling`. The scaling
is **advisory**: the NPZ keeps RAW arrays (every contract invariant intact); a
consumer standardizes at ingestion and denormalizes for metrics using the
persisted stats. New `core/scaling.py` (exact-inverse `standardize` /
`inverse_standardize`, std≈0 guard) and `core/meta.py::pop_scaling_meta`. The four
synthetic generators gain a `scaling: "identity" | "standardize"` parameter
(`standardize` descriptors fit on the **train split** only — no test leakage).

### Sequence contract foundation (WS-1) (#169, #170)

A per-entity sequence-windowing primitive with a Hypothesis leakage-property test
(#169), and a regression/sequence-tolerant dataset contract that makes class
metadata optional and dispatches on `task_type` (#170).

### Build provenance on the health surface (#180)

`/v1/health` and `/v1/health/ready` now report the source `git_sha` and ISO-8601
`build_date` baked into the image (`GIT_SHA` / `BUILD_DATE` / `APP_VERSION`
build-args → OCI labels + env vars; new `juniper_data.provenance` accessor; values
flow into `set_build_info(...)` and the shared `ReadinessResponse`). Foundation for
ecosystem stale-image detection. Requires `juniper-observability>=0.4.0`.

### Compatibility

- **fastapi 0.137** route-introspection compatibility (`_IncludedRouter`) (#181),
  `starlette>=1.0.1` floor (CVE-2026-48710), and routine dependency bumps.

---

## API Changes

### New / changed response fields

| Surface | Change | Breaking? |
| ------- | ------ | --------- |
| `DatasetMeta` | New optional `dt_scaling`, `target_scaling` descriptors | No |
| `/v1/health`, `/v1/health/ready` | New `git_sha`, `build_date` provenance fields | No |
| Dataset metadata | `n_classes` / `class_distribution` now optional (`task_type="regression"`) | No |

### New generators registered on the dataset route

`multi_sine`, `mackey_glass`, `ar_p`, `irregular_sine` (no extra) and `equities`,
`equities_seq` (`[equities]` extra). All emit the 3-D sequence NPZ contract:
`X (n,T,F)`, `y` / `y_reg`, `dt (n,T, dt[:,0]=0)`, `target_dt (n,)`,
`seq_lengths`, `observed_mask` — split-suffixed (`_train` / `_test` / `_full`).

---

## Upgrade Notes

This is a backward-compatible MINOR release. No migration steps are required for
existing classification datasets or consumers.

```bash
pip install --upgrade juniper-data            # synthetic generators, core, API
pip install --upgrade "juniper-data[equities]" # + equities / equities_seq
```

- The API server extra pulls `juniper-observability>=0.4.0` (provenance helpers).
- Synthetic Δt generators (`multi_sine`, `mackey_glass`, `ar_p`, `irregular_sine`)
  need **no** optional extra.

---

## Known Issues

- **`equities` / `equities_seq` require network access** to Yahoo Finance and SEC
  EDGAR at generation time; they are excluded from the offline test path. Not a
  functional defect.
- None blocking. All required CI checks (unit/integration across Python
  3.12–3.14, pre-commit, CodeQL, security, lockfile freshness, quality gate) pass.

---

## What's Next

- **Consumed downstream:** the `juniper-recurrence` benchmark and recurrence-model
  evaluation depend on these published generators (the Δt thesis was validated
  against `irregular_sine`).
- **Eval extensions:** noisy synthetic variants (`noise_std > 0`) and real
  `equities_seq` benchmarking.
- **Scaling/synthetic generator enhancements** tracked under WS-4.

---

## Version History

| Version | Date       | Description                                              |
| ------- | ---------- | ------------------------------------------------------- |
| 0.7.0   | 2026-06-19 | Synthetic dt-sequence generators + scaling meta channel |
| 0.6.0   | 2026-04-08 | Versioning, batch ops, systemd, PostgreSQL fixes        |

---

## Links

- [Full Changelog](../../CHANGELOG.md)
- [Previous Release](RELEASE_NOTES_v0.4.2.md)
- Pull requests: #187, #188, #189 (Δt generators + scaling), #171, #164 (equities), #169, #170 (sequence contract), #180 (provenance), #181 (fastapi 0.137), #191 (release)
