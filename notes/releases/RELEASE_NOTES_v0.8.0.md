# Juniper Data v0.8.0 Release Notes

**Release Date:** 2026-06-20
**Version:** 0.8.0
**Release Type:** MINOR

---

## Overview

Minor release adding a **configurable equities regression target**. `EquitiesParams` (inherited by
`EquitiesSeqParams`) gains a `regression_target` option so the `y_reg_*` array can be a **stationary
return** instead of the raw (non-stationary) next-day close. Backward-compatible: the default
(`next_close`) keeps every existing artifact byte-identical.

> **Status:** STABLE — backward-compatible minor. No migration required.

---

## Added

- **`regression_target: "next_close" | "return" | "log_return"`** on `EquitiesParams` (and, by
  inheritance, `EquitiesSeqParams`):
  - `next_close` (default) — the raw next-day close price, **byte-identical** to prior output;
  - `return` — the simple next-day return `next_close / close - 1`;
  - `log_return` — the log next-day return `ln(next_close / close)`.

  The raw close is non-stationary (it trends with the price level), which a bounded-memory recurrent
  regressor extrapolates poorly; the return variants are stationary — the standard conditioning for
  forecasting on trending price data. Both the flat `equities` and the windowed `equities_seq`
  generators honor it via a shared `EquitiesGenerator._regression_target` helper (computed in float64,
  cast to the contract's float32). No change to the one-hot `direction` target, the feature matrix, or
  any other array. (juniper-data#195)

### Motivation

Surfaced by the `juniper-recurrence` Δt-LMU evaluation: the `equities_seq` row showed the LMU
"failing" (r²≈−50) on the raw next-day-close target while a trivial `linear_ridge` won — a
target-conditioning (and, as later measured, readout-regularization) artifact, not a Δt-mechanism
flaw. The stationary return target is the data-side half of the fix. See juniper-ml
`notes/JUNIPER_RECURRENCE_EVALUATION_FINDINGS_2026-06-18.md` §3.2.

> "Normalization" in the ecosystem sense is already covered by the advisory `scaling` /
> `target_scaling` meta channel added in 0.7.0; this release adds the orthogonal *representation*
> change (level → return). r² is scale-invariant, so a z-scored variant would not change the measured
> outcome and is intentionally omitted.

---

## Upgrade

```bash
pip install --upgrade "juniper-data[equities]"   # 0.8.0
```

Backward-compatible; no migration. Existing callers get `regression_target="next_close"` by default
(byte-identical to 0.7.1). Opt into stationary returns with
`EquitiesParams(..., regression_target="log_return")`.

---

## Known Issues

None. All required CI checks pass; the equities unit tests cover default byte-identity,
`return` / `log_return` correctness, validation of the new field, and `equities_seq` pass-through.

---

## Version History

| Version | Date       | Description                                                   |
| ------- | ---------- | ------------------------------------------------------------- |
| 0.8.0   | 2026-06-20 | Configurable equities `regression_target` (returns/log-returns) |
| 0.7.1   | 2026-06-19 | Fix: equities constituents CSV now ships in the wheel        |
| 0.7.0   | 2026-06-19 | Synthetic dt-sequence generators + scaling meta channel      |

---

## Links

- [Full Changelog](../../CHANGELOG.md)
- [Previous Release](RELEASE_NOTES_v0.7.1.md)
- Feature PR: juniper-data#195
