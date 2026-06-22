# juniper-data v0.9.0 Release Notes

**Release Date:** 2026-06-22
**Version:** 0.9.0
**Codename:** DP-3 Capacity Dataset
**Release Type:** MINOR

> Authored from the canonical `juniper-ml/notes/templates/TEMPLATE_RELEASE_NOTES.md`.

---

## Overview

Adds the **`delay_product`** synthetic time-series generator — the *capacity-demonstrating* dataset
for the juniper-recurrence DP-3 readout spectrum — plus routine dependency / CI maintenance. The new
generator's regression target is a **bilinear product of two delayed in-window values**, a quadratic
form in the LMU memory state that a linear readout provably cannot fit (so it exposes a clear
nonlinear ≫ linear r² gap, unlike the near-linear forecasting synthetics). Backward-compatible:
purely additive (a new generator + dependency bumps).

> **Status:** STABLE — additive / backward-compatible; all existing generators and the 3-D NPZ
> contract are unchanged.

---

## Release Summary

- **Release type:** MINOR
- **Primary focus:** New `delay_product` capacity generator (DP-3) + dependency / CI maintenance
- **Breaking changes:** NO
- **Priority summary:** Unblocks the juniper-recurrence DP-3 P2 bench (the RFF-readout capacity gap)

---

## Features Summary

| ID         | Feature                                   | Status | Version | Phase |
| ---------- | ----------------------------------------- | ------ | ------- | ----- |
| DP-3 §8a   | `delay_product` capacity generator         | Done   | 0.9.0   | P2    |

---

## What's New

### `delay_product` synthetic generator (DP-3 capacity instrument)

An irregularly-sampled sinusoid superposition (the same non-uniform Δt sampling as `irregular_sine`)
whose regression target is the **bilinear product of two delayed in-window values**,
`y = x(t−τ₁)·x(t−τ₂)`, with `lag1` / `lag2` step-delays kept strictly inside the lookback.

**Changes:**

- The product is a **quadratic form in the (linear) LMU memory state**, so a **linear readout
  provably cannot fit it** (r² bounded below 1) while a **non-linear (random-Fourier-feature) readout
  can** — the capacity-demonstrating dataset that complements the near-linear synthetics (where the
  linear readout is already at its ceiling).
- Emits the standard additive 3-D NPZ contract
  (`{X, y, dt, target_dt, observed_mask}_{train,test,full}`, `task_type="regression"`,
  `time_unit="steps"`) and reuses the leakage-safe `window_timed_series` windowing (the target reads
  only the emitted window contents; `y_full == concat(train, test)`).
- Registered as `delay_product` in the generator registry; numpy-only, no extra. See juniper-ml
  `notes/JUNIPER_RECURRENCE_DP3_READOUT_SPECTRUM_DESIGN_2026-06-20.md` §8a.

---

## Bug Fixes

None.

---

## Improvements

Routine maintenance bundled into this release:

- **Dependency bumps** — `actions/checkout` 6 → 7, `anthropics/claude-code-action` 1.0.148 → 1.0.154,
  and the `python-minor` dependency group (16 updates).
- **CI / tooling** — local coverage reproduction (`make coverage` + util script), `asyncio_mode=auto`
  for the pytest-asyncio config, and pre-push pre-commit gates wired via `default_install_hook_types`.

---

## Test Results

The `delay_product` generator ships with a dedicated unit-test module (contract, genuinely
non-uniform `dt`, the known-answer bilinear target, determinism, parameter validation, and schema)
and is wired into the parametrized end-to-end synthetic-regression and scaling test suites. The full
juniper-data suite is green in CI.

---

## Upgrade Notes

This is a backward-compatible MINOR release. No migration steps required.

```bash
pip install --upgrade juniper-data==0.9.0
```

---

## Known Issues

None known at time of release.

---

## What's Next

- **juniper-recurrence DP-3 P2 bench** — the bench will delegate to `delay_product` (via
  `juniper_data.generators`) to demonstrate the RFF-readout capacity gap (`nonlinear ≫ linear` r²),
  alongside the *tie* on the existing near-linear datasets.

---

## Contributors

- Paul Calnon

---

## Version History

| Version | Date       | Description                                            |
| ------- | ---------- | ------------------------------------------------------ |
| 0.9.0   | 2026-06-22 | `delay_product` DP-3 capacity generator + maintenance  |
| 0.8.0   | 2026-06-19 | Configurable equities `regression_target`              |
| 0.7.1   | 2026-06-19 | equities wheel packaging fix                           |
| 0.7.0   | 2026-06-19 | Δt sequence data foundation                            |

---

## Links

- [Full Changelog](../../CHANGELOG.md)
- [DP-3 readout-spectrum design](https://github.com/pcalnon/juniper-ml/blob/main/notes/JUNIPER_RECURRENCE_DP3_READOUT_SPECTRUM_DESIGN_2026-06-20.md)
- [Previous Release](RELEASE_NOTES_v0.8.0.md)
