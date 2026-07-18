# juniper-data v0.10.0 Release Notes

**Release Date:** 2026-07-17
**Version:** 0.10.0
**Codename:** Generator Availability + MNIST Extra
**Release Type:** MINOR

> Authored from the canonical `juniper-ml/notes/templates/TEMPLATE_RELEASE_NOTES.md`.

---

## Overview

Ships the two juniper-data legs of the juniper-canopy **training-runtime-defects** plan (§4 I-5):
**D1** surfaces generator availability so a missing optional dependency returns an actionable
**`501 Not Implemented`** with an install hint instead of a masked generic `500`, and **D2** adds a
`[mnist]` extra so real MNIST / Fashion-MNIST generation works out of the box — including inside the
service Docker image. Also promotes the ecosystem **per-file coverage gate** (C-5) to blocking and
folds in routine dependency / CI maintenance since 0.9.0. Backward-compatible: the new `available`
field is additive, the public generator-parameter surface is unchanged, and the `501` replaces a
status only ever emitted on an already-failing request.

> **Status:** STABLE — additive / backward-compatible; the NPZ data contract and every existing
> generator's public parameter surface are unchanged.

---

## Release Summary

- **Release type:** MINOR
- **Primary focus:** Generator availability + actionable `501` (D1) and the `[mnist]` extra shipping
  real MNIST / Fashion-MNIST generation (D2) — the juniper-data legs of the canopy
  training-runtime-defects plan §4 I-5.
- **Breaking changes:** NO
- **Priority summary:** Unblocks end-to-end MNIST training through the stack and removes the masked
  `500` that hid a missing optional dependency from operators (observed live as 71 identical masked
  `500`s).

---

## Features Summary

| ID        | Feature                                                              | Status | Version | Phase |
| --------- | ------------------------------------------------------------------- | ------ | ------- | ----- |
| D1 / I-5  | Generator availability surfaced (`available` flag; `501` + hint)    | Done   | 0.10.0  | —     |
| D2 / I-5  | `[mnist]` extra ships real MNIST / Fashion-MNIST generation         | Done   | 0.10.0  | —     |

---

## What's New

### D1 — Generator availability surfaced

Every entry in `GET /v1/generators` and every `GET /v1/generators/{name}/schema` response now carries
an additive `available: bool` reporting whether the generator's optional dependencies are importable
in the running deployment.

**Changes:**

- Generator classes may declare an `is_available()` hook: `mnist` reports the Hugging Face `datasets`
  package, `equities` / `equities_seq` report the `equities` extra.
- Generators without the hook — the numpy-only synthetics, and `arc_agi` (whose HF need is
  parameter-conditional with a local-file fallback) — default to available.
- The field is purely additive; existing clients that ignore it are unaffected.

### D2 — `[mnist]` extra ships real MNIST / Fashion-MNIST generation

New optional dependency group `mnist = ["datasets[vision]>=4.0.0"]` (Hugging Face `datasets` +
Pillow for image decode); included in `[all]`.

**Changes:**

- The `requirements.lock` compile surface gains `--extra mnist` (lockfile-update workflow, CI
  freshness gate, and all documented copies of the command), so the service Docker image now ships
  the chain and MNIST generation works in containers out of the box.
- The image pins `HF_HOME=/app/data/hf-cache` so a mounted data volume can persist (or pre-seed, for
  offline deployments) the Hub download cache.
- A real-generation integration test (tiny `n_samples`, slow-marked) asserts the NPZ contract shapes
  — `(n, 784)` float32 `X`, `(n, 10)` one-hot `y` — and skips cleanly when `datasets` is not
  installed or the Hub is unreachable with no cache.

---

## Bug Fixes

- **`POST /v1/datasets` no longer masks a missing optional dependency as a generic `500`.** A
  generator raising `ImportError` (e.g. `mnist` without the HF `datasets` package — observed live as
  71 identical masked `500`s) now returns **`501 Not Implemented`** with the generator's actionable
  install hint in `detail` (`pip install datasets`). The unknown-generator and invalid-params `400`
  contracts and the generic `500` handler for genuinely unexpected errors are unchanged.
- **MNIST generator now loads the canonical namespaced Hub repositories** (`ylecun/mnist`,
  `zalando-datasets/fashion_mnist`). Bare canonical names (`load_dataset("mnist")`) are rejected by
  the huggingface-hub 1.x URI layer used by `datasets>=5` ("Repository id must be 'namespace/name'"),
  so the stable `MnistParams.dataset` values (`"mnist"` / `"fashion_mnist"` — the public parameter
  surface is unchanged) now map internally to the namespaced repos, which load identically on older
  `datasets` versions. Generator `VERSION` bumped 1.0.0 → 1.0.1.

---

## Improvements

Routine maintenance bundled into this release:

- **CI: per-file coverage is now a blocking gate (ecosystem per-file rollout C-5).** The advisory
  per-module step (`scripts/check_module_coverage.py`, which only *warned* below 85%) is replaced by
  the shared `juniper-coverage-gap-map --enforce` gate from `juniper-ci-tools>=0.6.0,<0.7.0`. CI now
  **fails** when any source file's statement coverage is below 90% or any packaged sub-module's
  pooled (statement-weighted) coverage is below 95%. `check_module_coverage.py` is retained for the
  pre-push hook and `util/run_coverage.bash`.
- **Test coverage lifted** for the `unit and not slow` lane from 86.8% to 98.3% to satisfy the new
  gate, with **no production-code changes** (unmarked genuine unit tests brought into the lane, plus
  new deterministic offline tests for API security, batch endpoints, sequence-windowing branches, the
  equities / windowed-equities helpers, and the `__init__` arc-agi fallback).
- **Dependency / CI bumps** — `setuptools` 82.0.1 → 83.0.0, the `python-minor` dependency group
  (multiple updates), `github/codeql-action` subactions aligned to `v4.37.0`, `actions/cache`,
  `actions/setup-python`, `anthropics/claude-code-action`, and dev type-stub bumps.
- **Publish hardening** — the TestPyPI install-verify step now runs strict `--no-deps` (release-train
  Phase 0.2, F-5).

---

## Test Results

No production code changed for the release bump itself. The D1/D2 feature work ships with dedicated
coverage: the additive `available` field is asserted on the generators list/schema routes, the `501`
+ install-hint path is exercised on `POST /v1/datasets`, and the `[mnist]` extra carries a
real-generation integration test (slow-marked, skips offline). The full juniper-data suite is green
in CI across Python 3.12 / 3.13 / 3.14 (plus the required macOS 3.12 leg), and the aggregate (80%) +
per-file (>=90% file / >=95% sub-module pooled) coverage gates pass.

---

## Upgrade Notes

This is a backward-compatible MINOR release. No migration steps required.

```bash
pip install --upgrade juniper-data==0.10.0
```

To generate MNIST / Fashion-MNIST datasets, install the new extra (heavy Hugging Face chain,
extra-gated):

```bash
pip install "juniper-data[mnist]==0.10.0"
```

The service Docker image already ships the `[mnist]` chain; offline deployments should seed the
`HF_HOME` cache (mounted at `/app/data/hf-cache`).

---

## Known Issues

None known at time of release.

---

## What's Next

- With D1 + D2 in place, end-to-end MNIST training through the stack (juniper-canopy →
  juniper-data → juniper-cascor) is unblocked; the remaining canopy training-runtime-defects
  units are tracked in juniper-ml
  `notes/JUNIPER_2026-07-11_JUNIPER-CANOPY_TRAINING-RUNTIME-DEFECTS-PLAN.md`.

---

## Contributors

- Paul Calnon

---

## Version History

| Version | Date       | Description                                               |
| ------- | ---------- | --------------------------------------------------------- |
| 0.10.0  | 2026-07-17 | Generator availability + `501` hint (D1); `[mnist]` extra (D2) |
| 0.9.0   | 2026-06-22 | `delay_product` DP-3 capacity generator + maintenance     |
| 0.8.0   | 2026-06-19 | Configurable equities `regression_target`                 |
| 0.7.1   | 2026-06-19 | equities wheel packaging fix                              |
| 0.7.0   | 2026-06-19 | Δt sequence data foundation                               |

---

## Links

- [Full Changelog](../../CHANGELOG.md)
- [Canopy training-runtime-defects plan](https://github.com/pcalnon/juniper-ml/blob/main/notes/JUNIPER_2026-07-11_JUNIPER-CANOPY_TRAINING-RUNTIME-DEFECTS-PLAN.md)
- [Previous Release](RELEASE_NOTES_v0.9.0.md)
