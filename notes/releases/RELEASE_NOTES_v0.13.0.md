# juniper-data v0.13.0 Release Notes

**Release Date:** 2026-09-05
**Version:** 0.13.0
**Codename:** Third Partition
**Release Type:** MINOR (with **breaking data-contract changes** — see below)

> Authored from the canonical `juniper-ml/notes/templates/TEMPLATE_RELEASE_NOTES.md`.

---

## Overview

Every dataset juniper-data produces is now partitioned **three ways** — `train`, `val`, `test` —
across all sixteen generators. `X_val` / `y_val` are new NPZ keys and are **not optional**.

The point is not the extra array. `val` is the split a trainer reads *during* the loop — early
stopping, candidate selection — while `test` is touched once, at the end. Until this release the
ecosystem had only two partitions, so a consumer that wanted an in-loop signal had to take it from
`X_test`: it selected on the split it then reported, and the reported score was not held out. That
was the actual state of juniper-cascor before juniper-cascor#620.

The design decision (O-1) had been closed for some time. The implementation had not been built.

> **Status:** BREAKING for consumers that assert `len(X_train) + len(X_test) == len(X_full)`,
> or that dispatch on `X.ndim` per the old "all arrays are 2-dimensional" guarantee. Additive for
> everyone else — the existing keys keep their names and meanings.

---

## Release Summary

- **Release type:** MINOR
- **Primary focus:** the `train` / `val` / `test` partition contract, producer side
- **Breaking changes:** **YES** — two, both in the published NPZ contract
- **Consumer requirement:** a juniper-data-client carrying #187. The released **0.4.2 does not** —
  its `NPZ_SPLITS` is `(train, test, full)`, so `val` is skipped by any split-generic reader. #187 is
  merged on client `main` and awaiting a release.

---

## Breaking changes

### 1. The length identity spans three partitions

```text
before:  len(X_train) + len(X_test)              == len(X_full)
now:     len(X_train) + len(X_val) + len(X_test) == len(X_full)
```

This was published as a numbered **guarantee** in `docs/api/JUNIPER_DATA_API.md` and
`docs/USER_MANUAL.md`, so consumers were entitled to rely on it. Both documents now state the
three-way form. Any consumer still asserting the two-way form fails against every artifact this
release produces — loudly, which is the intended outcome.

Two further corrections in the same guarantee list, both of which were simply **false** before:

- *"All arrays are 2-dimensional"* was never true of the six sequence generators, whose `X` is
  `(n, lookback, n_features)`. Dispatch on `meta.sequence`, not on `X.ndim`.
- `y_*` is `(n, 1)` for regression targets, not only `(n, n_classes)` one-hot.

### 2. Every generator is at `generator_version = "2.0.0"`

All sixteen. `generate_dataset_id` hashes the version, so a **seeded** request that previously
produced a two-way artifact cannot now resolve to that cached artifact. This is the deliberate
mitigation for stale-cache reads (risk R-1 of the rollout plan), not an incidental bump.

---

## What's New

### Two sizing models, chosen by `sizing_mode`

| Mode | The generator's native size knob denotes | val / test sized by |
| --- | --- | --- |
| `additive` *(default)* | the **train** row count | `val_percent` / `test_percent` — a percentage **of train**, defaulting to 40 / 30 |
| `carve` | the **total** row count | `train_ratio` / `val_ratio` / `test_ratio` |

Under `additive`, asking for more validation data does not take rows away from training.
`SpiralParams(n_points_per_spiral=97)` yields **194 train + 78 val + 58 test**, and the train count
is exactly what it would have been before the validation split existed.

`mnist`, `csv_import` and `arc_agi` accept **only** `carve` and refuse `additive` outright. Their row
count is not ours to choose, so offering a knob that pretends otherwise would be a lie the caller
cannot see. Refusing is §6.3's "not amenable to synthetic generation" clause, applied.

Ratios denote absolute dataset rows in both modes.

### Sequence generators split on two boundaries

`window_regular_series` and `window_timed_series` take a **required** keyword-only `val_ratio`;
`window_one_ticker` takes `val_cut_ordinal` alongside `cut_ordinal` and routes each window by which
interval its **target** time falls in.

`val_ratio` is required rather than defaulted deliberately: a call site that forgets it should fail
loudly, not silently emit an empty validation split.

The no-future-leak invariant is now **transitive** — every train target precedes every val target,
which precedes every test target. Checking only `train < test` would leave `val` free to overlap
either neighbour, and `val` is the split early stopping reads, so an overlap there is the most
consequential of the three. Embargo purging applies each split's own preceding cut.

Test takes the **remainder** rather than its own rounded share, so no row is lost to independent
rounding. `0.8 / 0.1 / 0.1` over four rows rounds to `3 + 0 + 0` and silently discards 25% of the
dataset; that is fixed.

### `equities` splits three ways per ticker

`val_ratio` defaults to `0.1` and `test_ratio` drops `0.2 → 0.1`. Rounding overflow is trimmed from
test first, then val, and **never** from train — shrinking train to fund a rounding artifact would
change what the model was fit on, which is the one thing a split-arithmetic fix must not do.

A `train_ratio` + `test_ratio` pair summing to exactly `1.0` is now **refused**: the validation share
comes out of the same 1.0, and the generator will not quietly shrink test to make room. State
`val_ratio=0.0` to get the old two-way division.

### `DatasetMeta.n_val`

Defaulted to `0` so an already-stored record still loads (risk R-3).

---

## Fixed

- **`delay_product` paired its validation windows with the wrong target.** The generator overwrites
  the forecast target `window_timed_series` emits with the in-window delay product, split by split —
  and its split list omitted `val`. `X_val` held delay-product windows while `y_val` kept the
  forecast target: features and target from two different problems, on the split early stopping
  reads. The test asserted the product identity on `y_full` only — a separate block with its own
  overwrite — so it passed. It now asserts per split.
- **`_classification_meta` under-counted** when `y_full` was absent, stacking only train + test.
- **`csv_import` shipped an unnormalised `X_val`** — missing from the train-fit min-max application.

Plus the equities work that landed in the same window: the filed-date look-ahead leak in shares
outstanding, the `companyfacts` rescue ladder (at least 28 of 37 unresolvable tickers rescued), the
14-symbol cap, and the refusal of datasets whose fundamentals cannot be resolved. See `CHANGELOG.md`
for the full list.

*(Corrected 2026-09-05: this said "37 unresolvable tickers → 1". The probe covered **29** names —
eight of the census's 37 were never probed — so the supported claim is ">= 28 of 37". The leak fix
shipped here was also incomplete; see the `[Unreleased]` entry in `CHANGELOG.md` for the same-day
restatement defect it left behind.)*

---

## Upgrade notes

**Order matters.** juniper-cascor#620 — merged on cascor `main`, not yet in a release (latest is
0.10.0) — refuses an artifact without `X_val` (§6.1 rule 1: it will not fall back to training rows
for the in-loop signal). So **juniper-data 0.13.0 must reach PyPI before a cascor release carrying
#620 is deployed**, or that cascor deployment needs
`JUNIPER_CASCOR_ALLOW_MISSING_VALIDATION_SPLIT=1` until this release is live.

The same ordering applies to juniper-data-client: its released 0.4.2 omits `val` from `NPZ_SPLITS`.

For consumers:

```python
with np.load("dataset.npz") as data:
    X_train, y_train = data["X_train"], data["y_train"]
    X_val,   y_val   = data["X_val"],   data["y_val"]      # new — always present
    X_test,  y_test  = data["X_test"],  data["y_test"]
```

Assert the three-way identity, not the two-way one. Dispatch tabular-vs-sequence on `meta.sequence`.

---

## Testing

- Full `juniper_data/tests/` suite green; 1460 unit tests.
- `ruff check` and `ruff format --check` clean.
- Mutation-checked: reverting the `delay_product` fix fails `test_target_is_in_window_product`;
  forcing an empty `val` fails 9 tests across the five regular-Δt generators and both windowing
  property suites.
- Cross-repo: a `multi_sine` artifact carrying `X_val` validates as `sequence` through
  `juniper_data_client.contract.validate_npz_contract`, all four splits.

---

## References

- Design of record: `juniper-ml/notes/JUNIPER_2026-08-29_JUNIPER-ECOSYSTEM_TRAIN-EVAL-TEST-PARTITION-DESIGN.md` §9.5 / §9.6
- Rollout plan: `juniper-ml/notes/JUNIPER_2026-08-30_JUNIPER-ECOSYSTEM_PARTITION-IMPLEMENTATION-PLAN.md`
- PRs: juniper-data #353, #358, #361, #367; juniper-data-client #187; juniper-cascor #620
