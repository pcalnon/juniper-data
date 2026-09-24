#!/usr/bin/env python
"""X8 — ``equities_seq`` is declared ``regression``, and the relabel moved its version.

``equities_seq`` emits two targets: a one-hot next-day direction (``y_*``) and a next-day close
(``y_reg_*``). The registry's ``task_type`` has no word for "both". juniper-data declared it
``classification``, while juniper-canopy labelled it ``regression``, because the one model that
trains on it -- the LMU -- reads ``y_reg_*``. The owner ruled on 2026-09-24 that juniper-data
changes its label.

The relabel changes what ``POST /v1/datasets`` emits and stores, and it leaves the arrays alone.
``compute_shape_meta`` fills ``n_classes`` / ``class_distribution`` only for
``classification``, so both become null. The dataset ID hashes the generator VERSION but not
the meta, and a cache hit serves the stored meta as-is. So the relabel came with a bump to
6.0.0: without it, a cached 5.0.0 artifact would keep serving classification meta under the
id a fresh request resolves to. That is the arc_agi precedent (#402 replaced ``task_type``
without a bump, and #427 had to repair it).

Flat ``equities`` is NOT relabelled. The ruling named ``equities_seq``, and flat ``equities``
keeps ``classification`` at 5.0.0. This pins that the pair now differs on purpose.
"""

from __future__ import annotations

import numpy as np
import pytest

from juniper_data.api.routes.generators import GENERATOR_REGISTRY
from juniper_data.core.dataset_id import generate_dataset_id
from juniper_data.core.meta import TASK_TYPE_CLASSIFICATION, TASK_TYPE_REGRESSION, compute_shape_meta
from juniper_data.generators.equities import VERSION as EQUITIES_VERSION
from juniper_data.generators.equities_seq import VERSION as EQUITIES_SEQ_VERSION

pytestmark = [pytest.mark.unit, pytest.mark.generators]


def _equities_seq_shaped_arrays(windows=12, lookback=5, features=15):
    """The shape ``equities_seq`` emits: 3-D ``X``, a one-hot direction ``y`` and a ``y_reg``."""
    rng = np.random.default_rng(0)
    arrays = {}
    for split, n in (("train", windows), ("val", windows // 3), ("test", windows // 3)):
        arrays[f"X_{split}"] = rng.normal(size=(n, lookback, features)).astype(np.float32)
        arrays[f"y_{split}"] = np.eye(2, dtype=np.float32)[np.arange(n) % 2]
        arrays[f"y_reg_{split}"] = rng.normal(size=(n, 1)).astype(np.float32)
    return arrays


class TestX8EquitiesSeqIsRegression:
    def test_the_registry_declares_regression(self):
        assert GENERATOR_REGISTRY["equities_seq"]["task_type"] == TASK_TYPE_REGRESSION

    def test_flat_equities_keeps_classification(self):
        # The ruling named ``equities_seq`` only. Flat ``equities`` emits the same two targets,
        # but its consumers (cascor) train on the one-hot direction.
        assert GENERATOR_REGISTRY["equities"]["task_type"] == TASK_TYPE_CLASSIFICATION

    def test_the_relabel_nulls_the_class_meta_and_nothing_else(self):
        arrays = _equities_seq_shaped_arrays()
        before = compute_shape_meta(arrays, TASK_TYPE_CLASSIFICATION)
        after = compute_shape_meta(arrays, GENERATOR_REGISTRY["equities_seq"]["task_type"])
        assert before["n_classes"] == 2 and before["class_distribution"]  # what 5.0.0 emitted
        assert after["n_classes"] is None and after["class_distribution"] is None
        unchanged = {k: v for k, v in before.items() if k not in {"n_classes", "class_distribution"}}
        assert {k: after[k] for k in unchanged} == unchanged
        assert after["n_features"] == 15  # the trailing axis of a 3-D X


class TestX8MovedTheVersion:
    def test_equities_seq_is_at_6_and_the_pair_now_differs(self):
        assert EQUITIES_SEQ_VERSION == "6.0.0"
        assert EQUITIES_VERSION == "5.0.0"

    def test_the_dataset_id_moved_with_it(self):
        # What the bump is FOR: the same request now resolves to a different id, so a cached
        # artifact carrying 5.0.0's classification meta cannot answer it.
        params = {"symbols": ["AAPL"], "lookback": 20}
        assert generate_dataset_id("equities_seq", EQUITIES_SEQ_VERSION, params) != generate_dataset_id("equities_seq", "5.0.0", params)
