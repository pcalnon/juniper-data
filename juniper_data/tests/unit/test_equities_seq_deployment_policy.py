"""equities_seq under the deployment policy and the incomplete-data contract.

Two gaps in the sequence generator, both found by the round-38 handoff validation
(2026-09-08) and both proven by execution before they were fixed:

* it defined no ``bind_deployment_defaults``, so ``generate_dataset_id`` hashed the
  schema defaults rather than the effective policy -- two requests under different
  ``JUNIPER_DATA_EQUITIES_ALLOW_TRUNCATION`` values collided on one ``dataset_id``;
* it never applied the fail / accept / drop policy the flat generator applies to rows
  no rescue path could recover, so an unrescued ticker shipped with fabricated
  fundamentals, no refusal and no annotation, whatever ``allow_truncation`` said.

Network sources are mocked exactly as ``test_equities_seq_generator`` mocks them; the
helpers are imported from there so the two files cannot drift in what "a ticker with
no shares data" means.
"""

# Project:       Juniper
# Sub-Project:   JuniperData
# Application:   juniper_data
# File Name:     test_equities_seq_deployment_policy.py
# Author:        Paul Calnon
# Version:       0.6.0
# License:       MIT License

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("yfinance")

from juniper_data.core import limits as eq_limits  # noqa: E402
from juniper_data.core.dataset_id import generate_dataset_id  # noqa: E402
from juniper_data.generators.equities import generator as eq_gen  # noqa: E402
from juniper_data.generators.equities_seq import VERSION as EQUITIES_SEQ_VERSION  # noqa: E402
from juniper_data.generators.equities_seq import EquitiesSeqGenerator, EquitiesSeqParams  # noqa: E402
from juniper_data.tests.unit.test_equities_seq_generator import _generate, _ohlcv, _shares  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.generators]


def _settings(*, allow: bool, cap: int = 7):
    settings = MagicMock()
    settings.equities_max_symbols = cap
    settings.equities_allow_truncation = allow
    settings.equities_incomplete_rows = "accept"
    return settings


class TestBindDeploymentDefaults:
    """The create route finds the binder by ``getattr``; a generator opts in by defining it."""

    def test_binder_exists_and_keeps_the_subclass(self) -> None:
        assert callable(getattr(EquitiesSeqGenerator, "bind_deployment_defaults", None))
        with patch("juniper_data.api.settings.get_settings", return_value=_settings(allow=True)):
            bound = EquitiesSeqGenerator.bind_deployment_defaults(EquitiesSeqParams(lookback=9))
        assert isinstance(bound, EquitiesSeqParams), "model_copy must keep the concrete class"
        assert bound.lookback == 9
        assert bound.max_symbols == 7
        assert bound.allow_truncation is True
        assert bound.model_dump()["allow_truncation"] is True

    def test_omitted_and_explicit_default_cap_hash_alike(self) -> None:
        """Mirror of the flat generator's binder test: the hashed cap is the RESOLVED one."""
        with patch("juniper_data.api.settings.get_settings", return_value=_settings(allow=False, cap=7)):
            omitted = EquitiesSeqGenerator.bind_deployment_defaults(EquitiesSeqParams(allow_truncation=True))
            explicit = EquitiesSeqGenerator.bind_deployment_defaults(EquitiesSeqParams(allow_truncation=True, max_symbols=eq_limits.EQUITIES_DEFAULT_MAX_SYMBOLS))
        assert omitted.model_dump()["max_symbols"] == explicit.model_dump()["max_symbols"] == 7

    def test_dataset_id_now_follows_the_deployment_policy(self) -> None:
        """THE REGRESSION. Without the binder both branches hash to the same id.

        Proven by execution on 2026-09-08 against main 03b7548f: env off and env on gave one
        ``equities_seq-3.0.0-…`` id twice, while ``equities`` gave two ids for the same pair --
        so toggling the deployment opt-in kept serving the artifact built under the old policy.
        """
        ids = {}
        for allow in (False, True):
            with patch("juniper_data.api.settings.get_settings", return_value=_settings(allow=allow)):
                bound = EquitiesSeqGenerator.bind_deployment_defaults(EquitiesSeqParams())
            ids[allow] = generate_dataset_id(generator="equities_seq", version=EQUITIES_SEQ_VERSION, params=bound.model_dump())
        assert ids[False] != ids[True], "the deployment opt-in must change the cache key"
        # The unbound dump is what used to be hashed -- deterministic (seed defaults to 42, so no
        # nonce) and blind to the policy. Kept as the statement of what the binder fixes.
        unbound = EquitiesSeqParams().model_dump()
        assert generate_dataset_id(generator="equities_seq", version=EQUITIES_SEQ_VERSION, params=unbound) == generate_dataset_id(generator="equities_seq", version=EQUITIES_SEQ_VERSION, params=unbound)
        assert unbound["allow_truncation"] is False


class TestIncompleteDataPolicy:
    """The fail / accept / drop contract, now applied by the sequence generator too."""

    @staticmethod
    def _no_shares(seed: int = 60):
        return {"AAPL": _ohlcv(seed=seed), "MSFT": _ohlcv(seed=seed + 1)}

    def test_default_is_refusal(self) -> None:
        """Unset gate ⇒ 422 with the remedy named -- what the flat generator already did."""
        with pytest.raises(eq_limits.IncompleteDataError) as excinfo:
            _generate(["AAPL", "MSFT"], self._no_shares(), shares=None)
        assert excinfo.value.unrescued == ["AAPL", "MSFT"]
        assert excinfo.value.rows_affected > 0
        assert "JUNIPER_DATA_EQUITIES_ALLOW_TRUNCATION" in str(excinfo.value)

    def test_refusal_is_a_value_error(self) -> None:
        """So a missed catch lands on 400, never a 500."""
        with pytest.raises(ValueError):
            _generate(["AAPL"], {"AAPL": _ohlcv(seed=66)}, shares=None)

    def test_accept_keeps_the_symbols_and_annotates(self) -> None:
        arrays = _generate(["AAPL", "MSFT"], self._no_shares(61), shares=None, allow_truncation=True)
        quality = arrays[eq_limits.DATA_QUALITY_META_KEY]
        assert quality["complete"] is False
        assert quality["policy"] == "accept"
        assert sorted(quality["unrescued"]) == ["AAPL", "MSFT"]
        assert quality["rows_affected"] > 0
        assert arrays["ticker_vocab"].tolist() == ["AAPL", "MSFT"], "accept must not remove the symbols"

    def test_drop_removes_the_symbol_from_every_window_and_says_so(self) -> None:
        """The policy runs BEFORE windowing, so a dropped ticker reaches no window and no normaliser fit."""
        good = _shares()
        ohlcv = {"AAPL": _ohlcv(seed=62), "MSFT": _ohlcv(seed=63)}

        def selective(cik, _use_cache):  # noqa: ANN001, ANN202
            return good if cik == 320193 else None

        params = EquitiesSeqParams(symbols=["AAPL", "MSFT"], start_date="2008-01-01", end_date="2011-01-01", use_cache=False, lookback=5, allow_truncation=True, incomplete_rows="drop")
        with patch.object(eq_gen.yf, "download", side_effect=lambda symbol, **_k: ohlcv[symbol].copy()), patch.object(eq_gen.EquitiesGenerator, "_fetch_shares", staticmethod(selective)):
            arrays = EquitiesSeqGenerator.generate(params)

        assert arrays["ticker_vocab"].tolist() == ["AAPL"], "the unresolvable symbol must be gone"
        for split in ("train", "val", "test"):
            assert np.all(arrays[f"ticker_code_{split}"] == 0), f"a {split} window carries the dropped ticker"
        quality = arrays[eq_limits.DATA_QUALITY_META_KEY]
        assert quality["policy"] == "drop"
        assert list(quality["unrescued"]) == ["MSFT"]
        assert quality["rows_affected"] == 0, "dropped rows are not IN the dataset to be affected"

    def test_drop_that_empties_the_dataset_still_fails(self) -> None:
        with pytest.raises(eq_limits.IncompleteDataError, match="leaves no dataset"):
            _generate(["AAPL"], {"AAPL": _ohlcv(seed=64)}, shares=None, allow_truncation=True, incomplete_rows="drop")

    def test_a_clean_dataset_carries_no_annotation(self) -> None:
        """Absence is the signal, exactly as for truncation.

        Worth knowing why this fixture is clean under the staleness annotation added on
        2026-09-11 (APD-DATA-039 / -045), because a first implementation made it dirty. Its last
        filing is 2009-01-15 and the frame ends 2009-07-13: 179 days of silence, well inside the
        365-day bound. It has a >365-day GAP earlier in the series, between the 2008-01-04 and
        2009-01-15 filings, and a per-row reading of staleness flagged the rows in that gap. That
        reading is wrong: an annual filer produces a 365-day gap once a year by definition. The
        ruled question is whether a series has STOPPED, so the annotation measures the silence
        before the window ends, not the age of each row's backing filing.
        """
        arrays = _generate(["AAPL"], {"AAPL": _ohlcv(seed=65)}, _shares())
        assert eq_limits.DATA_QUALITY_META_KEY not in arrays
