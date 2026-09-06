"""Equities time-series dataset generator (S&P 500, 2000 -> present).

Downloads daily OHLCV from Yahoo Finance (via ``yfinance``) and shares-
outstanding history from SEC EDGAR (XBRL company-concept API), conditions them
into a per-(ticker, day) record table -- name, ticker, date, OHLCV, 52-week
high/low, total shares, market capitalization, and a configurable-purchase-date
cost basis -- and formats the result into the JuniperData NPZ contract.

Targets (per the dataset spec) are dual:
  * canonical one-hot **next-day direction** (up/down) in ``y_*`` -- keeps
    ``n_classes == 2`` and the route's class distribution, taken over the
    partitions, correct;
  * auxiliary **next-day regression** target carried in extra ``y_reg_*``
    arrays (preserved through ``save_versioned`` / ``save_npz``); its
    representation is configurable via ``regression_target`` -- raw next-day
    close (default), simple return, or log return.

Compact, row-aligned identifier arrays (``ticker_code_*`` + ``ticker_vocab``,
``date_*`` as YYYYMMDD ints) keep every sample traceable without bloating the
artifact with per-row strings.

External data is fetched at ``generate()`` time (mnist-style) and cached under
``~/.cache/juniper_data/equities`` so re-runs are fast and offline-friendly.

Requires the optional ``equities`` extra: ``pip install "juniper-data[equities]"``.
"""

# Project:       Juniper
# Sub-Project:   JuniperData
# Application:   juniper_data
# File Name:     generator.py
# Author:        Paul Calnon
# Version:       0.6.0
# License:       MIT License

from __future__ import annotations

import contextlib
import csv
import json
import logging
import os
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from juniper_data.core.limits import DATA_QUALITY_META_KEY, INCOMPLETE_ACCEPT, INCOMPLETE_DROP, INCOMPLETE_FAIL, REASON_SYMBOL_CAP, TRUNCATION_META_KEY, UNIT_SYMBOLS, IncompleteDataError, InputTooLargeError, build_data_quality_meta, build_truncation_meta

from .defaults import CONSTITUENTS_FILENAME, EQUITIES_FEATURE_COLUMNS
from .params import EquitiesParams

VERSION = "3.0.0"

_logger = logging.getLogger(__name__)

try:
    import pandas as pd
    import yfinance as yf

    EQUITIES_DEPS_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without the extra installed
    EQUITIES_DEPS_AVAILABLE = False
    pd = None  # type: ignore[assignment]
    yf = None  # type: ignore[assignment]

# SEC fair-access policy: descriptive User-Agent + <10 requests/second.
_SEC_UA = {"User-Agent": "juniper-data (Juniper ML research; overtoad.research@gmail.com)"}
_SEC_CONCEPT_URL = "https://data.sec.gov/api/xbrl/companyconcept/CIK{cik:010d}/{taxonomy}/{tag}.json"
_SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_SEC_MIN_INTERVAL = 0.12  # seconds between SEC calls
# Reject SEC share-count points deviating from the series median by more than
# this factor: such jumps are XBRL filer scale typos (e.g. a cover-page value
# entered 1e6x too large), not real splits/buybacks/issuance.
_SHARES_OUTLIER_FACTOR = 100.0

# XBRL shares-outstanding concepts, tried in order (dei cover-page first).
_SHARES_CONCEPTS = (("dei", "EntityCommonStockSharesOutstanding"), ("us-gaap", "CommonStockSharesOutstanding"))

# Provenance values recorded on the shares frame and surfaced in DatasetMeta.
SHARES_SOURCE_CONCEPT = "companyconcept"
SHARES_SOURCE_FACTS = "companyfacts"
SHARES_QUALITY_POINT_IN_TIME = "point_in_time"
SHARES_QUALITY_PERIOD_AVERAGE = "period_average"
# ISSUED IS A DIFFERENT QUANTITY, not a weaker measurement of the same one.
# Issued includes treasury stock, so it is >= outstanding -- materially so for a
# company that has bought back stock. A ``market_cap`` built on it silently means
# something else for that symbol, which is why it rides in ``data_quality`` as a
# `degraded` entry naming this basis rather than being folded in silently.
SHARES_QUALITY_ISSUED = "issued_includes_treasury"
SHARES_QUALITY_UNRESCUED = "unrescued"

# THE RESCUE LADDER, walked only when ``companyconcept`` yields nothing.
#
# Probed 2026-09-05 (``util/ad-hoc/2026-09-05_probe_shares_rescue_paths.py``)
# against the constituents ``companyconcept`` reported as empty: **18 have the
# SAME dei concept, fully populated, in companyfacts** -- KO among them, with 71
# facts and a current count of 4.30 billion shares. Rung 3 rescues 10 more (META,
# RL, HRL, MKC, LEN, UHS, ABNB, TTD, TKO, XYZ). One name (STZ) has no share
# concept at all in companyfacts.
#
# **The supported claim is ">= 28 of 37 rescued", not "37 -> 1".** The probe's
# own gap list holds 29 entries (18 + 10 + 1), while its docstring says 37 --
# eight of the census's names were never probed, so their outcome is unknown, not
# rescued. Corrected 2026-09-05; the earlier "37 -> 1" is in this file's history,
# in CHANGELOG.md and in RELEASE_NOTES_v0.13.0.md.
#
# **The mechanism is NOT what this comment used to claim.** It said both
# endpoints exclude dimensional facts and that multi-class filers therefore go
# missing. That story does not fit its own example: KO is SINGLE-class and its
# rows carry no dimensional keys at all. What the evidence supports instead is an
# UPSTREAM REGRESSION in ``companyconcept`` between 2026-06 and 2026-09 -- a
# June-2026 cache holds KO's same 71 dei facts from the endpoint that returns
# nothing today, and across the probed names the ones rescued at rungs 1-2 are
# exactly the ones with a June cache entry. Population-level, "no data" went
# 15 -> 37 against a constituents list unchanged since 2026-06-03.
#
# The ladder is still right -- a fallback does not need to know why the primary
# failed -- but do not rely on it compensating for a PERMANENT property. If SEC
# restores the endpoint the framing changes; if it degrades further, this rung is
# not guaranteed either.
#
# Rung 3 is SEMANTICALLY WEAKER and is why provenance is recorded: a period
# average is not a point-in-time count, so a market cap derived from it is not
# the same quantity.
#
# companyfacts costs ~1.15 s and ~5 MB, so it must stay a FALLBACK. The
# comparison against companyconcept's "~600 B" was misleading: 600 B is the size
# of an EMPTY response, while a populated one has a median of ~10.9 KB across the
# 485-payload cache -- so the honest ratio is ~455x, not ~8,300x. Paying
# companyfacts per symbol would cut the 14-symbol cap to roughly 9 at the
# optimistic 2.1 s/symbol, or roughly 6 at the conservative 4.0 s recorded in
# ``juniper_data/core/limits.py``.
_SHARES_FACTS_LADDER = (
    ("dei", "EntityCommonStockSharesOutstanding", SHARES_QUALITY_POINT_IN_TIME),
    ("us-gaap", "CommonStockSharesOutstanding", SHARES_QUALITY_POINT_IN_TIME),
    ("us-gaap", "WeightedAverageNumberOfSharesOutstandingBasic", SHARES_QUALITY_PERIOD_AVERAGE),
    # LAST, and last deliberately. The weighted average above is the RIGHT
    # quantity measured over a period; issued is the WRONG quantity measured
    # exactly. For a market cap, a time-averaged count of outstanding shares is
    # nearer the truth than a precise count that includes treasury stock -- so
    # this rung only runs when nothing else reported anything at all.
    ("us-gaap", "CommonStockSharesIssued", SHARES_QUALITY_ISSUED),
)
_SEC_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"


_CONSTITUENTS_PATH = Path(__file__).resolve().parent / CONSTITUENTS_FILENAME
_CACHE_DIR = Path(os.environ.get("JUNIPER_DATA_EQUITIES_CACHE_DIR", str(Path.home() / ".cache" / "juniper_data" / "equities")))

# Monotonic timestamp of the last SEC request, for global throttling.
_last_sec_call = [0.0]


def _sec_get(url: str, retries: int = 3) -> dict[str, Any] | None:
    """GET a SEC JSON endpoint with throttling, retries, and a compliant User-Agent.

    Returns the parsed JSON, or None on 404 (concept not reported). Transient
    network errors are retried with linear backoff; the last error is raised if
    all attempts fail.
    """
    last_exc: Exception | None = None
    for attempt in range(retries):
        wait = _SEC_MIN_INTERVAL - (time.monotonic() - _last_sec_call[0])
        if wait > 0:
            time.sleep(wait)
        request = urllib.request.Request(url, headers=_SEC_UA)
        try:
            # Trusted, fixed SEC host; scheme is hardcoded https.
            with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310  # nosec B310
                return json.loads(response.read().decode())
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            last_exc = exc
        except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as exc:
            last_exc = exc
        finally:
            _last_sec_call[0] = time.monotonic()
        time.sleep(0.5 * (attempt + 1))
    if last_exc is not None:
        raise last_exc
    return None


class EquitiesGenerator:
    """Generator for S&P 500 equities time-series datasets.

    Downloads, conditions, and formats real market data into the JuniperData
    NPZ format with a temporal (date-ordered) train/test split per ticker.

    All methods are static to keep the generator stateless and side-effect free
    (aside from the on-disk download cache).
    """

    @staticmethod
    def is_available() -> bool:
        """Report whether this generator can run in this deployment (D1 / I-5).

        Returns:
            True when the optional ``equities`` extra (pandas + yfinance) is
            importable; False otherwise (``generate`` would raise ImportError).
        """
        return EQUITIES_DEPS_AVAILABLE

    @staticmethod
    def install_hint() -> str:
        """Report how to make this generator available (W-4, companion to ``is_available``).

        Single source of truth: ``generate`` raises this exact text, so the hint on
        ``GET /v1/generators`` and the 501 detail on ``POST /v1/datasets`` cannot drift.

        Returns:
            The curated, actionable install instruction for the missing extra.
        """
        return 'The "equities" extra is required. Install with: pip install "juniper-data[equities]"'

    @staticmethod
    def generate(params: EquitiesParams) -> dict[str, np.ndarray]:
        """Generate the equities dataset.

        Args:
            params: EquitiesParams instance defining the universe, date range,
                cost-basis purchase date, and conditioning options.

        Returns:
            Dictionary with the canonical NPZ keys (X_train, y_train, X_val,
            y_val, X_test, y_test) plus auxiliary arrays: y_reg_* (next-day
            close regression target), ticker_code_* / date_* (row-aligned
            identifiers), and ticker_vocab (code -> ticker lookup).

        Raises:
            ImportError: If the optional ``equities`` extra is not installed.
            ValueError: If no data could be retrieved for any requested symbol.
            InputTooLargeError: If the resolved universe exceeds its symbol cap
                and neither the request nor the deployment allowed truncation.
                Subclasses ValueError; the route maps it to 422.
        """
        if not EQUITIES_DEPS_AVAILABLE:
            raise ImportError(EquitiesGenerator.install_hint())

        constituents = EquitiesGenerator._load_constituents()
        symbols, meta_map, truncation = EquitiesGenerator._resolve_symbols(params, constituents)
        end_date = params.end_date or datetime.now(UTC).strftime("%Y-%m-%d")

        conditioned: dict[str, Any] = {}
        total = len(symbols)
        for index, ticker in enumerate(symbols, start=1):
            try:
                frame = EquitiesGenerator._condition_one(ticker, meta_map.get(ticker, {}), params, end_date)
            except Exception as exc:  # noqa: BLE001 - one bad ticker must not abort the whole batch
                _logger.warning("equities: skipping %s (%s)", ticker, exc)
                continue
            if frame is not None and not frame.empty:
                conditioned[ticker] = frame
                _logger.info("equities: [%d/%d] %s -> %d rows", index, total, ticker, len(frame))
            else:
                _logger.info("equities: [%d/%d] %s -> no data", index, total, ticker)

        if not conditioned:
            raise ValueError("No data could be retrieved for the requested symbols.")

        # APD-DATA-018 follow-up: classify what came back BEFORE assembling it.
        #
        # Two distinct problems, kept apart because a consumer has to ask about
        # them separately: `degraded` recovered a value from a weaker source (a
        # period average is not a point-in-time share count, so its market_cap is
        # a different quantity); `unrescued` recovered nothing at all, and under
        # fundamentals_fill="zero" would ship a market cap of 0.0 -- a number no
        # listed company can have.
        degraded: dict[str, str] = {}
        unrescued: dict[str, str] = {}
        for ticker, frame in conditioned.items():
            quality = str(frame["shares_quality"].iloc[0]) if "shares_quality" in frame.columns and len(frame) else SHARES_QUALITY_UNRESCUED
            if quality == SHARES_QUALITY_UNRESCUED:
                unrescued[ticker] = "no shares-outstanding concept in companyconcept or companyfacts"
            elif quality != SHARES_QUALITY_POINT_IN_TIME:
                degraded[ticker] = quality

        rows_affected = sum(len(conditioned[ticker]) for ticker in unrescued)
        policy = EquitiesGenerator._resolve_incomplete_policy(params, bool(unrescued))

        if unrescued:
            _logger.warning("equities: %d symbol(s) have no shares data after every rescue path: %s", len(unrescued), ", ".join(sorted(unrescued)))
        if policy == INCOMPLETE_FAIL:
            raise IncompleteDataError(
                detail="Shares outstanding could not be resolved for part of the requested universe, so total_shares and market_cap would be fabricated for those rows.",
                unrescued=sorted(unrescued),
                rows_affected=rows_affected,
                opt_in_env="JUNIPER_DATA_EQUITIES_ALLOW_TRUNCATION",
            )
        if policy == INCOMPLETE_DROP:
            for ticker in unrescued:
                conditioned.pop(ticker, None)
            _logger.warning("equities: dropped %d symbol(s) with unresolvable shares data per incomplete_rows='drop'", len(unrescued))
            if not conditioned:
                raise IncompleteDataError(
                    detail="Every requested symbol had unresolvable shares data, so dropping them leaves no dataset.",
                    unrescued=sorted(unrescued),
                    rows_affected=rows_affected,
                    opt_in_env="JUNIPER_DATA_EQUITIES_ALLOW_TRUNCATION",
                )

        vocab = sorted(conditioned)
        code_of = {ticker: code for code, ticker in enumerate(vocab)}

        # Three chronological partitions per ticker: train | val | test, earliest
        # first. Validation sits BETWEEN train and test in time, so early stopping
        # never reads rows from after the reported window.
        train_frames, val_frames, test_frames, full_frames = [], [], [], []
        for ticker in vocab:
            frame = conditioned[ticker].sort_index()
            frame["ticker_code"] = code_of[ticker]
            n_rows = len(frame)
            n_train = int(round(n_rows * params.train_ratio))
            n_val = int(round(n_rows * params.val_ratio))
            n_test = int(round(n_rows * params.test_ratio))
            # An over-subscribed request is trimmed from the END -- test first,
            # then val. Trimming train would silently shrink the partition every
            # existing baseline is measured against.
            overflow = n_train + n_val + n_test - n_rows
            if overflow > 0:
                taken = min(overflow, n_test)
                n_test -= taken
                overflow -= taken
            if overflow > 0:
                n_val -= min(overflow, n_val)
            train_frames.append(frame.iloc[:n_train])
            val_frames.append(frame.iloc[n_train : n_train + n_val])
            test_frames.append(frame.iloc[n_train + n_val : n_train + n_val + n_test])
            full_frames.append(frame)

        full = pd.concat(full_frames)
        train = pd.concat(train_frames) if train_frames else full.iloc[:0]
        val = pd.concat(val_frames) if val_frames else full.iloc[:0]
        test = pd.concat(test_frames) if test_frames else full.iloc[:0]

        # Fit normalization statistics on the TRAINING rows only (juniper-data#314).
        #
        # This previously fit on ``full`` -- every row, pooled across tickers, INCLUDING the
        # chronologically-later test rows -- and then applied those statistics to ``train``.
        # That is look-ahead leakage: the training features were scaled by a maximum that
        # exists only in the future relative to them.
        #
        # Fitting on ``train`` is decision 7 of the ecosystem partition design
        # (juniper-ml notes/JUNIPER_2026-08-29_JUNIPER-ECOSYSTEM_TRAIN-EVAL-TEST-PARTITION-DESIGN.md):
        # no quantity derived from a later partition may reach the training data.
        #
        # CONSEQUENCE, deliberate: ``X_val`` and ``X_test`` are no longer guaranteed to lie
        # within [0, 1]. They are scaled by train's statistics, and later rows legitimately
        # exceed the training range -- that excursion IS the information the old code was
        # leaking away. Only ``X_train`` is bounded now.
        #
        # ``train`` is empty only when train_ratio rounds every ticker to zero rows; fall back
        # to ``full`` there rather than emitting all-NaN statistics, since a degenerate request
        # has no training distribution to learn from and there is nothing to leak into.
        norm = None
        if params.normalize_features:
            fit_frame = train if len(train) > 0 else full
            norm = EquitiesGenerator._fit_normalizer(fit_frame)

        arrays: dict[str, np.ndarray] = {}
        # ``full`` left this tuple with decision 11. ``full`` the FRAME is still used
        # above as the normaliser fallback when train is empty -- that is a fit scope, not
        # an emitted array, and dropping the key does not change it.
        for name, frame in (("train", train), ("val", val), ("test", test)):
            features = EquitiesGenerator._features(frame, norm)
            arrays[f"X_{name}"] = features
            arrays[f"y_{name}"] = EquitiesGenerator._direction_onehot(frame)
            arrays[f"y_reg_{name}"] = EquitiesGenerator._regression_target(frame, params.regression_target)
            arrays[f"ticker_code_{name}"] = frame["ticker_code"].to_numpy(dtype=np.int32)
            arrays[f"date_{name}"] = EquitiesGenerator._dates_yyyymmdd(frame)
            # The three new dates ship as their own row-aligned YYYYMMDD arrays,
            # the same encoding as date_* -- 0 where unknown (no filing yet).
            for column, key in (("week52_high_date", "week52_high_date"), ("week52_low_date", "week52_low_date"), ("report_date", "report_date")):
                arrays[f"{key}_{name}"] = EquitiesGenerator._column_yyyymmdd(frame, column)

        arrays["ticker_vocab"] = np.array(vocab, dtype=np.str_)

        # APD-DATA-018: hand the route the permanent truncation annotation over
        # the reserved channel key, the same way _synthetic.py hands over
        # "scaling" and csv_import hands over its own. Popped BEFORE checksum +
        # NPZ persist, so the stored arrays stay array-only. Absent entirely
        # when nothing was cut.
        #
        # records_imported is filled in HERE and not at the cut, because rows
        # are not known until conditioning finishes -- and note it counts rows
        # that SURVIVED conditioning, so it is legitimately lower than
        # (symbols x sessions) when a ticker returns no data.
        if truncation is not None:
            truncation["records_imported"] = int(len(full))
            arrays[TRUNCATION_META_KEY] = truncation

        # The permanent data-quality annotation. Absent entirely when nothing is
        # wrong, so its presence alone answers "is anything degraded here".
        # DROP still annotates. The symbols are gone from the arrays, but a
        # dataset that quietly contains fewer symbols than were asked for is the
        # same silent-partial-data problem in a different costume -- the record of
        # WHAT was dropped is the whole point. Only `rows_affected` goes to zero,
        # because those rows are genuinely not in the dataset to be affected.
        quality_meta = build_data_quality_meta(
            unrescued=unrescued,
            degraded=degraded,
            rows_affected=0 if policy == INCOMPLETE_DROP else rows_affected,
            policy=policy,
        )
        if quality_meta is not None:
            arrays[DATA_QUALITY_META_KEY] = quality_meta

        return arrays

    # ------------------------------------------------------------------ #
    # Universe resolution                                                #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _load_constituents() -> dict[str, dict[str, Any]]:
        """Load the bundled S&P 500 constituents snapshot."""
        rows: dict[str, dict[str, Any]] = {}
        with open(_CONSTITUENTS_PATH, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                ticker = row["ticker"].strip().upper()
                cik_raw = (row.get("cik") or "").strip()
                rows[ticker] = {
                    "name": row.get("name", ticker).strip(),
                    "cik": int(cik_raw) if cik_raw.isdigit() else None,
                    "sector": (row.get("sector") or "").strip(),
                }
        return rows

    @staticmethod
    def _resolve_bounds(params: EquitiesParams) -> tuple[int, bool]:
        """Resolve the effective symbol cap and truncation opt-in for this request.

        **A request may only LOWER the cap, never raise it** -- the deployment
        value is a ceiling and the effective cap is the minimum of the two. This
        is the same rule ``csv_import`` applies to its byte cap, and for the same
        two reasons: a bound the bounded party can raise is not a bound, and a
        generated client that serialises schema defaults would otherwise send
        the schema's own ``max_symbols`` on every request and silently override
        a *lower* operator ceiling.

        ``max_symbols=None`` means "no request-side limit", not "unbounded" --
        the deployment ceiling still applies. There is deliberately no way for a
        caller to ask for an unbounded universe.

        ``allow_truncation`` is a logical OR: either the caller opts in for this
        request, or the deployment has opted in for every request. A client
        cannot opt *out* of the operator's choice.

        Returns:
            ``(cap_symbols, allow_truncation)``.
        """
        # Imported HERE, not at module scope, deliberately. juniper-data carries
        # a circular import that csv_import already sits inside: importing a
        # generator package runs its __init__ -> generator -> api.settings ->
        # api/__init__ -> app -> routes.generators -> back into the half-built
        # package. csv_import pays that cost at module scope and is therefore
        # un-runnable in isolation; there is no reason to add a second entry
        # point to the same cycle for one settings lookup.
        from juniper_data.api.settings import get_settings

        settings = get_settings()
        ceiling = settings.equities_max_symbols
        requested = params.max_symbols if params.max_symbols is not None else ceiling
        cap = min(requested, ceiling)
        allow = bool(params.allow_truncation or settings.equities_allow_truncation)
        return cap, allow

    @staticmethod
    def _resolve_incomplete_policy(params: EquitiesParams, has_unrescued: bool) -> str:
        """Decide fail / accept / drop for rows that no rescue path recovered.

        Two knobs, because the owner's spec needs two different shapes from one
        contract:

        * ``allow_truncation`` is the **gate** -- the same boolean that governs an
          over-cap universe, and settable the same three ways (request parameter,
          ``JUNIPER_DATA_EQUITIES_ALLOW_TRUNCATION``, matching ``.env`` entry).
          Unset means **fail**, which is the default and the safe direction: a
          dataset silently carrying a fabricated market cap is the failure this
          area exists to prevent.
        * ``incomplete_rows`` says what to do once the gate is open -- ``accept``
          the affected rows or ``drop`` them. An interactive consumer maps its
          three choices onto these two knobs; a command-line consumer that only
          sets the boolean gets ``accept``, which is the documented CLI behaviour.

        Returns ``INCOMPLETE_ACCEPT`` unchanged when nothing is wrong, so a clean
        dataset never depends on either knob.
        """
        if not has_unrescued:
            return INCOMPLETE_ACCEPT

        from juniper_data.api.settings import get_settings

        settings = get_settings()
        allowed = bool(params.allow_truncation or settings.equities_allow_truncation)
        if not allowed:
            return INCOMPLETE_FAIL
        choice = params.incomplete_rows or settings.equities_incomplete_rows
        return INCOMPLETE_DROP if choice == INCOMPLETE_DROP else INCOMPLETE_ACCEPT

    @staticmethod
    def bind_deployment_defaults(params: EquitiesParams) -> EquitiesParams:
        """Copy the effective symbol cap and truncation opt-in onto the params object.

        The equities half of the cache-key defect ``csv_import`` fixed for bytes.
        ``generate_dataset_id`` hashes ``params.model_dump()``, and dump fills Field
        defaults -- so an omitted ``max_symbols`` is stored as its schema default even
        when the deployment ceiling is tighter, and ``allow_truncation`` is stored as
        false even when the operator opted in globally. Two requests that will run under
        DIFFERENT policies therefore hash to the SAME dataset_id, and a restart that
        raises the ceiling keeps serving the artifact truncated under the old one.

        The create route calls this through a ``getattr`` hook, so a generator opts in
        simply by defining it; there is no registry to keep in step.

        Idempotent: ``_resolve_bounds`` clamps with ``min(requested, ceiling)`` and ORs
        the opt-in, so re-binding an already-bound params object returns the same values.
        """
        cap, allow = EquitiesGenerator._resolve_bounds(params)
        return params.model_copy(update={"max_symbols": cap, "allow_truncation": allow})

    @staticmethod
    def _resolve_symbols(params: EquitiesParams, constituents: dict[str, dict[str, Any]]) -> tuple[list[str], dict[str, dict[str, Any]], dict[str, Any] | None]:
        """Resolve the ticker list and the ticker -> (name, cik) metadata map."""
        if params.symbols:
            ordered = [symbol.strip().upper() for symbol in params.symbols if symbol.strip()]
            meta = {symbol: constituents[symbol] for symbol in ordered if symbol in constituents}
            missing = [symbol for symbol in ordered if symbol not in constituents]
            if missing:
                sec_map = EquitiesGenerator._load_sec_ticker_map()
                for symbol in missing:
                    match = sec_map.get(symbol) or sec_map.get(symbol.replace(".", "-"))
                    meta[symbol] = {"name": match["name"] if match else symbol, "cik": match["cik"] if match else None, "sector": ""}
        else:
            ordered = sorted(constituents)
            meta = constituents

        # APD-DATA-018. This was `ordered = ordered[: params.max_symbols]` -- a
        # bare slice that truncated SILENTLY, recorded nothing, and returned a
        # dataset indistinguishable from a complete one. The default was `None`,
        # so in practice it never fired and every request fanned out over all
        # 503 bundled constituents: 18-34 minutes against a 30 s budget.
        #
        # The cap is in SYMBOLS, not bytes, because measurement (2026-09-04)
        # showed the cost is per request. See juniper_data/core/limits.py.
        cap_symbols, allow_truncation = EquitiesGenerator._resolve_bounds(params)
        requested_count = len(ordered)

        if requested_count <= cap_symbols:
            return ordered, meta, None

        if not allow_truncation:
            raise InputTooLargeError(
                source="The requested universe",
                unit=UNIT_SYMBOLS,
                cap=cap_symbols,
                actual=requested_count,
                opt_in_env="JUNIPER_DATA_EQUITIES_ALLOW_TRUNCATION",
            )

        # Truncation is authorised. Which symbols survive is not arbitrary: the
        # list is already deterministically ordered (sorted constituents, or the
        # caller's own sequence), so the prefix is reproducible across runs
        # rather than depending on dict iteration or download completion order.
        kept = ordered[:cap_symbols]
        truncation = build_truncation_meta(
            reason=REASON_SYMBOL_CAP,
            unit=UNIT_SYMBOLS,
            cap=cap_symbols,
            requested=requested_count,
            imported=len(kept),
            # Rows are not known until conditioning finishes; generate() fills
            # this in. Recording 0 here would be a lie the caller could read.
            records_imported=-1,
        )
        return kept, meta, truncation

    @staticmethod
    def _load_sec_ticker_map() -> dict[str, dict[str, Any]]:
        """Fetch SEC's ticker -> {name, cik} map (cached) for unknown symbols."""
        cache = _CACHE_DIR / "company_tickers.json"
        data = None
        if cache.exists():
            try:
                data = json.loads(cache.read_text())
            except (OSError, json.JSONDecodeError):
                data = None
        if data is None:
            data = _sec_get(_SEC_TICKERS_URL)
            if data is not None:
                cache.parent.mkdir(parents=True, exist_ok=True)
                with contextlib.suppress(OSError):
                    cache.write_text(json.dumps(data))
        if not data:
            return {}
        return {row["ticker"].upper(): {"name": row.get("title", row["ticker"]), "cik": int(row["cik_str"])} for row in data.values()}

    # ------------------------------------------------------------------ #
    # Per-ticker conditioning                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _condition_one(ticker: str, info: dict[str, Any], params: EquitiesParams, end_date: str) -> Any:
        """Download and condition a single ticker into a record DataFrame.

        Returns None if no usable price data is available.
        """
        frame = EquitiesGenerator._download_ohlcv(ticker, params.start_date, end_date, params.use_cache)
        if frame is None or frame.empty:
            return None

        # yfinance omits the action columns entirely for a ticker with no
        # dividends or splits in range; absent means "none happened", which is
        # 0.0, not missing.
        for action_column in ("dividend", "split_ratio"):
            if action_column not in frame.columns:
                frame[action_column] = 0.0
        frame[["dividend", "split_ratio"]] = frame[["dividend", "split_ratio"]].fillna(0.0)
        if "adj_close" not in frame.columns:
            frame["adj_close"] = frame["close"]

        window = params.week52_window
        frame["week52_high"] = frame["high"].rolling(window, min_periods=1).max()
        frame["week52_low"] = frame["low"].rolling(window, min_periods=1).min()

        # WHEN the 52-week extreme happened, not just what it was. Free: it comes
        # from the window already being computed one line above.
        high_at = EquitiesGenerator._rolling_extreme_positions(frame["high"].to_numpy(dtype="float64"), window, take_max=True)
        low_at = EquitiesGenerator._rolling_extreme_positions(frame["low"].to_numpy(dtype="float64"), window, take_min=True)
        positions = np.arange(len(frame))
        frame["week52_high_date"] = frame.index[high_at]
        frame["week52_low_date"] = frame.index[low_at]
        # Days SINCE is the model-usable form -- a raw YYYYMMDD in a float32
        # feature column is a number whose magnitude means nothing. The dates
        # themselves ship as their own row-aligned arrays.
        frame["days_since_week52_high"] = (positions - high_at).astype("float64")
        frame["days_since_week52_low"] = (positions - low_at).astype("float64")

        cik = info.get("cik")
        shares = None
        if cik:
            try:
                shares = EquitiesGenerator._fetch_shares(cik, params.use_cache)
            except Exception as exc:  # noqa: BLE001 - shares are best-effort; never drop price data over a SEC blip
                _logger.warning("equities: shares fetch failed for %s (cik=%s): %s", ticker, cik, exc)
        if shares is not None and not shares.empty:
            # ALIGN ON THE FILING DATE, NOT THE PERIOD END.
            #
            # This used to reindex on the period-end index and forward-fill, which
            # is a look-ahead leak: every trade date between a figure's `end` and
            # the filing that disclosed it was handed a share count that was not
            # public yet -- and `market_cap`, derived from it, inherited the leak.
            # Measured on a live 2013-2021 AAPL run, `days_since_report` came back
            # as low as **-19 days**: a negative age is the leak stating itself
            # out loud. Over the whole AAPL series 325 of 2,266 rows (14.3%)
            # carried a negative age; after this change, zero do.
            #
            # `end` ON THIS TAG IS AN AS-OF DATE, NOT A FISCAL PERIOD END --
            # SEC's own description is "as stated on cover of related periodic
            # report". An earlier version of this comment claimed AAPL's quarter
            # ending 2021-03-27 was not filed until 2021-04-29, "five weeks". That
            # is wrong twice: AAPL's dei series has no 2021-03 point at all, and
            # the 2021-04-29 filing carries `end=2021-04-16` -- a 13-day gap. The
            # -19 comes from 2015-2016 (four filings tie at 19 days); 2021's widest
            # gap is 14. The genuine outlier is a 10-K/A: `end=2009-10-16` filed
            # 2010-01-25, **101 days**. The leak was real; only the example was not.
            #
            # It is the same class as the normalisation leak fixed in
            # juniper-data#314, and the same rule from decision 7 of the ecosystem
            # partition design: no quantity that was not knowable at a row's date
            # may reach that row.
            #
            # Points with no `filed` are DROPPED rather than fallen back to their
            # period end. When a figure became public is exactly what is unknown
            # for them, and guessing reinstates the leak; if that empties the
            # series the ticker simply has no shares data, which
            # `fundamentals_fill` already handles.
            # BREAK SAME-DAY TIES ON THE PERIOD END, EXPLICITLY.
            #
            # Two facts can share one `filed`: an 8-K restating an old quarter
            # lands the same day as the current 10-Q. Only one survives
            # `duplicated(keep="last")`, and the right one is the LATER period --
            # it is what "shares outstanding" means on that filing date.
            #
            # This used to be `set_index("filed").sort_index()`, which got the
            # right answer only by accident: rows arrive `end`-ascending from
            # `_fetch_shares`, so `keep="last"` picked the later period ONLY if
            # the re-sort preserved that order among ties. `sort_index()` defaults
            # to `kind="quicksort"`, which is not stable, so it did not. Measured
            # over the 485-payload SEC cache: 54 tickers have such a collision and
            # **15 of them, across 9 tickers, kept the restated OLD figure** --
            # DVA by 10.4% (2013-03-01), O'Reilly by 6.8%, KO by 0.9%, and ADSK --
            # which sits at position 12 of the default 14-symbol universe -- by
            # 0.26%. Every one of those is a silently wrong `market_cap`.
            #
            # Sorting on (filed, end) states the tie-break instead of inheriting
            # it. pandas resolves a multi-column `sort_values` with `np.lexsort`,
            # which is stable regardless of `kind`, so this no longer depends on
            # an upstream ordering invariant that a refactor could quietly drop.
            known = shares.dropna(subset=["filed"])
            known = known.rename_axis("end").reset_index().sort_values(["filed", "end"]).set_index("filed")
            known = known[~known.index.duplicated(keep="last")]

            frame["shares_quality"] = str(shares["shares_quality"].iloc[0]) if "shares_quality" in shares.columns else SHARES_QUALITY_POINT_IN_TIME
            frame["shares_origin"] = str(shares["shares_origin"].iloc[0]) if "shares_origin" in shares.columns else SHARES_SOURCE_CONCEPT
            if len(known):
                union = frame.index.union(known.index)
                frame["total_shares"] = known["shares"].reindex(union).sort_index().ffill().reindex(frame.index).astype("float64")
                as_of = pd.Series(known.index, index=known.index).reindex(union).sort_index().ffill().reindex(frame.index)
                frame["report_date"] = pd.to_datetime(as_of)
            else:
                frame["total_shares"] = np.nan
                frame["report_date"] = pd.NaT
        else:
            frame["total_shares"] = np.nan
            frame["report_date"] = pd.NaT
            frame["shares_quality"] = SHARES_QUALITY_UNRESCUED
            frame["shares_origin"] = SHARES_QUALITY_UNRESCUED

        if frame["total_shares"].isna().all():
            frame["shares_quality"] = SHARES_QUALITY_UNRESCUED
            frame["shares_origin"] = SHARES_QUALITY_UNRESCUED
            # SAY SO. Under the default fundamentals_fill="zero" this ticker's
            # total_shares and market_cap become 0.0 for every row -- a value no
            # listed company can have, and one nothing downstream distinguishes
            # from a measurement. Roughly 4-6% of the bundled S&P 500 universe
            # reports no shares concept to SEC at all (KO and ABT among them);
            # before this line, that produced a silently zero-filled feature
            # column and no signal anywhere.
            _logger.warning("equities: %s has NO shares-outstanding data from SEC; total_shares/market_cap will be filled per fundamentals_fill=%r", ticker, params.fundamentals_fill)

        frame["market_cap"] = frame["close"] * frame["total_shares"]

        if params.fundamentals_fill == "zero":
            frame[["total_shares", "market_cap"]] = frame[["total_shares", "market_cap"]].fillna(0.0)
        elif params.fundamentals_fill == "drop":
            frame = frame.dropna(subset=["total_shares"])

        # Days since the most recent filing. NaN where no filing precedes the row
        # (pre-2009 for most names, and every row for a ticker SEC returns nothing
        # for) -- filled per the same fundamentals_fill policy as total_shares,
        # because it is missing for exactly the same reason.
        age = (frame.index - frame["report_date"]).dt.days if frame["report_date"].notna().any() else pd.Series(np.nan, index=frame.index)
        frame["days_since_report"] = pd.to_numeric(age, errors="coerce").astype("float64")
        if params.fundamentals_fill == "zero":
            frame["days_since_report"] = frame["days_since_report"].fillna(0.0)

        basis_field = params.basis_price_field if params.basis_price_field in frame.columns else "close"
        purchase = pd.to_datetime(params.purchase_date)
        on_or_before = frame.loc[frame.index <= purchase]
        basis = float(on_or_before[basis_field].iloc[-1]) if len(on_or_before) else float(frame[basis_field].iloc[0])
        frame["cost_basis"] = basis

        frame["name"] = info.get("name", ticker)
        frame["ticker"] = ticker

        frame["next_close"] = frame["close"].shift(-1)
        frame = frame.iloc[:-1]  # last row has no next-day target
        frame = frame.dropna(subset=["open", "high", "low", "close", "volume", "next_close"])
        if frame.empty:
            return None
        frame["direction_up"] = (frame["next_close"] > frame["close"]).astype(np.float32)
        return frame

    @staticmethod
    def _download_ohlcv(ticker: str, start: str, end: str, use_cache: bool) -> Any:
        """Download daily OHLCV for a ticker (cached), normalized to lower-snake columns."""
        cache = _CACHE_DIR / "ohlcv" / f"{ticker}_{start}_{end}.csv"
        if use_cache and cache.exists():
            with contextlib.suppress(Exception):  # a corrupt cache must never be fatal
                cached = pd.read_csv(cache, index_col=0)
                cached.index = pd.to_datetime(cached.index)
                return cached

        # yfinance uses dashes for class shares (BRK.B -> BRK-B).
        # ``actions=True`` adds Dividends and Stock Splits to the SAME response --
        # no extra request, no extra latency. Verified against AAPL: 7:1 on
        # 2014-06-09 and 4:1 on 2020-08-31.
        downloaded = yf.download(ticker.replace(".", "-"), start=start, end=end, interval="1d", auto_adjust=False, actions=True, progress=False, threads=False)
        if downloaded is None or len(downloaded) == 0:
            return None
        frame = EquitiesGenerator._normalize_ohlcv_columns(downloaded)
        if frame.empty:
            return None

        if use_cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            with contextlib.suppress(Exception):  # cache write is best-effort
                frame.to_csv(cache)
        return frame

    @staticmethod
    def _normalize_ohlcv_columns(frame: Any) -> Any:
        """Flatten any MultiIndex columns and rename to lower-snake OHLCV."""
        if isinstance(frame.columns, pd.MultiIndex):
            frame = frame.copy()
            frame.columns = frame.columns.get_level_values(0)
        rename = {"Open": "open", "High": "high", "Low": "low", "Close": "close", "Adj Close": "adj_close", "Volume": "volume", "Dividends": "dividend", "Stock Splits": "split_ratio"}
        frame = frame.rename(columns=rename)
        keep = [column for column in ["open", "high", "low", "close", "adj_close", "volume", "dividend", "split_ratio"] if column in frame.columns]
        frame = frame[keep].copy()
        frame.index = pd.to_datetime(frame.index)
        return frame

    @staticmethod
    def _fetch_shares_from_facts(cik: int) -> tuple[dict[str, Any] | None, str]:
        """Rescue rung: find a shares series in ``companyfacts``.

        ``companyconcept`` returns a present-but-EMPTY concept for a substantial
        minority of filers -- 37 of the 503 bundled S&P 500 constituents -- while
        ``companyfacts`` carries the identical concept, populated, for 18 of them.
        KO is the worked example: ``companyconcept`` says 0 facts,
        ``companyfacts`` says 71, current count 4.30 billion shares. Same CIK,
        same taxonomy, same tag.

        Walks :data:`_SHARES_FACTS_LADDER` in order and returns the first rung
        with facts, reshaped to look exactly like a ``companyconcept`` payload so
        the caller's parsing is unchanged.

        Returns:
            ``(payload_or_None, quality)`` where ``quality`` distinguishes a
            point-in-time count from a period average -- the two are not the same
            quantity and a market cap built on each means something different.
        """
        payload = _sec_get(_SEC_FACTS_URL.format(cik=int(cik)))
        if not payload or not payload.get("facts"):
            return None, SHARES_QUALITY_POINT_IN_TIME

        for taxonomy, tag, quality in _SHARES_FACTS_LADDER:
            entry = payload["facts"].get(taxonomy, {}).get(tag)
            if not entry or not any(entry.get("units", {}).values()):
                continue
            if quality != SHARES_QUALITY_POINT_IN_TIME:
                _logger.warning(
                    "equities: cik=%s has no point-in-time shares-outstanding concept; falling back to %s/%s (%s) -- market_cap for this symbol is not directly comparable with the others, and the dataset is annotated as degraded",
                    cik,
                    taxonomy,
                    tag,
                    "a PERIOD AVERAGE, not a point-in-time count" if quality == SHARES_QUALITY_PERIOD_AVERAGE else "shares ISSUED, which includes treasury stock and so exceeds shares outstanding",
                )
            return {"units": entry["units"]}, quality
        return None, SHARES_QUALITY_POINT_IN_TIME

    @staticmethod
    def _fetch_shares(cik: int, use_cache: bool) -> Any:
        """Fetch shares outstanding AND their filing dates from SEC EDGAR.

        Returns a ``DataFrame`` indexed by period-end date with two columns:

        * ``shares`` -- the outstanding share count (what this always returned).
        * ``filed`` -- the date SEC received the filing that reported it.

        ``filed`` is free: it is already in every fact of the payload this method
        downloads, and was previously used only to pick the latest filing per
        period-end and then discarded. Surfacing it is what makes the caller's
        "reporting date" field cost no extra request.

        The distinction between the two dates matters and is easy to lose: the
        index is the period the figure DESCRIBES, ``filed`` is when it became
        publicly knowable. Only the second is safe to condition on at a given
        trade date -- using the period end would leak, sometimes by months.
        """
        cache = _CACHE_DIR / "shares" / f"{int(cik):010d}.json"
        data = None
        if use_cache and cache.exists():
            try:
                data = json.loads(cache.read_text())
            except (OSError, json.JSONDecodeError):
                data = None
        quality = SHARES_QUALITY_POINT_IN_TIME
        origin = SHARES_SOURCE_CONCEPT
        if data is None:
            for taxonomy, tag in _SHARES_CONCEPTS:
                payload = _sec_get(_SEC_CONCEPT_URL.format(cik=int(cik), taxonomy=taxonomy, tag=tag))
                # ACCEPT ONLY A CONCEPT THAT ACTUALLY HAS FACTS.
                #
                # This was `if payload and payload.get("units")`, and SEC returns
                # a present-but-EMPTY concept for some filers: KO, ABT and others
                # answer 200 with ``{"units": {"shares": {}}}``. That dict is
                # truthy, so the loop accepted it and **broke before trying the
                # us-gaap fallback** -- which for BIIB holds 42 perfectly good
                # facts. The ticker then got no shares at all, and
                # ``fundamentals_fill="zero"`` turned that into a total_shares of
                # 0.0 and a market_cap of 0.0, indistinguishable downstream from
                # a real measurement.
                #
                # Truthiness is the wrong test for "has data" whenever the API can
                # return an empty container; count the facts instead.
                if payload and any(payload.get("units", {}).values()):
                    data = payload
                    break

            if data is None:
                # RESCUE RUNG. companyconcept found nothing; companyfacts often
                # has the very same concept populated. Costs ~1.15 s and ~5 MB,
                # which is why it runs only here.
                data, quality = EquitiesGenerator._fetch_shares_from_facts(cik)
                origin = SHARES_SOURCE_FACTS if data is not None else origin

            if data and use_cache:
                cache.parent.mkdir(parents=True, exist_ok=True)
                with contextlib.suppress(OSError):
                    cache.write_text(json.dumps(data))
        if not data or not any(data.get("units", {}).values()):
            return None

        # Keep the latest-filed value per period-end date, and the filing date
        # that supplied it -- the sort key already orders by (end, filed), so the
        # last write per end date wins and both facts come from the same point.
        best: dict[str, float] = {}
        filed_on: dict[str, str] = {}
        for unit_points in data["units"].values():
            for point in sorted(unit_points, key=lambda item: (item.get("end", ""), item.get("filed", ""))):
                if point.get("val") is not None and point.get("end"):
                    best[point["end"]] = float(point["val"])
                    if point.get("filed"):
                        filed_on[point["end"]] = point["filed"]
        if not best:
            return None
        series = pd.Series(best)
        series.index = pd.to_datetime(series.index)
        series = series.sort_index()
        # Drop XBRL filer scale errors (isolated points ~1e6x off): forward-fill
        # then carries the last good value across the dropped point. Median is
        # robust to the minority of bad points.
        median = float(series.median())
        if median > 0:
            series = series[(series >= median / _SHARES_OUTLIER_FACTOR) & (series <= median * _SHARES_OUTLIER_FACTOR)]
        if not len(series):
            return None

        frame = series.to_frame(name="shares")
        # Provenance rides ALONG with the values, as a constant column, so it
        # survives every reindex/ffill downstream without a second return value
        # or a fragile ``.attrs``. A consumer must be able to tell a market cap
        # built on point-in-time shares from one built on a period average.
        frame["shares_quality"] = quality
        frame["shares_origin"] = origin
        # A point with no ``filed`` (rare, older filings) becomes NaT rather than
        # a guess -- the consumer sees "unknown", not a fabricated date.
        frame["filed"] = pd.to_datetime(pd.Series({pd.Timestamp(end): filed_on.get(end) for end in best}, dtype="object")).reindex(frame.index)
        return frame

    # ------------------------------------------------------------------ #
    # Array assembly                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _fit_normalizer(full: Any) -> tuple[np.ndarray, np.ndarray]:
        """Compute per-feature min and range over the full matrix for [0, 1] scaling."""
        matrix = EquitiesGenerator._raw_features(full)
        minimum = np.nanmin(matrix, axis=0, keepdims=True)
        maximum = np.nanmax(matrix, axis=0, keepdims=True)
        span = maximum - minimum
        span[span == 0] = 1.0
        return minimum.astype(np.float32), span.astype(np.float32)

    @staticmethod
    def _raw_features(frame: Any) -> np.ndarray:
        """Stack the ordered feature columns into a float32 (n, 10) matrix."""
        return np.column_stack([frame[column].to_numpy(dtype=np.float32) for column in EQUITIES_FEATURE_COLUMNS])

    @staticmethod
    def _features(frame: Any, norm: tuple[np.ndarray, np.ndarray] | None) -> np.ndarray:
        """Build the (optionally normalized) feature matrix for a frame."""
        if frame.empty:
            return np.zeros((0, len(EQUITIES_FEATURE_COLUMNS)), dtype=np.float32)
        matrix = EquitiesGenerator._raw_features(frame)
        if norm is not None:
            minimum, span = norm
            matrix = (matrix - minimum) / span
        return matrix.astype(np.float32)

    @staticmethod
    def _direction_onehot(frame: Any) -> np.ndarray:
        """One-hot encode next-day direction as [down, up] (n, 2) float32."""
        if frame.empty:
            return np.zeros((0, 2), dtype=np.float32)
        up = frame["direction_up"].to_numpy(dtype=np.float32)
        onehot = np.zeros((len(up), 2), dtype=np.float32)
        onehot[:, 1] = up
        onehot[:, 0] = 1.0 - up
        return onehot

    @staticmethod
    def _regression_target(frame: Any, mode: str) -> np.ndarray:
        """Build the ``(n, 1)`` float32 ``y_reg`` regression target.

        ``next_close`` is the raw next-day close (non-stationary; the original
        target, preserved byte-for-byte). ``return`` is the simple next-day
        return ``next_close / close - 1`` and ``log_return`` is
        ``ln(next_close / close)`` -- both stationary, the standard conditioning
        for a regressor on trending price data. Computed in float64 then cast to
        the contract's float32.
        """
        if frame.empty:
            return np.zeros((0, 1), dtype=np.float32)
        next_close = frame["next_close"].to_numpy(dtype=np.float64)
        if mode == "next_close":
            values = next_close
        else:
            close = frame["close"].to_numpy(dtype=np.float64)
            ratio = next_close / close
            values = np.log(ratio) if mode == "log_return" else ratio - 1.0
        return values.astype(np.float32).reshape(-1, 1)

    @staticmethod
    def _rolling_extreme_positions(values: np.ndarray, window: int, *, take_max: bool = False, take_min: bool = False) -> np.ndarray:
        """Index of the max/min within each trailing window, as absolute positions.

        Returns, for every row i, the position of the extreme value in
        ``values[max(0, i-window+1) : i+1]`` -- the same window
        ``Series.rolling(window, min_periods=1)`` uses, so the position this
        returns always points at the value ``week52_high`` / ``week52_low``
        reports.

        Uses a strided view rather than ``rolling(...).apply()``: the latter is a
        Python call per window, which on 6,708 rows x a 252-day window is ~1.7M
        invocations per ticker. The per-symbol budget here is ~2 s of network, so
        seconds of avoidable compute would be a real regression.

        The front pad carries -inf (or +inf) so a padded slot can never win the
        comparison, which makes the ramp-up rows -- where fewer than ``window``
        observations exist -- fall out without a special case.
        """
        if take_max == take_min:
            raise ValueError("_rolling_extreme_positions needs exactly one of take_max / take_min")
        count = len(values)
        if count == 0:
            return np.zeros((0,), dtype=np.int64)
        span = max(1, min(window, count))
        fill = -np.inf if take_max else np.inf
        padded = np.concatenate([np.full(span - 1, fill, dtype="float64"), values])
        windows = np.lib.stride_tricks.sliding_window_view(padded, span)
        offsets = windows.argmax(axis=1) if take_max else windows.argmin(axis=1)
        return np.arange(count) - (span - 1) + offsets

    @staticmethod
    def _dates_yyyymmdd(frame: Any) -> np.ndarray:
        """Row-aligned trade dates encoded as YYYYMMDD int32."""
        if frame.empty:
            return np.zeros((0,), dtype=np.int32)
        return frame.index.strftime("%Y%m%d").astype(np.int32).to_numpy()

    @staticmethod
    def _column_yyyymmdd(frame: Any, column: str) -> np.ndarray:
        """A date COLUMN encoded as YYYYMMDD int32, with 0 for unknown.

        0 rather than a sentinel date: it is out of range for any real trade date,
        so it cannot be mistaken for one, and it survives the int32 round-trip
        that a NaT would not.
        """
        if frame.empty or column not in frame.columns:
            return np.zeros((0 if frame.empty else len(frame),), dtype=np.int32)
        values = pd.to_datetime(frame[column], errors="coerce")
        encoded = values.dt.strftime("%Y%m%d")
        return pd.to_numeric(encoded, errors="coerce").fillna(0).astype(np.int32).to_numpy()


def get_schema() -> dict:
    """Return the JSON schema describing the generator parameters."""
    return EquitiesParams.model_json_schema()
