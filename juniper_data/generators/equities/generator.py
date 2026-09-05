"""Equities time-series dataset generator (S&P 500, 2000 -> present).

Downloads daily OHLCV from Yahoo Finance (via ``yfinance``) and shares-
outstanding history from SEC EDGAR (XBRL company-concept API), conditions them
into a per-(ticker, day) record table -- name, ticker, date, OHLCV, 52-week
high/low, total shares, market capitalization, and a configurable-purchase-date
cost basis -- and formats the result into the JuniperData NPZ contract.

Targets (per the dataset spec) are dual:
  * canonical one-hot **next-day direction** (up/down) in ``y_*`` -- keeps
    ``n_classes == 2`` and the route's ``argmax(y_full)`` class distribution
    correct;
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

from juniper_data.core.limits import REASON_SYMBOL_CAP, TRUNCATION_META_KEY, UNIT_SYMBOLS, InputTooLargeError, build_truncation_meta

from .defaults import CONSTITUENTS_FILENAME, EQUITIES_FEATURE_COLUMNS
from .params import EquitiesParams

VERSION = "1.0.0"

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
            Dictionary with the canonical NPZ keys (X_train, y_train, X_test,
            y_test, X_full, y_full) plus auxiliary arrays: y_reg_* (next-day
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

        vocab = sorted(conditioned)
        code_of = {ticker: code for code, ticker in enumerate(vocab)}

        train_frames, test_frames, full_frames = [], [], []
        for ticker in vocab:
            frame = conditioned[ticker].sort_index()
            frame["ticker_code"] = code_of[ticker]
            n_rows = len(frame)
            n_train = int(round(n_rows * params.train_ratio))
            n_test = int(round(n_rows * params.test_ratio))
            if n_train + n_test > n_rows:
                n_test = n_rows - n_train
            train_frames.append(frame.iloc[:n_train])
            test_frames.append(frame.iloc[n_train : n_train + n_test])
            full_frames.append(frame)

        full = pd.concat(full_frames)
        train = pd.concat(train_frames) if train_frames else full.iloc[:0]
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
        # CONSEQUENCE, deliberate: ``X_full`` and ``X_test`` are no longer guaranteed to lie
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
        for name, frame in (("full", full), ("train", train), ("test", test)):
            features = EquitiesGenerator._features(frame, norm)
            arrays[f"X_{name}"] = features
            arrays[f"y_{name}"] = EquitiesGenerator._direction_onehot(frame)
            arrays[f"y_reg_{name}"] = EquitiesGenerator._regression_target(frame, params.regression_target)
            arrays[f"ticker_code_{name}"] = frame["ticker_code"].to_numpy(dtype=np.int32)
            arrays[f"date_{name}"] = EquitiesGenerator._dates_yyyymmdd(frame)

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

        window = params.week52_window
        frame["week52_high"] = frame["high"].rolling(window, min_periods=1).max()
        frame["week52_low"] = frame["low"].rolling(window, min_periods=1).min()

        cik = info.get("cik")
        shares = None
        if cik:
            try:
                shares = EquitiesGenerator._fetch_shares(cik, params.use_cache)
            except Exception as exc:  # noqa: BLE001 - shares are best-effort; never drop price data over a SEC blip
                _logger.warning("equities: shares fetch failed for %s (cik=%s): %s", ticker, cik, exc)
        if shares is not None and not shares.empty:
            aligned = shares.reindex(frame.index.union(shares.index)).sort_index().ffill().reindex(frame.index)
            frame["total_shares"] = aligned.astype("float64")
        else:
            frame["total_shares"] = np.nan
        frame["market_cap"] = frame["close"] * frame["total_shares"]

        if params.fundamentals_fill == "zero":
            frame[["total_shares", "market_cap"]] = frame[["total_shares", "market_cap"]].fillna(0.0)
        elif params.fundamentals_fill == "drop":
            frame = frame.dropna(subset=["total_shares"])

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
        downloaded = yf.download(ticker.replace(".", "-"), start=start, end=end, interval="1d", auto_adjust=False, progress=False, threads=False)
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
        rename = {"Open": "open", "High": "high", "Low": "low", "Close": "close", "Adj Close": "adj_close", "Volume": "volume"}
        frame = frame.rename(columns=rename)
        keep = [column for column in ["open", "high", "low", "close", "adj_close", "volume"] if column in frame.columns]
        frame = frame[keep].copy()
        frame.index = pd.to_datetime(frame.index)
        return frame

    @staticmethod
    def _fetch_shares(cik: int, use_cache: bool) -> Any:
        """Fetch a shares-outstanding time series (Series[date -> shares]) from SEC EDGAR."""
        cache = _CACHE_DIR / "shares" / f"{int(cik):010d}.json"
        data = None
        if use_cache and cache.exists():
            try:
                data = json.loads(cache.read_text())
            except (OSError, json.JSONDecodeError):
                data = None
        if data is None:
            for taxonomy, tag in _SHARES_CONCEPTS:
                payload = _sec_get(_SEC_CONCEPT_URL.format(cik=int(cik), taxonomy=taxonomy, tag=tag))
                if payload and payload.get("units"):
                    data = payload
                    break
            if data and use_cache:
                cache.parent.mkdir(parents=True, exist_ok=True)
                with contextlib.suppress(OSError):
                    cache.write_text(json.dumps(data))
        if not data or not data.get("units"):
            return None

        # Keep the latest-filed value per period-end date.
        best: dict[str, float] = {}
        for unit_points in data["units"].values():
            for point in sorted(unit_points, key=lambda item: (item.get("end", ""), item.get("filed", ""))):
                if point.get("val") is not None and point.get("end"):
                    best[point["end"]] = float(point["val"])
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
        return series if len(series) else None

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
    def _dates_yyyymmdd(frame: Any) -> np.ndarray:
        """Row-aligned trade dates encoded as YYYYMMDD int32."""
        if frame.empty:
            return np.zeros((0,), dtype=np.int32)
        return frame.index.strftime("%Y%m%d").astype(np.int32).to_numpy()


def get_schema() -> dict:
    """Return the JSON schema describing the generator parameters."""
    return EquitiesParams.model_json_schema()
