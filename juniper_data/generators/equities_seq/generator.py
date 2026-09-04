"""Windowed (3-D sequence) equities generator -- irregular-Δt time-series variant.

A sibling of the flat 2-D ``equities`` generator (#164). It reuses
``EquitiesGenerator``'s data pipeline -- universe resolution, per-ticker
conditioning, and feature normalization -- unchanged, then windows each ticker's
rows into fixed-length lookback sequences via ``window_one_ticker`` and
concatenates across tickers. The flat 2-D ``equities`` generator is untouched
(its artifact stays byte-identical).

The result is the additive 3-D NPZ contract (WS-1 / juniper-data#168): per split
``X`` is ``(W, L, F)``; each window carries its per-step elapsed time ``dt``
(calendar days derived from the trading-day dates -- weekend/holiday gaps ARE the
irregular Δt), the irregular forecast horizon ``target_dt``, an all-ones
``observed_mask`` (trading-day-native, nothing imputed), the per-step ``date``
and the per-window ``window_end_date`` / ``ticker_code``, plus the targets
``y`` (one-hot next-day direction) and ``y_reg`` (the configurable next-day
regression target -- raw close / return / log-return, per ``regression_target``).
``full`` is each ticker's train windows followed by its test windows.

See ``juniper-ml/notes/JUNIPER_2026-06-05_JUNIPER-RECURRENCE_RECURSE-DELTA-T-HANDLING.md`` §3
(schema delta) and §6 (the dt / observed_mask contract).
"""

# Project:       Juniper
# Sub-Project:   JuniperData
# Application:   juniper_data
# File Name:     generator.py
# Author:        Paul Calnon
# Version:       0.6.0
# License:       MIT License

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

import numpy as np

try:
    import pandas as pd
except ImportError:  # pragma: no cover - exercised only without the equities extra
    pd = None  # type: ignore[assignment]

from juniper_data.core.split import temporal_split_index
from juniper_data.generators._sequence import _yyyymmdd_to_ordinal, window_one_ticker
from juniper_data.generators.equities.generator import EQUITIES_DEPS_AVAILABLE, EquitiesGenerator

from .params import EquitiesSeqParams

VERSION = "1.0.0"

_logger = logging.getLogger(__name__)

# Per-window keys produced by ``window_one_ticker`` (also the NPZ key stems).
_WINDOW_KEYS = ("X", "y", "y_reg", "date", "dt", "target_dt", "window_end_date", "ticker_code", "observed_mask")


class EquitiesSeqGenerator:
    """Generator for windowed (3-D) S&P 500 equities sequence datasets.

    All methods are static (stateless, side-effect free aside from the shared
    on-disk download cache reused from ``EquitiesGenerator``).
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
    def generate(params: EquitiesSeqParams) -> dict[str, np.ndarray]:
        """Generate the windowed equities sequence dataset.

        Returns the additive 3-D NPZ contract for train/test/full: ``X_{split}``
        ``(W, L, F)`` plus ``dt`` / ``target_dt`` / ``observed_mask`` / ``date`` /
        ``window_end_date`` / ``ticker_code`` and the one-hot ``y_{split}`` +
        regression ``y_reg_{split}`` targets, with a code -> ticker
        ``ticker_vocab``.

        Args:
            params: ``EquitiesSeqParams`` (equities config + ``lookback``).

        Raises:
            ImportError: if the optional ``equities`` extra is not installed.
            ValueError: if no symbol yielded data, or none had more than
                ``lookback + 1`` rows (so no window could be built).
        """
        if not EQUITIES_DEPS_AVAILABLE:
            raise ImportError(EquitiesSeqGenerator.install_hint())

        # Reuse the flat generator's data pipeline (intentional internal reuse of
        # the sibling generator -- keeps a single source of truth for fetching,
        # conditioning, and normalization; the 2-D generator is not modified).
        constituents = EquitiesGenerator._load_constituents()
        symbols, meta_map = EquitiesGenerator._resolve_symbols(params, constituents)
        end_date = params.end_date or datetime.now(UTC).strftime("%Y-%m-%d")

        conditioned: dict[str, Any] = {}
        total = len(symbols)
        for index, ticker in enumerate(symbols, start=1):
            try:
                frame = EquitiesGenerator._condition_one(ticker, meta_map.get(ticker, {}), params, end_date)
            except Exception as exc:  # noqa: BLE001 - one bad ticker must not abort the whole batch
                _logger.warning("equities_seq: skipping %s (%s)", ticker, exc)
                continue
            if frame is not None and not frame.empty:
                conditioned[ticker] = frame.sort_index()
                _logger.info("equities_seq: [%d/%d] %s -> %d rows", index, total, ticker, len(frame))
            else:
                _logger.info("equities_seq: [%d/%d] %s -> no data", index, total, ticker)

        if not conditioned:
            raise ValueError("No data could be retrieved for the requested symbols.")

        vocab = sorted(conditioned)
        code_of = {ticker: code for code, ticker in enumerate(vocab)}

        # Fit normalization on the TRAINING rows across all tickers (juniper-data#314),
        # the same statistic the flat generator uses -- which was also fixed there.
        #
        # This previously fit on the concatenated FULL frames, including each ticker's
        # chronologically-later test rows, and then applied those statistics to the training
        # windows: look-ahead leakage. The per-ticker split boundary is already computed below
        # as ``temporal_split_index(n_rows, params.train_ratio)``, so the training rows are
        # available here and the same boundary is reused rather than re-derived.
        #
        # CONSEQUENCE, deliberate: test windows are no longer bounded by [0, 1]. Later rows
        # legitimately exceed the training range, and that excursion is real signal rather
        # than something to normalise away.
        norm = None
        if params.normalize_features:
            train_frames = []
            for ticker in vocab:
                frame = conditioned[ticker]
                cut = temporal_split_index(len(frame), params.train_ratio)
                if cut > 0:
                    train_frames.append(frame.iloc[:cut])
            # Every ticker rounding to zero training rows leaves nothing to fit on; fall back
            # to the full frames rather than emitting all-NaN statistics.
            fit_frames = train_frames if train_frames else [conditioned[ticker] for ticker in vocab]
            norm = EquitiesGenerator._fit_normalizer(pd.concat(fit_frames))

        per_ticker: list[dict[str, dict[str, np.ndarray]]] = []
        for ticker in vocab:
            frame = conditioned[ticker]
            n_rows = len(frame)
            if n_rows <= params.lookback + 1:
                _logger.info("equities_seq: %s has %d rows (need > lookback+1=%d) -> no windows", ticker, n_rows, params.lookback + 1)
                continue
            feats = EquitiesGenerator._features(frame, norm)
            dates = EquitiesGenerator._dates_yyyymmdd(frame)
            y_dir = EquitiesGenerator._direction_onehot(frame)
            y_reg = EquitiesGenerator._regression_target(frame, params.regression_target)
            ords = _yyyymmdd_to_ordinal(dates)
            cut_ordinal = int(ords[temporal_split_index(n_rows, params.train_ratio)])
            out = window_one_ticker(
                feats,
                dates,
                y_dir,
                y_reg,
                code_of[ticker],
                lookback=params.lookback,
                cut_ordinal=cut_ordinal,
            )
            per_ticker.append(out)

        if not per_ticker:
            raise ValueError(f"No symbol had more than lookback+1={params.lookback + 1} rows; cannot build any window.")

        arrays = EquitiesSeqGenerator._assemble(per_ticker)
        arrays["ticker_vocab"] = np.array(vocab, dtype=np.str_)
        return arrays

    @staticmethod
    def _assemble(per_ticker: list[dict[str, dict[str, np.ndarray]]]) -> dict[str, np.ndarray]:
        """Concatenate per-ticker windows into the train/test/full NPZ arrays.

        ``full`` is each ticker's train windows followed by its test windows
        (chronological within ticker), concatenated across tickers -- so
        ``full`` == ``train`` + ``test`` with no dropped or duplicated window.
        """
        arrays: dict[str, np.ndarray] = {}
        for split in ("train", "test"):
            for key in _WINDOW_KEYS:
                arrays[f"{key}_{split}"] = np.concatenate([out[split][key] for out in per_ticker], axis=0)
        for key in _WINDOW_KEYS:
            blocks: list[np.ndarray] = []
            for out in per_ticker:
                blocks.append(out["train"][key])
                blocks.append(out["test"][key])
            arrays[f"{key}_full"] = np.concatenate(blocks, axis=0)
        return arrays


def get_schema() -> dict:
    """Return the JSON schema describing the generator parameters."""
    return EquitiesSeqParams.model_json_schema()
