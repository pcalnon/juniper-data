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

Recurrence-ready request (W1.1(a))
----------------------------------
juniper-recurrence cannot train on the bare defaults. Send the dataset half of the bundle
explicitly to ``POST /v1/datasets``::

    {"generator": "equities_seq",
     "params": {"symbols": ["AAPL"], "fundamentals_fill": "drop",
                "normalize_features": false, "regression_target": "log_return"}}

``symbols`` is an explicit list within the 14-symbol cap: omitting it asks for the 503 bundled
S&P 500 names, which the cap refuses. ``normalize_features: false`` is also the default and
hashes to the same ``dataset_id``; it is stated so the request says what it relies on.

The model half is the consumer's -- the juniper-recurrence ``POST /v1/train`` or
``POST /v1/crossval`` body -- and is written here so an operator does not stop at the dataset::

    {"readout": "rff", "ridge": 1.0, "rff_features": 256, "rff_gamma": "median"}

Those four are not juniper-data params. ``EquitiesSeqParams`` ignores keys it does not declare,
so sending them to ``POST /v1/datasets`` does nothing, silently. An omitted ``readout`` is the
linear rung at the service's ``default_ridge`` (0.0 by default), and an RFF request that omits
``ridge`` gets GCV.

Why: at the bare defaults (``fundamentals_fill="nan"``, ``regression_target="next_close"``)
columns 7, 8 and 14 of ``X`` (``total_shares``, ``market_cap``, ``days_since_report``) carry NaN
before each ticker's first SEC filing, which the recurrence service refuses as non-finite
``X_train`` (F-P1, juniper-data#409). Producer normalisation is a convenience for the happy
path, not a cross-validation control -- it is fitted on the pooled ``train`` partition, and the
walk-forward folds inside that partition inherit its statistics (the plan's R4 note) -- and the
2026-10-04 measurement gave no reason to enable it (RFF at ridge 1.0: aggregate cross-validated
r² -0.115 raw against -0.142 normalised).

Under ``fundamentals_fill="drop"`` keep ``purchase_date`` on or before ``start_date``, or leave
both at their defaults (a Saturday start and the Monday that opens the window): a purchase after
the start is refused (W1.8). Pin ``start_date`` and ``end_date`` when the artifact must be
reproducible, because ``end_date`` defaults to today.

Sources, both in juniper-ml ``notes/``:
``JUNIPER_2026-10-03_JUNIPER-RECURRENCE_EQUITIES-END-TO-END-AUDIT-AND-DEVELOPMENT-PLAN.md``
(W1.1, F-P1, the R4 note) and ``JUNIPER_2026-10-04_JUNIPER-RECURRENCE_EQUITIES-CV-BLOWUP-INVESTIGATION.md``
(§2.3(b), §3.3). Pinned by ``tests/unit/test_equities_seq_generator.py::TestRecurrenceReadyBundle``.

The ``dataset_id`` is request-addressed (W5.7, F-P8)
---------------------------------------------------
``dataset_id`` hashes the generator, its ``VERSION`` and the request params -- not the arrays.
Two mints of one id can therefore differ in content whenever the upstream response differs:
the 2026-10-05 review found a second mint of ``equities_seq-6.0.0-15505731cba5b86d`` whose
price response had no action columns, so its ``dividend`` column was all zero and the
2020-08-31 AAPL split was absent. The meta ``checksum`` IS the content fingerprint (sha256 over
the arrays), so a consumer that needs "the same data" compares checksums, not ids. The producer
guards here record what it can see -- ``data_quality.actions_present`` says whether each
symbol's response carried the action columns, ``data_quality.dropped_symbols`` names every
resolved symbol that is absent from the arrays and why -- and refuse a requested symbol that is
lost unless ``allow_truncation`` opens the gate. Refusing a mint whose actions MAY be missing is
a request-level choice: the generator cannot know whether a symbol had any actions in range.
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

import numpy as np

try:
    import pandas as pd
except ImportError:  # pragma: no cover - exercised only without the equities extra
    pd = None  # type: ignore[assignment]

from juniper_data.core.limits import DATA_QUALITY_META_KEY, TRUNCATION_META_KEY
from juniper_data.core.split import temporal_split_indices
from juniper_data.generators._sequence import _yyyymmdd_to_ordinal, window_one_ticker
from juniper_data.generators.equities.generator import EQUITIES_DEPS_AVAILABLE, EquitiesGenerator

from .params import EquitiesSeqParams

# 6.0.0 (X8, owner ruling 2026-09-24): the registry's ``task_type`` became ``regression``,
# so the emitted meta changed (``n_classes`` / ``class_distribution`` are now null). The
# arrays did not. The dataset ID hashes this version but not the meta, so without the
# bump a cached 5.0.0 artifact would keep serving classification meta under the same id.
VERSION = "6.0.0"

_logger = logging.getLogger(__name__)

# Per-window keys produced by ``window_one_ticker`` (also the NPZ key stems).
_WINDOW_KEYS = ("X", "y", "y_reg", "date", "dt", "target_dt", "window_end_date", "ticker_code", "observed_mask")


class EquitiesSeqGenerator:
    """Generator for windowed (3-D) S&P 500 equities sequence datasets.

    All methods are static (stateless, side-effect free aside from the shared
    on-disk download cache reused from ``EquitiesGenerator``).

    For juniper-recurrence, do not call it with the bare defaults: send the
    recurrence-ready bundle in the module docstring -- ``fundamentals_fill="drop"``,
    ``normalize_features=False``, ``regression_target="log_return"`` and explicit
    ``symbols`` -- and train with ``readout="rff"``, ``ridge=1.0``,
    ``rff_features=256``, ``rff_gamma="median"`` on the recurrence side.
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
    def bind_deployment_defaults(params: EquitiesSeqParams) -> EquitiesSeqParams:
        """Copy the effective symbol cap and truncation opt-in onto the params object.

        Delegates to :meth:`EquitiesGenerator.bind_deployment_defaults`, which is the
        single implementation of the clamp (``min(request, deployment)``) and of the
        truncation tri-state (``deployment if request is None else request`` -- a
        logical OR until APD-DATA-052). ``model_copy`` returns the CONCRETE class it was
        called on, so an ``EquitiesSeqParams`` in is an ``EquitiesSeqParams`` out, with
        ``lookback`` and every other subclass field intact -- pinned by
        ``test_equities_seq_deployment_policy.py`` rather than assumed.

        **The defect this closes.** The create route looks the binder up with
        ``getattr(generator_class, "bind_deployment_defaults", None)``
        (``api/routes/datasets.py``), so a generator opts in simply by defining it --
        and this one did not. ``generate_dataset_id`` hashes ``params.model_dump()``,
        and dump fills Field defaults, so an omitted ``max_symbols`` hashed as the
        schema default and ``allow_truncation`` hashed as ``false`` no matter what the
        deployment said. Proven by execution on 2026-09-08: two ``equities_seq``
        requests, identical but for ``JUNIPER_DATA_EQUITIES_ALLOW_TRUNCATION`` being
        off then on, hashed to the SAME ``equities_seq-3.0.0-e9b10e26ed01ae0e`` (the id is quoted as
        it was measured, under the generator version current at the time; the binder fix and the
        4.0.0 bump have both changed it since), so
        toggling the deployment opt-in kept serving the artifact built under the old
        policy. The flat ``equities`` generator produced two different ids for the
        same pair.

        **DELIBERATE CONSEQUENCE: every ``equities_seq`` dataset_id changes once.**
        ``max_symbols=None`` now hashes as the bound cap and ``allow_truncation`` as
        the effective value, so the first request after this ships resolves to a new
        id and regenerates. That one-time cache turnover IS the point -- the old ids
        were computed from a policy that was not the one the generator ran under.
        """
        return EquitiesGenerator.bind_deployment_defaults(params)  # type: ignore[return-value]  # model_copy preserves the concrete subclass

    @staticmethod
    def generate(params: EquitiesSeqParams) -> dict[str, np.ndarray]:
        """Generate the windowed equities sequence dataset.

        Returns the additive 3-D NPZ contract for train/val/test: ``X_{split}``
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
        symbols, meta_map, truncation = EquitiesGenerator._resolve_symbols(params, constituents)
        end_date = params.end_date or datetime.now(UTC).strftime("%Y-%m-%d")

        # The flat generator's conditioning loop (W5.7 made it shared), which records every
        # ticker it loses and why. Both generators had their own copy, and both lost tickers
        # in silence.
        conditioned, lost = EquitiesGenerator._condition_universe(symbols, meta_map, params, end_date, log_prefix="equities_seq")

        if not conditioned:
            raise ValueError("No data could be retrieved for the requested symbols.")

        # W5.7 (F-P5): a ticker with too few rows for one window is lost from the arrays
        # exactly as a ticker with no data is, and was just as silent -- the `continue` in
        # the windowing loop below logs at INFO and records nothing. Classified HERE, before
        # the policy, so a requested symbol that cannot contribute a window is refused under
        # the shut gate like one that returned nothing, and annotated under the open one. It
        # stays in `ticker_vocab` with zero windows: the code allocation is unchanged, so an
        # existing dataset_id keeps the arrays it had.
        for ticker, frame in conditioned.items():
            if len(frame) <= params.lookback + 1:
                lost[ticker] = f"{len(frame)} row(s) after conditioning, need more than lookback+1={params.lookback + 1}: no window"

        # The incomplete-data policy, the flat generator's own helper (APD-DATA-018).
        # This generator had no copy of it, so an unrescued ticker shipped here with a
        # fabricated total_shares / market_cap, no refusal and no annotation -- while
        # the identical request to `equities` was refused. Called HERE, after
        # conditioning and before the normaliser fit and the windowing, so a dropped
        # ticker reaches neither: `vocab` below is derived from the returned mapping.
        conditioned, quality_meta = EquitiesGenerator._apply_incomplete_policy(params, conditioned, lost=lost)

        vocab = sorted(conditioned)
        code_of = {ticker: code for code, ticker in enumerate(vocab)}

        # Fit normalization on the TRAINING rows across all tickers (juniper-data#314),
        # the same statistic the flat generator uses -- which was also fixed there.
        #
        # This previously fit on the concatenated FULL frames, including each ticker's
        # chronologically-later test rows, and then applied those statistics to the training
        # windows: look-ahead leakage. The per-ticker split boundary is already computed below
        # as the FIRST cut of ``temporal_split_indices(n_rows, params.train_ratio,
        # params.val_ratio)``, so the training rows are available here and the same
        # boundary is reused rather than re-derived.
        #
        # CONSEQUENCE, deliberate: test windows are no longer bounded by [0, 1]. Later rows
        # legitimately exceed the training range, and that excursion is real signal rather
        # than something to normalise away.
        norm = None
        if params.normalize_features:
            train_frames = []
            for ticker in vocab:
                frame = conditioned[ticker]
                # The FIRST of the two cuts is the train boundary; val and test both
                # sit after it, so the fit scope is unchanged by the third partition.
                cut, _ = temporal_split_indices(len(frame), params.train_ratio, params.val_ratio)
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
            EquitiesGenerator._check_return_target_inputs(ticker, frame, params.regression_target)
            y_reg = EquitiesGenerator._regression_target(frame, params.regression_target)
            ords = _yyyymmdd_to_ordinal(dates)
            # Two chronological cuts per ticker: train | val | test, in time order.
            # Derived from the SAME helper the normaliser fit scope uses above, so
            # the boundary the statistics are fit on and the boundary the windows
            # are split at cannot drift apart.
            train_end, val_end = temporal_split_indices(n_rows, params.train_ratio, params.val_ratio)
            out = window_one_ticker(
                feats,
                dates,
                y_dir,
                y_reg,
                code_of[ticker],
                lookback=params.lookback,
                cut_ordinal=int(ords[train_end]),
                val_cut_ordinal=int(ords[val_end]),
            )
            per_ticker.append(out)

        if not per_ticker:
            raise ValueError(f"No symbol had more than lookback+1={params.lookback + 1} rows; cannot build any window.")

        arrays = EquitiesSeqGenerator._assemble(per_ticker)
        arrays["ticker_vocab"] = np.array(vocab, dtype=np.str_)

        # APD-DATA-018: this generator reuses the flat one's universe resolution,
        # so it inherits the symbol cap -- and must therefore also carry the
        # annotation. Inheriting the bound while dropping the record of it is the
        # worse of the two halves to skip: the dataset would be silently partial
        # with nothing anywhere saying so.
        #
        # ``EquitiesSeqParams`` subclasses ``EquitiesParams``, so ``max_symbols``
        # and ``allow_truncation`` need no redeclaration here.
        if truncation is not None:
            # The partition sum, not a retired `X_full`. `records_imported` is defined in
            # `core/limits.py` as "rows in the resulting dataset", and the flat generator
            # fills it from the CONDITIONED frame (`len(full)`) for the same reason -- it
            # counts what survived, which is exactly what the three partitions hold.
            # Reading `arrays["X_full"]` here raised KeyError against this generator's own
            # `_assemble`, which stopped producing that key.
            truncation["records_imported"] = int(sum(arrays[f"X_{name}"].shape[0] for name in ("train", "val", "test")))
            arrays[TRUNCATION_META_KEY] = truncation

        # The permanent data-quality annotation, over the same reserved-key channel and
        # in the same order as the flat generator. Absent entirely when nothing is
        # wrong, so its presence alone answers "is anything degraded here".
        if quality_meta is not None:
            arrays[DATA_QUALITY_META_KEY] = quality_meta

        return arrays

    @staticmethod
    def _assemble(per_ticker: list[dict[str, dict[str, np.ndarray]]]) -> dict[str, np.ndarray]:
        """Concatenate per-ticker windows into the train / val / test NPZ arrays.

        Decision 11 retired the ``*_full`` family, and with it the second loop that
        built it. The row ORDER those two loops differed in is worth recording, because
        consumers reconstructing "the whole dataset" inherit the difference:

        - these per-split arrays are SPLIT-major -- every ticker's train windows, then
          every ticker's val, then every ticker's test;
        - ``*_full`` was ENTITY-major -- each ticker's train, val and test in turn,
          concatenated across tickers.

        Same windows, different permutation, and identical only for a single-ticker
        request. A consumer that slices the whole view by row index -- walk-forward
        cross-validation does exactly that -- gets different folds from the two orders.
        ``juniper_recurrence_model.data.derive_full_split`` rebuilds the entity-major
        order deliberately, by stable-sorting the concatenation on ``ticker_code``.

        The split list is a single tuple rather than a literal per loop. Enumerating the
        splits twice is how a partition silently goes missing from one view but not the
        other.
        """
        splits = ("train", "val", "test")
        arrays: dict[str, np.ndarray] = {}
        for split in splits:
            for key in _WINDOW_KEYS:
                arrays[f"{key}_{split}"] = np.concatenate([out[split][key] for out in per_ticker], axis=0)
        return arrays


def get_schema() -> dict:
    """Return the JSON schema describing the generator parameters."""
    return EquitiesSeqParams.model_json_schema()
