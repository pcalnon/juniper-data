"""Shared request parameters for the three-way train / val / test partition.

Every tabular generator takes the same partition-sizing vocabulary, so it is
declared once here and mixed into each generator's params model rather than
transcribed nine times. The nine-way transcription is the failure this module
exists to prevent -- ``juniper-data#320`` was exactly that shape in the Postgres
store, where five copies of one field list had each drifted differently.

Two sizing models, per section 6.3 of the partition design of record
(juniper-ml ``notes/JUNIPER_2026-08-29_JUNIPER-ECOSYSTEM_TRAIN-EVAL-TEST-PARTITION-DESIGN.md``):

* **additive** (default) -- the generator's native size knob denotes the TRAIN
  row count, and ``val`` / ``test`` are generated as ADDITIONAL rows, sized as
  percentages of it. Decisions 2 and 8. A request for 1000 training points
  yields a 1000-point training set.
* **carve** -- the conventional division of a fixed N by ratios. Section 6.3
  admits this "when any of these holds: an explicit CLI switch, environment
  variable or config setting; the dataset has no generator or no generator
  specs; or the dataset type is not amenable to synthetic generation".

The last clause is why :class:`CarveOnlyPartitionParams` exists. A generator
reading real data -- ``mnist``, ``csv_import``, ``arc_agi`` -- cannot conjure
additional rows to honour a train count, so additive sizing is not merely
undesirable there, it is unimplementable. Rejecting it explicitly is better than
accepting the parameter and quietly carving anyway.
"""

# Project:       Juniper
# Sub-Project:   JuniperData
# Application:   juniper_data
# File Name:     partition_params.py
# Author:        Paul Calnon
# Version:       0.6.0
# License:       MIT License

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, Field, model_validator

from juniper_data.core.split import (
    DEFAULT_TEST_PERCENT,
    DEFAULT_VAL_PERCENT,
    MAX_PARTITION_PERCENT,
    SIZING_MODE_ADDITIVE,
    SIZING_MODE_CARVE,
)

#: Carve-mode validation share for generators that CAN synthesise extra rows.
#:
#: Zero, deliberately. For those generators carve is the opt-in mode, and a
#: non-zero default here would silently halve the test partition of an existing
#: caller who selected carve and never asked for a validation split. Asking for
#: `val` in carve mode is an explicit act: set `val_ratio`, and lower
#: `train_ratio` / `test_ratio` to make room (the cross-field validator says so
#: if you do not).
DEFAULT_CARVE_VAL_RATIO: float = 0.0

#: Carve-mode validation share for the real-data generators, where carve is the
#: ONLY mode. Non-zero, because a default of 0 would mean those generators never
#: emit a validation partition at all -- which is the defect this whole change
#: exists to remove. Their `test_ratio` defaults drop to match.
DEFAULT_CARVE_ONLY_VAL_RATIO: float = 0.1


class PartitionParams(BaseModel):
    """Partition-sizing parameters shared by every tabular generator.

    Attributes:
        sizing_mode: ``"additive"`` (default) or ``"carve"``.
        val_percent: additive mode -- val rows as a percentage of train.
        test_percent: additive mode -- test rows as a percentage of train.
        val_ratio: carve mode -- val's share of the fixed N.
    """

    #: Whether this generator can synthesise additional rows. Real-data
    #: generators override it to False; see :class:`CarveOnlyPartitionParams`.
    supports_additive_sizing: ClassVar[bool] = True

    sizing_mode: str = Field(
        default=SIZING_MODE_ADDITIVE,
        description="'additive' honours the size knob as the TRAIN count and generates val/test as extra rows; 'carve' divides a fixed N by ratios",
    )
    val_percent: float = Field(
        default=DEFAULT_VAL_PERCENT,
        ge=0.0,
        le=MAX_PARTITION_PERCENT,
        description="Additive mode: validation rows as a percentage of the train count (default 40)",
    )
    test_percent: float = Field(
        default=DEFAULT_TEST_PERCENT,
        ge=0.0,
        le=MAX_PARTITION_PERCENT,
        description="Additive mode: test rows as a percentage of the train count (default 30)",
    )
    val_ratio: float = Field(
        default=DEFAULT_CARVE_VAL_RATIO,
        ge=0.0,
        le=1.0,
        description="Carve mode: fraction of the fixed dataset used for in-loop validation",
    )

    @model_validator(mode="after")
    def validate_sizing_mode(self) -> PartitionParams:
        """Reject an unknown mode, and additive sizing where it cannot be honoured."""
        if self.sizing_mode not in (SIZING_MODE_ADDITIVE, SIZING_MODE_CARVE):
            raise ValueError(f"sizing_mode must be '{SIZING_MODE_ADDITIVE}' or '{SIZING_MODE_CARVE}', got {self.sizing_mode!r}")

        if self.sizing_mode == SIZING_MODE_ADDITIVE and not self.supports_additive_sizing:
            raise ValueError(f"sizing_mode '{SIZING_MODE_ADDITIVE}' is not available for this generator: it reads real data and cannot generate additional rows to honour a train count. Use '{SIZING_MODE_CARVE}'.")
        return self

    @model_validator(mode="after")
    def validate_carve_ratios_sum(self) -> PartitionParams:
        """A carve cannot allocate more rows than the dataset has.

        Declared once here rather than per generator. Before the third
        partition, only two of the nine tabular generators carried a cross-field
        ratio check and both tested ``train_ratio + test_ratio``, so the length
        identity was violable through request params -- design section 9.2
        recorded exactly that. The check now spans all three partitions and
        applies to every generator that mixes this in.

        It runs in BOTH sizing modes, not just carve. Additive sizing ignores
        these ratios, so a nonsensical pair does nothing there -- but accepting
        ``train_ratio=0.9, test_ratio=0.9`` silently, purely because the current
        mode happens not to read them, hands the caller a request that will fail
        the moment they switch mode. A malformed value is a malformed value.

        ``train_ratio`` / ``test_ratio`` are read with ``getattr`` because they
        are declared by the concrete generator params, not by this mixin.
        """
        train_ratio = getattr(self, "train_ratio", 0.0)
        test_ratio = getattr(self, "test_ratio", 0.0)
        total = train_ratio + self.val_ratio + test_ratio
        if total > 1.0:
            raise ValueError(f"train_ratio ({train_ratio}) + val_ratio ({self.val_ratio}) + test_ratio ({test_ratio}) must be <= 1.0, got {total}")
        return self


def rescale_generator_params(params: PartitionParams, **update: object) -> PartitionParams:
    """Apply a computed size knob to a params model **and re-validate it**.

    ``model_copy(update=...)`` does not re-run validation in pydantic v2, so a
    computed size knob written that way silently bypasses the generator's own
    bounds. That is not hypothetical here: additive sizing multiplies the knob
    by 1.7 at the default breakdown, so a spiral request at the documented
    maximum (``n_points_per_spiral=10000``) lands at 17 000 against a
    ``le=MAX_POINTS`` of 10 000 -- a value the same model rejects outright when
    constructed directly.

    Re-validating turns that into a 422 at the API boundary instead of an
    allocation the worker may not survive. The consequence is deliberate and
    worth stating: under additive sizing a generator's effective maximum size
    knob is its declared cap divided by the realised multiplier, because the cap
    governs the rows actually generated rather than the rows requested.

    Args:
        params: the validated request params.
        **update: computed field values to apply.

    Returns:
        A new, fully validated params model of the same type.

    Raises:
        pydantic.ValidationError: if the computed values violate the model's
            own field constraints.
    """
    return type(params).model_validate({**params.model_dump(), **update})


class CarveOnlyPartitionParams(PartitionParams):
    """Partition parameters for generators that cannot synthesise extra rows.

    ``mnist``, ``csv_import`` and ``arc_agi`` read a fixed corpus, so section
    6.3's "not amenable to synthetic generation" clause applies and carve is the
    only available model. The default flips accordingly, and asking for additive
    raises rather than silently carving.
    """

    supports_additive_sizing: ClassVar[bool] = False

    sizing_mode: str = Field(
        default=SIZING_MODE_CARVE,
        description="Fixed to 'carve': this generator reads real data and cannot generate additional rows",
    )
    val_ratio: float = Field(
        default=DEFAULT_CARVE_ONLY_VAL_RATIO,
        ge=0.0,
        le=1.0,
        description="Carve mode: fraction of the fixed dataset used for in-loop validation",
    )
