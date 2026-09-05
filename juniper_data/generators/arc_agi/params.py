"""Parameters for the ARC-AGI dataset generator."""

from typing import Literal

from pydantic import Field

from juniper_data.core.constants import DEFAULT_GENERATOR_SEED
from juniper_data.core.partition_params import CarveOnlyPartitionParams

from .defaults import (
    ARC_AGI_DEFAULT_FLATTEN_PAIRS,
    ARC_AGI_DEFAULT_INCLUDE_TEST,
    ARC_AGI_DEFAULT_PAD_TO,
    ARC_AGI_DEFAULT_PAD_VALUE,
    ARC_AGI_DEFAULT_SOURCE,
    ARC_AGI_DEFAULT_SUBSET,
    ARC_AGI_DEFAULT_TEST_RATIO,
    ARC_AGI_DEFAULT_TRAIN_RATIO,
    MAX_PAD_TO,
    MAX_PAD_VALUE,
    MIN_PAD_TO,
    MIN_PAD_VALUE,
)


class ArcAgiParams(CarveOnlyPartitionParams):
    """Configuration parameters for ARC-AGI dataset loading.

    Loads ARC-AGI tasks from Hugging Face Hub or local JSON files.
    The ARC (Abstraction and Reasoning Corpus) contains grid-based
    reasoning tasks with input/output pairs.
    """

    source: Literal["huggingface", "local"] = Field(
        default=ARC_AGI_DEFAULT_SOURCE,
        description="Data source: 'huggingface' or 'local' JSON files",
    )
    local_path: str | None = Field(
        default=None,
        description="Path to local ARC JSON files (required if source='local')",
    )
    subset: Literal["training", "evaluation", "all"] = Field(
        default=ARC_AGI_DEFAULT_SUBSET,
        description="Which subset to load: 'training', 'evaluation', or 'all'",
    )
    n_tasks: int | None = Field(
        default=None,
        ge=1,
        description="Limit number of tasks to load (None for all)",
    )
    pad_to: int = Field(
        default=ARC_AGI_DEFAULT_PAD_TO,
        ge=MIN_PAD_TO,
        le=MAX_PAD_TO,
        description="Pad all grids to this size (max ARC grid is 30x30)",
    )
    pad_value: int = Field(
        default=ARC_AGI_DEFAULT_PAD_VALUE,
        ge=MIN_PAD_VALUE,
        le=MAX_PAD_VALUE,
        description="Value to use for padding (-1 recommended for masking)",
    )
    include_test: bool = Field(
        default=ARC_AGI_DEFAULT_INCLUDE_TEST,
        description="Include test input/output pairs (in addition to train pairs)",
    )
    flatten_pairs: bool = Field(
        default=ARC_AGI_DEFAULT_FLATTEN_PAIRS,
        description="Flatten all input/output pairs into single arrays",
    )
    seed: int | None = Field(default=DEFAULT_GENERATOR_SEED, ge=0, description="Random seed for reproducibility. Defaults to DEFAULT_GENERATOR_SEED so the documented default configuration is REPRODUCIBLE (juniper-data#319); pass None explicitly to opt into a fresh draw per call.")
    train_ratio: float = Field(default=ARC_AGI_DEFAULT_TRAIN_RATIO, gt=0, le=1, description="Fraction of data for training")
    test_ratio: float = Field(default=ARC_AGI_DEFAULT_TEST_RATIO, ge=0, le=1, description="Fraction of data for testing")
    shuffle: bool = Field(default=True, description="Shuffle before splitting")
