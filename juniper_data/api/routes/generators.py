"""Generator endpoints for listing and describing available generators."""

from typing import Any

from fastapi import APIRouter, HTTPException
from starlette import status

from juniper_data.core.models import GeneratorInfo
from juniper_data.generators.ar_p import VERSION as AR_P_VERSION
from juniper_data.generators.ar_p import ArPGenerator, ArPParams
from juniper_data.generators.arc_agi import VERSION as ARC_AGI_VERSION
from juniper_data.generators.arc_agi import ArcAgiGenerator, ArcAgiParams
from juniper_data.generators.checkerboard import VERSION as CHECKERBOARD_VERSION
from juniper_data.generators.checkerboard import CheckerboardGenerator, CheckerboardParams
from juniper_data.generators.circles import VERSION as CIRCLES_VERSION
from juniper_data.generators.circles import CirclesGenerator, CirclesParams
from juniper_data.generators.csv_import import VERSION as CSV_IMPORT_VERSION
from juniper_data.generators.csv_import import CsvImportGenerator, CsvImportParams
from juniper_data.generators.delay_product import VERSION as DELAY_PRODUCT_VERSION
from juniper_data.generators.delay_product import DelayProductGenerator, DelayProductParams
from juniper_data.generators.equities import VERSION as EQUITIES_VERSION
from juniper_data.generators.equities import EquitiesGenerator, EquitiesParams
from juniper_data.generators.equities_seq import VERSION as EQUITIES_SEQ_VERSION
from juniper_data.generators.equities_seq import EquitiesSeqGenerator, EquitiesSeqParams
from juniper_data.generators.gaussian import VERSION as GAUSSIAN_VERSION
from juniper_data.generators.gaussian import GaussianGenerator, GaussianParams
from juniper_data.generators.irregular_sine import VERSION as IRREGULAR_SINE_VERSION
from juniper_data.generators.irregular_sine import IrregularSineGenerator, IrregularSineParams
from juniper_data.generators.mackey_glass import VERSION as MACKEY_GLASS_VERSION
from juniper_data.generators.mackey_glass import MackeyGlassGenerator, MackeyGlassParams
from juniper_data.generators.mnist import VERSION as MNIST_VERSION
from juniper_data.generators.mnist import MnistGenerator, MnistParams
from juniper_data.generators.moon import VERSION as MOON_VERSION
from juniper_data.generators.moon import MoonGenerator, MoonParams
from juniper_data.generators.multi_sine import VERSION as MULTI_SINE_VERSION
from juniper_data.generators.multi_sine import MultiSineGenerator, MultiSineParams
from juniper_data.generators.spiral import VERSION as SPIRAL_VERSION
from juniper_data.generators.spiral import SpiralGenerator, SpiralParams
from juniper_data.generators.xor import VERSION as XOR_VERSION
from juniper_data.generators.xor import XorGenerator, XorParams

router = APIRouter(prefix="/generators", tags=["generators"])

GENERATOR_REGISTRY: dict[str, dict[str, Any]] = {
    # ``task_type`` is declared per generator and drives the dataset route's
    # metadata dispatch (WS-1 / juniper-data#168): "classification" generators
    # get n_classes + class_distribution from their one-hot y; "regression"
    # generators leave those None and carry the target directly in y_*. The
    # classifiers include equities (canonical one-hot next-day direction, with an
    # auxiliary y_reg_* close rider); the multi_sine / mackey_glass / ar_p /
    # irregular_sine synthetics are the regression generators (juniper-data#179
    # §A) -- numpy-only (W, L, 1) sequences with a per-step dt (regular for the
    # first three; genuinely non-uniform for irregular_sine).
    "spiral": {
        "generator": SpiralGenerator,
        "params_class": SpiralParams,
        "version": SPIRAL_VERSION,
        "task_type": "classification",
        "description": "Multi-spiral classification dataset generator. Generates N interleaved spiral arms with configurable points, rotations, and noise.",
    },
    "xor": {
        "generator": XorGenerator,
        "params_class": XorParams,
        "version": XOR_VERSION,
        "task_type": "classification",
        "description": "XOR classification dataset generator. Generates points in 4 quadrants with opposite classes in diagonal quadrants.",
    },
    "gaussian": {
        "generator": GaussianGenerator,
        "params_class": GaussianParams,
        "version": GAUSSIAN_VERSION,
        "task_type": "classification",
        "description": "Gaussian blobs classification dataset generator. Generates mixture-of-Gaussians with configurable centers and covariance.",
    },
    "circles": {
        "generator": CirclesGenerator,
        "params_class": CirclesParams,
        "version": CIRCLES_VERSION,
        "task_type": "classification",
        "description": "Concentric circles classification dataset generator. Generates binary classification with inner and outer circle classes.",
    },
    # XREPO-01b / DC-02 (2026-04-24): added to align with the
    # juniper-data-client ``GENERATOR_MOON`` constant, which previously
    # referenced a server generator that did not exist.
    "moon": {
        "generator": MoonGenerator,
        "params_class": MoonParams,
        "version": MOON_VERSION,
        "task_type": "classification",
        "description": "Two interleaving half-moons classification dataset generator. Generates binary classification with upper and lower half-circle classes.",
    },
    "checkerboard": {
        "generator": CheckerboardGenerator,
        "params_class": CheckerboardParams,
        "version": CHECKERBOARD_VERSION,
        "task_type": "classification",
        "description": "Checkerboard pattern classification dataset generator. Generates 2D grid with alternating class squares.",
    },
    "csv_import": {
        "generator": CsvImportGenerator,
        "params_class": CsvImportParams,
        "version": CSV_IMPORT_VERSION,
        "task_type": "classification",
        "description": "CSV/JSON import generator for custom datasets. Import data from CSV or JSON files with configurable feature and label columns.",
    },
    "equities": {
        "generator": EquitiesGenerator,
        "params_class": EquitiesParams,
        "version": EQUITIES_VERSION,
        "task_type": "classification",
        "description": "S&P 500 equities time-series generator. Daily OHLCV (2000->present) from Yahoo Finance plus SEC EDGAR shares/market-cap, with 52-week high/low, configurable-purchase-date cost basis, and dual targets: next-day direction (one-hot y_*) and next-day close (y_reg_*).",
    },
    "equities_seq": {
        "generator": EquitiesSeqGenerator,
        "params_class": EquitiesSeqParams,
        "version": EQUITIES_SEQ_VERSION,
        "task_type": "classification",
        "time_unit": "calendar_days",
        "description": "Windowed (3-D sequence) S&P 500 equities variant. Slides a per-ticker lookback window over the daily OHLCV rows to produce (W, L, F) sequences with a per-step calendar-day dt (weekend/holiday gaps are the irregular Δt), an irregular forecast horizon target_dt, an all-ones observed_mask, and the next-day direction (one-hot y_*) + next-day close (y_reg_*) targets.",
    },
    "multi_sine": {
        "generator": MultiSineGenerator,
        "params_class": MultiSineParams,
        "version": MULTI_SINE_VERSION,
        "task_type": "regression",
        "time_unit": "steps",
        "description": "Multi-sine synthetic time-series regression generator (numpy-only, no extra). Superposition of K sinusoids sampled at a regular Δt, windowed into (W, L, 1) sequences with a per-step dt and a horizon-ahead regression target y_*. Deterministic given the seed.",
    },
    "mackey_glass": {
        "generator": MackeyGlassGenerator,
        "params_class": MackeyGlassParams,
        "version": MACKEY_GLASS_VERSION,
        "task_type": "regression",
        "time_unit": "steps",
        "description": "Mackey-Glass synthetic time-series regression generator (numpy-only). Discrete-Euler integration of the chaotic delay-differential equation (β=0.2, γ=0.1, n=10, τ=17), windowed into (W, L, 1) sequences with a per-step dt and a horizon-ahead regression target y_*. Deterministic.",
    },
    "ar_p": {
        "generator": ArPGenerator,
        "params_class": ArPParams,
        "version": AR_P_VERSION,
        "task_type": "regression",
        "time_unit": "steps",
        "description": "Autoregressive AR(p) synthetic time-series regression generator (numpy-only). xₜ=c+Σ φᵢ xₜ₋ᵢ+εₜ with Gaussian innovations (default stable AR(2)), windowed into (W, L, 1) sequences with a per-step dt and a horizon-ahead regression target y_*. Deterministic given the seed.",
    },
    "irregular_sine": {
        "generator": IrregularSineGenerator,
        "params_class": IrregularSineParams,
        "version": IRREGULAR_SINE_VERSION,
        "task_type": "regression",
        "time_unit": "steps",
        "description": "Irregular-Δt sine synthetic time-series regression generator (numpy-only). K sinusoids sampled at NON-uniform (jittered) times, windowed into (W, L, 1) sequences with a non-uniform per-step dt and variable target_dt. Offline known-answer counterpart to equities' calendar gaps.",
    },
    "delay_product": {
        "generator": DelayProductGenerator,
        "params_class": DelayProductParams,
        "version": DELAY_PRODUCT_VERSION,
        "task_type": "regression",
        "time_unit": "steps",
        "description": "Delay-product synthetic time-series regression generator (numpy-only). Irregularly-sampled sinusoid superposition (the same non-uniform Δt as irregular_sine) whose target is the BILINEAR product of two delayed in-window values y=x(t−τ₁)·x(t−τ₂) (lag1/lag2 step-delays inside the lookback). A quadratic form in the LMU memory that a linear readout provably cannot fit but a non-linear (RFF) readout can — the DP-3 capacity instrument that exposes a clear nonlinear ≫ linear r² gap.",
    },
    "mnist": {
        "generator": MnistGenerator,
        "params_class": MnistParams,
        "version": MNIST_VERSION,
        "task_type": "classification",
        "description": "MNIST and Fashion-MNIST dataset generator. Downloads and prepares standard handwritten digit or fashion item classification datasets.",
    },
    "arc_agi": {
        "generator": ArcAgiGenerator,
        "params_class": ArcAgiParams,
        "version": ARC_AGI_VERSION,
        "task_type": "classification",
        "description": "ARC-AGI (Abstraction and Reasoning Corpus) dataset generator. Generates visual reasoning tasks from the ARC benchmark.",
    },
}


def generator_available(info: dict[str, Any]) -> bool:
    """Return whether a registered generator is usable in this deployment (D1 / I-5).

    A generator class MAY declare an ``is_available()`` static method reporting
    whether its optional dependencies are importable (mnist -> HF ``datasets``;
    equities / equities_seq -> the ``equities`` extra). Generators that do not
    declare the hook are considered available: that is the correct posture for
    the numpy-only synthetics, and for generators whose dependency need is
    parameter-conditional (arc_agi's HF source has a local-file fallback), where
    the request-time ImportError -> 501 mapping in the datasets route is the
    backstop.

    Args:
        info: A GENERATOR_REGISTRY entry.

    Returns:
        True when the generator declares no availability hook or its hook
        reports True; False when the hook reports a missing capability.
    """
    is_available = getattr(info["generator"], "is_available", None)
    if is_available is None:
        return True
    return bool(is_available())


@router.get("", response_model=list[GeneratorInfo])
async def list_generators() -> list[GeneratorInfo]:
    """List all registered dataset generators with their info.

    Returns:
        List of generator information objects including name, version,
        description, deployment availability, and parameter schema.
    """
    generators: list[GeneratorInfo] = []
    generators.extend(
        GeneratorInfo(
            name=name,
            version=info["version"],
            description=info["description"],
            available=generator_available(info),
            schema=info["params_class"].model_json_schema(),
        )
        for name, info in GENERATOR_REGISTRY.items()
    )
    return generators


@router.get("/{name}/schema")
async def get_generator_schema(name: str) -> dict[str, Any]:
    """Get the JSON schema for a generator's parameters.

    Args:
        name: Generator name (e.g., "spiral").

    Returns:
        JSON schema dictionary describing the generator's parameters, plus a
        top-level ``available`` boolean reporting whether the generator's
        optional dependencies are present in this deployment (D1 / I-5; an
        additive key — JSON Schema consumers ignore unknown keywords).

    Raises:
        HTTPException: 404 if generator not found.
    """
    if name not in GENERATOR_REGISTRY:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Generator '{name}' not found")

    info = GENERATOR_REGISTRY[name]
    return {**info["params_class"].model_json_schema(), "available": generator_available(info)}
