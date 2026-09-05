#!/usr/bin/env python
"""The generator registry's ``task_type`` / ``time_unit`` must stay complete.

``POST /v1/datasets`` does not ask the generator what it produced. It reads the
handwritten ``GENERATOR_REGISTRY`` entry:

    task_type = generator_info.get("task_type", "classification")
    seq_meta = derive_sequence_meta(arrays, generator_info.get("time_unit"))

A missing ``task_type`` therefore silently classifies the artifact.
``compute_shape_meta`` then applies ``argmax`` to whatever ``y_*`` it finds --
continuous regression targets included -- and persists a fake ``n_classes``.
That is the same failure class as juniper-data#320 / #343: an independently
transcribed field list the serving path trusts, with a default that hides the
omission.

``time_unit`` has no default-to-wrong-value, but the same shape of hole: it is
echoed onto ``DatasetMeta`` only when the registry declares it. A lookback-
bearing params class (every 3-D sequence generator) that forgets the key
persists ``time_unit=None`` on a sequence artifact. The synthetics restate
``SYNTHETIC_TIME_UNIT`` as the string ``"steps"`` in five registry entries;
nothing previously asserted those cannot drift apart.

These pins are written against ``model_fields`` / on-disk packages / the
``SyntheticSequenceParams`` base, not against today's count of 16 generators.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from juniper_data.api.routes.generators import GENERATOR_REGISTRY
from juniper_data.generators._synthetic import SYNTHETIC_TIME_UNIT, SyntheticSequenceParams

pytestmark = [pytest.mark.unit, pytest.mark.generators]

_ALLOWED_TASK_TYPES = frozenset({"classification", "regression"})


def _on_disk_generator_packages() -> set[str]:
    root = Path(__file__).resolve().parents[2] / "generators"
    return {path.name for path in root.iterdir() if path.is_dir() and not path.name.startswith("_") and (path / "__init__.py").exists()}


class TestTaskTypeIsExplicit:
    """The route default must never fire: every entry names its task type."""

    @pytest.mark.parametrize("name", sorted(GENERATOR_REGISTRY))
    def test_entry_declares_task_type(self, name: str) -> None:
        assert "task_type" in GENERATOR_REGISTRY[name], f"{name}: missing task_type; the create route defaults the omission to classification"

    @pytest.mark.parametrize("name", sorted(GENERATOR_REGISTRY))
    def test_task_type_is_a_known_value(self, name: str) -> None:
        task_type = GENERATOR_REGISTRY[name]["task_type"]
        assert task_type in _ALLOWED_TASK_TYPES, f"{name}: task_type={task_type!r} is not classification or regression"


class TestTimeUnitFollowsLookback:
    """3-D sequence generators declare a time unit; tabular ones must not invent one."""

    @pytest.mark.parametrize("name", sorted(GENERATOR_REGISTRY))
    def test_lookback_params_declare_time_unit(self, name: str) -> None:
        info = GENERATOR_REGISTRY[name]
        has_lookback = "lookback" in info["params_class"].model_fields
        if has_lookback:
            assert "time_unit" in info, f"{name}: params have lookback but the registry has no time_unit"
            assert isinstance(info["time_unit"], str) and info["time_unit"], f"{name}: time_unit must be a non-empty string"
        else:
            assert "time_unit" not in info, f"{name}: tabular generator must not declare time_unit"


class TestSyntheticsStayOnTheSharedContract:
    """``_synthetic.py`` is the source of truth for the numpy-only sequence gens."""

    @pytest.mark.parametrize(
        "name",
        sorted(name for name, info in GENERATOR_REGISTRY.items() if issubclass(info["params_class"], SyntheticSequenceParams)),
    )
    def test_synthetic_is_regression_with_shared_time_unit(self, name: str) -> None:
        info = GENERATOR_REGISTRY[name]
        assert info["task_type"] == "regression", f"{name}: SyntheticSequenceParams generators are regression (WS-1 / #179)"
        assert info["time_unit"] == SYNTHETIC_TIME_UNIT, f"{name}: registry time_unit={info['time_unit']!r} drifted from SYNTHETIC_TIME_UNIT={SYNTHETIC_TIME_UNIT!r}"


class TestRegistryMatchesThePackageTree:
    def test_registry_keys_are_exactly_the_public_generator_packages(self) -> None:
        """A new on-disk generator that is not registered is invisible to POST /v1/datasets.

        The inverse -- a registry entry with no package -- is an import error at
        module load, but only if someone types the import. Deriving the expected
        set from the tree means this test does not have to be edited by the same
        person who forgot the registry row.
        """
        assert set(GENERATOR_REGISTRY) == _on_disk_generator_packages()
