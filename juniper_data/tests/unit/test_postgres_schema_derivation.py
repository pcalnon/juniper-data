#!/usr/bin/env python
"""The Postgres schema and row mappers are derived from ``DatasetMeta`` (juniper-data#320).

The store used to carry FIVE independent transcriptions of the same field set -- the DDL,
``_meta_to_row``, ``_row_to_meta``, the upsert and the update -- and every one had drifted:

* the DDL declared ``n_classes`` / ``class_distribution`` ``NOT NULL`` after the model made
  both nullable, so the first regression dataset written would have failed the INSERT;
* the other four enumerated 23 of the model's 30 fields, silently dropping ``task_type``,
  ``sequence``, ``lookback``, ``time_unit``, ``dt_scaling``, ``target_scaling`` and
  ``truncation``;
* ``_row_to_meta`` raised ``TypeError`` on a NULL ``class_distribution``.

**No test asserted the round trip**, which is why it accumulated. That is the central test
here, and it needs no database: the mappers are pure functions.

The count assertions are deliberately written against ``DatasetMeta.model_fields`` rather
than a hardcoded 30 -- a test that pins today's number would have to be edited by the same
person who forgot to add the column, which is the failure being prevented.
"""

import json
from datetime import UTC, datetime

import pytest

from juniper_data.core.models import DatasetMeta
from juniper_data.storage.postgres_store import (
    _JSONB_FIELDS,
    _TEXT_ARRAY_FIELDS,
    _admits_none,
    _column_specs,
    build_schema_sql,
    build_update_sql,
    build_upsert_sql,
)

pytestmark = [pytest.mark.unit, pytest.mark.storage]

_MODEL_FIELDS = set(DatasetMeta.model_fields)


def _classification_meta() -> DatasetMeta:
    return DatasetMeta(
        dataset_id="spiral-1.0.0-abc123",
        generator="spiral",
        generator_version="1.0.0",
        params={"n_points_per_spiral": 500, "seed": 42},
        n_samples=1000,
        n_features=2,
        n_classes=2,
        n_train=800,
        n_test=200,
        class_distribution={"0": 500, "1": 500},
        created_at=datetime(2026, 9, 4, 12, 0, tzinfo=UTC),
        checksum="deadbeef",
        tags=["a", "b"],
    )


def _regression_sequence_meta() -> DatasetMeta:
    """The artifact shape the old NOT NULL schema would have rejected outright."""
    return DatasetMeta(
        dataset_id="mackey_glass-1.0.0-def456",
        generator="mackey_glass",
        generator_version="1.0.0",
        params={"n_steps": 2000},
        n_samples=968,
        n_features=1,
        task_type="regression",
        n_classes=None,
        class_distribution=None,
        n_train=774,
        n_test=194,
        sequence=True,
        lookback=32,
        time_unit="step",
        dt_scaling={"mode": "identity"},
        target_scaling={"mode": "standardize", "mean": 0.9},
        truncation={"applied": False},
        created_at=datetime(2026, 9, 4, 12, 0, tzinfo=UTC),
    )


class TestSchemaCoversTheWholeModel:
    def test_every_model_field_has_a_column(self):
        assert {name for name, _, _ in _column_specs()} == _MODEL_FIELDS

    @pytest.mark.parametrize("field", sorted({"task_type", "sequence", "lookback", "time_unit", "dt_scaling", "target_scaling", "truncation"}))
    def test_the_previously_dropped_fields_are_present(self, field):
        """These seven were absent from all five transcriptions."""
        ddl = build_schema_sql()
        assert f" {field} " in ddl or f"    {field} " in ddl

    def test_nullability_follows_the_model(self):
        """The defect that would have hard-failed on the first regression dataset."""
        for name, _, nullable in _column_specs():
            expected = _admits_none(DatasetMeta.model_fields[name].annotation)
            assert nullable == expected, f"{name}: column nullable={nullable}, model admits None={expected}"

    def test_nullable_columns_are_not_declared_not_null(self):
        ddl = build_schema_sql()
        for name, _, nullable in _column_specs():
            if nullable:
                assert f"{name} " in ddl
                assert f"    {name} INTEGER NOT NULL" not in ddl
                assert f"    {name} JSONB NOT NULL" not in ddl

    def test_n_classes_and_class_distribution_are_nullable(self):
        """Named explicitly because these two are the regression-artifact blockers."""
        ddl = build_schema_sql()
        assert "n_classes INTEGER NOT NULL" not in ddl
        assert "class_distribution JSONB NOT NULL" not in ddl

    def test_added_columns_are_migration_safe(self):
        """``SCHEMA_SQL`` runs on every init, so a second boot must not error."""
        ddl = build_schema_sql()
        for name, _, _ in _column_specs():
            if name == "dataset_id":
                continue
            assert f"ADD COLUMN IF NOT EXISTS {name} " in ddl

    def test_a_not_null_added_column_always_carries_a_default(self):
        """``ADD COLUMN ... NOT NULL`` without a default fails on a POPULATED table."""
        for line in build_schema_sql().splitlines():
            if line.startswith("ALTER TABLE") and "NOT NULL" in line:
                assert "DEFAULT" in line, f"unsafe migration: {line}"

    def test_primary_key_is_declared_once(self):
        assert build_schema_sql().count("PRIMARY KEY") == 1


class TestStatementsCoverTheWholeModel:
    def test_upsert_names_every_column(self):
        sql = build_upsert_sql()
        for name in _MODEL_FIELDS:
            assert f"%({name})s" in sql, f"{name} missing from the INSERT values"

    def test_update_names_every_mutable_column(self):
        sql = build_update_sql()
        for name in _MODEL_FIELDS - {"created_at"}:
            assert f"%({name})s" in sql, f"{name} missing from the UPDATE"

    def test_created_at_is_preserved_on_conflict(self):
        """A re-save is not a re-creation."""
        assert "created_at = EXCLUDED.created_at" not in build_upsert_sql()

    def test_jsonb_columns_are_cast(self):
        sql = build_upsert_sql()
        for name in _JSONB_FIELDS:
            assert f"%({name})s::jsonb" in sql

    @pytest.mark.parametrize("builder,keyword", [(build_upsert_sql, "INSERT INTO"), (build_update_sql, "UPDATE ")], ids=["upsert", "update"])
    def test_statement_begins_with_its_keyword(self, builder, keyword):
        """Guards a real mistake made while writing this change.

        A ``# nosec`` comment was first placed on the ``return f\"\"\"`` line -- which is INSIDE
        the multi-line string, so the comment text was silently prepended to every generated
        statement. Every structural assertion above still passed, because the columns were all
        present; only the prefix was wrong.
        """
        assert builder().strip().startswith(keyword)

    @pytest.mark.parametrize("builder", [build_upsert_sql, build_update_sql, build_schema_sql], ids=["upsert", "update", "schema"])
    def test_no_python_comment_leaks_into_generated_sql(self, builder):
        assert "nosec" not in builder()
        assert "#" not in builder()

    def test_a_hostile_table_name_is_refused(self):
        """The identifier position cannot be parameterised, so it is validated instead."""
        for hostile in ("datasets; DROP TABLE users", 'datasets"', "datasets--", "data sets", ""):
            with pytest.raises(ValueError, match="unsafe SQL identifier"):
                build_schema_sql(hostile)

    def test_a_legitimate_table_name_is_accepted(self):
        assert "CREATE TABLE IF NOT EXISTS juniper_datasets_v2" in build_schema_sql("juniper_datasets_v2")


class TestRoundTrip:
    """THE test whose absence let five transcriptions drift apart."""

    @pytest.fixture
    def store(self):
        """The mappers do not touch instance state, so no connection is needed."""
        from juniper_data.storage.postgres_store import PostgresDatasetStore

        return PostgresDatasetStore.__new__(PostgresDatasetStore)

    @pytest.mark.parametrize("factory", [_classification_meta, _regression_sequence_meta], ids=["classification", "regression_sequence"])
    def test_meta_survives_a_round_trip(self, store, factory):
        original = factory()
        row = store._meta_to_row(original)
        restored = store._row_to_meta(row)
        assert restored == original

    def test_round_trip_carries_the_seven_previously_dropped_fields(self, store):
        """Explicit, because an equality assertion alone would not say WHICH field moved."""
        original = _regression_sequence_meta()
        restored = store._row_to_meta(store._meta_to_row(original))
        for field in ("task_type", "sequence", "lookback", "time_unit", "dt_scaling", "target_scaling", "truncation"):
            assert getattr(restored, field) == getattr(original, field), f"{field} did not survive"

    def test_row_has_a_key_for_every_model_field(self, store):
        assert set(store._meta_to_row(_classification_meta())) == _MODEL_FIELDS

    def test_null_class_distribution_does_not_raise(self, store):
        """The old mapper raised TypeError here -- ``json.loads(None)``."""
        row = store._meta_to_row(_regression_sequence_meta())
        assert row["class_distribution"] is None, "None must stay SQL NULL, not the string 'null'"
        assert store._row_to_meta(row).class_distribution is None

    def test_jsonb_values_are_serialised_not_passed_raw(self, store):
        row = store._meta_to_row(_classification_meta())
        assert isinstance(row["params"], str)
        assert json.loads(row["params"]) == {"n_points_per_spiral": 500, "seed": 42}

    def test_driver_may_return_json_already_decoded(self, store):
        """psycopg2 decodes JSONB to dict; the mapper must accept both shapes."""
        row = store._meta_to_row(_classification_meta())
        row["params"] = json.loads(row["params"])
        assert store._row_to_meta(row).params == {"n_points_per_spiral": 500, "seed": 42}

    def test_a_row_from_an_older_table_still_loads(self, store):
        """Columns absent because the table predates a field must fall back to the default,
        not raise a KeyError. This is what makes a code-ahead-of-schema deploy survivable."""
        row = store._meta_to_row(_classification_meta())
        for missing in ("truncation", "dt_scaling", "sequence", "task_type"):
            row.pop(missing, None)
        restored = store._row_to_meta(row)
        assert restored.task_type == "classification"
        assert restored.sequence is False
        assert restored.truncation is None

    def test_a_null_in_a_non_nullable_column_falls_back_to_the_default(self, store):
        """An older table can hold NULL where the model now refuses one."""
        row = store._meta_to_row(_classification_meta())
        row["task_type"] = None
        row["sequence"] = None
        restored = store._row_to_meta(row)
        assert restored.task_type == "classification"
        assert restored.sequence is False

    def test_text_array_columns_round_trip(self, store):
        original = _classification_meta()
        restored = store._row_to_meta(store._meta_to_row(original))
        assert restored.tags == ["a", "b"]
        assert restored.artifact_formats == original.artifact_formats
        assert set(_TEXT_ARRAY_FIELDS) == {"artifact_formats", "tags"}
