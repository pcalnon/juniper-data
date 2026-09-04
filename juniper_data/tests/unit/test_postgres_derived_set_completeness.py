#!/usr/bin/env python
"""The remaining handwritten Postgres sets must stay derived from ``DatasetMeta``.

PR #343 (juniper-data#320) derived the DDL, both mappers, the INSERT and the UPDATE
from ``model_fields``. Three classification sets were left transcribed by hand:

* ``_JSONB_FIELDS`` -- dump / load / ``::jsonb`` cast
* ``_TEXT_ARRAY_FIELDS`` -- TEXT[] encode / decode
* ``_SQL_DEFAULTS`` -- ``ADD COLUMN ... NOT NULL`` against a populated table

``_sql_type_for`` already falls back to the annotation origin, so a new ``dict``
or ``list`` field on the model would appear in the DDL automatically and still be
dropped or uncast at the mapper and the statements. That is the same drift class
#343 closed for the column *list*. These tests pin the leftover sets against the
model, not against today's count.

#343's ``test_postgres_schema_derivation.py`` already owns the round trip, the
seven previously-dropped fields, nullability, ALTER ``IF NOT EXISTS``, the
schema-builder hostile name, upsert ``::jsonb``, and the nosec-leak. This file
does not restack those.
"""

from datetime import datetime
from typing import get_origin

import pytest

from juniper_data.core.models import DatasetMeta
from juniper_data.storage.postgres_store import (
    _JSONB_FIELDS,
    _SQL_DEFAULTS,
    _TEXT_ARRAY_FIELDS,
    _admits_none,
    _column_specs,
    _non_none_type,
    _safe_identifier,
    _sql_type_for,
    build_schema_sql,
    build_update_sql,
    build_upsert_sql,
)

pytestmark = [pytest.mark.unit, pytest.mark.storage]

_HOSTILE_IDENTIFIERS = (
    "datasets; DROP TABLE users",
    'datasets"',
    "datasets--",
    "data sets",
    "datasets;x",
    "1datasets",
    "",
)


def _annotation_origin(name: str) -> object:
    """Container origin of a model field after stripping ``None`` from a union."""
    base = _non_none_type(DatasetMeta.model_fields[name].annotation)
    return get_origin(base) or base


def _dict_fields() -> set[str]:
    return {name for name in DatasetMeta.model_fields if _annotation_origin(name) is dict}


def _list_fields() -> set[str]:
    return {name for name in DatasetMeta.model_fields if _annotation_origin(name) is list}


def _not_null_defaulted_fields() -> set[str]:
    """NOT NULL columns whose model field supplies a default (or factory).

    ``ADD COLUMN ... NOT NULL`` against a populated table fails without a SQL
    default. Required fields are created with the table and do not need one.
    """
    return {name for name, field in DatasetMeta.model_fields.items() if not field.is_required() and not _admits_none(field.annotation)}


class TestClassificationSetsFollowTheModel:
    def test_jsonb_set_is_every_dict_field(self):
        """A new dict on DatasetMeta must join ``_JSONB_FIELDS`` or the mapper will not dump it."""
        assert _dict_fields() == _JSONB_FIELDS

    def test_text_array_set_is_every_list_field(self):
        assert _list_fields() == _TEXT_ARRAY_FIELDS

    def test_sql_defaults_cover_every_not_null_defaulted_field(self):
        """A new ``foo: str = \"bar\"`` without a SQL default cannot be added NOT NULL later."""
        assert set(_SQL_DEFAULTS) == _not_null_defaulted_fields()

    def test_sql_defaults_name_real_model_fields(self):
        assert set(_SQL_DEFAULTS) <= set(DatasetMeta.model_fields)

    def test_dict_fields_are_jsonb_in_the_schema(self):
        types = {name: sql_type for name, sql_type, _ in _column_specs()}
        for name in _dict_fields():
            assert types[name] == "JSONB", f"{name} is a dict but the schema says {types[name]}"

    def test_list_fields_are_text_arrays_in_the_schema(self):
        types = {name: sql_type for name, sql_type, _ in _column_specs()}
        for name in _list_fields():
            assert types[name] == "TEXT[]", f"{name} is a list but the schema says {types[name]}"


class TestSqlTypeFallback:
    """The annotation fallback is what makes a forgotten set-update still emit the right DDL.

    The set-completeness tests above then fail, forcing the mapper and the
    statements to catch up. Both sides are required: fallback-only would hide
    the mapper drift; set-only would hide a broken fallback.
    """

    def test_unknown_dict_field_is_still_jsonb(self):
        assert _sql_type_for("not_a_model_field", dict[str, int]) == "JSONB"

    def test_unknown_list_field_is_still_text_array(self):
        assert _sql_type_for("not_a_model_field", list[str]) == "TEXT[]"

    def test_unknown_optional_dict_is_still_jsonb(self):
        assert _sql_type_for("not_a_model_field", dict[str, int] | None) == "JSONB"

    @pytest.mark.parametrize(
        "annotation,expected",
        [
            (bool, "BOOLEAN"),
            (int, "INTEGER"),
            (float, "DOUBLE PRECISION"),
            (str, "TEXT"),
            (datetime, "TIMESTAMP WITH TIME ZONE"),
            (object, "TEXT"),
        ],
        ids=["bool", "int", "float", "str", "datetime", "unknown"],
    )
    def test_scalar_annotation_maps_to_sql(self, annotation, expected):
        assert _sql_type_for("not_a_model_field", annotation) == expected


class TestUpdateCastsAndClassAttribute:
    def test_update_casts_every_jsonb_column(self):
        """#343 asserted ``::jsonb`` on the INSERT; the UPDATE is a fifth transcription."""
        sql = build_update_sql()
        for name in _JSONB_FIELDS:
            assert f"{name} = %({name})s::jsonb" in sql, f"{name} is not cast on UPDATE"

    def test_schema_sql_class_attr_is_the_builder(self):
        """``SCHEMA_SQL`` must not be replaced by a third handwritten DDL string."""
        from juniper_data.storage.postgres_store import PostgresDatasetStore

        assert build_schema_sql() == PostgresDatasetStore.SCHEMA_SQL

    def test_indexes_use_the_validated_table_name(self):
        sql = build_schema_sql("juniper_datasets_v2")
        for suffix in ("generator", "created_at", "expires_at", "name"):
            assert f"idx_juniper_datasets_v2_{suffix}" in sql
        assert "idx_datasets_" not in sql


class TestIdentifierGatingOnEveryBuilder:
    """#343 gated the schema builder. The UPDATE and INSERT interpolate the same position."""

    @pytest.mark.parametrize(
        "builder",
        [build_schema_sql, build_update_sql, build_upsert_sql],
        ids=["schema", "update", "upsert"],
    )
    @pytest.mark.parametrize("hostile", _HOSTILE_IDENTIFIERS)
    def test_hostile_table_name_is_refused(self, builder, hostile):
        with pytest.raises(ValueError, match="unsafe SQL identifier"):
            builder(hostile)

    def test_safe_identifier_uses_fullmatch_not_search(self):
        """A valid prefix must not launder a hostile suffix."""
        with pytest.raises(ValueError, match="unsafe SQL identifier"):
            _safe_identifier("datasets;x")
        assert _safe_identifier("juniper_datasets_v2") == "juniper_datasets_v2"

    def test_builders_emit_the_validated_table_name(self):
        assert "UPDATE custom_table " in build_update_sql("custom_table")
        assert "INSERT INTO custom_table" in build_upsert_sql("custom_table")
