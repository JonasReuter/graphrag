# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Unit tests for relationship→source and text-unit→relationship provenance.

Tests the "linked tables" design:
- build_relationship_context() adds a source_id column (short_id of primary source text unit)
- build_text_unit_context() adds a relationship_ids column (short_ids of contained relationships)
"""

from __future__ import annotations

import pytest

from graphrag.data_model.entity import Entity
from graphrag.data_model.relationship import Relationship
from graphrag.data_model.text_unit import TextUnit
from graphrag.query.context_builder.local_context import build_relationship_context
from graphrag.query.context_builder.source_context import build_text_unit_context


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def text_units() -> dict[str, TextUnit]:
    return {
        "tu-1": TextUnit(
            id="tu-1",
            short_id="s1",
            text="Alice and Bob work at Acme.",
            relationship_ids=["rel-1", "rel-2"],
        ),
        "tu-2": TextUnit(
            id="tu-2",
            short_id="s2",
            text="Carol leads Beta division.",
            relationship_ids=["rel-3"],
        ),
    }


@pytest.fixture
def relationships() -> list[Relationship]:
    return [
        Relationship(
            id="rel-1",
            short_id="r1",
            source="Alice",
            target="Acme",
            description="works at",
            text_unit_ids=["tu-1"],
            rank=3,
        ),
        Relationship(
            id="rel-2",
            short_id="r2",
            source="Bob",
            target="Acme",
            description="works at",
            text_unit_ids=["tu-1"],
            rank=2,
        ),
        Relationship(
            id="rel-3",
            short_id="r3",
            source="Carol",
            target="Beta",
            description="leads",
            text_unit_ids=["tu-2"],
            rank=1,
        ),
    ]


@pytest.fixture
def entities() -> list[Entity]:
    return [
        Entity(id="e1", short_id="e1", title="Alice", rank=3, text_unit_ids=["tu-1"]),
        Entity(id="e2", short_id="e2", title="Bob", rank=2, text_unit_ids=["tu-1"]),
        Entity(id="e3", short_id="e3", title="Carol", rank=1, text_unit_ids=["tu-2"]),
    ]


# ---------------------------------------------------------------------------
# Tests: Relationship → source_id
# ---------------------------------------------------------------------------

class TestRelationshipSourceId:
    def test_source_id_column_present_when_text_units_provided(
        self, entities, relationships, text_units
    ):
        """source_id column appears in context when text_units dict is passed."""
        _, df = build_relationship_context(
            selected_entities=entities,
            relationships=relationships,
            text_units=text_units,
        )
        assert "source_id" in df.columns

    def test_source_id_resolves_to_short_id(self, entities, relationships, text_units):
        """source_id is the short_id of the primary source text unit."""
        _, df = build_relationship_context(
            selected_entities=entities,
            relationships=relationships,
            text_units=text_units,
        )
        # rel-1 → tu-1 → short_id="s1"
        row = df[df["id"] == "r1"]
        assert not row.empty
        assert row.iloc[0]["source_id"] == "s1"

    def test_source_id_absent_without_text_units(self, entities, relationships):
        """source_id column is NOT added when text_units is not provided (backward compat)."""
        _, df = build_relationship_context(
            selected_entities=entities,
            relationships=relationships,
        )
        assert "source_id" not in df.columns

    def test_source_id_empty_when_relationship_has_no_text_unit_ids(
        self, entities, text_units
    ):
        """source_id is empty string when the relationship has no text_unit_ids."""
        rels_without_source = [
            Relationship(
                id="rel-x",
                short_id="rx",
                source="Alice",
                target="Bob",
                description="knows",
                text_unit_ids=None,
                rank=1,
            )
        ]
        _, df = build_relationship_context(
            selected_entities=entities,
            relationships=rels_without_source,
            text_units=text_units,
        )
        assert df.iloc[0]["source_id"] == ""


# ---------------------------------------------------------------------------
# Tests: Text Unit → relationship_ids
# ---------------------------------------------------------------------------

class TestTextUnitRelationshipIds:
    def test_relationship_ids_column_present_when_relationships_provided(
        self, text_units, relationships
    ):
        """relationship_ids column appears when relationships dict is passed."""
        rel_dict = {r.id: r for r in relationships}
        _, ctx = build_text_unit_context(
            text_units=list(text_units.values()),
            shuffle_data=False,
            relationships=rel_dict,
        )
        df = ctx["sources"]
        assert "relationship_ids" in df.columns

    def test_relationship_ids_resolves_to_short_ids(self, text_units, relationships):
        """relationship_ids contains comma-separated short_ids from text_unit.relationship_ids."""
        rel_dict = {r.id: r for r in relationships}
        _, ctx = build_text_unit_context(
            text_units=list(text_units.values()),
            shuffle_data=False,
            relationships=rel_dict,
        )
        df = ctx["sources"]
        row = df[df["id"] == "s1"]
        assert not row.empty
        rel_ids = set(row.iloc[0]["relationship_ids"].split(","))
        assert rel_ids == {"r1", "r2"}

    def test_relationship_ids_absent_without_relationships(self, text_units):
        """relationship_ids column is NOT added when relationships is not provided (backward compat)."""
        _, ctx = build_text_unit_context(
            text_units=list(text_units.values()),
            shuffle_data=False,
        )
        df = ctx["sources"]
        assert "relationship_ids" not in df.columns

    def test_relationship_ids_empty_when_text_unit_has_none(self, relationships):
        """relationship_ids is empty string when text_unit has no relationship_ids."""
        units = [TextUnit(id="tu-x", short_id="sx", text="No relations here.", relationship_ids=None)]
        rel_dict = {r.id: r for r in relationships}
        _, ctx = build_text_unit_context(
            text_units=units,
            shuffle_data=False,
            relationships=rel_dict,
        )
        df = ctx["sources"]
        assert df.iloc[0]["relationship_ids"] == ""
