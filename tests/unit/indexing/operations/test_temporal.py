# Copyright (C) 2026 Microsoft Corporation.
# Licensed under the MIT License

"""Tests for temporal architecture features.

Covers:
- Data model temporal fields and from_dict deserialization
- Temporal scope normalization (temporal_utils)
- Timestamp propagation during entity/relationship merge
- Graph extractor temporal_scope parsing
- Context builder temporal columns
"""

import pandas as pd
import pytest

from graphrag.data_model.entity import Entity
from graphrag.data_model.relationship import Relationship
from graphrag.data_model.text_unit import TextUnit
from graphrag.index.operations.extract_graph.extract_graph import (
    _compute_temporal_bounds,
    _merge_entities,
    _merge_relationships,
    _resolve_temporal_scopes,
)
from graphrag.index.operations.extract_graph.graph_extractor import (
    GraphExtractor,
    TUPLE_DELIMITER,
    RECORD_DELIMITER,
    COMPLETION_DELIMITER,
)
from graphrag.index.operations.extract_graph.temporal_utils import (
    resolve_temporal_scope,
)


# ---------------------------------------------------------------
# Phase 1: Data model temporal fields
# ---------------------------------------------------------------


class TestEntityTemporalFields:
    def test_default_values(self):
        e = Entity(id="1", short_id=None, title="test")
        assert e.observed_at is None
        assert e.last_observed_at is None
        assert e.valid_from is None
        assert e.valid_until is None

    def test_from_dict_with_temporal(self):
        d = {
            "id": "1",
            "title": "ACME",
            "observed_at": "2024-01-01",
            "last_observed_at": "2024-06-15",
            "valid_from": "2020-01-01",
            "valid_until": "2024-12-31",
        }
        e = Entity.from_dict(d)
        assert e.observed_at == "2024-01-01"
        assert e.last_observed_at == "2024-06-15"
        assert e.valid_from == "2020-01-01"
        assert e.valid_until == "2024-12-31"

    def test_from_dict_without_temporal(self):
        """Backwards compatibility: old data without temporal fields."""
        d = {"id": "1", "title": "OLD_ENTITY"}
        e = Entity.from_dict(d)
        assert e.observed_at is None
        assert e.last_observed_at is None


class TestRelationshipTemporalFields:
    def test_default_values(self):
        r = Relationship(id="1", short_id=None, source="A", target="B")
        assert r.observed_at is None
        assert r.last_observed_at is None

    def test_from_dict_with_temporal(self):
        d = {
            "id": "1",
            "source": "A",
            "target": "B",
            "observed_at": "2024-03-01",
            "last_observed_at": "2024-09-01",
            "valid_from": "2024-03-01",
            "valid_until": None,
        }
        r = Relationship.from_dict(d)
        assert r.observed_at == "2024-03-01"
        assert r.valid_from == "2024-03-01"


class TestTextUnitTemporalFields:
    def test_default_values(self):
        t = TextUnit(id="1", short_id=None, text="hello")
        assert t.created_at is None

    def test_from_dict_with_created_at(self):
        d = {"id": "1", "text": "hello", "created_at": "2024-01-15"}
        t = TextUnit.from_dict(d)
        assert t.created_at == "2024-01-15"


# ---------------------------------------------------------------
# Phase 2: Temporal scope normalization
# ---------------------------------------------------------------


class TestResolveTemporalScope:
    def test_year_range(self):
        vf, vu, q = resolve_temporal_scope("2020-2023")
        assert vf == "2020-01-01"
        assert vu == "2023-12-31"
        assert q is None

    def test_year_range_with_spaces(self):
        vf, vu, q = resolve_temporal_scope("2020 - 2023")
        assert vf == "2020-01-01"
        assert vu == "2023-12-31"

    def test_year_to_present(self):
        vf, vu, q = resolve_temporal_scope("2020-present")
        assert vf == "2020-01-01"
        assert vu is None

    def test_since_month_year(self):
        vf, vu, q = resolve_temporal_scope("since January 2024")
        assert vf == "2024-01-01"
        assert vu is None

    def test_since_year(self):
        vf, vu, q = resolve_temporal_scope("since 2024")
        assert vf == "2024-01-01"
        assert vu is None

    def test_until_month_year(self):
        vf, vu, q = resolve_temporal_scope("until March 2025")
        assert vf is None
        assert vu == "2025-03-01"

    def test_quarter(self):
        vf, vu, q = resolve_temporal_scope("Q3 2024")
        assert vf == "2024-07-01"
        assert vu == "2024-09-30"

    def test_bare_year(self):
        vf, vu, q = resolve_temporal_scope("2024")
        assert vf == "2024-01-01"
        assert vu == "2024-12-31"

    def test_month_year(self):
        vf, vu, q = resolve_temporal_scope("January 2024")
        assert vf == "2024-01-01"
        assert vu is None

    def test_formerly(self):
        vf, vu, q = resolve_temporal_scope("formerly")
        assert vf is None
        assert vu is None
        assert q == "formerly"

    def test_current(self):
        vf, vu, q = resolve_temporal_scope("current")
        assert q == "current"

    def test_empty(self):
        vf, vu, q = resolve_temporal_scope("")
        assert vf is None
        assert vu is None
        assert q is None

    def test_none(self):
        vf, vu, q = resolve_temporal_scope(None)
        assert vf is None

    def test_unresolvable(self):
        vf, vu, q = resolve_temporal_scope("during the Renaissance period")
        assert vf is None
        assert vu is None
        assert q == "during the Renaissance period"

    def test_iso_date(self):
        vf, vu, q = resolve_temporal_scope("2024-01-15")
        assert vf == "2024-01-15"
        assert vu is None

    def test_german_seit(self):
        vf, vu, q = resolve_temporal_scope("seit Januar 2024")
        assert vf == "2024-01-01"

    def test_german_bis(self):
        vf, vu, q = resolve_temporal_scope("bis 2025")
        assert vu == "2025-12-31"


# ---------------------------------------------------------------
# Phase 2: Timestamp computation during merge
# ---------------------------------------------------------------


class TestComputeTemporalBounds:
    def test_with_timestamps(self):
        timestamps = {"tu1": "2024-01-01", "tu2": "2024-06-15", "tu3": "2024-03-01"}
        obs, last_obs = _compute_temporal_bounds(["tu1", "tu2", "tu3"], timestamps)
        assert obs == "2024-01-01"
        assert last_obs == "2024-06-15"

    def test_no_timestamps(self):
        obs, last_obs = _compute_temporal_bounds(["tu1"], None)
        assert obs is None
        assert last_obs is None

    def test_missing_ids(self):
        timestamps = {"tu1": "2024-01-01"}
        obs, last_obs = _compute_temporal_bounds(["tu_missing"], timestamps)
        assert obs is None
        assert last_obs is None

    def test_empty_timestamp_values(self):
        timestamps = {"tu1": "", "tu2": "2024-01-01"}
        obs, last_obs = _compute_temporal_bounds(["tu1", "tu2"], timestamps)
        assert obs == "2024-01-01"
        assert last_obs == "2024-01-01"


class TestMergeWithTimestamps:
    def test_entities_get_observed_at(self):
        df1 = pd.DataFrame([{
            "title": "A", "type": "ORG", "description": "desc", "source_id": "tu1",
        }])
        df2 = pd.DataFrame([{
            "title": "A", "type": "ORG", "description": "desc2", "source_id": "tu2",
        }])
        timestamps = {"tu1": "2024-01-01", "tu2": "2024-06-15"}
        merged = _merge_entities([df1, df2], timestamps)

        assert len(merged) == 1
        assert merged.iloc[0]["observed_at"] == "2024-01-01"
        assert merged.iloc[0]["last_observed_at"] == "2024-06-15"

    def test_entities_without_timestamps(self):
        df = pd.DataFrame([{
            "title": "A", "type": "ORG", "description": "desc", "source_id": "tu1",
        }])
        merged = _merge_entities([df])

        assert merged.iloc[0]["observed_at"] is None
        assert merged.iloc[0]["last_observed_at"] is None

    def test_relationships_get_observed_at(self):
        df = pd.DataFrame([{
            "source": "A", "target": "B", "weight": 1.0,
            "description": "linked", "source_id": "tu1",
        }])
        timestamps = {"tu1": "2024-03-01"}
        merged = _merge_relationships([df], timestamps)

        assert merged.iloc[0]["observed_at"] == "2024-03-01"
        assert merged.iloc[0]["last_observed_at"] == "2024-03-01"


class TestMergeRelationshipsWithTemporalScope:
    """Temporal scope is extracted for relationships, not entities."""

    def test_temporal_scope_preserved_on_relationship(self):
        df = pd.DataFrame([{
            "source": "A", "target": "B", "weight": 1.0,
            "description": "linked", "source_id": "tu1", "temporal_scope": "2024",
        }])
        merged = _merge_relationships([df])
        assert merged.iloc[0]["temporal_scope"] == "2024"

    def test_temporal_scope_first_nonempty_on_relationship(self):
        df1 = pd.DataFrame([{
            "source": "A", "target": "B", "weight": 1.0,
            "description": "d1", "source_id": "tu1", "temporal_scope": "",
        }])
        df2 = pd.DataFrame([{
            "source": "A", "target": "B", "weight": 1.0,
            "description": "d2", "source_id": "tu2", "temporal_scope": "since 2023",
        }])
        merged = _merge_relationships([df1, df2])
        assert merged.iloc[0]["temporal_scope"] == "since 2023"

    def test_entity_merge_has_no_temporal_scope(self):
        """Entities do not accumulate temporal_scope -- only observed_at."""
        df = pd.DataFrame([{
            "title": "A", "type": "ORG", "description": "desc", "source_id": "tu1",
        }])
        merged = _merge_entities([df])
        assert "temporal_scope" not in merged.columns


class TestResolveTemporalScopes:
    def test_resolves_to_valid_from_until(self):
        df = pd.DataFrame([{
            "title": "A", "temporal_scope": "2020-2023",
        }])
        result = _resolve_temporal_scopes(df)
        assert result.iloc[0]["valid_from"] == "2020-01-01"
        assert result.iloc[0]["valid_until"] == "2023-12-31"

    def test_no_temporal_scope_column(self):
        df = pd.DataFrame([{"title": "A"}])
        result = _resolve_temporal_scopes(df)
        assert result.iloc[0]["valid_from"] is None
        assert result.iloc[0]["valid_until"] is None


# ---------------------------------------------------------------
# Phase 2: Graph extractor temporal_scope parsing
# ---------------------------------------------------------------


class TestGraphExtractorTemporalParsing:
    """Test that _process_result correctly parses temporal_scope fields."""

    def _make_extractor(self):
        """Create a minimal GraphExtractor for testing _process_result."""
        # We need a mock model that won't be called
        class MockModel:
            pass
        return GraphExtractor(
            model=MockModel(),  # type: ignore
            prompt="",
            max_gleanings=0,
        )

    def test_entity_has_no_temporal_scope(self):
        """Entities never get temporal_scope -- only relationships do."""
        extractor = self._make_extractor()
        # Even if LLM outputs a 5th field, it is ignored for entities
        result = (
            '("entity"<|>ACME CORP<|>ORGANIZATION<|>A company)'
            f"\n##\n{COMPLETION_DELIMITER}"
        )
        entities_df, _, _ = extractor._process_result(
            result, "tu1", TUPLE_DELIMITER, RECORD_DELIMITER,
            extract_evidence=False, extract_temporal=True,
        )
        assert len(entities_df) == 1
        assert "temporal_scope" not in entities_df.columns

    def test_relationship_temporal_scope(self):
        extractor = self._make_extractor()
        result = (
            '("relationship"<|>ALICE<|>ACME<|>Alice works at Acme<|>8<|>since 2020)'
            f"\n##\n{COMPLETION_DELIMITER}"
        )
        _, rels_df, _ = extractor._process_result(
            result, "tu1", TUPLE_DELIMITER, RECORD_DELIMITER,
            extract_evidence=False, extract_temporal=True,
        )
        assert len(rels_df) == 1
        assert rels_df.iloc[0]["temporal_scope"] == "since 2020"

    def test_no_temporal_when_not_enabled(self):
        extractor = self._make_extractor()
        result = (
            '("relationship"<|>ALICE<|>ACME<|>Alice works at Acme<|>8<|>since 2020)'
            f"\n##\n{COMPLETION_DELIMITER}"
        )
        _, rels_df, _ = extractor._process_result(
            result, "tu1", TUPLE_DELIMITER, RECORD_DELIMITER,
            extract_evidence=False, extract_temporal=False,
        )
        # When not temporal, the 6th field is not parsed as temporal_scope
        assert "temporal_scope" not in rels_df.columns
