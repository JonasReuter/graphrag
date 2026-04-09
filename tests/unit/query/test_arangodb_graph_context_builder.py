# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Unit tests for ArangoDBGraphContextBuilder.

ArangoDB I/O, the LLM embedder, and the vector store are all mocked.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from graphrag.data_model.community_report import CommunityReport
from graphrag.data_model.entity import Entity
from graphrag.data_model.relationship import Relationship
from graphrag.data_model.text_unit import TextUnit
from graphrag.query.context_builder.builders import ContextBuilderResult
from graphrag.query.structured_search.local_search.graph_context import (
    ArangoDBGraphContextBuilder,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_entities(n: int = 3) -> list[Entity]:
    return [
        Entity(
            id=f"ent-{i}",
            short_id=str(i),
            title=f"Entity{i}",
            type="person",
            description=f"Description {i}",
            rank=i + 1,
            community_ids=[str(i)],
            text_unit_ids=[f"tu-{i}"],
        )
        for i in range(n)
    ]


def _make_relationships(entities: list[Entity]) -> list[Relationship]:
    rels = []
    for i in range(len(entities) - 1):
        rels.append(Relationship(
            id=f"rel-{i}",
            short_id=str(i),
            source=entities[i].title,
            target=entities[i + 1].title,
            weight=1.0,
            rank=i + 2,
        ))
    return rels


def _make_community_reports(n: int = 2) -> list[CommunityReport]:
    return [
        CommunityReport(
            id=f"rep-{i}",
            short_id=str(i),
            title=f"Report {i}",
            community_id=str(i),
            summary=f"Summary {i}",
            full_content=f"Full content {i}",
            rank=float(i + 1),
        )
        for i in range(n)
    ]


def _make_text_units(n: int = 3) -> list[TextUnit]:
    return [
        TextUnit(
            id=f"tu-{i}",
            short_id=str(i),
            text=f"Source text {i}",
            n_tokens=50,
        )
        for i in range(n)
    ]


def _make_mock_graph_store(
    entities: list[Entity] | None = None,
    relationships: list[Relationship] | None = None,
    reports: list[CommunityReport] | None = None,
) -> MagicMock:
    """Mock graph store whose traverse/hybrid returns the provided domain rows as dicts."""
    gs = MagicMock()

    ent_list = entities or _make_entities()
    rel_list = relationships or _make_relationships(ent_list)
    rep_list = reports or _make_community_reports()

    # Convert domain objects back to dict (simulating raw ArangoDB docs)
    def _entity_doc(e: Entity) -> dict:
        return {
            "id": e.id,
            "_key": e.id,
            "title": e.title,
            "type": e.type,
            "description": e.description,
            "degree": e.rank,
            "community_ids": e.community_ids,
            "text_unit_ids": e.text_unit_ids,
        }

    def _rel_doc(r: Relationship) -> dict:
        return {
            "id": r.id,
            "_key": r.id,
            "source": r.source,
            "target": r.target,
            "weight": r.weight,
            "combined_degree": r.rank,
            "description": r.description,
        }

    def _rep_doc(r: CommunityReport) -> dict:
        return {
            "id": r.id,
            "_key": r.id,
            "title": r.title,
            "community": r.community_id,
            "summary": r.summary,
            "full_content": r.full_content,
            "rank": r.rank,
        }

    ent_docs = [_entity_doc(e) for e in ent_list]
    rel_docs = [_rel_doc(r) for r in rel_list]
    rep_docs = [_rep_doc(r) for r in rep_list]

    gs.hybrid_search.return_value = (ent_docs, rel_docs)
    gs.traverse_neighbors.return_value = (ent_docs, rel_docs)
    gs.get_community_reports_for_entities.return_value = rep_docs
    return gs


def _make_builder(
    graph_store: MagicMock | None = None,
    text_units: list[TextUnit] | None = None,
    community_reports: list[CommunityReport] | None = None,
    use_hybrid_search: bool = True,
) -> ArangoDBGraphContextBuilder:
    mock_gs = graph_store or _make_mock_graph_store()

    mock_embedder = MagicMock()
    mock_embedder.embed_query.return_value = [0.1] * 4

    mock_vector_store = MagicMock()

    return ArangoDBGraphContextBuilder(
        graph_store=mock_gs,
        entity_text_embeddings=mock_vector_store,
        text_embedder=mock_embedder,
        community_reports=community_reports,
        text_units=text_units,
        traversal_depth=1,
        top_k_seeds=5,
        use_hybrid_search=use_hybrid_search,
    )


# ---------------------------------------------------------------------------
# Tests: return type
# ---------------------------------------------------------------------------

class TestBuildContextReturnType:
    def test_returns_context_builder_result(self):
        builder = _make_builder()
        result = builder.build_context(query="Who is Entity0?")

        assert isinstance(result, ContextBuilderResult)
        assert isinstance(result.context_chunks, str)
        assert isinstance(result.context_records, dict)

    def test_context_chunks_is_non_empty_string(self):
        builder = _make_builder(text_units=_make_text_units())
        result = builder.build_context(query="test query", max_context_tokens=4000)

        assert len(result.context_chunks) > 0


# ---------------------------------------------------------------------------
# Tests: hybrid search path
# ---------------------------------------------------------------------------

class TestHybridSearchPath:
    def test_calls_hybrid_search_when_enabled(self):
        gs = _make_mock_graph_store()
        builder = _make_builder(graph_store=gs, use_hybrid_search=True)

        builder.build_context(query="test query")

        gs.hybrid_search.assert_called_once()
        gs.traverse_neighbors.assert_not_called()

    def test_hybrid_search_receives_query_embedding(self):
        gs = _make_mock_graph_store()
        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = [0.9, 0.8, 0.7, 0.6]

        builder = ArangoDBGraphContextBuilder(
            graph_store=gs,
            entity_text_embeddings=MagicMock(),
            text_embedder=mock_embedder,
            traversal_depth=1,
            top_k_seeds=3,
            use_hybrid_search=True,
        )
        builder.build_context(query="test query")

        # query_vector is passed positionally, k as keyword
        call_args, call_kwargs = gs.hybrid_search.call_args
        assert list(call_args[0]) == [0.9, 0.8, 0.7, 0.6]
        assert call_kwargs["k"] == 3

    def test_uses_traverse_when_hybrid_disabled(self):
        gs = _make_mock_graph_store()
        gs.hybrid_search.side_effect = AssertionError("should not be called")

        mock_vs = MagicMock()
        mock_vs.similarity_search_by_vector.return_value = []

        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = [0.1] * 4

        builder = ArangoDBGraphContextBuilder(
            graph_store=gs,
            entity_text_embeddings=mock_vs,
            text_embedder=mock_embedder,
            use_hybrid_search=False,
        )
        # Should not raise
        builder.build_context(query="test query")


# ---------------------------------------------------------------------------
# Tests: community context
# ---------------------------------------------------------------------------

class TestCommunityContext:
    def test_community_reports_appear_in_context_records(self):
        builder = _make_builder()
        result = builder.build_context(query="test", community_prop=0.3)

        assert "reports" in result.context_records

    def test_falls_back_to_preloaded_reports_when_graph_returns_empty(self):
        gs = _make_mock_graph_store()
        gs.get_community_reports_for_entities.return_value = []

        preloaded = _make_community_reports(2)
        entities = _make_entities(2)
        # entity.community_ids = ["0"] and ["1"] — match community_id of reports
        gs.hybrid_search.return_value = (
            [{"id": e.id, "_key": e.id, "title": e.title, "type": e.type,
              "description": e.description, "degree": e.rank,
              "community_ids": e.community_ids, "text_unit_ids": e.text_unit_ids,
              "human_readable_id": e.short_id}
             for e in entities],
            [],
        )

        builder = _make_builder(graph_store=gs, community_reports=preloaded)
        result = builder.build_context(query="test", community_prop=0.3, max_context_tokens=4000)

        # The context_records should have the reports section populated
        reports_df = result.context_records.get("reports")
        assert reports_df is not None


# ---------------------------------------------------------------------------
# Tests: text unit context
# ---------------------------------------------------------------------------

class TestTextUnitContext:
    def test_text_units_not_in_context_when_not_provided(self):
        builder = _make_builder(text_units=None)
        result = builder.build_context(query="test", text_unit_prop=0.5, max_context_tokens=4000)

        # No sources section since text_units=None
        assert "sources" not in result.context_records

    def test_text_units_included_when_provided(self):
        builder = _make_builder(text_units=_make_text_units(3))
        result = builder.build_context(query="test", text_unit_prop=0.5, max_context_tokens=4000)

        # Sources section should be present
        assert "sources" in result.context_records


# ---------------------------------------------------------------------------
# Tests: token budget validation
# ---------------------------------------------------------------------------

class TestTokenBudgetValidation:
    def test_raises_when_props_exceed_one(self):
        builder = _make_builder()
        with pytest.raises(ValueError, match="must not exceed 1"):
            builder.build_context(
                query="test",
                community_prop=0.6,
                text_unit_prop=0.6,
            )

    def test_accepts_props_summing_to_exactly_one(self):
        builder = _make_builder()
        # Should not raise
        builder.build_context(
            query="test",
            community_prop=0.5,
            text_unit_prop=0.5,
        )


# ---------------------------------------------------------------------------
# Tests: traversal_depth override
# ---------------------------------------------------------------------------

class TestTraversalDepth:
    def test_custom_depth_passed_to_hybrid_search(self):
        gs = _make_mock_graph_store()
        builder = _make_builder(graph_store=gs, use_hybrid_search=True)

        builder.build_context(query="test", traversal_depth=4)

        call_kwargs = gs.hybrid_search.call_args[1]
        assert call_kwargs["depth"] == 4

    def test_default_depth_used_when_not_overridden(self):
        gs = _make_mock_graph_store()
        builder = ArangoDBGraphContextBuilder(
            graph_store=gs,
            entity_text_embeddings=MagicMock(),
            text_embedder=MagicMock(),
            traversal_depth=3,    # set at construction
            use_hybrid_search=True,
        )
        builder.text_embedder.embed_query.return_value = [0.1] * 4

        builder.build_context(query="test")

        call_kwargs = gs.hybrid_search.call_args[1]
        assert call_kwargs["depth"] == 3
