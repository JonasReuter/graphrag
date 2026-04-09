# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Unit tests for the index_graph_store operation.

ArangoDB I/O and the DataReader are fully mocked — no running database required.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from graphrag.index.operations.index_graph_store.index_graph_store import (
    index_graph_store,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_graph_store() -> MagicMock:
    gs = MagicMock()
    gs.setup_graph.return_value = None
    gs._ensure_entity_vector_index.return_value = None
    gs.upsert_entities.return_value = 3
    gs.upsert_relationships.return_value = 2
    gs.upsert_communities.return_value = (2, {"0": "c-uuid-0", "1": "c-uuid-1"})
    gs.upsert_community_reports.return_value = 2
    gs.upsert_text_units.return_value = 3
    gs.upsert_entity_community_edges_from_communities.return_value = 4
    gs.upsert_entity_text_unit_edges.return_value = 5
    return gs


def _make_entities_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"id": "ent-0", "title": "Entity0", "degree": 1, "community_ids": ["0"], "text_unit_ids": ["tu-0"]},
        {"id": "ent-1", "title": "Entity1", "degree": 2, "community_ids": ["1"], "text_unit_ids": ["tu-1"]},
        {"id": "ent-2", "title": "Entity2", "degree": 3, "community_ids": [],    "text_unit_ids": []},
    ])


def _make_relationships_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"id": "rel-0", "source": "Entity0", "target": "Entity1", "weight": 1.0},
        {"id": "rel-1", "source": "Entity1", "target": "Entity2", "weight": 0.8},
    ])


def _make_communities_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"id": "c-uuid-0", "community": "0", "title": "C0"},
        {"id": "c-uuid-1", "community": "1", "title": "C1"},
    ])


def _make_reports_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"id": "rep-0", "community": "0", "title": "Report 0", "summary": "s", "full_content": "fc", "rank": 1.0},
    ])


def _make_text_units_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"id": "tu-0", "text": "Text 0", "n_tokens": 50},
        {"id": "tu-1", "text": "Text 1", "n_tokens": 60},
    ])


def _make_reader(
    entities_df: pd.DataFrame | None = None,
    relationships_df: pd.DataFrame | None = None,
    communities_df: pd.DataFrame | None = None,
    reports_df: pd.DataFrame | None = None,
    text_units_df: pd.DataFrame | None = None,
) -> MagicMock:
    """Build a mock DataReader returning the given DataFrames."""
    reader = MagicMock()
    reader.entities = AsyncMock(return_value=entities_df if entities_df is not None else _make_entities_df())
    reader.relationships = AsyncMock(return_value=relationships_df if relationships_df is not None else _make_relationships_df())
    reader.communities = AsyncMock(return_value=communities_df if communities_df is not None else _make_communities_df())
    reader.community_reports = AsyncMock(return_value=reports_df if reports_df is not None else _make_reports_df())
    reader.text_units = AsyncMock(return_value=text_units_df if text_units_df is not None else _make_text_units_df())
    return reader


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestIndexGraphStoreOperation:
    def _run(self, graph_store, reader, **kwargs):
        async def _inner():
            with patch(
                "graphrag.index.operations.index_graph_store.index_graph_store.DataReader",
                return_value=reader,
            ):
                return await index_graph_store(graph_store, table_provider=MagicMock(), **kwargs)
        return asyncio.run(_inner())

    def test_returns_counts_dict(self):
        gs = _make_graph_store()
        counts = self._run(gs, _make_reader(), store_vectors=False)

        assert "entities" in counts
        assert "relationships" in counts
        assert "communities" in counts
        assert "community_reports" in counts
        assert "text_units" in counts
        assert "entity_community_membership" in counts
        assert "entity_text_unit" in counts

    def test_setup_graph_called_first(self):
        call_order = []
        gs = _make_graph_store()
        gs.setup_graph.side_effect = lambda: call_order.append("setup_graph")
        gs.upsert_entities.side_effect = lambda *a, **kw: call_order.append("upsert_entities") or 3

        self._run(gs, _make_reader(), store_vectors=False)

        assert call_order.index("setup_graph") < call_order.index("upsert_entities")

    def test_entities_upserted_before_relationships(self):
        call_order = []
        gs = _make_graph_store()
        gs.upsert_entities.side_effect = lambda *a, **kw: call_order.append("entities") or 3
        gs.upsert_relationships.side_effect = lambda *a, **kw: call_order.append("relationships") or 2

        self._run(gs, _make_reader(), store_vectors=False)

        assert call_order.index("entities") < call_order.index("relationships")

    def test_communities_upserted_before_membership_edges(self):
        call_order = []
        gs = _make_graph_store()
        gs.upsert_communities.side_effect = lambda rows: (call_order.append("communities") or (2, {"0": "c0"}))
        gs.upsert_entity_community_edges_from_communities.side_effect = lambda *a, **kw: call_order.append("membership") or 2

        self._run(gs, _make_reader(), store_vectors=False)

        assert call_order.index("communities") < call_order.index("membership")

    def test_title_to_id_map_passed_to_relationships(self):
        gs = _make_graph_store()
        self._run(gs, _make_reader(), store_vectors=False)

        _, kwargs = gs.upsert_relationships.call_args
        title_to_id = kwargs.get("title_to_id") or gs.upsert_relationships.call_args[1].get("title_to_id", {})
        # All 3 entities should be resolvable
        assert title_to_id.get("Entity0") == "ent-0"
        assert title_to_id.get("Entity1") == "ent-1"
        assert title_to_id.get("Entity2") == "ent-2"

    def test_community_rows_passed_to_membership_edges(self):
        """Community rows from the reader are forwarded to edge creation."""
        gs = _make_graph_store()
        communities_df = _make_communities_df()
        reader = _make_reader(communities_df=communities_df)

        self._run(gs, reader, store_vectors=False)

        call_args = gs.upsert_entity_community_edges_from_communities.call_args
        assert call_args is not None
        passed_rows = call_args[0][0]
        assert len(passed_rows) == len(communities_df)

    def test_skip_vectors_when_store_vectors_false(self):
        gs = _make_graph_store()
        self._run(gs, _make_reader(), store_vectors=False)

        _, kwargs = gs.upsert_entities.call_args
        vector_map = kwargs.get("vector_map")
        assert not vector_map  # None or empty dict

    def test_graceful_degradation_when_communities_missing(self):
        reader = _make_reader()
        reader.communities = AsyncMock(side_effect=Exception("table not found"))

        gs = _make_graph_store()
        # Should not raise — just skip communities and membership edges
        counts = self._run(gs, reader, store_vectors=False)

        assert counts.get("entities", 0) > 0  # entities still indexed


class TestIndexGraphStoreVectorCopy:
    def test_reads_vectors_from_arangodb_when_store_vectors_true(self):
        mock_db = MagicMock()
        mock_cursor = iter([
            {"id": "ent-0", "vector": [0.1, 0.2, 0.3, 0.4]},
            {"id": "ent-1", "vector": [0.5, 0.6, 0.7, 0.8]},
        ])
        mock_db.aql.execute.return_value = mock_cursor

        captured_vector_map = {}

        gs = _make_graph_store()
        def capture_entities(rows, vector_map=None):
            if vector_map:
                captured_vector_map.update(vector_map)
            return len(rows)
        gs.upsert_entities.side_effect = capture_entities

        async def _inner():
            with patch(
                "graphrag.index.operations.index_graph_store.index_graph_store.DataReader",
                return_value=_make_reader(),
            ):
                return await index_graph_store(
                    gs,
                    table_provider=MagicMock(),
                    store_vectors=True,
                    db=mock_db,
                )
        asyncio.run(_inner())

        assert "ent-0" in captured_vector_map
        assert captured_vector_map["ent-0"] == [0.1, 0.2, 0.3, 0.4]
