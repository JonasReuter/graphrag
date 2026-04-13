# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Unit tests for get_graph_global_search_engine and graph_global_search API.

ArangoDB I/O, the LLM, and the vector store are all mocked so the tests run
without an ArangoDB instance.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from graphrag.data_model.community_report import CommunityReport
from graphrag.query.structured_search.global_search.search import GlobalSearch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_report(i: int, has_embedding: bool = True) -> CommunityReport:
    r = CommunityReport(
        id=f"rep-{i}",
        short_id=str(i),
        title=f"Report {i}",
        community_id=str(i),
        summary=f"Summary {i}",
        full_content=f"Full content {i}",
        rank=float(i),
    )
    if has_embedding:
        r.full_content_embedding = [0.1] * 4
    return r


def _make_config() -> MagicMock:
    cfg = MagicMock()
    cfg.graph_store.url = "http://localhost:8529"
    cfg.graph_store.username = "root"
    cfg.graph_store.password = "test"
    cfg.graph_store.db_name = "graphrag"
    cfg.graph_store.graph_name = "knowledge_graph"
    cfg.graph_store.vector_size = 4
    cfg.global_search.completion_model_id = "default"
    cfg.global_search.data_max_tokens = 8000
    cfg.global_search.max_context_tokens = 8000
    cfg.global_search.map_max_length = 1000
    cfg.global_search.reduce_max_length = 2000
    cfg.global_search.map_prompt = None
    cfg.global_search.reduce_prompt = None
    cfg.global_search.knowledge_prompt = None
    cfg.concurrent_requests = 4
    model_settings = MagicMock()
    model_settings.call_args = {}
    cfg.get_completion_model_config.return_value = model_settings
    return cfg


# ---------------------------------------------------------------------------
# Tests: get_graph_global_search_engine
# ---------------------------------------------------------------------------

class TestGetGraphGlobalSearchEngine:
    """get_graph_global_search_engine builds a GlobalSearch with ArangoDB data."""

    def _run(self, reports: list[CommunityReport], embedding_store=None):
        from graphrag.query.factory import get_graph_global_search_engine

        config = _make_config()
        mock_model = MagicMock()
        mock_model.tokenizer = MagicMock()

        with (
            patch("graphrag.query.factory.create_completion", return_value=mock_model),
            patch("graphrag_vectors.arangodb_graph.ArangoDBGraphStore") as MockStore,
            patch(
                "graphrag.query.input.retrieval.arangodb_graph_retriever.ArangoDBGraphRetriever"
            ) as MockRetriever,
        ):
            mock_store_instance = MagicMock()
            MockStore.return_value = mock_store_instance
            mock_retriever_instance = MagicMock()
            mock_retriever_instance.get_all_community_reports.return_value = reports
            MockRetriever.return_value = mock_retriever_instance

            engine = get_graph_global_search_engine(
                config=config,
                full_content_embedding_store=embedding_store,
                response_type="Multiple Paragraphs",
            )
            return engine, mock_store_instance, mock_retriever_instance

    def test_returns_global_search_instance(self):
        reports = [_make_report(i) for i in range(3)]
        engine, _, _ = self._run(reports)
        assert isinstance(engine, GlobalSearch)

    def test_connects_to_arangodb(self):
        reports = [_make_report(0)]
        _, store, _ = self._run(reports)
        store.connect.assert_called_once()

    def test_loads_all_community_reports(self):
        reports = [_make_report(i) for i in range(5)]
        _, _, retriever = self._run(reports)
        retriever.get_all_community_reports.assert_called_once()

    def test_filters_reports_without_embedding_when_store_given(self):
        """Reports without full_content_embedding are dropped when a store is provided."""
        from graphrag.query.factory import get_graph_global_search_engine
        from graphrag.query.indexer_adapters import read_indexer_report_embeddings

        reports_with = [_make_report(i, has_embedding=True) for i in range(3)]
        reports_without = [_make_report(i + 10, has_embedding=False) for i in range(2)]
        all_reports = reports_with + reports_without

        config = _make_config()
        mock_model = MagicMock()
        mock_model.tokenizer = MagicMock()
        mock_store = MagicMock()

        with (
            patch("graphrag.query.factory.create_completion", return_value=mock_model),
            patch("graphrag_vectors.arangodb_graph.ArangoDBGraphStore"),
            patch(
                "graphrag.query.input.retrieval.arangodb_graph_retriever.ArangoDBGraphRetriever"
            ) as MockRetriever,
            patch("graphrag.query.factory.read_indexer_report_embeddings"),
        ):
            MockRetriever.return_value.get_all_community_reports.return_value = all_reports

            engine = get_graph_global_search_engine(
                config=config,
                full_content_embedding_store=mock_store,
                response_type="Multiple Paragraphs",
            )

        # Only the 3 reports with embeddings should reach the context builder
        context_reports = engine.context_builder.community_reports
        assert len(context_reports) == 3
        assert all(r.full_content_embedding is not None for r in context_reports)

    def test_no_embedding_store_keeps_all_reports_with_embeddings(self):
        """Without a store, only reports that already have embeddings are kept."""
        reports = [_make_report(i, has_embedding=(i % 2 == 0)) for i in range(6)]
        engine, _, _ = self._run(reports, embedding_store=None)
        # Reports with indices 0, 2, 4 have embeddings
        assert len(engine.context_builder.community_reports) == 3

    def test_empty_report_list_yields_empty_context(self):
        engine, _, _ = self._run([])
        assert engine.context_builder.community_reports == []

