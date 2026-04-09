# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Unit tests for CohereReranker.

The Cohere client is mocked — no real API calls are made.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from graphrag.query.context_builder.reranker import CohereReranker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_cohere_client(ranked_indices: list[int]) -> MagicMock:
    """Build a mock Cohere ClientV2 whose rerank() returns results in the given index order."""
    result_items = [MagicMock(index=i) for i in ranked_indices]
    response = MagicMock()
    response.results = result_items
    client = MagicMock()
    client.rerank.return_value = response
    return client


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCohereReranker:
    def _make_reranker(self, ranked_indices: list[int]) -> tuple[CohereReranker, MagicMock]:
        mock_client = _make_mock_cohere_client(ranked_indices)
        with patch("graphrag.query.context_builder.reranker.CohereReranker.__init__", lambda self, **kw: None):
            reranker = CohereReranker.__new__(CohereReranker)
            reranker._client = mock_client
            reranker.model = "rerank-v3.5"
            reranker.top_n = None
        return reranker, mock_client

    def test_rerank_reorders_metadata(self):
        """Metadata is returned in the order determined by the reranker scores."""
        reranker, _ = self._make_reranker(ranked_indices=[2, 0, 1])
        documents = ["doc A", "doc B", "doc C"]
        metadata = ["meta A", "meta B", "meta C"]

        result = reranker.rerank(query="query", documents=documents, metadata=metadata)

        assert result == ["meta C", "meta A", "meta B"]

    def test_rerank_passes_correct_args_to_api(self):
        """Correct query, documents, model, and top_n are forwarded to the Cohere client."""
        reranker, mock_client = self._make_reranker(ranked_indices=[0, 1])
        documents = ["first", "second"]
        metadata = [{"id": 1}, {"id": 2}]

        reranker.rerank(query="test query", documents=documents, metadata=metadata)

        mock_client.rerank.assert_called_once_with(
            model="rerank-v3.5",
            query="test query",
            documents=documents,
            top_n=2,  # top_n=None → falls back to len(documents)
        )

    def test_rerank_empty_documents_returns_original(self):
        """Empty input is returned unchanged without calling the API."""
        reranker, mock_client = self._make_reranker([])
        result = reranker.rerank(query="q", documents=[], metadata=[])
        assert result == []
        mock_client.rerank.assert_not_called()

    def test_rerank_api_failure_returns_original_order(self):
        """If the Cohere API raises, original order is preserved (graceful degradation)."""
        mock_client = MagicMock()
        mock_client.rerank.side_effect = RuntimeError("API unavailable")
        with patch("graphrag.query.context_builder.reranker.CohereReranker.__init__", lambda self, **kw: None):
            reranker = CohereReranker.__new__(CohereReranker)
            reranker._client = mock_client
            reranker.model = "rerank-v3.5"
            reranker.top_n = None

        metadata = ["a", "b", "c"]
        result = reranker.rerank(query="q", documents=["x", "y", "z"], metadata=metadata)
        assert result == metadata

    def test_rerank_top_n_is_forwarded(self):
        """When top_n is set, it is passed to the API and limits the result."""
        mock_client = _make_mock_cohere_client(ranked_indices=[1, 0])  # top_n=2 results
        mock_client.rerank.return_value.results = [MagicMock(index=1), MagicMock(index=0)]
        with patch("graphrag.query.context_builder.reranker.CohereReranker.__init__", lambda self, **kw: None):
            reranker = CohereReranker.__new__(CohereReranker)
            reranker._client = mock_client
            reranker.model = "rerank-v3.5"
            reranker.top_n = 2

        reranker.rerank(query="q", documents=["x", "y", "z"], metadata=["a", "b", "c"])
        mock_client.rerank.assert_called_once_with(
            model="rerank-v3.5",
            query="q",
            documents=["x", "y", "z"],
            top_n=2,
        )

    def test_import_error_without_cohere(self):
        """ImportError is raised with a helpful message if cohere is not installed."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "cohere":
                raise ImportError("No module named 'cohere'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(ImportError, match="cohere package is required"):
                CohereReranker(api_key="test")
