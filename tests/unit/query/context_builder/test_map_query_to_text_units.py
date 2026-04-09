# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Unit tests for map_query_to_text_units (Path B direct text unit retrieval)."""

from __future__ import annotations

from typing import Any

from graphrag.data_model.text_unit import TextUnit
from graphrag.query.context_builder.entity_extraction import map_query_to_text_units
from graphrag_llm.config import LLMProviderType, ModelConfig
from graphrag_llm.embedding import create_embedding
from graphrag_vectors import (
    TextEmbedder,
    VectorStore,
    VectorStoreDocument,
    VectorStoreSearchResult,
)

embedding_model = create_embedding(
    ModelConfig(
        type=LLMProviderType.MockLLM,
        model_provider="openai",
        model="text-embedding-3-small",
        mock_responses=[1.0, 1.0, 1.0],
    )
)


class MockTextUnitVectorStore(VectorStore):
    """VectorStore that returns documents sorted by text-length distance to query."""

    def __init__(self, documents: list[VectorStoreDocument]) -> None:
        super().__init__(index_name="mock_tu")
        self.documents = documents

    def connect(self, **kwargs: Any) -> None:
        raise NotImplementedError

    def create_index(self) -> None:
        raise NotImplementedError

    def load_documents(self, documents: list[VectorStoreDocument]) -> None:
        raise NotImplementedError

    def insert(self, document: VectorStoreDocument) -> None:
        raise NotImplementedError

    def similarity_search_by_vector(
        self,
        query_embedding: list[float],
        k: int = 10,
        select: list[str] | None = None,
        filters: Any = None,
        include_vectors: bool = True,
    ) -> list[VectorStoreSearchResult]:
        return [VectorStoreSearchResult(document=d, score=1) for d in self.documents[:k]]

    def similarity_search_by_text(
        self,
        text: str,
        text_embedder: TextEmbedder,
        k: int = 10,
        select: list[str] | None = None,
        filters: Any = None,
        include_vectors: bool = True,
    ) -> list[VectorStoreSearchResult]:
        return sorted(
            [VectorStoreSearchResult(document=d, score=abs(len(text) - len(str(d.id)))) for d in self.documents],
            key=lambda x: x.score,
        )[:k]

    def search_by_id(self, id: str, select: list[str] | None = None, include_vectors: bool = True) -> VectorStoreDocument:
        return self.documents[0]

    def count(self) -> int:
        return len(self.documents)

    def remove(self, ids: list[str]) -> None:
        raise NotImplementedError

    def update(self, document: VectorStoreDocument) -> None:
        raise NotImplementedError


def _make_text_units() -> dict[str, TextUnit]:
    return {
        f"tu-{i}": TextUnit(id=f"tu-{i}", short_id=f"s{i}", text=f"Text unit {i} content.")
        for i in range(5)
    }


def test_map_query_to_text_units_returns_matched_units():
    """Retrieved text units are looked up by ID in all_text_units dict."""
    all_units = _make_text_units()
    vectorstore = MockTextUnitVectorStore([
        VectorStoreDocument(id=uid, vector=None) for uid in all_units
    ])

    result = map_query_to_text_units(
        query="some query",
        text_unit_vectorstore=vectorstore,
        text_embedder=embedding_model,
        all_text_units=all_units,
        k=3,
    )

    assert len(result) == 3
    assert all(isinstance(u, TextUnit) for u in result)
    assert all(u.id in all_units for u in result)


def test_map_query_to_text_units_empty_query_returns_empty():
    """Empty query returns an empty list (no vector search performed)."""
    all_units = _make_text_units()
    vectorstore = MockTextUnitVectorStore([
        VectorStoreDocument(id=uid, vector=None) for uid in all_units
    ])

    result = map_query_to_text_units(
        query="",
        text_unit_vectorstore=vectorstore,
        text_embedder=embedding_model,
        all_text_units=all_units,
        k=3,
    )

    assert result == []


def test_map_query_to_text_units_respects_k():
    """Result length is bounded by k."""
    all_units = _make_text_units()
    vectorstore = MockTextUnitVectorStore([
        VectorStoreDocument(id=uid, vector=None) for uid in all_units
    ])

    result = map_query_to_text_units(
        query="query",
        text_unit_vectorstore=vectorstore,
        text_embedder=embedding_model,
        all_text_units=all_units,
        k=2,
    )

    assert len(result) <= 2


def test_map_query_to_text_units_skips_unknown_ids():
    """Vector store results with IDs not in all_text_units are silently skipped."""
    all_units = {"tu-1": TextUnit(id="tu-1", short_id="s1", text="Known unit.")}
    vectorstore = MockTextUnitVectorStore([
        VectorStoreDocument(id="tu-1", vector=None),
        VectorStoreDocument(id="tu-unknown", vector=None),
    ])

    result = map_query_to_text_units(
        query="query",
        text_unit_vectorstore=vectorstore,
        text_embedder=embedding_model,
        all_text_units=all_units,
        k=5,
    )

    assert len(result) == 1
    assert result[0].id == "tu-1"
