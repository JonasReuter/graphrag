# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Reranker implementations for GraphRAG query pipeline."""

from __future__ import annotations

import logging
import os
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CohereReranker:
    """Reranker using Cohere's Rerank API (synchronous client).

    Uses a single API call per rerank request — not one call per document.
    Suitable for use inside synchronous context builders.
    """

    def __init__(
        self,
        api_key: str = "",
        model: str = "rerank-v3.5",
        top_n: int | None = None,
    ) -> None:
        """Initialize CohereReranker.

        Args:
            api_key: Cohere API key. Falls back to the ``COHERE_API_KEY``
                environment variable when not provided explicitly.
            model: Cohere reranker model name.
            top_n: Maximum number of results to return. Defaults to all candidates.
        """
        try:
            import cohere
        except ImportError as e:
            msg = "cohere package is required for CohereReranker. Install with: pip install cohere"
            raise ImportError(msg) from e
        resolved_key = api_key or os.environ.get("COHERE_API_KEY", "")
        self._client = cohere.ClientV2(api_key=resolved_key)
        self.model = model
        self.top_n = top_n

    def rerank(
        self,
        query: str,
        documents: list[str],
        metadata: list[T],
    ) -> list[T]:
        """Rerank documents using Cohere Rerank API.

        Single API call for all documents — O(1) requests regardless of N.

        Args:
            query: The search query.
            documents: Text representations of each candidate (parallel to metadata).
            metadata: Original objects corresponding to each document.

        Returns:
            metadata list sorted by reranker relevance score (highest first).
            Falls back to original order if the API call fails.
        """
        if not documents or not metadata:
            return metadata

        try:
            response = self._client.rerank(
                model=self.model,
                query=query,
                documents=documents,
                top_n=self.top_n or len(documents),
            )
            n = len(metadata)
            return [metadata[r.index] for r in response.results if r.index < n]
        except Exception:
            logger.exception("Cohere reranking failed — returning original order")
            return metadata
