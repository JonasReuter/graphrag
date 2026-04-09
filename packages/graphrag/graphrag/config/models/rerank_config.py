# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Reranker configuration model."""

from typing import Literal

from pydantic import BaseModel, Field


class RerankConfig(BaseModel):
    """Configuration for the optional reranker pipeline step."""

    enabled: bool = Field(
        description="Whether to enable reranking of retrieved context candidates.",
        default=False,
    )
    provider: Literal["cohere"] = Field(
        description="Reranker provider. Currently only 'cohere' is supported.",
        default="cohere",
    )
    api_key: str = Field(
        description="API key for the reranker provider.",
        default="",
    )
    model: str = Field(
        description="Reranker model name.",
        default="rerank-v3.5",
    )
    rerank_text_units: bool = Field(
        description=(
            "Rerank text unit (source chunk) candidates before token-budget fill. "
            "Highest impact: text units receive 50%% of the context token budget."
        ),
        default=True,
    )
    rerank_communities: bool = Field(
        description=(
            "Rerank community report candidates before token-budget fill. "
            "Community reports receive 25%% of the context token budget."
        ),
        default=True,
    )
    direct_text_unit_search_k: int = Field(
        description=(
            "Number of text units to retrieve via direct vector search (Path B). "
            "These are unioned with entity-derived text units (Path A) before reranking, "
            "replicating the classic RAG oversample-then-rerank pattern."
        ),
        default=50,
    )
