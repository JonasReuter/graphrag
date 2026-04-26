# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Parameterization settings for entity resolution configuration."""

from pathlib import Path

from pydantic import BaseModel, Field
from graphrag_vectors import VectorStoreConfig

from graphrag.config.defaults import graphrag_config_defaults
from graphrag.prompts.index.entity_resolution import ENTITY_RESOLUTION_PROMPT


class EntityResolutionConfig(BaseModel):
    """Configuration section for entity resolution."""

    enabled: bool = Field(
        description="Whether to run entity resolution after graph extraction.",
        default=graphrag_config_defaults.entity_resolution.enabled,
    )
    completion_model_id: str = Field(
        description="The model ID to use for merge confirmation.",
        default=graphrag_config_defaults.entity_resolution.completion_model_id,
    )
    embedding_model_id: str = Field(
        description="The model ID to use for generating entity embeddings.",
        default=graphrag_config_defaults.entity_resolution.embedding_model_id,
    )
    model_instance_name: str = Field(
        description="Cache partition name for the completion model.",
        default=graphrag_config_defaults.entity_resolution.model_instance_name,
    )
    strategy: str = Field(
        description="Resolution strategy: 'llm_context_window' sends sorted entity windows to the LLM for disambiguation. 'embedding_search' uses vector neighbours plus per-group LLM confirmation. 'ann_hybrid' uses vector neighbours as blocking, graph-aware scoring for automatic high-confidence merges, and LLM review only for borderline groups.",
        default=graphrag_config_defaults.entity_resolution.strategy,
    )
    similarity_threshold: float = Field(
        description="Minimum similarity to consider two entities as candidates for merging.",
        default=graphrag_config_defaults.entity_resolution.similarity_threshold,
    )
    top_k: int = Field(
        description="Number of nearest neighbours to retrieve per entity from the vector store.",
        default=graphrag_config_defaults.entity_resolution.top_k,
    )
    window_tokens: int = Field(
        description="Maximum token budget per LLM context window (llm_context_window strategy). Increase as LLM context lengths grow.",
        default=graphrag_config_defaults.entity_resolution.window_tokens,
    )
    prompt: str | None = Field(
        description="Path to a custom entity resolution prompt file.",
        default=graphrag_config_defaults.entity_resolution.prompt,
    )
    vector_store: VectorStoreConfig | None = Field(
        description="Optional vector store override for entity resolution. Falls back to the top-level vector_store config.",
        default=None,
    )
    auto_merge_threshold: float = Field(
        description="For ann_hybrid, automatically merge candidate pairs whose combined title, embedding, type, and graph-context score is at or above this threshold.",
        default=graphrag_config_defaults.entity_resolution.auto_merge_threshold,
    )
    llm_review_threshold: float = Field(
        description="For ann_hybrid, send groups with at least one pair at or above this score to the LLM when they are below the auto-merge threshold.",
        default=graphrag_config_defaults.entity_resolution.llm_review_threshold,
    )
    max_llm_groups: int = Field(
        description="For ann_hybrid, maximum number of borderline groups to review with the LLM in one run.",
        default=graphrag_config_defaults.entity_resolution.max_llm_groups,
    )
    profile_neighbor_limit: int = Field(
        description="For ann_hybrid, maximum number of neighbouring entity titles and predicates to include in each entity profile before embedding.",
        default=graphrag_config_defaults.entity_resolution.profile_neighbor_limit,
    )

    def resolved_prompt(self) -> str:
        """Return the prompt string (from file or built-in default)."""
        if self.prompt:
            return Path(self.prompt).read_text(encoding="utf-8")
        return ENTITY_RESOLUTION_PROMPT
