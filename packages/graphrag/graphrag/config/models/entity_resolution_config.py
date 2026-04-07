# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Parameterization settings for entity resolution configuration."""

from pathlib import Path

from pydantic import BaseModel, Field

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
    similarity_threshold: float = Field(
        description="Minimum cosine similarity to consider two entities as candidates for merging.",
        default=graphrag_config_defaults.entity_resolution.similarity_threshold,
    )
    top_k: int = Field(
        description="Number of nearest neighbours to retrieve per entity from the vector store.",
        default=graphrag_config_defaults.entity_resolution.top_k,
    )
    prompt: str | None = Field(
        description="Path to a custom entity resolution prompt file.",
        default=graphrag_config_defaults.entity_resolution.prompt,
    )

    def resolved_prompt(self) -> str:
        """Return the prompt string (from file or built-in default)."""
        if self.prompt:
            return Path(self.prompt).read_text(encoding="utf-8")
        return ENTITY_RESOLUTION_PROMPT
