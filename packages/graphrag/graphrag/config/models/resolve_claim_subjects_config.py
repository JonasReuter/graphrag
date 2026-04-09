# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Parameterization settings for resolve_claim_subjects configuration."""

from graphrag_vectors import VectorStoreConfig
from pydantic import BaseModel, Field

from graphrag.config.defaults import graphrag_config_defaults


class ResolveClaimSubjectsConfig(BaseModel):
    """Configuration for post-extraction claim subject resolution.

    Runs after extract_covariates and matches claim subject_id / object_id
    values against the resolved entity list using embeddings, with optional
    LLM confirmation for near-matches.
    """

    enabled: bool = Field(
        description="Whether to resolve claim subject/object IDs against the entity list.",
        default=graphrag_config_defaults.resolve_claim_subjects.enabled,
    )
    embedding_model_id: str = Field(
        description="The model ID to use for embedding claim subject names.",
        default=graphrag_config_defaults.resolve_claim_subjects.embedding_model_id,
    )
    completion_model_id: str = Field(
        description="The model ID to use for LLM confirmation of near-matches.",
        default=graphrag_config_defaults.resolve_claim_subjects.completion_model_id,
    )
    model_instance_name: str = Field(
        description="Cache partition name for the completion model.",
        default=graphrag_config_defaults.resolve_claim_subjects.model_instance_name,
    )
    high_threshold: float = Field(
        description="Cosine similarity above which a match is accepted without LLM confirmation.",
        default=graphrag_config_defaults.resolve_claim_subjects.high_threshold,
    )
    mid_threshold: float = Field(
        description="Cosine similarity above which a match is sent to LLM for confirmation.",
        default=graphrag_config_defaults.resolve_claim_subjects.mid_threshold,
    )
    top_k: int = Field(
        description="Number of nearest neighbours to retrieve per claim subject from the entity index.",
        default=graphrag_config_defaults.resolve_claim_subjects.top_k,
    )
    entity_index_name: str = Field(
        description="Name of the vector store index holding entity embeddings (populated by resolve_entities).",
        default=graphrag_config_defaults.resolve_claim_subjects.entity_index_name,
    )
    vector_store: VectorStoreConfig | None = Field(
        description="Optional vector store override. Falls back to the top-level vector_store config.",
        default=None,
    )
