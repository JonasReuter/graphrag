# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Parameterization settings for evidence extraction and verification."""

from graphrag.config.defaults import (
    DEFAULT_COMPLETION_MODEL_ID,
)
from pydantic import BaseModel, Field


class EvidenceConfig(BaseModel):
    """Configuration for evidence extraction, verification, and confidence scoring."""

    enabled: bool = Field(
        description="Whether to capture evidence metadata during graph extraction.",
        default=False,
    )

    capture_source_spans: bool = Field(
        description="Whether to extract exact source text spans as evidence.",
        default=True,
    )

    capture_confidence: bool = Field(
        description="Whether to extract LLM self-assessed confidence scores.",
        default=True,
    )

    prompt: str | None = Field(
        description="The evidence-enhanced extraction prompt to use. None uses the default.",
        default=None,
    )

    # --- Verification settings ---

    verification_enabled: bool = Field(
        description="Whether to run the post-extraction LLM verification workflow.",
        default=False,
    )

    verification_model_id: str = Field(
        description="The completion model to use for evidence verification.",
        default=DEFAULT_COMPLETION_MODEL_ID,
    )

    verification_model_instance_name: str = Field(
        description="The model instance name for caching verification calls.",
        default="verify_evidence",
    )

    # --- Cross-document consolidation ---

    cross_doc_similarity_threshold: float = Field(
        description="Embedding similarity threshold below which descriptions from different sources are flagged as contradictions.",
        default=0.6,
    )

    # --- Confidence scoring ---

    confidence_weights: dict[str, float] = Field(
        description="Weights for the multi-factor confidence formula.",
        default={
            "extraction": 0.3,
            "source_agreement": 0.3,
            "cross_doc": 0.25,
            "contradiction_penalty": 0.15,
        },
    )

    # --- Quality metrics ---

    quality_metrics_enabled: bool = Field(
        description="Whether to compute quality metrics at the end of the pipeline.",
        default=True,
    )

    low_confidence_threshold: float = Field(
        description="Confidence score below which entities/relationships are flagged as low-confidence.",
        default=0.3,
    )
