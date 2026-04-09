# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Parameterization settings for temporal metadata extraction and propagation."""

from pydantic import BaseModel, Field


class TemporalConfig(BaseModel):
    """Configuration for temporal metadata in the graph extraction pipeline."""

    enabled: bool = Field(
        description="Whether to enable temporal metadata extraction and propagation.",
        default=False,
    )

    temporal_extraction: bool = Field(
        description="Whether the LLM should extract temporal_scope during graph extraction.",
        default=True,
    )

    temporal_prompt: str | None = Field(
        description="Path to a custom temporal extraction prompt. None uses the default.",
        default=None,
    )

    propagate_document_timestamps: bool = Field(
        description="Whether to propagate document creation_date to text units, entities, and relationships.",
        default=True,
    )
