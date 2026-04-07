# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A package containing the 'Evidence' model."""

from dataclasses import dataclass
from typing import Any

from graphrag.data_model.identified import Identified


@dataclass
class Evidence(Identified):
    """
    A piece of evidence supporting a graph assertion (entity or relationship).

    Evidence records link extracted graph elements back to the specific text
    that supports them, along with confidence and completeness metadata.
    """

    subject_type: str
    """The type of graph element this evidence supports: 'entity' or 'relationship'."""

    subject_id: str
    """Identifier of the subject. Entity title for entities, 'SOURCE::TARGET' for relationships."""

    text_unit_id: str
    """The ID of the text unit from which this evidence was extracted."""

    source_span: str | None = None
    """The exact quoted text from the source chunk that supports this extraction."""

    extraction_confidence: float = 0.5
    """LLM self-assessed confidence in this extraction (0.0-1.0)."""

    completeness_status: str = "complete"
    """Whether the extraction is 'complete', 'partial', or 'inferred'."""

    time_scope: str | None = None
    """Temporal qualifier if mentioned (e.g. '2020-2023', 'formerly')."""

    qualifier: str | None = None
    """Conditional language qualifier (e.g. 'allegedly', 'reportedly', 'interim')."""

    verification_status: str = "unverified"
    """Verification result: 'unverified', 'confirmed', 'weakened', 'refuted', or 'insufficient'."""

    verification_method: str | None = None
    """Method used for verification: 'llm_verified', 'cross_doc', or None."""

    attributes: dict[str, Any] | None = None

    @classmethod
    def from_dict(
        cls,
        d: dict[str, Any],
        id_key: str = "id",
        short_id_key: str = "human_readable_id",
        subject_type_key: str = "subject_type",
        subject_id_key: str = "subject_id",
        text_unit_id_key: str = "text_unit_id",
        source_span_key: str = "source_span",
        extraction_confidence_key: str = "extraction_confidence",
        completeness_status_key: str = "completeness_status",
        time_scope_key: str = "time_scope",
        qualifier_key: str = "qualifier",
        verification_status_key: str = "verification_status",
        verification_method_key: str = "verification_method",
        attributes_key: str = "attributes",
    ) -> "Evidence":
        """Create a new evidence from the dict data."""
        return Evidence(
            id=d[id_key],
            short_id=d.get(short_id_key),
            subject_type=d[subject_type_key],
            subject_id=d[subject_id_key],
            text_unit_id=d[text_unit_id_key],
            source_span=d.get(source_span_key),
            extraction_confidence=d.get(extraction_confidence_key, 0.5),
            completeness_status=d.get(completeness_status_key, "complete"),
            time_scope=d.get(time_scope_key),
            qualifier=d.get(qualifier_key),
            verification_status=d.get(verification_status_key, "unverified"),
            verification_method=d.get(verification_method_key),
            attributes=d.get(attributes_key),
        )
