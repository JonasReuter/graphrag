# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A package containing the 'Contradiction' model."""

from dataclasses import dataclass
from typing import Any

from graphrag.data_model.identified import Identified


@dataclass
class Contradiction(Identified):
    """
    A tracked contradiction or semantic relationship between two graph elements.

    Used to record cases where evidence from different sources conflicts,
    or where two elements may refer to the same real-world concept.
    """

    relation_type: str
    """Type of relationship: 'same_as', 'possibly_same_as', 'contradicts', or 'supersedes'."""

    subject_a_type: str
    """Type of first subject: 'entity', 'relationship', or 'evidence'."""

    subject_a_id: str
    """ID or key of the first subject."""

    subject_b_type: str
    """Type of second subject: 'entity', 'relationship', or 'evidence'."""

    subject_b_id: str
    """ID or key of the second subject."""

    description: str | None = None
    """Human-readable description of the contradiction or similarity."""

    confidence: float = 0.5
    """Confidence in this contradiction/similarity (0.0-1.0)."""

    detection_method: str = "embedding_similarity"
    """Method used to detect this: 'embedding_similarity' or 'llm_verified'."""

    attributes: dict[str, Any] | None = None

    @classmethod
    def from_dict(
        cls,
        d: dict[str, Any],
        id_key: str = "id",
        short_id_key: str = "human_readable_id",
    ) -> "Contradiction":
        """Create a new contradiction from the dict data."""
        return Contradiction(
            id=d[id_key],
            short_id=d.get(short_id_key),
            relation_type=d["relation_type"],
            subject_a_type=d["subject_a_type"],
            subject_a_id=d["subject_a_id"],
            subject_b_type=d["subject_b_type"],
            subject_b_id=d["subject_b_id"],
            description=d.get("description"),
            confidence=d.get("confidence", 0.5),
            detection_method=d.get("detection_method", "embedding_similarity"),
            attributes=d.get("attributes"),
        )
