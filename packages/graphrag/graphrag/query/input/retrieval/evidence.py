# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Evidence retrieval utilities for query context building."""

import pandas as pd

from graphrag.data_model.entity import Entity
from graphrag.data_model.evidence import Evidence
from graphrag.data_model.relationship import Relationship


def get_candidate_evidence(
    selected_entities: list[Entity],
    evidence: list[Evidence],
    relationships: list[Relationship] | None = None,
) -> list[Evidence]:
    """Retrieve evidence records relevant to the selected entities and their relationships.

    Args:
        selected_entities: Entities selected for the current query context.
        evidence: All available evidence records.
        relationships: Optional list of relationships to also include evidence for.

    Returns:
        Filtered list of Evidence records relevant to selected_entities.
    """
    entity_titles = {entity.title for entity in selected_entities}
    rel_keys: set[str] = set()
    if relationships:
        for rel in relationships:
            if rel.source in entity_titles or rel.target in entity_titles:
                rel_keys.add(f"{rel.source}::{rel.target}")

    return [
        ev for ev in evidence
        if (ev.subject_type == "entity" and ev.subject_id in entity_titles)
        or (ev.subject_type == "relationship" and ev.subject_id in rel_keys)
    ]


def to_evidence_dataframe(evidence: list[Evidence]) -> pd.DataFrame:
    """Convert a list of Evidence objects to a DataFrame for tracking."""
    if not evidence:
        return pd.DataFrame(columns=["id", "subject_id", "subject_type",
                                     "verification_status", "extraction_confidence"])
    return pd.DataFrame([{
        "id": ev.short_id or ev.id,
        "subject_id": ev.subject_id,
        "subject_type": ev.subject_type,
        "verification_status": ev.verification_status,
        "extraction_confidence": ev.extraction_confidence,
    } for ev in evidence])
