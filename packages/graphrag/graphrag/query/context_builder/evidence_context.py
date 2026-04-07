# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Evidence Context Builder for local search."""

from typing import cast, Any

import pandas as pd
from graphrag_llm.tokenizer import Tokenizer

from graphrag.data_model.entity import Entity
from graphrag.data_model.evidence import Evidence
from graphrag.data_model.relationship import Relationship
from graphrag.tokenizer.get_tokenizer import get_tokenizer

# Priority order for verification status when sorting evidence
_VERIFICATION_PRIORITY = {
    "confirmed": 0,
    "unverified": 1,
    "insufficient": 2,
    "weakened": 3,
    "refuted": 4,
}


def build_evidence_context(
    selected_entities: list[Entity],
    evidence: list[Evidence],
    relationships: list[Relationship] | None = None,
    tokenizer: Tokenizer | None = None,
    max_context_tokens: int = 2000,
    column_delimiter: str = "|",
    context_name: str = "Evidence",
    min_confidence: float = 0.0,
    exclude_refuted: bool = False,
) -> tuple[str, pd.DataFrame]:
    """Prepare evidence data as context for the system prompt.

    Retrieves evidence records for the selected entities and their relationships,
    prioritises confirmed and high-confidence evidence, and formats them as a table.

    Returns:
        (context_text, record_df)
    """
    tokenizer = tokenizer or get_tokenizer()

    if not selected_entities or not evidence:
        return "", pd.DataFrame()

    # Collect entity titles and relationship keys for the selected entities
    entity_titles = {entity.title for entity in selected_entities}
    rel_keys: set[str] = set()
    if relationships:
        for rel in relationships:
            if rel.source in entity_titles or rel.target in entity_titles:
                rel_keys.add(f"{rel.source}::{rel.target}")

    # Filter relevant evidence
    relevant: list[Evidence] = []
    for ev in evidence:
        if ev.subject_type == "entity" and ev.subject_id in entity_titles:
            relevant.append(ev)
        elif ev.subject_type == "relationship" and ev.subject_id in rel_keys:
            relevant.append(ev)

    if not relevant:
        return "", pd.DataFrame()

    # Filter by minimum confidence and optionally exclude refuted
    if min_confidence > 0:
        relevant = [ev for ev in relevant if ev.extraction_confidence >= min_confidence]
    if exclude_refuted:
        relevant = [ev for ev in relevant if ev.verification_status != "refuted"]

    # Sort: confirmed first, then by extraction_confidence descending
    relevant.sort(
        key=lambda ev: (
            _VERIFICATION_PRIORITY.get(ev.verification_status, 99),
            -ev.extraction_confidence,
        )
    )

    # Build context table
    current_context_text = f"-----{context_name}-----\n"
    header = ["id", "subject", "type", "source_quote", "confidence", "status"]
    current_context_text += column_delimiter.join(header) + "\n"
    current_tokens = tokenizer.num_tokens(current_context_text)

    all_context_records = [header]

    for ev in relevant:
        short_id = ev.short_id or ev.id[:8]
        source_quote = (ev.source_span or "")[:200]  # truncate very long quotes
        confidence_str = f"{ev.extraction_confidence:.2f}"
        row = [
            short_id,
            ev.subject_id,
            ev.subject_type,
            source_quote,
            confidence_str,
            ev.verification_status,
        ]
        row_text = column_delimiter.join(row) + "\n"
        new_tokens = tokenizer.num_tokens(row_text)
        if current_tokens + new_tokens > max_context_tokens:
            break
        current_context_text += row_text
        all_context_records.append(row)
        current_tokens += new_tokens

    if len(all_context_records) > 1:
        record_df = pd.DataFrame(
            all_context_records[1:], columns=cast("Any", all_context_records[0])
        )
    else:
        record_df = pd.DataFrame()

    return current_context_text, record_df
