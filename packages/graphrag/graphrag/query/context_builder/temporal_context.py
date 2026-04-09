# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Timeline context builder for chronological entity queries."""

from typing import Any, cast

import pandas as pd
from graphrag_llm.tokenizer import Tokenizer

from graphrag.data_model.relationship import Relationship
from graphrag.data_model.text_unit import TextUnit
from graphrag.tokenizer.get_tokenizer import get_tokenizer


def build_timeline_context(
    entity_title: str,
    relationships: list[Relationship],
    text_units: dict[str, TextUnit] | None = None,
    tokenizer: Tokenizer | None = None,
    max_context_tokens: int = 8000,
    column_delimiter: str = "|",
    context_name: str = "Timeline",
) -> tuple[str, pd.DataFrame]:
    """Build a chronologically sorted context table for a single entity.

    Produces a table sorted by observed_at ascending:
        date | event | related_entities | source

    Used for queries like "What did customer X buy over time?" or
    "What is the latest status of negotiation with Y?"
    """
    tokenizer = tokenizer or get_tokenizer()

    # Filter relationships involving the entity
    entity_rels = [
        rel for rel in relationships
        if rel.source.upper() == entity_title.upper()
        or rel.target.upper() == entity_title.upper()
    ]

    if not entity_rels:
        return "", pd.DataFrame()

    # Sort by observed_at (None values go last)
    entity_rels.sort(
        key=lambda r: r.observed_at or "9999-99-99",
    )

    # Build context table
    current_context_text = f"-----{context_name}: {entity_title}-----\n"
    header = ["date", "event", "related_entity", "source"]
    current_context_text += column_delimiter.join(header) + "\n"
    current_tokens = tokenizer.num_tokens(current_context_text)

    all_records: list[list[str]] = [header]
    for rel in entity_rels:
        # Determine the "other" entity
        other = rel.target if rel.source.upper() == entity_title.upper() else rel.source

        # Resolve source text unit short_id
        source_ref = ""
        if rel.text_unit_ids and text_units:
            unit = text_units.get(rel.text_unit_ids[0])
            if unit and unit.short_id:
                source_ref = unit.short_id

        row = [
            rel.observed_at or "",
            rel.description or "",
            other,
            source_ref,
        ]
        row_text = column_delimiter.join(row) + "\n"
        row_tokens = tokenizer.num_tokens(row_text)

        if current_tokens + row_tokens > max_context_tokens:
            break

        current_context_text += row_text
        all_records.append(row)
        current_tokens += row_tokens

    if len(all_records) > 1:
        record_df = pd.DataFrame(
            all_records[1:], columns=cast("Any", all_records[0])
        )
    else:
        record_df = pd.DataFrame()

    return current_context_text, record_df
