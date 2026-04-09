# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Context Build utility methods."""

import random
from typing import Any, cast

import pandas as pd
from graphrag_llm.tokenizer import Tokenizer

from graphrag.data_model.relationship import Relationship
from graphrag.data_model.text_unit import TextUnit
from graphrag.tokenizer.get_tokenizer import get_tokenizer

_RELATIONSHIP_IDS_COL = "relationship_ids"

"""
Contain util functions to build text unit context for the search's system prompt
"""


def build_text_unit_context(
    text_units: list[TextUnit],
    tokenizer: Tokenizer | None = None,
    column_delimiter: str = "|",
    shuffle_data: bool = True,
    max_context_tokens: int = 8000,
    context_name: str = "Sources",
    random_state: int = 86,
    relationships: dict[str, Relationship] | None = None,
) -> tuple[str, dict[str, pd.DataFrame]]:
    """Prepare text-unit data table as context data for system prompt.

    When relationships is provided, a relationship_ids column is appended to each row,
    listing the short_ids of relationships that appear in that text unit.
    This implements the "linked tables" provenance design: the Sources table cross-references
    the Relationships table, making provenance explicit for the LLM.
    """
    tokenizer = tokenizer or get_tokenizer()
    if text_units is None or len(text_units) == 0:
        return ("", {})

    if shuffle_data:
        random.seed(random_state)
        random.shuffle(text_units)

    include_relationship_ids = relationships is not None
    has_temporal = any(u.created_at for u in text_units)

    # build a lookup: relationship UUID → short_id (for cross-reference)
    rel_id_to_short: dict[str, str] = {}
    if include_relationship_ids:
        for rel_id, rel in (relationships or {}).items():
            if rel.short_id:
                rel_id_to_short[rel_id] = rel.short_id

    # add context header
    current_context_text = f"-----{context_name}-----" + "\n"

    # add header
    header = ["id", "text"]
    if has_temporal:
        header.append("date")
    attribute_cols = (
        list(text_units[0].attributes.keys()) if text_units[0].attributes else []
    )
    attribute_cols = [col for col in attribute_cols if col not in header]
    header.extend(attribute_cols)
    if include_relationship_ids:
        header.append(_RELATIONSHIP_IDS_COL)

    current_context_text += column_delimiter.join(header) + "\n"
    current_tokens = tokenizer.num_tokens(current_context_text)
    all_context_records = [header]

    for unit in text_units:
        new_context = [
            unit.short_id,
            unit.text,
        ]
        if has_temporal:
            new_context.append(unit.created_at or "")
        new_context.extend([
            str(unit.attributes.get(field, "")) if unit.attributes else ""
            for field in attribute_cols
        ])
        if include_relationship_ids:
            rel_short_ids = [
                rel_id_to_short[rid]
                for rid in (unit.relationship_ids or [])
                if rid in rel_id_to_short
            ]
            new_context.append(",".join(rel_short_ids))

        new_context_text = column_delimiter.join(new_context) + "\n"
        new_tokens = tokenizer.num_tokens(new_context_text)

        if current_tokens + new_tokens > max_context_tokens:
            break

        current_context_text += new_context_text
        all_context_records.append(new_context)
        current_tokens += new_tokens

    if len(all_context_records) > 1:
        record_df = pd.DataFrame(
            all_context_records[1:], columns=cast("Any", all_context_records[0])
        )
    else:
        record_df = pd.DataFrame()
    return current_context_text, {context_name.lower(): record_df}


def count_relationships(
    entity_relationships: list[Relationship], text_unit: TextUnit
) -> int:
    """Count the number of relationships of the selected entity that are associated with the text unit."""
    if not text_unit.relationship_ids:
        # Use list comprehension to count relationships where the text_unit.id is in rel.text_unit_ids
        return sum(
            1
            for rel in entity_relationships
            if rel.text_unit_ids and text_unit.id in rel.text_unit_ids
        )

    # Use a set for faster lookups if entity_relationships is large
    entity_relationship_ids = {rel.id for rel in entity_relationships}

    # Count matching relationship ids efficiently
    return sum(
        1 for rel_id in text_unit.relationship_ids if rel_id in entity_relationship_ids
    )
