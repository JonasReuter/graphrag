# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing extract_graph method."""

import logging
from typing import TYPE_CHECKING

import pandas as pd

from graphrag.callbacks.workflow_callbacks import WorkflowCallbacks
from graphrag.config.enums import AsyncType
from graphrag.index.operations.extract_graph.graph_extractor import GraphExtractor
from graphrag.index.operations.extract_graph.temporal_utils import (
    resolve_temporal_scope,
)
from graphrag.index.operations.extract_graph.utils import filter_orphan_relationships
from graphrag.index.utils.derive_from_rows import derive_from_rows

if TYPE_CHECKING:
    from graphrag_llm.completion import LLMCompletion

logger = logging.getLogger(__name__)


async def extract_graph(
    text_units: pd.DataFrame,
    callbacks: WorkflowCallbacks,
    text_column: str,
    id_column: str,
    model: "LLMCompletion",
    prompt: str,
    entity_types: list[str],
    max_gleanings: int,
    num_threads: int,
    async_type: AsyncType,
    extract_evidence: bool = False,
    extract_temporal: bool = False,
    text_unit_timestamps: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Extract a graph from a piece of text using a language model.

    Parameters
    ----------
    text_unit_timestamps
        Mapping from text_unit id to ISO-8601 created_at timestamp.
        Used to compute observed_at/last_observed_at on merged entities
        and relationships when temporal extraction is enabled.
    """
    num_started = 0

    async def run_strategy(row):
        nonlocal num_started
        text = row[text_column]
        id = row[id_column]
        result = await _run_extract_graph(
            text=text,
            source_id=id,
            entity_types=entity_types,
            model=model,
            prompt=prompt,
            max_gleanings=max_gleanings,
            extract_evidence=extract_evidence,
            extract_temporal=extract_temporal,
        )
        num_started += 1
        return result

    results = await derive_from_rows(
        text_units,
        run_strategy,
        callbacks,
        num_threads=num_threads,
        async_type=async_type,
        progress_msg="extract graph progress: ",
    )

    entity_dfs = []
    relationship_dfs = []
    evidence_dfs = []
    for result in results:
        if result:
            entity_dfs.append(result[0])
            relationship_dfs.append(result[1])
            evidence_dfs.append(result[2])

    entities = _merge_entities(entity_dfs, text_unit_timestamps)
    relationships = _merge_relationships(relationship_dfs, text_unit_timestamps)
    relationships = filter_orphan_relationships(relationships, entities)
    evidence = _collect_evidence(evidence_dfs)

    # Resolve temporal_scope -> valid_from/valid_until for relationships only.
    # Entities (persons, organizations) do NOT get valid_from/valid_until from
    # the extraction prompt -- their existence is not time-bounded by document dates.
    if extract_temporal:
        relationships = _resolve_temporal_scopes(relationships)

    return (entities, relationships, evidence)


async def _run_extract_graph(
    text: str,
    source_id: str,
    entity_types: list[str],
    model: "LLMCompletion",
    prompt: str,
    max_gleanings: int,
    extract_evidence: bool = False,
    extract_temporal: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the graph intelligence entity extraction strategy."""
    extractor = GraphExtractor(
        model=model,
        prompt=prompt,
        max_gleanings=max_gleanings,
        on_error=lambda e, s, d: logger.error(
            "Entity Extraction Error", exc_info=e, extra={"stack": s, "details": d}
        ),
    )
    text = text.strip()

    entities_df, relationships_df, evidence_df = await extractor(
        text,
        entity_types=entity_types,
        source_id=source_id,
        extract_evidence=extract_evidence,
        extract_temporal=extract_temporal,
    )

    return (entities_df, relationships_df, evidence_df)


def _compute_temporal_bounds(
    source_ids: list[str],
    text_unit_timestamps: dict[str, str] | None,
) -> tuple[str | None, str | None]:
    """Compute observed_at (min) and last_observed_at (max) from text unit timestamps."""
    if not text_unit_timestamps:
        return None, None
    timestamps = [
        text_unit_timestamps[sid]
        for sid in source_ids
        if sid in text_unit_timestamps and text_unit_timestamps[sid]
    ]
    if not timestamps:
        return None, None
    return min(timestamps), max(timestamps)


def _first_non_empty(values: list) -> str | None:
    """Return the first non-empty temporal_scope from a list, or None."""
    for v in values:
        if v and str(v).strip():
            return str(v).strip()
    return None


def _merge_entities(
    entity_dfs: list[pd.DataFrame],
    text_unit_timestamps: dict[str, str] | None = None,
) -> pd.DataFrame:
    all_entities = pd.concat(entity_dfs, ignore_index=True)

    merged = (
        all_entities
        .groupby(["title", "type"], sort=False)
        .agg(
            description=("description", list),
            text_unit_ids=("source_id", list),
            frequency=("source_id", "count"),
        )
        .reset_index()
    )

    # Compute document-level temporal bounds (when was this entity first/last seen)
    observed_at_list = []
    last_observed_at_list = []
    for _, row in merged.iterrows():
        obs, last_obs = _compute_temporal_bounds(
            row["text_unit_ids"], text_unit_timestamps
        )
        observed_at_list.append(obs)
        last_observed_at_list.append(last_obs)
    merged["observed_at"] = observed_at_list
    merged["last_observed_at"] = last_observed_at_list

    return merged


def _merge_relationships(
    relationship_dfs: list[pd.DataFrame],
    text_unit_timestamps: dict[str, str] | None = None,
) -> pd.DataFrame:
    all_relationships = pd.concat(relationship_dfs, ignore_index=False)

    if "temporal_scope" not in all_relationships.columns:
        all_relationships["temporal_scope"] = None

    merged = (
        all_relationships
        .groupby(["source", "target"], sort=False)
        .agg(
            description=("description", list),
            text_unit_ids=("source_id", list),
            weight=("weight", "sum"),
            _temporal_scopes=("temporal_scope", list),
        )
        .reset_index()
    )

    observed_at_list = []
    last_observed_at_list = []
    for _, row in merged.iterrows():
        obs, last_obs = _compute_temporal_bounds(
            row["text_unit_ids"], text_unit_timestamps
        )
        observed_at_list.append(obs)
        last_observed_at_list.append(last_obs)
    merged["observed_at"] = observed_at_list
    merged["last_observed_at"] = last_observed_at_list

    merged["temporal_scope"] = merged["_temporal_scopes"].apply(_first_non_empty)
    merged = merged.drop(columns=["_temporal_scopes"])

    return merged


def _resolve_temporal_scopes(df: pd.DataFrame) -> pd.DataFrame:
    """Resolve temporal_scope column into valid_from/valid_until columns."""
    if "temporal_scope" not in df.columns:
        df["valid_from"] = None
        df["valid_until"] = None
        return df

    valid_from_list = []
    valid_until_list = []
    for scope in df["temporal_scope"]:
        vf, vu, qualifier = resolve_temporal_scope(scope)
        valid_from_list.append(vf)
        valid_until_list.append(vu)
        # If unresolved, qualifier stays in temporal_scope column already
    df["valid_from"] = valid_from_list
    df["valid_until"] = valid_until_list
    return df


def _collect_evidence(evidence_dfs) -> pd.DataFrame:
    """Collect evidence records without merging — each is a distinct observation."""
    non_empty = [df for df in evidence_dfs if len(df) > 0]
    if not non_empty:
        return pd.DataFrame(
            columns=[
                "subject_type",
                "subject_id",
                "text_unit_id",
                "source_span",
                "extraction_confidence",
                "completeness_status",
            ]
        )
    return pd.concat(non_empty, ignore_index=True)
