# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Resolve claim subject_id / object_id to canonical entity titles.

After claim extraction the LLM sometimes produces entity names that differ
from the canonical titles in the knowledge graph (name variants, ß/SS,
honorifics, company suffixes, etc.).  This operation embeds each unique
claim subject/object name, searches the existing entity embedding index,
and applies confirmed matches — either directly (high similarity) or after
LLM confirmation (mid-range similarity).
"""

import json
import logging
import re
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from graphrag_llm.completion import LLMCompletion
    from graphrag_llm.embedding import LLMEmbedding
    from graphrag_vectors import VectorStore

logger = logging.getLogger(__name__)

_EMBED_BATCH = 32
_JSON_RE = re.compile(r"\{[^}]*\}", re.DOTALL)


async def resolve_claim_subjects(
    covariates_df: pd.DataFrame,
    embedding_model: "LLMEmbedding",
    completion_model: "LLMCompletion",
    vector_store: "VectorStore",
    prompt: str,
    high_threshold: float,
    mid_threshold: float,
    top_k: int,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Resolve claim subject/object IDs to canonical entity titles.

    Returns the updated DataFrame and the resolution map for logging.
    """
    # Collect all unique non-null subject/object IDs
    subjects = set(covariates_df["subject_id"].dropna().unique())
    objects = set(covariates_df["object_id"].dropna().unique())
    unique_ids = sorted(subjects | objects)

    if not unique_ids:
        return covariates_df, {}

    # Guard: check entity index is populated
    try:
        count = vector_store.count()
    except Exception:  # noqa: BLE001
        count = 0
    if count == 0:
        logger.warning(
            "resolve_claim_subjects: entity embedding index is empty — "
            "run resolve_entities first. Skipping."
        )
        return covariates_df, {}

    logger.info(
        "resolve_claim_subjects: resolving %d unique claim subject/object IDs "
        "against %d entity embeddings",
        len(unique_ids),
        count,
    )

    resolution_map: dict[str, str] = {}
    auto_resolved = 0
    llm_confirmed = 0
    llm_rejected = 0

    # Embed in batches — no upsert, read-only against existing index
    for start in range(0, len(unique_ids), _EMBED_BATCH):
        batch = unique_ids[start : start + _EMBED_BATCH]
        response = await embedding_model.embedding_async(input=batch)
        vectors = response.embeddings or []

        for raw_id, vec in zip(batch, vectors, strict=False):
            if vec is None:
                continue
            vec_list = vec.tolist() if hasattr(vec, "tolist") else list(vec)

            results = vector_store.similarity_search_by_vector(
                query_embedding=vec_list,
                k=top_k,
                include_vectors=False,
            )
            if not results:
                continue

            best = results[0]
            candidate_title = best.document.data.get("title", "")
            score = best.score

            if not candidate_title or candidate_title == raw_id:
                continue

            if score >= high_threshold:
                resolution_map[raw_id] = candidate_title
                auto_resolved += 1
                logger.debug(
                    "  AUTO  %r → %r  (%.3f)", raw_id, candidate_title, score
                )
            elif score >= mid_threshold:
                confirmed = await _llm_confirm(
                    raw_id, candidate_title, score, completion_model, prompt
                )
                if confirmed:
                    resolution_map[raw_id] = candidate_title
                    llm_confirmed += 1
                    logger.debug(
                        "  LLM✓  %r → %r  (%.3f)", raw_id, candidate_title, score
                    )
                else:
                    llm_rejected += 1
                    logger.debug(
                        "  LLM✗  %r ≠ %r  (%.3f)", raw_id, candidate_title, score
                    )

    logger.info(
        "resolve_claim_subjects: %d auto-resolved, %d LLM-confirmed, %d LLM-rejected",
        auto_resolved,
        llm_confirmed,
        llm_rejected,
    )

    if resolution_map:
        covariates_df = covariates_df.copy()
        covariates_df["subject_id"] = covariates_df["subject_id"].map(
            lambda x: resolution_map.get(x, x)
        )
        covariates_df["object_id"] = covariates_df["object_id"].map(
            lambda x: resolution_map.get(x, x) if x else x
        )

    return covariates_df, resolution_map


async def _llm_confirm(
    subject_id: str,
    candidate_title: str,
    score: float,
    model: "LLMCompletion",
    prompt: str,
) -> bool:
    """Ask the LLM whether subject_id and candidate_title refer to the same entity."""
    formatted = prompt.format(
        subject_id=subject_id,
        candidate_title=candidate_title,
        score=score,
    )
    try:
        response = await model.completion_async(
            messages=[{"role": "user", "content": formatted}]
        )
        content = response.content.strip()
        # Strip markdown fences if present
        match = _JSON_RE.search(content)
        if match:
            content = match.group(0)
        parsed = json.loads(content)
        return bool(parsed.get("match"))
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "resolve_claim_subjects: LLM confirmation failed for %r → %r: %s",
            subject_id,
            candidate_title,
            exc,
        )
        return False
