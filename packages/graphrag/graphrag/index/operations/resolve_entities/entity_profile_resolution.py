# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Graph-aware ANN candidate scoring for entity resolution.

This module intentionally avoids maintained alias/stop-word lists. It uses
entity profile embeddings for blocking and cheap local scores for deciding
whether a candidate pair is safe to merge automatically or should be reviewed
by the LLM.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
import math
import re
from typing import Any

from graphrag.index.operations.resolve_entities.entity_grouping import _UnionFind
from graphrag.index.operations.resolve_entities.entity_merge_extractor import (
    EntityMergeExtractor,
)
from graphrag_vectors import VectorStore


@dataclass(frozen=True)
class EntityContext:
    """Small graph context profile for one entity title."""

    neighbours: frozenset[str]
    predicates: frozenset[str]


@dataclass(frozen=True)
class ScoredPair:
    """Candidate pair plus component scores."""

    left_idx: int
    right_idx: int
    score: float
    embedding_score: float
    title_score: float
    context_score: float
    type_score: float


_WORD_RE = re.compile(r"[\w]+", re.UNICODE)


def build_entity_profile_texts(
    entities: list[dict[str, Any]],
    relationships: list[dict[str, Any]],
    *,
    neighbour_limit: int,
) -> list[str]:
    """Build text profiles for embedding-based blocking.

    Profiles include the title, type, description, and a bounded amount of graph
    context. This gives the vector store enough signal to retrieve likely
    duplicates without comparing all entity pairs.
    """
    contexts = build_entity_contexts(relationships)
    texts: list[str] = []

    for entity in entities:
        title = entity["title"]
        context = contexts.get(title, EntityContext(frozenset(), frozenset()))
        neighbours = sorted(context.neighbours)[:neighbour_limit]
        predicates = sorted(context.predicates)[:neighbour_limit]
        texts.append(
            "\n".join(
                [
                    f"Title: {title}",
                    f"Type: {entity.get('type', '')}",
                    f"Description: {entity.get('description', '')}",
                    f"Predicates: {', '.join(predicates)}",
                    f"Neighbours: {', '.join(neighbours)}",
                ]
            )
        )

    return texts


def build_entity_contexts(
    relationships: list[dict[str, Any]],
) -> dict[str, EntityContext]:
    """Collect neighbouring titles and predicates for each entity title."""
    neighbours: dict[str, set[str]] = defaultdict(set)
    predicates: dict[str, set[str]] = defaultdict(set)

    for relationship in relationships:
        source = str(relationship.get("source") or "")
        target = str(relationship.get("target") or "")
        if not source or not target:
            continue

        description = str(relationship.get("description") or "")
        rel_type = str(relationship.get("type") or relationship.get("predicate") or "")
        predicate = rel_type or description

        neighbours[source].add(target)
        neighbours[target].add(source)
        if predicate:
            predicates[source].add(predicate)
            predicates[target].add(predicate)

    return {
        title: EntityContext(
            neighbours=frozenset(neighbours[title]),
            predicates=frozenset(predicates[title]),
        )
        for title in set(neighbours) | set(predicates)
    }


async def build_ann_hybrid_merge_map(
    *,
    all_entities: list[dict[str, Any]],
    relationships: list[dict[str, Any]],
    embeddings: list[list[float]],
    vector_store: VectorStore,
    extractor: EntityMergeExtractor,
    top_k: int,
    candidate_threshold: float,
    auto_merge_threshold: float,
    llm_review_threshold: float,
    max_llm_groups: int,
) -> dict[str, str]:
    """Build an alias to canonical mapping with ANN blocking and hybrid scoring.

    The strategy is designed to scale with O(n * top_k) candidate generation
    instead of O(n^2) pairwise comparison.
    """
    titles = [e["title"] for e in all_entities]
    title_to_idx = {title: i for i, title in enumerate(titles)}
    contexts = build_entity_contexts(relationships)

    scored_pairs = _score_ann_candidate_pairs(
        all_entities=all_entities,
        embeddings=embeddings,
        vector_store=vector_store,
        title_to_idx=title_to_idx,
        contexts=contexts,
        top_k=top_k,
        candidate_threshold=candidate_threshold,
    )

    if not scored_pairs:
        return {}

    auto_pairs = [
        pair for pair in scored_pairs if pair.score >= auto_merge_threshold
    ]
    review_pairs = [
        pair
        for pair in scored_pairs
        if llm_review_threshold <= pair.score < auto_merge_threshold
    ]

    merge_map: dict[str, str] = {}
    merge_map.update(_merge_map_from_pairs(auto_pairs, all_entities))

    review_groups = _groups_from_scored_pairs(review_pairs, len(all_entities))
    review_groups = sorted(
        review_groups,
        key=lambda group: _max_pair_score(group, review_pairs),
        reverse=True,
    )[:max_llm_groups]

    for group_indices in review_groups:
        candidates = [
            {
                "title": all_entities[i]["title"],
                "type": all_entities[i].get("type", ""),
                "description": _description_with_context(
                    all_entities[i], contexts.get(all_entities[i]["title"])
                ),
            }
            for i in group_indices
        ]
        decision = await extractor(candidates)
        if decision.merge and decision.canonical:
            canonical = decision.canonical
            group_titles = {c["title"] for c in candidates}
            if canonical not in group_titles:
                canonical = _choose_canonical(group_indices, all_entities)
            for candidate in candidates:
                if candidate["title"] != canonical:
                    merge_map[candidate["title"]] = canonical

    return merge_map


def _score_ann_candidate_pairs(
    *,
    all_entities: list[dict[str, Any]],
    embeddings: list[list[float]],
    vector_store: VectorStore,
    title_to_idx: dict[str, int],
    contexts: dict[str, EntityContext],
    top_k: int,
    candidate_threshold: float,
) -> list[ScoredPair]:
    pairs: dict[tuple[int, int], ScoredPair] = {}

    for i, (entity, vector) in enumerate(zip(all_entities, embeddings, strict=True)):
        if not vector:
            continue
        results = vector_store.similarity_search_by_vector(
            query_embedding=vector,
            k=top_k + 1,
            include_vectors=False,
        )
        for result in results:
            if result.score < candidate_threshold:
                continue
            neighbour_title = result.document.data.get("title", "")
            j = title_to_idx.get(neighbour_title)
            if j is None or j == i:
                continue

            lo, hi = (i, j) if i < j else (j, i)
            scored = _score_pair(
                all_entities[lo],
                all_entities[hi],
                embedding_score=float(result.score),
                contexts=contexts,
            )
            existing = pairs.get((lo, hi))
            if existing is None or scored.score > existing.score:
                pairs[(lo, hi)] = scored

    return list(pairs.values())


def _score_pair(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    embedding_score: float,
    contexts: dict[str, EntityContext],
) -> ScoredPair:
    left_title = left["title"]
    right_title = right["title"]
    title_score = _title_similarity(left_title, right_title)
    context_score = _context_similarity(
        contexts.get(left_title), contexts.get(right_title)
    )
    type_score = _type_compatibility(left.get("type"), right.get("type"))
    score = (
        0.40 * embedding_score
        + 0.30 * title_score
        + 0.20 * context_score
        + 0.10 * type_score
    )
    return ScoredPair(
        left_idx=-1,
        right_idx=-1,
        score=score,
        embedding_score=embedding_score,
        title_score=title_score,
        context_score=context_score,
        type_score=type_score,
    )


def _title_similarity(left: str, right: str) -> float:
    left_norm = _normalise_for_similarity(left)
    right_norm = _normalise_for_similarity(right)
    if not left_norm or not right_norm:
        return 0.0
    if left_norm == right_norm:
        return 1.0

    left_tokens = set(left_norm.split())
    right_tokens = set(right_norm.split())
    token_score = _jaccard(left_tokens, right_tokens)
    sequence_score = SequenceMatcher(None, left_norm, right_norm).ratio()

    if left_norm in right_norm or right_norm in left_norm:
        containment_score = min(len(left_norm), len(right_norm)) / max(
            len(left_norm), len(right_norm)
        )
    else:
        containment_score = 0.0

    return max(token_score, sequence_score, containment_score)


def _context_similarity(
    left: EntityContext | None, right: EntityContext | None
) -> float:
    if left is None or right is None:
        return 0.0

    neighbour_score = _jaccard(left.neighbours, right.neighbours)
    predicate_score = _jaccard(left.predicates, right.predicates)
    return 0.60 * neighbour_score + 0.40 * predicate_score


def _type_compatibility(left: Any, right: Any) -> float:
    left_type = str(left or "").strip().casefold()
    right_type = str(right or "").strip().casefold()
    if not left_type or not right_type:
        return 0.5
    if left_type == right_type:
        return 1.0
    return 0.0


def _normalise_for_similarity(text: str) -> str:
    return " ".join(_WORD_RE.findall(text.casefold()))


def _jaccard(left: set[str] | frozenset[str], right: set[str] | frozenset[str]) -> float:
    if not left and not right:
        return 0.0
    intersection = len(left & right)
    union = len(left | right)
    return intersection / union if union else 0.0


def _groups_from_scored_pairs(
    pairs: list[ScoredPair], n: int) -> list[list[int]]:
    uf = _UnionFind(n)
    involved: set[int] = set()

    for pair in pairs:
        uf.union(pair.left_idx, pair.right_idx)
        involved.add(pair.left_idx)
        involved.add(pair.right_idx)

    groups: dict[int, list[int]] = defaultdict(list)
    for idx in involved:
        groups[uf.find(idx)].append(idx)

    return [group for group in groups.values() if len(group) >= 2]


def _merge_map_from_pairs(
    pairs: list[ScoredPair], all_entities: list[dict[str, Any]]
) -> dict[str, str]:
    groups = _groups_from_scored_pairs(pairs, len(all_entities))
    merge_map: dict[str, str] = {}

    for group_indices in groups:
        canonical = _choose_canonical(group_indices, all_entities)
        for idx in group_indices:
            title = all_entities[idx]["title"]
            if title != canonical:
                merge_map[title] = canonical

    return merge_map


def _choose_canonical(
    group_indices: list[int], all_entities: list[dict[str, Any]]) -> str:
    """Choose the most informative title without relying on a curated alias list."""

    def score(idx: int) -> tuple[float, int, int, str]:
        entity = all_entities[idx]
        title = entity["title"]
        description = str(entity.get("description") or "")
        text_units = entity.get("text_unit_ids") or []
        if hasattr(text_units, "tolist"):
            text_units = text_units.tolist()
        title_tokens = _WORD_RE.findall(title)
        acronym_penalty = 1.0 if len(title) <= 4 and title.isupper() else 0.0
        return (
            -acronym_penalty,
            len(title_tokens),
            len(description) + len(text_units),
            title,
        )

    return max(group_indices, key=score) |> lambda idx: all_entities[idx]["title"]


def _description_with_context(
    entity: dict[str, Any], context: EntityContext | None
) -> str:
    description = str(entity.get("description") or "")
    if context is None:
        return description
    predicates = ", ".join(sorted(context.predicates)[:20])
    neighbours = ", ".join(sorted(context.neighbours)[:20])
    return (
        f"{description}\n"
        f"Graph predicates: {predicates}\n"
        f"Graph neighbours: {neighbours}"
    )


def _max_pair_score(group: list[int], pairs: list[ScoredPair]) -> float:
    group_set = set(group)
    scores = [
        pair.score
        for pair in pairs
        if pair.left_idx in group_set and pair.right_idx in group_set
    ]
    return max(scores) if scores else -math.inf
