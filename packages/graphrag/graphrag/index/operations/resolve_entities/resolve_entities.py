# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Entity resolution operation: embed → search → group → LLM confirm → merge → upsert."""

import json
import logging
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import pandas as pd

from graphrag_storage.tables.table import Table
from graphrag_vectors import VectorStore, VectorStoreDocument

from graphrag.data_model.row_transformers import transform_entity_row
from graphrag.data_model.schemas import ENTITIES_FINAL_COLUMNS
from graphrag.index.operations.resolve_entities.entity_grouping import (
    build_candidate_groups,
)
from graphrag.index.operations.resolve_entities.entity_merge_extractor import (
    EntityMergeExtractor,
)
from graphrag.index.operations.resolve_entities.typing import MergeGroup

if TYPE_CHECKING:
    from graphrag_llm.completion import LLMCompletion
    from graphrag_llm.embedding import LLMEmbedding

logger = logging.getLogger(__name__)

_EMBED_BATCH = 32


async def resolve_entities(
    entities_table: Table,
    relationships_table: Table,
    embedding_model: "LLMEmbedding",
    completion_model: "LLMCompletion",
    vector_store: VectorStore,
    prompt: str,
    similarity_threshold: float,
    top_k: int,
    strategy: str = "llm_context_window",
    window_tokens: int = 100_000,
) -> dict[str, list[dict[str, Any]]]:
    """Run the full entity resolution pass.

    For strategy ``llm_context_window`` (default):
    1. Load all entities from the table.
    2. Embed each entity (title: description).
    3. Sort entities by greedy nearest-neighbor traversal of embedding space.
    4. Pack into windows of ``window_tokens`` and let the LLM identify duplicates.
    5. Apply confirmed merges to entities and relationships tables.

    For strategy ``embedding_search`` (legacy):
    1. Load all entities from the table.
    2. Embed each entity and upsert into the persistent store.
    3. For every entity, search the store for the top-k most similar neighbours.
    4. Group pairs above the threshold using Union-Find.
    5. Confirm each group via LLM.
    6. Apply confirmed merges to entities and relationships tables.

    Returns sample rows for logging.
    """
    # --- Phase 1: load all entities into memory ---
    all_entities: list[dict[str, Any]] = []
    async for row in entities_table:
        if row.get("title"):
            all_entities.append(dict(row))

    if not all_entities:
        logger.warning("resolve_entities: no entities found — skipping.")
        return {"entities": [], "relationships": []}

    logger.info("resolve_entities: loaded %d entities", len(all_entities))

    # --- Phase 2: embed entities inline and upsert into vector store ---
    texts = [f"{e['title']}: {e.get('description', '')}" for e in all_entities]
    ids = [str(e["id"]) for e in all_entities]
    embeddings = await _embed_texts(texts, ids, embedding_model, vector_store)

    # --- Phase 3 & 4: resolve duplicates based on strategy ---
    if strategy == "llm_context_window":
        merge_map = await _resolve_via_context_window(
            all_entities=all_entities,
            embeddings=embeddings,
            completion_model=completion_model,
            prompt=prompt,
            window_tokens=window_tokens,
        )
    else:
        merge_map = await _build_merge_map(
            all_entities=all_entities,
            embeddings=embeddings,
            vector_store=vector_store,
            extractor=EntityMergeExtractor(model=completion_model, prompt=prompt),
            similarity_threshold=similarity_threshold,
            top_k=top_k,
        )

    if not merge_map:
        logger.info("resolve_entities: no merges confirmed — graph unchanged.")
        return {"entities": [], "relationships": [], "contradictions": pd.DataFrame()}

    logger.info("resolve_entities: applying %d merge mappings", len(merge_map))

    # --- Phase 5: apply merges ---
    entity_samples = await _rewrite_entities(entities_table, all_entities, merge_map)
    relationship_samples = await _rewrite_relationships(relationships_table, merge_map)
    contradictions_df = _build_contradictions_from_merge_map(merge_map)

    logger.info(
        "resolve_entities: recorded %d same_as contradiction(s)", len(contradictions_df)
    )
    return {
        "entities": entity_samples,
        "relationships": relationship_samples,
        "contradictions": contradictions_df,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _embed_texts(
    texts: list[str],
    ids: list[str],
    model: "LLMEmbedding",
    vector_store: VectorStore,
) -> list[list[float]]:
    """Embed texts in batches, upsert to vector store, and return all vectors."""
    all_vectors: list[list[float] | None] = [None] * len(texts)

    for start in range(0, len(texts), _EMBED_BATCH):
        batch_texts = texts[start : start + _EMBED_BATCH]
        batch_ids = ids[start : start + _EMBED_BATCH]

        response = await model.embedding_async(input=batch_texts)
        vectors = response.embeddings or []

        docs: list[VectorStoreDocument] = []
        for i, (doc_id, vec) in enumerate(zip(batch_ids, vectors, strict=False)):
            if vec is None:
                continue
            vec_list = vec.tolist() if hasattr(vec, "tolist") else list(vec)
            all_vectors[start + i] = vec_list
            docs.append(
                VectorStoreDocument(
                    id=doc_id,
                    vector=vec_list,
                    data={"title": texts[start + i].split(":")[0]},
                )
            )

        if docs:
            vector_store.load_documents(docs)

    # Fill any None with empty list so indices remain stable
    return [v if v is not None else [] for v in all_vectors]


async def _build_merge_map(
    all_entities: list[dict[str, Any]],
    embeddings: list[list[float]],
    vector_store: VectorStore,
    extractor: EntityMergeExtractor,
    similarity_threshold: float,
    top_k: int,
) -> dict[str, str]:
    """Return alias→canonical mapping for confirmed merges.

    Uses the vector store for neighbour lookup (persistent across runs),
    then groups candidates via Union-Find, then confirms via LLM.
    """
    titles = [e["title"] for e in all_entities]
    title_to_idx = {t: i for i, t in enumerate(titles)}

    # Collect candidate pairs from vector store search
    candidate_pairs: list[tuple[int, int]] = []
    for i, (entity, vec) in enumerate(zip(all_entities, embeddings, strict=True)):
        if not vec:
            continue
        results = vector_store.similarity_search_by_vector(
            query_embedding=vec,
            k=top_k + 1,  # +1 because the entity itself will be returned
            include_vectors=False,
        )
        for result in results:
            if result.score < similarity_threshold:
                continue
            neighbour_title = result.document.data.get("title", "")
            j = title_to_idx.get(neighbour_title)
            if j is None or j == i:
                continue
            lo, hi = (i, j) if i < j else (j, i)
            candidate_pairs.append((lo, hi))

    if not candidate_pairs:
        return {}

    # Deduplicate pairs and build groups
    unique_pairs = list({(lo, hi) for lo, hi in candidate_pairs})
    # Build simple adjacency and use build_candidate_groups for transitive closure
    # We already have pairs — pass embeddings to get Union-Find grouping
    groups = _groups_from_pairs(unique_pairs, len(titles))

    merge_map: dict[str, str] = {}
    for group_indices in groups:
        candidates = [
            {
                "title": all_entities[i]["title"],
                "type": all_entities[i].get("type", ""),
                "description": all_entities[i].get("description", ""),
            }
            for i in group_indices
        ]
        decision = await extractor(candidates)
        if decision.merge and decision.canonical:
            canonical = decision.canonical
            # Validate canonical is actually in the group
            group_titles = {c["title"] for c in candidates}
            if canonical not in group_titles:
                canonical = candidates[0]["title"]
            for c in candidates:
                if c["title"] != canonical:
                    merge_map[c["title"]] = canonical

    return merge_map


def _groups_from_pairs(
    pairs: list[tuple[int, int]], n: int
) -> list[list[int]]:
    """Build transitive closure groups from a list of (i, j) pairs via Union-Find."""
    from graphrag.index.operations.resolve_entities.entity_grouping import _UnionFind

    uf = _UnionFind(n)
    for lo, hi in pairs:
        uf.union(lo, hi)

    from collections import defaultdict

    groups: dict[int, list[int]] = defaultdict(list)
    # Only include indices that appear in at least one pair
    involved = {idx for pair in pairs for idx in pair}
    for idx in involved:
        groups[uf.find(idx)].append(idx)

    return [g for g in groups.values() if len(g) >= 2]


def _sort_by_embedding(embeddings: list[list[float]]) -> list[int]:
    """Greedy nearest-neighbor sort — returns indices in traversal order.

    Adjacent indices in the result have maximally similar embeddings,
    which clusters potential duplicates close together in the LLM window.
    """
    import numpy as np

    if not embeddings or not embeddings[0]:
        return list(range(len(embeddings)))

    E = np.array(embeddings, dtype="float32")
    norms = np.linalg.norm(E, axis=1, keepdims=True)
    E = E / (norms + 1e-8)

    n = len(E)
    visited = np.zeros(n, dtype=bool)
    order: list[int] = [0]
    visited[0] = True

    for _ in range(n - 1):
        sims = E @ E[order[-1]]
        sims[visited] = -np.inf
        next_idx = int(np.argmax(sims))
        order.append(next_idx)
        visited[next_idx] = True

    return order


def _pack_windows(
    entities: list[dict], window_tokens: int
) -> list[list[dict]]:
    """Pack entities into windows not exceeding window_tokens (estimated)."""
    windows: list[list[dict]] = []
    current: list[dict] = []
    current_tokens = 0

    for entity in entities:
        text = json.dumps(entity, ensure_ascii=False)
        tokens = len(text) // 4
        if current and current_tokens + tokens > window_tokens:
            windows.append(current)
            current = []
            current_tokens = 0
        current.append(entity)
        current_tokens += tokens

    if current:
        windows.append(current)

    return windows


async def _resolve_via_context_window(
    all_entities: list[dict[str, Any]],
    embeddings: list[list[float]],
    completion_model: "LLMCompletion",
    prompt: str,
    window_tokens: int,
) -> dict[str, str]:
    """Sort entities by embedding similarity, pack into windows, let LLM identify duplicates."""
    from graphrag.index.operations.resolve_entities.entity_disambiguation_extractor import (
        EntityDisambiguationExtractor,
    )

    sorted_indices = _sort_by_embedding(embeddings)
    sorted_entities = [
        {
            "title": all_entities[i]["title"],
            "type": all_entities[i].get("type", ""),
            "description": all_entities[i].get("description", ""),
        }
        for i in sorted_indices
    ]

    windows = _pack_windows(sorted_entities, window_tokens)
    extractor = EntityDisambiguationExtractor(model=completion_model, prompt=prompt)

    # Track which titles appear in which window to handle cross-window duplicates
    # (not handled in this version — within-window disambiguation only)
    merge_map: dict[str, str] = {}
    for window in windows:
        groups = await extractor(window)
        all_titles = {e["title"] for e in window}
        for group in groups:
            canonical = group.canonical
            if canonical not in all_titles:
                continue
            for alias in group.aliases:
                if alias in all_titles and alias != canonical:
                    merge_map[alias] = canonical

    return merge_map


async def _rewrite_entities(
    entities_table: Table,
    all_entities: list[dict[str, Any]],
    merge_map: dict[str, str],
) -> list[dict[str, Any]]:
    """Rewrite the entities table: drop aliases, merge text_unit_ids into canonical."""
    # Build a lookup: canonical title → merged entity data
    canonical_data: dict[str, dict[str, Any]] = {}
    for entity in all_entities:
        title = entity["title"]
        canonical = merge_map.get(title, title)
        if canonical not in canonical_data:
            canonical_data[canonical] = dict(entity)
            canonical_data[canonical]["text_unit_ids"] = list(
                entity.get("text_unit_ids") or []
            )
        else:
            # Merge text_unit_ids from alias into canonical
            existing_ids = set(canonical_data[canonical]["text_unit_ids"])
            for uid in entity.get("text_unit_ids") or []:
                existing_ids.add(uid)
            canonical_data[canonical]["text_unit_ids"] = list(existing_ids)

    # Update frequency for canonical entities
    for data in canonical_data.values():
        data["frequency"] = len(data["text_unit_ids"])

    # Stream-write resolved entities back
    samples: list[dict[str, Any]] = []
    human_readable_id = 0
    seen: set[str] = set()

    async for row in entities_table:
        title = row.get("title")
        if not title or title in seen:
            continue
        canonical = merge_map.get(title, title)
        if canonical in seen:
            continue
        seen.add(canonical)

        merged = canonical_data.get(canonical, row)
        merged["title"] = canonical
        merged["id"] = str(uuid4())
        merged["human_readable_id"] = human_readable_id
        human_readable_id += 1
        out = {col: merged.get(col) for col in ENTITIES_FINAL_COLUMNS}
        await entities_table.write(out)
        if len(samples) < 5:
            samples.append(out)

    return samples


async def _rewrite_relationships(
    relationships_table: Table,
    merge_map: dict[str, str],
) -> list[dict[str, Any]]:
    """Rewrite relationship source/target using the merge map, remove self-loops, re-deduplicate."""
    seen: dict[tuple[str, str], dict[str, Any]] = {}

    async for row in relationships_table:
        source = merge_map.get(row.get("source", ""), row.get("source", ""))
        target = merge_map.get(row.get("target", ""), row.get("target", ""))

        # Drop self-loops
        if source == target:
            continue

        key = (source, target)
        if key not in seen:
            row["source"] = source
            row["target"] = target
            seen[key] = dict(row)
        else:
            # Merge text_unit_ids
            existing_ids = set(seen[key].get("text_unit_ids") or [])
            for uid in row.get("text_unit_ids") or []:
                existing_ids.add(uid)
            seen[key]["text_unit_ids"] = list(existing_ids)
            seen[key]["weight"] = (seen[key].get("weight") or 0) + (
                row.get("weight") or 0
            )

    samples: list[dict[str, Any]] = []
    human_readable_id = 0
    for row in seen.values():
        row["human_readable_id"] = human_readable_id
        human_readable_id += 1
        await relationships_table.write(row)
        if len(samples) < 5:
            samples.append(row)

    return samples


_CONTRADICTION_COLUMNS = [
    "id", "human_readable_id", "relation_type",
    "subject_a_type", "subject_a_id",
    "subject_b_type", "subject_b_id",
    "description", "confidence", "detection_method",
]


def _build_contradictions_from_merge_map(
    merge_map: dict[str, str],
) -> pd.DataFrame:
    """Build a contradictions DataFrame from LLM-confirmed entity merges.

    Each alias→canonical pair becomes a ``same_as`` Contradiction record,
    preserving the provenance of every entity deduplication decision.
    """
    if not merge_map:
        return pd.DataFrame(columns=_CONTRADICTION_COLUMNS)

    rows = []
    for i, (alias, canonical) in enumerate(merge_map.items()):
        rows.append({
            "id": str(uuid4()),
            "human_readable_id": str(i),
            "relation_type": "same_as",
            "subject_a_type": "entity",
            "subject_a_id": alias,
            "subject_b_type": "entity",
            "subject_b_id": canonical,
            "description": f'"{alias}" is a duplicate of "{canonical}" (LLM-confirmed)',
            "confidence": 0.9,
            "detection_method": "llm_verified",
        })
    return pd.DataFrame(rows)
