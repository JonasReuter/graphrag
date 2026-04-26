# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Guided ArangoDB graph retrieval helpers for query-time retrieval."""

from __future__ import annotations

from typing import Any

from graphrag.config.models.graph_store_config import GraphStoreConfig
from graphrag_vectors import (
    GuidedArangoGraphRetriever,
    GuidedGraphRetrievalConfig,
    QueryGraphPlan,
)


def guided_config_from_graph_store_config(
    graph_store_config: GraphStoreConfig,
) -> GuidedGraphRetrievalConfig:
    """Build guided retrieval runtime config from the GraphRAG graph store config."""
    return GuidedGraphRetrievalConfig(
        seed_k=graph_store_config.guided_seed_k,
        max_depth=graph_store_config.guided_max_depth,
        max_edges_per_node=graph_store_config.guided_max_edges_per_node,
        max_expansions=graph_store_config.guided_max_expansions,
        max_frontier_size=graph_store_config.guided_max_frontier_size,
        max_results=graph_store_config.guided_max_results,
        min_path_score=graph_store_config.guided_min_path_score,
        depth_decay=graph_store_config.guided_depth_decay,
        community_report_limit=graph_store_config.guided_community_report_limit,
        allow_vector_scan_fallback=graph_store_config.guided_allow_vector_scan_fallback,
    )


def retrieve_guided_graph_context(
    *,
    graph_store: Any,
    query_vector: list[float],
    query: str,
    graph_store_config: GraphStoreConfig,
    entity_types: list[str] | tuple[str, ...] | None = None,
    relation_types: list[str] | tuple[str, ...] | None = None,
    hard_entity_type_filter: bool = False,
    hard_relation_type_filter: bool = False,
) -> dict[str, Any]:
    """Run guided vector-first graph retrieval against an ArangoDBGraphStore.

    This helper keeps query-time callers away from broad k-hop retrieval. It uses
    the Arango entity vector index for seeds and expands only a bounded number of
    high-scoring graph paths.
    """
    if not graph_store_config.guided_retrieval_enabled:
        return {"vertices": [], "edges": [], "paths": [], "community_reports": [], "stats": {"disabled": True}}

    retriever = GuidedArangoGraphRetriever(
        db=graph_store._db,
        graph_name=graph_store.graph_name,
    )
    plan = QueryGraphPlan.from_query_text(
        query,
        entity_types=entity_types,
        relation_types=relation_types,
        hard_entity_type_filter=hard_entity_type_filter,
        hard_relation_type_filter=hard_relation_type_filter,
    )
    config = guided_config_from_graph_store_config(graph_store_config)
    return retriever.retrieve(
        query_vector=query_vector,
        query=query,
        plan=plan,
        config=config,
    )
