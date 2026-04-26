# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Parameterization settings for the ArangoDB graph store configuration."""

from pydantic import BaseModel, Field


class GraphStoreConfig(BaseModel):
    """Configuration for native ArangoDB graph storage and retrieval."""

    enabled: bool = Field(
        default=False,
        description="Whether ArangoDB graph indexing and retrieval is enabled.",
    )
    url: str = Field(
        default="http://localhost:8529",
        description="ArangoDB server URL.",
    )
    username: str = Field(
        default="root",
        description="ArangoDB username.",
    )
    password: str = Field(
        default="",
        description="ArangoDB password.",
    )
    db_name: str = Field(
        default="graphrag",
        description="ArangoDB database name.",
    )
    graph_name: str = Field(
        default="knowledge_graph",
        description="Named graph name in ArangoDB.",
    )
    batch_size: int = Field(
        default=500,
        description="Upsert batch size for bulk import operations.",
    )
    store_vectors: bool = Field(
        default=True,
        description=(
            "Whether to copy entity description vectors into the entities collection. "
            "Enables combined vector+graph AQL queries (hybrid search). "
            "Only effective when vector_store.type is also 'arangodb'."
        ),
    )
    traversal_depth: int = Field(
        default=2,
        description="Default k-hop depth for legacy graph traversal retrieval.",
    )
    top_k_seeds: int = Field(
        default=10,
        description="Number of vector seed entities for legacy hybrid retrieval.",
    )

    guided_retrieval_enabled: bool = Field(
        default=True,
        description="Whether query-time Arango retrieval should prefer guided vector-first traversal over legacy k-hop expansion.",
    )
    guided_seed_k: int = Field(
        default=12,
        description="Number of semantic seed entities for guided graph retrieval.",
    )
    guided_max_depth: int = Field(
        default=3,
        description="Maximum depth for guided best-first traversal. This is a cap, not a broad k-hop expansion.",
    )
    guided_max_edges_per_node: int = Field(
        default=8,
        description="Maximum number of scored edges expanded per frontier node.",
    )
    guided_max_expansions: int = Field(
        default=128,
        description="Global cap on node expansions for guided retrieval.",
    )
    guided_max_frontier_size: int = Field(
        default=256,
        description="Maximum number of pending path states kept in the best-first frontier.",
    )
    guided_max_results: int = Field(
        default=80,
        description="Maximum number of ranked path results returned by guided retrieval.",
    )
    guided_min_path_score: float = Field(
        default=0.05,
        description="Minimum score required to keep and expand a guided graph path.",
    )
    guided_depth_decay: float = Field(
        default=0.72,
        description="Per-hop score decay for guided traversal.",
    )
    guided_community_report_limit: int = Field(
        default=8,
        description="Maximum number of community reports returned with guided graph retrieval results.",
    )
    guided_allow_vector_scan_fallback: bool = Field(
        default=False,
        description="Allow full entity vector scan when the Arango vector index is unavailable. Keep false in production to protect latency.",
    )
