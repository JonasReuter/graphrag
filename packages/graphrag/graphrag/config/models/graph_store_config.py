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
    vector_size: int = Field(
        default=3072,
        description="Dimension of entity embedding vectors (must match embedding model output).",
    )
    traversal_depth: int = Field(
        default=2,
        description="Default k-hop depth for graph traversal retrieval.",
    )
    top_k_seeds: int = Field(
        default=10,
        description="Number of vector seed entities for hybrid retrieval.",
    )
