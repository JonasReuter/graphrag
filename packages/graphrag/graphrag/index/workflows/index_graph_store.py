# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing run_workflow method definition for ArangoDB graph store indexing."""

import logging

from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.index.operations.index_graph_store import index_graph_store as run_index
from graphrag.index.typing.context import PipelineRunContext
from graphrag.index.typing.workflow import WorkflowFunctionOutput

logger = logging.getLogger(__name__)


async def run_workflow(
    config: GraphRagConfig,
    context: PipelineRunContext,
) -> WorkflowFunctionOutput:
    """Index GraphRAG pipeline output into ArangoDB as a named graph.

    Stores entities, relationships (as edges), communities, community_reports,
    and text_units in ArangoDB collections, and creates:
    - Named graph 'knowledge_graph' with edge definitions
    - HNSW vector index on entities.vector (if store_vectors=True)
    - entity_community_membership and entity_text_unit edge collections

    Skipped (no-op) when graph_store.enabled is false.
    """
    logger.info("Workflow started: index_graph_store")

    graph_cfg = getattr(config, "graph_store", None)
    if not graph_cfg or not graph_cfg.enabled:
        logger.info("Workflow skipped: index_graph_store (graph_store.enabled=false)")
        return WorkflowFunctionOutput(result=None)

    from graphrag_vectors.arangodb_graph import ArangoDBGraphStore
    from graphrag_vectors.vector_store_type import VectorStoreType

    # Only copy vectors into entity docs when the vector store is also ArangoDB
    # (so the entity_description_embedding collection already exists there)
    store_vectors = graph_cfg.store_vectors
    vector_store_type = getattr(config.vector_store, "type", None)
    if store_vectors and str(vector_store_type) != str(VectorStoreType.ArangoDB):
        logger.warning(
            "graph_store.store_vectors=true but vector_store.type is '%s', not 'arangodb'. "
            "Entity vectors will NOT be copied into graph store entity documents. "
            "Set graph_store.store_vectors: false or switch vector_store.type to arangodb.",
            vector_store_type,
        )
        store_vectors = False

    graph_store = ArangoDBGraphStore(
        url=graph_cfg.url,
        username=graph_cfg.username,
        password=graph_cfg.password,
        db_name=graph_cfg.db_name,
        graph_name=graph_cfg.graph_name,
        batch_size=graph_cfg.batch_size,
        vector_size=graph_cfg.vector_size,
    )
    graph_store.connect()

    counts = await run_index(
        graph_store=graph_store,
        table_provider=context.output_table_provider,
        store_vectors=store_vectors,
        db=graph_store._db if store_vectors else None,
    )

    logger.info("Workflow completed: index_graph_store — %s", counts)
    return WorkflowFunctionOutput(result=counts)
