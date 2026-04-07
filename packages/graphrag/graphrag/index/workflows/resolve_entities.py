# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Entity resolution workflow: persistent embedding-based deduplication."""

import logging

from graphrag_llm.completion import create_completion
from graphrag_llm.embedding import create_embedding
from graphrag_vectors import create_vector_store
from graphrag_vectors.index_schema import IndexSchema

from graphrag.cache.cache_key_creator import cache_key_creator
from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.data_model.row_transformers import transform_entity_row
from graphrag.index.operations.resolve_entities.resolve_entities import (
    resolve_entities,
)
from graphrag.index.typing.context import PipelineRunContext
from graphrag.index.typing.workflow import WorkflowFunctionOutput

logger = logging.getLogger(__name__)

_ENTITY_RESOLUTION_INDEX = "entity_resolution_embeddings"


async def run_workflow(
    config: GraphRagConfig,
    context: PipelineRunContext,
) -> WorkflowFunctionOutput:
    """Run the entity resolution workflow.

    No-ops when ``config.entity_resolution.enabled`` is False.
    """
    if not config.entity_resolution.enabled:
        logger.info("Workflow skipped: resolve_entities (disabled in config)")
        return WorkflowFunctionOutput(result=None)

    logger.info("Workflow started: resolve_entities")

    er_config = config.entity_resolution

    # --- Embedding model ---
    embedding_model_config = config.get_embedding_model_config(
        er_config.embedding_model_id
    )
    embedding_model = create_embedding(
        embedding_model_config,
        cache=context.cache.child("entity_resolution_embeddings"),
        cache_key_creator=cache_key_creator,
    )

    # --- Completion model for LLM confirmation ---
    completion_model_config = config.get_completion_model_config(
        er_config.completion_model_id
    )
    completion_model = create_completion(
        completion_model_config,
        cache=context.cache.child(er_config.model_instance_name),
        cache_key_creator=cache_key_creator,
    )

    # --- Vector store for persistent entity embeddings ---
    # Uses a dedicated index on the configured vector store backend.
    vs_config = er_config.vector_store or config.vector_store
    index_schema = IndexSchema(
        index_name=_ENTITY_RESOLUTION_INDEX,
        id_field="id",
        vector_field="vector",
        vector_size=embedding_model_config.vector_size
        if hasattr(embedding_model_config, "vector_size")
        else 1536,
        fields={"title": "str"},
    )
    vector_store = create_vector_store(vs_config, index_schema)
    vector_store.connect()

    # --- Prompt ---
    prompt = er_config.resolved_prompt()

    async with (
        context.output_table_provider.open(
            "entities",
            transformer=transform_entity_row,
        ) as entities_table,
        context.output_table_provider.open(
            "relationships",
        ) as relationships_table,
    ):
        result = await resolve_entities(
            entities_table=entities_table,
            relationships_table=relationships_table,
            embedding_model=embedding_model,
            completion_model=completion_model,
            vector_store=vector_store,
            prompt=prompt,
            similarity_threshold=er_config.similarity_threshold,
            top_k=er_config.top_k,
        )

    logger.info("Workflow completed: resolve_entities")
    return WorkflowFunctionOutput(result=result)
