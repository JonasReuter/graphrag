# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing run_workflow method definition."""

import logging
from typing import TYPE_CHECKING
from uuid import uuid4

import pandas as pd
from graphrag_llm.completion import create_completion
from graphrag_llm.embedding import create_embedding
from graphrag_vectors import create_vector_store
from graphrag_vectors.index_schema import IndexSchema

from graphrag.cache.cache_key_creator import cache_key_creator
from graphrag.callbacks.workflow_callbacks import WorkflowCallbacks
from graphrag.config.defaults import DEFAULT_ENTITY_TYPES
from graphrag.config.enums import AsyncType
from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.data_model.data_reader import DataReader
from graphrag.data_model.schemas import COVARIATES_FINAL_COLUMNS
from graphrag.index.operations.extract_covariates.extract_covariates import (
    extract_covariates as extractor,
)
from graphrag.index.typing.context import PipelineRunContext
from graphrag.index.typing.workflow import WorkflowFunctionOutput

if TYPE_CHECKING:
    from graphrag_llm.completion import LLMCompletion

logger = logging.getLogger(__name__)


async def run_workflow(
    config: GraphRagConfig,
    context: PipelineRunContext,
) -> WorkflowFunctionOutput:
    """All the steps to extract and format covariates."""
    logger.info("Workflow started: extract_covariates")
    output = None
    if config.extract_claims.enabled:
        reader = DataReader(context.output_table_provider)
        text_units = await reader.text_units()

        # Build entity lookup map from resolved entities so _clean_claim can
        # normalize LLM-produced name variants (e.g. ß vs SS) back to canonical titles.
        resolved_entities_map: dict[str, str] = {}
        try:
            entities = await reader.entities()
            if entities is not None and "title" in entities.columns:
                for title in entities["title"].dropna():
                    canonical = str(title)
                    # exact match
                    resolved_entities_map[canonical] = canonical
                    # ß-normalized variant → canonical
                    normalized = canonical.upper().replace("ß", "SS").strip()
                    resolved_entities_map[normalized] = canonical
        except Exception:  # noqa: BLE001
            logger.debug("No entities table found; skipping entity name normalization for claims.")

        model_config = config.get_completion_model_config(
            config.extract_claims.completion_model_id
        )

        model = create_completion(
            model_config,
            cache=context.cache.child(config.extract_claims.model_instance_name),
            cache_key_creator=cache_key_creator,
        )

        prompts = config.extract_claims.resolved_prompts()

        output = await extract_covariates(
            text_units=text_units,
            callbacks=context.callbacks,
            model=model,
            covariate_type="claim",
            max_gleanings=config.extract_claims.max_gleanings,
            claim_description=config.extract_claims.description,
            prompt=prompts.extraction_prompt,
            entity_types=DEFAULT_ENTITY_TYPES,
            num_threads=config.concurrent_requests,
            async_type=config.async_mode,
            resolved_entities_map=resolved_entities_map,
        )

        # Optional: resolve claim subject/object IDs to canonical entity titles
        # using embedding similarity + LLM confirmation. Controlled by
        # config.resolve_claim_subjects. Runs inline so extract_covariates
        # produces fully-linked covariates in a single pipeline step.
        if config.resolve_claim_subjects.enabled and len(output) > 0:
            from graphrag.index.operations.resolve_claim_subjects.resolve_claim_subjects import (
                resolve_claim_subjects,
            )
            from graphrag.prompts.index.resolve_claim_subjects import (
                RESOLVE_CLAIM_SUBJECT_PROMPT,
            )

            rcs_cfg = config.resolve_claim_subjects
            emb_cfg = config.get_embedding_model_config(rcs_cfg.embedding_model_id)
            embedding_model = create_embedding(
                emb_cfg,
                cache=context.cache.child("resolve_claim_subjects_embeddings"),
                cache_key_creator=cache_key_creator,
            )
            llm_cfg = config.get_completion_model_config(rcs_cfg.completion_model_id)
            llm_model = create_completion(
                llm_cfg,
                cache=context.cache.child(rcs_cfg.model_instance_name),
                cache_key_creator=cache_key_creator,
            )
            vs_config = rcs_cfg.vector_store or config.vector_store
            vector_size = getattr(emb_cfg, "vector_size", None) or 1536
            index_schema = IndexSchema(
                index_name=rcs_cfg.entity_index_name,
                id_field="id",
                vector_field="vector",
                vector_size=vector_size,
                fields={"title": "str"},
            )
            vector_store = create_vector_store(vs_config, index_schema)
            vector_store.connect()

            output, _ = await resolve_claim_subjects(
                covariates_df=output,
                embedding_model=embedding_model,
                completion_model=llm_model,
                vector_store=vector_store,
                prompt=RESOLVE_CLAIM_SUBJECT_PROMPT,
                high_threshold=rcs_cfg.high_threshold,
                mid_threshold=rcs_cfg.mid_threshold,
                top_k=rcs_cfg.top_k,
            )

        await context.output_table_provider.write_dataframe("covariates", output)

    logger.info("Workflow completed: extract_covariates")
    return WorkflowFunctionOutput(result=output)


async def extract_covariates(
    text_units: pd.DataFrame,
    callbacks: WorkflowCallbacks,
    model: "LLMCompletion",
    covariate_type: str,
    max_gleanings: int,
    claim_description: str,
    prompt: str,
    entity_types: list[str],
    num_threads: int,
    async_type: AsyncType,
    resolved_entities_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """All the steps to extract and format covariates."""
    # reassign the id because it will be overwritten in the output by a covariate one
    # this also results in text_unit_id being copied to the output covariate table
    text_units["text_unit_id"] = text_units["id"]

    covariates = await extractor(
        input=text_units,
        callbacks=callbacks,
        model=model,
        column="text",
        covariate_type=covariate_type,
        max_gleanings=max_gleanings,
        claim_description=claim_description,
        prompt=prompt,
        entity_types=entity_types,
        num_threads=num_threads,
        async_type=async_type,
        resolved_entities_map=resolved_entities_map,
    )
    text_units.drop(columns=["text_unit_id"], inplace=True)  # don't pollute the global
    covariates["id"] = covariates["covariate_type"].apply(lambda _x: str(uuid4()))
    covariates["human_readable_id"] = covariates.index

    return covariates.loc[:, COVARIATES_FINAL_COLUMNS]
