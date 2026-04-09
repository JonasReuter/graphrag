# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing run_workflow method definition for confidence scoring."""

import logging

import pandas as pd

from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.data_model.data_reader import DataReader
from graphrag.index.operations.compute_confidence.compute_confidence import (
    compute_entity_relationship_confidence,
)
from graphrag.index.typing.context import PipelineRunContext
from graphrag.index.typing.workflow import WorkflowFunctionOutput

logger = logging.getLogger(__name__)


async def run_workflow(
    config: GraphRagConfig,
    context: PipelineRunContext,
) -> WorkflowFunctionOutput:
    """Compute multi-factor confidence scores for all entities and relationships."""
    logger.info("Workflow started: compute_confidence")

    evidence_cfg = getattr(config, "evidence", None)
    if not evidence_cfg or not evidence_cfg.enabled:
        logger.info("Workflow skipped: compute_confidence (evidence disabled)")
        return WorkflowFunctionOutput(result=None)

    reader = DataReader(context.output_table_provider)

    try:
        evidence_df = await context.output_table_provider.read_dataframe("evidence")
    except Exception:
        logger.info("Workflow skipped: compute_confidence (no evidence table found)")
        return WorkflowFunctionOutput(result=None)

    if evidence_df is None or len(evidence_df) == 0:
        logger.info("Workflow skipped: compute_confidence (empty evidence table)")
        return WorkflowFunctionOutput(result=None)

    entities_df = await reader.entities()
    relationships_df = await reader.relationships()
    text_units_df = await reader.text_units()

    enriched_entities, enriched_relationships, contradictions_df = (
        compute_entity_relationship_confidence(
            entities_df=entities_df,
            relationships_df=relationships_df,
            evidence_df=evidence_df,
            text_units_df=text_units_df,
            weights=evidence_cfg.confidence_weights,
            cross_doc_similarity_threshold=evidence_cfg.cross_doc_similarity_threshold,
        )
    )

    await context.output_table_provider.write_dataframe("entities", enriched_entities)
    await context.output_table_provider.write_dataframe("relationships", enriched_relationships)

    # Merge with any contradictions already written by earlier workflows (e.g. resolve_entities)
    existing_contradictions: pd.DataFrame | None = None
    try:
        existing_contradictions = await context.output_table_provider.read_dataframe(
            "contradictions"
        )
    except Exception:
        pass

    frames = [
        df
        for df in [existing_contradictions, contradictions_df if len(contradictions_df) > 0 else None]
        if df is not None and len(df) > 0
    ]
    if frames:
        combined = pd.concat(frames, ignore_index=True)
        await context.output_table_provider.write_dataframe("contradictions", combined)

    logger.info(
        "Workflow completed: compute_confidence — %d contradictions found",
        len(contradictions_df),
    )
    return WorkflowFunctionOutput(
        result={
            "entities": enriched_entities,
            "relationships": enriched_relationships,
            "contradictions": contradictions_df,
        }
    )
