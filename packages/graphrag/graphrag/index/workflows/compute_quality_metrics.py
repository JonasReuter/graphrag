# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing run_workflow method definition for quality metrics."""

import logging

from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.data_model.data_reader import DataReader
from graphrag.index.operations.compute_quality_metrics.compute_quality_metrics import (
    compute_quality_metrics as run_metrics,
)
from graphrag.index.typing.context import PipelineRunContext
from graphrag.index.typing.workflow import WorkflowFunctionOutput

logger = logging.getLogger(__name__)


async def run_workflow(
    config: GraphRagConfig,
    context: PipelineRunContext,
) -> WorkflowFunctionOutput:
    """Compute quality metrics for the knowledge graph."""
    logger.info("Workflow started: compute_quality_metrics")

    evidence_cfg = getattr(config, "evidence", None)
    if not evidence_cfg or not evidence_cfg.enabled or not evidence_cfg.quality_metrics_enabled:
        logger.info("Workflow skipped: compute_quality_metrics (disabled)")
        return WorkflowFunctionOutput(result=None)

    reader = DataReader(context.output_table_provider)

    entities_df = await reader.entities()
    relationships_df = await reader.relationships()

    # Evidence and contradictions tables may not exist
    evidence_df = None
    contradictions_df = None
    try:
        evidence_df = await context.output_table_provider.read_dataframe("evidence")
    except Exception:
        pass
    try:
        contradictions_df = await context.output_table_provider.read_dataframe("contradictions")
    except Exception:
        pass

    quality_metrics_df, quality_details_df = run_metrics(
        entities_df=entities_df,
        relationships_df=relationships_df,
        evidence_df=evidence_df,
        contradictions_df=contradictions_df,
        low_confidence_threshold=evidence_cfg.low_confidence_threshold,
    )

    await context.output_table_provider.write_dataframe("quality_metrics", quality_metrics_df)
    if len(quality_details_df) > 0:
        await context.output_table_provider.write_dataframe("quality_details", quality_details_df)

    # Log a quick summary
    if len(quality_metrics_df) > 0:
        row = quality_metrics_df.iloc[0]
        logger.info(
            "Quality metrics: %d entities, %d relationships, %.1f%% with evidence, "
            "avg confidence=%.2f, %.1f%% low confidence, %.1f%% contradicted",
            int(row.get("total_entities", 0)),
            int(row.get("total_relationships", 0)),
            float(row.get("pct_entities_with_evidence", 0)) * 100,
            float(row.get("avg_confidence", 0)),
            float(row.get("pct_low_confidence", 0)) * 100,
            float(row.get("pct_contradicted", 0)) * 100,
        )

    logger.info("Workflow completed: compute_quality_metrics")
    return WorkflowFunctionOutput(
        result={
            "quality_metrics": quality_metrics_df,
            "quality_details": quality_details_df,
        }
    )
