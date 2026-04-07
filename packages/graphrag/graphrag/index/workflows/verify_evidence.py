# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing run_workflow method definition for evidence verification."""

import logging

from graphrag_llm.completion import create_completion

from graphrag.cache.cache_key_creator import cache_key_creator
from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.data_model.data_reader import DataReader
from graphrag.index.operations.verify_evidence.verify_evidence import (
    verify_evidence as run_verification,
)
from graphrag.index.typing.context import PipelineRunContext
from graphrag.index.typing.workflow import WorkflowFunctionOutput

logger = logging.getLogger(__name__)


async def run_workflow(
    config: GraphRagConfig,
    context: PipelineRunContext,
) -> WorkflowFunctionOutput:
    """Verify all extracted evidence records against their source text units."""
    logger.info("Workflow started: verify_evidence")

    evidence_cfg = getattr(config, "evidence", None)
    if not evidence_cfg or not evidence_cfg.enabled or not evidence_cfg.verification_enabled:
        logger.info("Workflow skipped: verify_evidence (disabled)")
        return WorkflowFunctionOutput(result=None)

    reader = DataReader(context.output_table_provider)

    # Load evidence table - may not exist if evidence is disabled
    try:
        evidence_df = await context.output_table_provider.read_dataframe("evidence")
    except Exception:
        logger.info("Workflow skipped: verify_evidence (no evidence table found)")
        return WorkflowFunctionOutput(result=None)

    if evidence_df is None or len(evidence_df) == 0:
        logger.info("Workflow skipped: verify_evidence (empty evidence table)")
        return WorkflowFunctionOutput(result=None)

    text_units = await reader.text_units()

    verification_model_config = config.get_completion_model_config(
        evidence_cfg.verification_model_id
    )
    verification_model = create_completion(
        verification_model_config,
        cache=context.cache.child(evidence_cfg.verification_model_instance_name),
        cache_key_creator=cache_key_creator,
    )

    verified_evidence = await run_verification(
        evidence_df=evidence_df,
        text_units_df=text_units,
        model=verification_model,
        callbacks=context.callbacks,
        num_threads=config.concurrent_requests,
        async_type=config.async_mode,
    )

    await context.output_table_provider.write_dataframe("evidence", verified_evidence)

    logger.info("Workflow completed: verify_evidence")
    return WorkflowFunctionOutput(result={"evidence": verified_evidence})
