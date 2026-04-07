# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Evidence verification operation — LLM-based verification of extracted evidence."""

import json
import logging
import traceback
from typing import TYPE_CHECKING

import pandas as pd
from graphrag_llm.utils import CompletionMessagesBuilder

from graphrag.callbacks.workflow_callbacks import WorkflowCallbacks
from graphrag.config.enums import AsyncType
from graphrag.index.typing.error_handler import ErrorHandlerFn
from graphrag.index.utils.derive_from_rows import derive_from_rows
from graphrag.prompts.index.verify_evidence import VERIFY_EVIDENCE_PROMPT

if TYPE_CHECKING:
    from graphrag_llm.completion import LLMCompletion
    from graphrag_llm.types import LLMCompletionResponse

logger = logging.getLogger(__name__)


async def verify_evidence(
    evidence_df: pd.DataFrame,
    text_units_df: pd.DataFrame,
    model: "LLMCompletion",
    callbacks: WorkflowCallbacks,
    num_threads: int,
    async_type: AsyncType,
) -> pd.DataFrame:
    """Verify all evidence records against their source text units using LLM.

    Evidence records are grouped by text_unit_id so that all claims from a
    single chunk are verified in one LLM call (batching for efficiency).

    Returns the evidence_df with updated verification_status and verification_method columns.
    """
    if evidence_df is None or len(evidence_df) == 0:
        return evidence_df

    # Build a lookup: text_unit_id -> text
    text_lookup: dict[str, str] = {}
    if text_units_df is not None and len(text_units_df) > 0:
        for _, row in text_units_df.iterrows():
            text_lookup[str(row["id"])] = str(row.get("text", ""))

    # Group evidence by text_unit_id
    grouped = evidence_df.groupby("text_unit_id", sort=False)
    text_unit_ids = list(grouped.groups.keys())

    # Build rows for derive_from_rows: one row per text unit
    batch_rows = pd.DataFrame({
        "text_unit_id": text_unit_ids,
        "text": [text_lookup.get(tid, "") for tid in text_unit_ids],
        "evidence_indices": [
            grouped.groups[tid].tolist() for tid in text_unit_ids
        ],
    })

    # Verification results: index -> status
    verification_results: dict[int, str] = {}

    async def verify_batch(row):
        tid = row["text_unit_id"]
        source_text = row["text"]
        indices = row["evidence_indices"]

        if not source_text:
            for idx in indices:
                verification_results[idx] = "insufficient"
            return None

        batch_evidence = evidence_df.loc[indices]

        # Build claim list for the prompt
        claims_list = []
        for i, (df_idx, ev_row) in enumerate(batch_evidence.iterrows()):
            subject = ev_row.get("subject_id", "")
            subject_type = ev_row.get("subject_type", "entity")
            description = ev_row.get("source_span") or f"{subject_type}: {subject}"
            claims_list.append(f'{{"id": "{i}", "claim": "{description}"}}')

        claims_json = "\n".join(claims_list)

        try:
            messages = CompletionMessagesBuilder().add_user_message(
                VERIFY_EVIDENCE_PROMPT.format(
                    source_text=source_text,
                    claims=claims_json,
                )
            ).build()

            response: LLMCompletionResponse = await model.completion_async(
                messages=messages,
            )  # type: ignore
            content = response.content.strip()

            # Parse JSON response
            parsed = json.loads(content)
            id_to_status = {item["id"]: item["status"] for item in parsed}

            for i, df_idx in enumerate(indices):
                status = id_to_status.get(str(i), "insufficient")
                if status not in ("confirmed", "weakened", "refuted", "insufficient"):
                    status = "insufficient"
                verification_results[df_idx] = status

        except Exception as e:
            logger.warning(
                "Evidence verification failed for text_unit %s: %s",
                tid,
                e,
                exc_info=True,
            )
            for idx in indices:
                verification_results[idx] = "insufficient"

        return None

    await derive_from_rows(
        batch_rows,
        verify_batch,
        callbacks,
        num_threads=num_threads,
        async_type=async_type,
        progress_msg="verify evidence progress: ",
    )

    # Apply results back to the DataFrame
    result_df = evidence_df.copy()
    for idx, status in verification_results.items():
        result_df.at[idx, "verification_status"] = status
        result_df.at[idx, "verification_method"] = "llm_verified"

    return result_df
