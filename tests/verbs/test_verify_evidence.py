# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Workflow-level tests for evidence verification."""

import json

import pandas as pd
import pytest
from graphrag_llm.config import LLMProviderType

from graphrag.index.workflows.verify_evidence import run_workflow
from tests.unit.config.utils import get_default_graphrag_config
from tests.verbs.util import create_test_context


def _make_evidence_df(n: int = 3) -> pd.DataFrame:
    """Build a minimal evidence DataFrame."""
    rows = []
    for i in range(n):
        rows.append({
            "id": f"ev{i}",
            "human_readable_id": str(i),
            "subject_type": "entity",
            "subject_id": f"ENTITY_{i}",
            "text_unit_id": "tu1",  # all point to same text unit → 1 LLM call
            "source_span": f"Entity {i} was mentioned here",
            "extraction_confidence": 0.8,
            "completeness_status": "complete",
            "verification_status": "unverified",
            "verification_method": None,
        })
    return pd.DataFrame(rows)


def _make_text_units_df() -> pd.DataFrame:
    return pd.DataFrame([{
        "id": "tu1",
        "human_readable_id": "0",
        "text": "Entity 0 was mentioned here. Entity 1 was mentioned here. Entity 2 was mentioned here.",
        "document_id": "doc1",
        "n_tokens": 20,
    }])


# Mock response: one JSON array with statuses for all 3 claims
MOCK_VERIFY_RESPONSE = json.dumps([
    {"id": "0", "status": "confirmed"},
    {"id": "1", "status": "weakened"},
    {"id": "2", "status": "insufficient"},
])


@pytest.mark.asyncio
async def test_verify_evidence_updates_status():
    context = await create_test_context()

    evidence_df = _make_evidence_df(3)
    text_units_df = _make_text_units_df()

    await context.output_table_provider.write_dataframe("evidence", evidence_df)
    await context.output_table_provider.write_dataframe("text_units", text_units_df)

    config = get_default_graphrag_config()
    config.evidence.enabled = True
    config.evidence.verification_enabled = True

    llm_settings = config.get_completion_model_config(
        config.evidence.verification_model_id
    )
    llm_settings.type = LLMProviderType.MockLLM
    llm_settings.mock_responses = [MOCK_VERIFY_RESPONSE]  # type: ignore

    await run_workflow(config, context)

    result = await context.output_table_provider.read_dataframe("evidence")

    assert len(result) == 3
    assert "verification_status" in result.columns
    assert "verification_method" in result.columns

    # Check that statuses were updated correctly
    statuses = dict(zip(result["subject_id"], result["verification_status"]))
    assert statuses["ENTITY_0"] == "confirmed"
    assert statuses["ENTITY_1"] == "weakened"
    assert statuses["ENTITY_2"] == "insufficient"

    # All verified via LLM
    assert (result["verification_method"] == "llm_verified").all()


@pytest.mark.asyncio
async def test_verify_evidence_skipped_when_disabled():
    context = await create_test_context()

    # Write evidence with unverified status
    evidence_df = _make_evidence_df(2)
    await context.output_table_provider.write_dataframe("evidence", evidence_df)

    config = get_default_graphrag_config()
    config.evidence.enabled = True
    config.evidence.verification_enabled = False  # disabled

    await run_workflow(config, context)

    # Evidence should be untouched (workflow was a no-op)
    result = await context.output_table_provider.read_dataframe("evidence")
    assert (result["verification_status"] == "unverified").all()


@pytest.mark.asyncio
async def test_verify_evidence_skipped_when_evidence_disabled():
    context = await create_test_context()
    config = get_default_graphrag_config()
    config.evidence.enabled = False  # top-level disabled

    result = await run_workflow(config, context)
    assert result.result is None
