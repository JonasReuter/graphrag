# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

from unittest.mock import AsyncMock, MagicMock, patch

from graphrag.data_model.schemas import COVARIATES_FINAL_COLUMNS
from graphrag.index.operations.extract_covariates.claim_extractor import (
    ClaimExtractor,
    _normalize,
)
from graphrag.index.workflows.extract_covariates import (
    run_workflow,
)
from graphrag_llm.config import LLMProviderType
from pandas.testing import assert_series_equal

from tests.unit.config.utils import get_default_graphrag_config

from .util import (
    create_test_context,
    load_test_table,
)


def test_normalize():
    """_normalize uppercases and replaces ß with SS."""
    assert _normalize("Customer A") == "CUSTOMER A"
    assert _normalize("ME-Meßsysteme GmbH") == "ME-MESSSYSTEME GMBH"
    assert _normalize("ME-MESSSYSTEME GMBH") == "ME-MESSSYSTEME GMBH"
    assert _normalize("  spaces  ") == "SPACES"
    assert _normalize(None) == ""
    assert _normalize("") == ""


def test_clean_claim_normalizes_ss_variant():
    """_clean_claim resolves ß-variant names to their canonical entity titles."""
    resolved = {
        "CUSTOMER A": "CUSTOMER A",       # canonical
        "ME-MESSSYSTEME GMBH": "ME-MESSSYSTEME GMBH",  # canonical
    }
    claim = {
        "subject_id": "CUSTOMER A",           # ß-variant produced by LLM
        "object_id": "ME-MEßSYSTEME GMBH",     # ß-variant produced by LLM
    }
    # ClaimExtractor needs a model — use None and call _clean_claim directly
    extractor = ClaimExtractor.__new__(ClaimExtractor)
    result = extractor._clean_claim(claim, "d0", resolved)  # noqa: SLF001
    assert result["subject_id"] == "CUSTOMER A"
    assert result["object_id"] == "ME-MESSSYSTEME GMBH"


def test_clean_claim_unknown_entity_unchanged():
    """_clean_claim leaves names unchanged when not in the resolved map."""
    resolved = {"CUSTOMER A": "CUSTOMER A"}
    claim = {"subject_id": "UNKNOWN ENTITY", "object_id": "NONE"}
    extractor = ClaimExtractor.__new__(ClaimExtractor)
    result = extractor._clean_claim(claim, "d0", resolved)  # noqa: SLF001
    assert result["subject_id"] == "UNKNOWN ENTITY"
    assert result["object_id"] == "NONE"


MOCK_LLM_RESPONSES = [
    """
(COMPANY A<|>GOVERNMENT AGENCY B<|>ANTI-COMPETITIVE PRACTICES<|>TRUE<|>2022-01-10T00:00:00<|>2022-01-10T00:00:00<|>Company A was found to engage in anti-competitive practices because it was fined for bid rigging in multiple public tenders published by Government Agency B according to an article published on 2022/01/10<|>According to an article published on 2022/01/10, Company A was fined for bid rigging while participating in multiple public tenders published by Government Agency B.)
    """.strip()
]


async def test_extract_covariates():
    input = load_test_table("text_units")

    context = await create_test_context(
        storage=["text_units"],
    )

    config = get_default_graphrag_config()
    config.extract_claims.enabled = True
    config.extract_claims.description = "description"
    llm_settings = config.get_completion_model_config(
        config.extract_claims.completion_model_id
    )
    llm_settings.type = LLMProviderType.MockLLM
    llm_settings.mock_responses = MOCK_LLM_RESPONSES  # type: ignore

    await run_workflow(config, context)

    actual = await context.output_table_provider.read_dataframe("covariates")

    for column in COVARIATES_FINAL_COLUMNS:
        assert column in actual.columns

    # our mock only returns one covariate per text unit, so that's a 1:1 mapping versus the LLM-extracted content in the test data
    assert len(actual) == len(input)

    # assert all of the columns that covariates copied from the input
    assert_series_equal(actual["text_unit_id"], input["id"], check_names=False)

    # make sure the human ids are incrementing
    assert actual["human_readable_id"][0] == 0
    assert actual["human_readable_id"][1] == 1

    # check that the mock data is parsed and inserted into the correct columns
    assert actual["covariate_type"][0] == "claim"
    assert actual["subject_id"][0] == "COMPANY A"
    assert actual["object_id"][0] == "GOVERNMENT AGENCY B"
    assert actual["type"][0] == "ANTI-COMPETITIVE PRACTICES"
    assert actual["status"][0] == "TRUE"
    assert actual["start_date"][0] == "2022-01-10T00:00:00"
    assert actual["end_date"][0] == "2022-01-10T00:00:00"
    assert (
        actual["description"][0]
        == "Company A was found to engage in anti-competitive practices because it was fined for bid rigging in multiple public tenders published by Government Agency B according to an article published on 2022/01/10"
    )
    assert (
        actual["source_text"][0]
        == "According to an article published on 2022/01/10, Company A was fined for bid rigging while participating in multiple public tenders published by Government Agency B."
    )


async def test_extract_covariates_with_inline_resolution():
    """When resolve_claim_subjects is enabled, run_workflow resolves subject IDs inline."""
    context = await create_test_context(storage=["text_units"])

    config = get_default_graphrag_config()
    config.extract_claims.enabled = True
    config.extract_claims.description = "description"
    config.resolve_claim_subjects.enabled = True

    llm_settings = config.get_completion_model_config(
        config.extract_claims.completion_model_id
    )
    llm_settings.type = LLMProviderType.MockLLM
    llm_settings.mock_responses = MOCK_LLM_RESPONSES  # type: ignore

    # Mock vector store: count > 0, returns high-score match for "COMPANY A"
    mock_result = MagicMock()
    mock_result.score = 0.97
    mock_result.document.data = {"title": "COMPANY A CANONICAL"}

    mock_store = MagicMock()
    mock_store.count.return_value = 5
    mock_store.similarity_search_by_vector.return_value = [mock_result]
    mock_store.connect.return_value = None

    # Mock embedding model: returns a fixed vector
    mock_emb_response = MagicMock()
    mock_emb_response.embeddings = [MagicMock(tolist=lambda: [0.1] * 4)]
    mock_emb_model = MagicMock()
    mock_emb_model.embedding_async = AsyncMock(return_value=mock_emb_response)

    with (
        patch(
            "graphrag.index.workflows.extract_covariates.create_vector_store",
            return_value=mock_store,
        ),
        patch(
            "graphrag.index.workflows.extract_covariates.create_embedding",
            return_value=mock_emb_model,
        ),
    ):
        await run_workflow(config, context)

    actual = await context.output_table_provider.read_dataframe("covariates")

    # The mock LLM extract returns "COMPANY A"; the mock vector store maps it
    # to "COMPANY A CANONICAL" at score 0.97 (above high_threshold=0.95)
    assert actual["subject_id"][0] == "COMPANY A CANONICAL"
    mock_emb_model.embedding_async.assert_awaited()


async def test_extract_covariates_resolution_disabled_skips_embedding():
    """When resolve_claim_subjects is disabled, no embedding calls are made."""
    context = await create_test_context(storage=["text_units"])

    config = get_default_graphrag_config()
    config.extract_claims.enabled = True
    config.extract_claims.description = "description"
    config.resolve_claim_subjects.enabled = False  # explicitly off

    llm_settings = config.get_completion_model_config(
        config.extract_claims.completion_model_id
    )
    llm_settings.type = LLMProviderType.MockLLM
    llm_settings.mock_responses = MOCK_LLM_RESPONSES  # type: ignore

    mock_emb_model = MagicMock()
    mock_emb_model.embedding_async = AsyncMock()

    with patch(
        "graphrag.index.workflows.extract_covariates.create_embedding",
        return_value=mock_emb_model,
    ):
        await run_workflow(config, context)

    mock_emb_model.embedding_async.assert_not_awaited()
