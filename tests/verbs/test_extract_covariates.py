# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from graphrag.data_model.schemas import COVARIATES_FINAL_COLUMNS
from graphrag.index.operations.extract_covariates.claim_extractor import (
    ClaimExtractor,
    _normalize,
)
from graphrag.index.workflows.extract_covariates import (
    _entity_specs_per_chunk,
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
    """_normalize handles Unicode diacritics across languages."""
    # German ß → SS (via casefold)
    assert _normalize("Customer A") == "CUSTOMER A"
    assert _normalize("ME-Meßsysteme GmbH") == "ME-MESSSYSTEME GMBH"
    assert _normalize("ME-MESSSYSTEME GMBH") == "ME-MESSSYSTEME GMBH"
    # German umlauts
    assert _normalize("Zürich Köln") == "ZURICH KOLN"
    # French accents
    assert _normalize("naïve café") == "NAIVE CAFE"
    # Spanish tilde
    assert _normalize("Ñoño López") == "NONO LOPEZ"
    # Whitespace handling
    assert _normalize("  spaces  ") == "SPACES"
    # Edge cases
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


# ---------------------------------------------------------------------------
# _entity_specs_per_chunk — embedding pre-search helper
# ---------------------------------------------------------------------------

def _make_search_result(title: str):
    r = MagicMock()
    r.document.data = {"title": title}
    return r


def _make_emb_model(n_per_call: int = 1):
    """Return a mock embedding model that yields fixed vectors per batch."""
    def _embed(input):
        resp = MagicMock()
        resp.embeddings = [MagicMock(tolist=lambda v=[0.1]*4: v) for _ in input]
        return resp
    m = MagicMock()
    m.embedding_async = AsyncMock(side_effect=_embed)
    return m


@pytest.mark.asyncio
async def test_entity_specs_returns_fallback_when_index_empty():
    store = MagicMock()
    store.count.return_value = 0
    model = _make_emb_model()

    result = await _entity_specs_per_chunk(
        texts=["chunk 1", "chunk 2"],
        embedding_model=model,
        vector_store=store,
        top_k=5,
        fallback=["person", "org"],
    )

    assert result == [["person", "org"], ["person", "org"]]
    model.embedding_async.assert_not_awaited()


@pytest.mark.asyncio
async def test_entity_specs_returns_top_k_titles():
    store = MagicMock()
    store.count.return_value = 10
    store.similarity_search_by_vector.return_value = [
        _make_search_result("CUSTOMER A"),
        _make_search_result("CUSTOMER B"),
    ]
    model = _make_emb_model()

    result = await _entity_specs_per_chunk(
        texts=["chunk about Customer A"],
        embedding_model=model,
        vector_store=store,
        top_k=5,
        fallback=["person"],
    )

    assert result == [["CUSTOMER A", "CUSTOMER B"]]


@pytest.mark.asyncio
async def test_entity_specs_filters_stale_titles_via_known_titles():
    store = MagicMock()
    store.count.return_value = 10
    # Index has pre-resolution title "CUSTOMER B" but canonical is "HERR LINDNER"
    store.similarity_search_by_vector.return_value = [
        _make_search_result("CUSTOMER B"),   # stale — not in known_titles
        _make_search_result("HERR LINDNER"),   # canonical ✓
    ]
    model = _make_emb_model()
    known = {"HERR LINDNER", "CUSTOMER A"}

    result = await _entity_specs_per_chunk(
        texts=["chunk"],
        embedding_model=model,
        vector_store=store,
        top_k=5,
        fallback=["person"],
        known_titles=known,
    )

    assert result == [["HERR LINDNER"]]


@pytest.mark.asyncio
async def test_entity_specs_falls_back_when_all_filtered():
    store = MagicMock()
    store.count.return_value = 5
    store.similarity_search_by_vector.return_value = [
        _make_search_result("STALE ENTITY"),
    ]
    model = _make_emb_model()

    result = await _entity_specs_per_chunk(
        texts=["chunk"],
        embedding_model=model,
        vector_store=store,
        top_k=3,
        fallback=["person", "org"],
        known_titles={"CUSTOMER A"},  # STALE ENTITY not in known
    )

    assert result == [["person", "org"]]


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


async def test_extract_covariates_entity_candidates_injected():
    """When entity_candidates_k > 0, pre-search injects entity titles per chunk."""
    context = await create_test_context(storage=["text_units"])

    config = get_default_graphrag_config()
    config.extract_claims.enabled = True
    config.extract_claims.description = "description"
    config.extract_claims.entity_candidates_k = 5

    llm_settings = config.get_completion_model_config(
        config.extract_claims.completion_model_id
    )
    llm_settings.type = LLMProviderType.MockLLM
    llm_settings.mock_responses = MOCK_LLM_RESPONSES  # type: ignore

    mock_result = MagicMock()
    mock_result.document.data = {"title": "COMPANY A"}
    mock_store = MagicMock()
    mock_store.count.return_value = 3
    mock_store.similarity_search_by_vector.return_value = [mock_result]
    mock_store.connect.return_value = None

    def _fake_embed(input):
        resp = MagicMock()
        resp.embeddings = [MagicMock(tolist=lambda v=[0.1] * 4: v) for _ in input]
        return resp

    mock_emb_model = MagicMock()
    mock_emb_model.embedding_async = AsyncMock(side_effect=_fake_embed)

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

    # Embedding was called for the pre-search
    mock_emb_model.embedding_async.assert_awaited()
    # Vector store was searched
    mock_store.similarity_search_by_vector.assert_called()


async def test_extract_covariates_entity_candidates_disabled_skips_embedding():
    """When entity_candidates_k == 0, no embedding pre-search is performed."""
    context = await create_test_context(storage=["text_units"])

    config = get_default_graphrag_config()
    config.extract_claims.enabled = True
    config.extract_claims.description = "description"
    config.extract_claims.entity_candidates_k = 0  # disabled

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
