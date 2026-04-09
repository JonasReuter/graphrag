# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Unit tests for resolve_claim_subjects operation."""

from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest
from graphrag.index.operations.resolve_claim_subjects.resolve_claim_subjects import (
    _llm_confirm,
    resolve_claim_subjects,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_covariates(**overrides) -> pd.DataFrame:
    base = {
        "id": ["c1", "c2"],
        "subject_id": ["JOCHEN KREß", "ME-MESSYSTEME GMBH"],
        "object_id": ["K6D40", None],
        "type": ["SUPPORT_CASE", "TECHNICAL_CONSULTATION"],
        "status": ["OPEN", "CONFIRMED"],
        "description": ["desc1", "desc2"],
        "source_text": ["src1", "src2"],
        "start_date": [None, None],
        "end_date": [None, None],
        "covariate_type": ["claim", "claim"],
        "human_readable_id": [0, 1],
        "text_unit_id": ["t1", "t2"],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def _make_search_result(title: str, score: float):
    result = MagicMock()
    result.score = score
    result.document.data = {"title": title}
    return result


def _make_embedding_model(vectors=None):
    model = MagicMock()
    vectors = vectors or [[0.1] * 4, [0.2] * 4, [0.3] * 4]
    response = MagicMock()
    response.embeddings = [MagicMock(tolist=lambda v=v: v) for v in vectors]
    model.embedding_async = AsyncMock(return_value=response)
    return model


def _make_vector_store(results_by_query=None, count=10):
    store = MagicMock()
    store.count.return_value = count
    if results_by_query is None:
        store.similarity_search_by_vector.return_value = []
    else:
        store.similarity_search_by_vector.side_effect = results_by_query
    return store


def _make_completion_model(match: bool):
    model = MagicMock()
    response = MagicMock()
    response.content = f'{{"match": {"true" if match else "false"}}}'
    model.completion_async = AsyncMock(return_value=response)
    return model


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_high_score_auto_resolved():
    """Score above high_threshold → replace without LLM call."""
    df = _make_covariates()
    embedding_model = _make_embedding_model([[0.1] * 4, [0.2] * 4, [0.3] * 4])

    # unique ids sorted alphabetically: "JOCHEN KREß", "K6D40", "ME-MESSYSTEME GMBH"
    store = _make_vector_store(results_by_query=[
        [_make_search_result("JOCHEN KRESS", 0.97)],        # JOCHEN KREß → high → auto
        [_make_search_result("K6D40", 1.0)],                # K6D40 → itself, skipped
        [_make_search_result("ME-MESSSYSTEME GMBH", 0.96)], # ME-MESSYSTEME → high → auto
    ])
    completion_model = _make_completion_model(match=True)

    result_df, resolution_map = await resolve_claim_subjects(
        covariates_df=df,
        embedding_model=embedding_model,
        completion_model=completion_model,
        vector_store=store,
        prompt="irrelevant",
        high_threshold=0.95,
        mid_threshold=0.80,
        top_k=3,
    )

    assert result_df["subject_id"].tolist() == ["JOCHEN KRESS", "ME-MESSSYSTEME GMBH"]
    assert resolution_map["JOCHEN KREß"] == "JOCHEN KRESS"
    assert resolution_map["ME-MESSYSTEME GMBH"] == "ME-MESSSYSTEME GMBH"
    completion_model.completion_async.assert_not_called()


@pytest.mark.asyncio
async def test_mid_score_llm_confirmed():
    """Score in mid range + LLM confirms → replace."""
    df = _make_covariates(
        subject_id=["HERR DROGAN", "HERR DROGAN"],
        object_id=[None, None],
    )
    embedding_model = _make_embedding_model([[0.1] * 4])
    store = _make_vector_store(results_by_query=[
        [_make_search_result("HEIKO DROGAN", 0.88)],
    ])
    completion_model = _make_completion_model(match=True)

    result_df, _resolution_map = await resolve_claim_subjects(
        covariates_df=df,
        embedding_model=embedding_model,
        completion_model=completion_model,
        vector_store=store,
        prompt="{subject_id} {candidate_title} {score}",
        high_threshold=0.95,
        mid_threshold=0.80,
        top_k=3,
    )

    assert all(result_df["subject_id"] == "HEIKO DROGAN")
    completion_model.completion_async.assert_awaited_once()


@pytest.mark.asyncio
async def test_mid_score_llm_rejected():
    """Score in mid range + LLM rejects → leave unchanged."""
    df = _make_covariates(subject_id=["K6D40", "K6D40"], object_id=[None, None])
    embedding_model = _make_embedding_model([[0.1] * 4])
    store = _make_vector_store(results_by_query=[
        [_make_search_result("K6D80", 0.85)],
    ])
    completion_model = _make_completion_model(match=False)

    result_df, resolution_map = await resolve_claim_subjects(
        covariates_df=df,
        embedding_model=embedding_model,
        completion_model=completion_model,
        vector_store=store,
        prompt="{subject_id} {candidate_title} {score}",
        high_threshold=0.95,
        mid_threshold=0.80,
        top_k=3,
    )

    assert all(result_df["subject_id"] == "K6D40")
    assert "K6D40" not in resolution_map


@pytest.mark.asyncio
async def test_low_score_unchanged():
    """Score below mid_threshold → leave unchanged, no LLM call."""
    df = _make_covariates(subject_id=["PATIENT/DROGAN", "PATIENT/DROGAN"], object_id=[None, None])
    embedding_model = _make_embedding_model([[0.1] * 4])
    store = _make_vector_store(results_by_query=[
        [_make_search_result("HERR DROGAN", 0.60)],
    ])
    completion_model = _make_completion_model(match=True)

    result_df, _resolution_map = await resolve_claim_subjects(
        covariates_df=df,
        embedding_model=embedding_model,
        completion_model=completion_model,
        vector_store=store,
        prompt="irrelevant",
        high_threshold=0.95,
        mid_threshold=0.80,
        top_k=3,
    )

    assert all(result_df["subject_id"] == "PATIENT/DROGAN")
    completion_model.completion_async.assert_not_called()


@pytest.mark.asyncio
async def test_empty_index_returns_unchanged():
    """Empty vector store → skip resolution, return original df."""
    df = _make_covariates()
    embedding_model = _make_embedding_model()
    store = _make_vector_store(count=0)
    completion_model = _make_completion_model(match=True)

    result_df, resolution_map = await resolve_claim_subjects(
        covariates_df=df,
        embedding_model=embedding_model,
        completion_model=completion_model,
        vector_store=store,
        prompt="irrelevant",
        high_threshold=0.95,
        mid_threshold=0.80,
        top_k=3,
    )

    assert result_df.equals(df)
    assert resolution_map == {}
    embedding_model.embedding_async.assert_not_called()


@pytest.mark.asyncio
async def test_object_id_also_resolved():
    """object_id column is resolved in parallel with subject_id."""
    df = _make_covariates(
        subject_id=["TILL LINDNER", "TILL LINDNER"],
        object_id=["K6D 40", "K6D 40"],  # typo with space
    )
    # unique sorted ids: K6D 40, TILL LINDNER
    embedding_model = _make_embedding_model([[0.1] * 4, [0.2] * 4])
    store = _make_vector_store(results_by_query=[
        [_make_search_result("K6D40", 0.97)],       # K6D 40 → auto
        [_make_search_result("TILL LINDNER", 1.0)], # identical → skipped
    ])
    completion_model = _make_completion_model(match=True)

    result_df, _resolution_map = await resolve_claim_subjects(
        covariates_df=df,
        embedding_model=embedding_model,
        completion_model=completion_model,
        vector_store=store,
        prompt="irrelevant",
        high_threshold=0.95,
        mid_threshold=0.80,
        top_k=3,
    )

    assert all(result_df["object_id"] == "K6D40")


@pytest.mark.asyncio
async def test_deduplication_embeds_each_id_once():
    """Identical subject_id values are embedded only once."""
    df = pd.DataFrame({
        "id": ["c1", "c2", "c3"],
        "subject_id": ["SAME", "SAME", "SAME"],
        "object_id": [None, None, None],
        "type": ["T", "T", "T"],
        "status": ["OPEN", "OPEN", "OPEN"],
        "description": ["d", "d", "d"],
        "source_text": ["s", "s", "s"],
        "start_date": [None, None, None],
        "end_date": [None, None, None],
        "covariate_type": ["claim", "claim", "claim"],
        "human_readable_id": [0, 1, 2],
        "text_unit_id": ["t1", "t2", "t3"],
    })
    embedding_model = _make_embedding_model([[0.1] * 4])
    store = _make_vector_store(results_by_query=[
        [_make_search_result("CANONICAL", 0.97)],
    ])
    completion_model = _make_completion_model(match=True)

    await resolve_claim_subjects(
        covariates_df=df,
        embedding_model=embedding_model,
        completion_model=completion_model,
        vector_store=store,
        prompt="irrelevant",
        high_threshold=0.95,
        mid_threshold=0.80,
        top_k=3,
    )

    # embedding_async called once for the single unique id
    embedding_model.embedding_async.assert_awaited_once()


@pytest.mark.asyncio
async def test_llm_confirm_malformed_json_returns_false():
    """_llm_confirm returns False when LLM response is not valid JSON."""
    model = MagicMock()
    response = MagicMock()
    response.content = "I think yes"
    model.completion_async = AsyncMock(return_value=response)

    result = await _llm_confirm("A", "B", 0.85, model, "{subject_id} {candidate_title} {score}")
    assert result is False
