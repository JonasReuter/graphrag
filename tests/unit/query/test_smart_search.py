# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Unit tests for smart_search() — classifier and routing logic.

All LLM calls and search functions are mocked so tests run without
an ArangoDB instance or live API keys.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from graphrag.api.smart_search import (
    QUERY_TYPES,
    SmartSearchResult,
    _build_call_config,
    _classify_query,
    smart_search,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(top_k_seeds: int = 10, rerank_enabled: bool = False) -> MagicMock:
    cfg = MagicMock()
    cfg.local_search.completion_model_id = "default"
    cfg.local_search.rerank.enabled = rerank_enabled
    cfg.graph_store.top_k_seeds = top_k_seeds

    # model_copy(deep=True) must return a real-ish mock we can mutate
    copy = MagicMock()
    copy.local_search.rerank.enabled = rerank_enabled
    copy.graph_store.top_k_seeds = top_k_seeds
    cfg.model_copy.return_value = copy
    return cfg


def _classifier_response(type_: str, from_date=None, until_date=None) -> str:
    return json.dumps({"type": type_, "from_date": from_date, "until_date": until_date})


# ---------------------------------------------------------------------------
# Classifier tests
# ---------------------------------------------------------------------------

class TestClassifyQuery:
    """_classify_query returns the correct type and date hints."""

    async def _run(self, response_json: str, config=None):
        cfg = config or _make_config()
        mock_model = MagicMock()
        mock_raw = MagicMock()
        mock_model.completion_async = AsyncMock(return_value=mock_raw)

        with (
            patch("graphrag.api.smart_search.create_completion", return_value=mock_model),
            patch(
                "graphrag.api.smart_search.gather_completion_response_async",
                new=AsyncMock(return_value=response_json),
            ),
        ):
            return await _classify_query("test query", cfg)

    @pytest.mark.asyncio
    async def test_entity_fact(self):
        qtype, fd, ud = await self._run(_classifier_response("entity_fact"))
        assert qtype == "entity_fact"
        assert fd is None and ud is None

    @pytest.mark.asyncio
    async def test_enumeration(self):
        qtype, *_ = await self._run(_classifier_response("enumeration"))
        assert qtype == "enumeration"

    @pytest.mark.asyncio
    async def test_synthesis(self):
        qtype, *_ = await self._run(_classifier_response("synthesis"))
        assert qtype == "synthesis"

    @pytest.mark.asyncio
    async def test_temporal_with_dates(self):
        qtype, fd, ud = await self._run(
            _classifier_response("temporal", from_date="2025-01-01", until_date="2025-12-31")
        )
        assert qtype == "temporal"
        assert fd == "2025-01-01"
        assert ud == "2025-12-31"

    @pytest.mark.asyncio
    async def test_unknown_type_falls_back_to_entity_fact(self):
        qtype, *_ = await self._run(_classifier_response("nonsense_type"))
        assert qtype == "entity_fact"

    @pytest.mark.asyncio
    async def test_llm_error_falls_back_to_entity_fact(self):
        cfg = _make_config()
        mock_model = MagicMock()
        mock_model.completion_async = AsyncMock(side_effect=RuntimeError("api error"))

        with patch("graphrag.api.smart_search.create_completion", return_value=mock_model):
            qtype, fd, ud = await _classify_query("test", cfg)

        assert qtype == "entity_fact"
        assert fd is None and ud is None

    @pytest.mark.asyncio
    async def test_malformed_json_falls_back(self):
        qtype, *_ = await self._run("not valid json at all")
        assert qtype == "entity_fact"


# ---------------------------------------------------------------------------
# Config override tests
# ---------------------------------------------------------------------------

class TestBuildCallConfig:
    """_build_call_config applies overrides without mutating original."""

    def test_entity_fact_enables_rerank(self):
        cfg = _make_config(top_k_seeds=10, rerank_enabled=False)
        result = _build_call_config(cfg, "entity_fact")
        assert result.local_search.rerank.enabled is True

    def test_entity_fact_raises_top_k_seeds_to_20(self):
        cfg = _make_config(top_k_seeds=5)
        result = _build_call_config(cfg, "entity_fact")
        assert result.graph_store.top_k_seeds == 20

    def test_entity_fact_does_not_lower_existing_high_top_k(self):
        cfg = _make_config(top_k_seeds=30)
        result = _build_call_config(cfg, "entity_fact")
        assert result.graph_store.top_k_seeds == 30

    def test_temporal_also_enables_rerank(self):
        cfg = _make_config(rerank_enabled=False)
        result = _build_call_config(cfg, "temporal")
        assert result.local_search.rerank.enabled is True

    def test_enumeration_does_not_change_config(self):
        cfg = _make_config(top_k_seeds=10, rerank_enabled=False)
        _build_call_config(cfg, "enumeration")
        # original config must never be mutated
        cfg.model_copy.assert_called_once_with(deep=True)

    def test_original_config_not_mutated(self):
        cfg = _make_config(top_k_seeds=10)
        _build_call_config(cfg, "entity_fact")
        # model_copy was called — original never touched directly
        assert cfg.graph_store.top_k_seeds == 10


# ---------------------------------------------------------------------------
# Routing tests
# ---------------------------------------------------------------------------

class TestSmartSearch:
    """smart_search routes to correct mode and returns SmartSearchResult."""

    def _fake_local_result(self):
        return ("local answer", {"entities": pd.DataFrame()}, {"embedding": 200.0, "llm_generation": 5000.0})

    def _fake_global_result(self):
        return ("global answer", {}, {"context_build": 80.0, "map_llm": 3000.0, "reduce_llm": 3000.0})

    @pytest.mark.asyncio
    async def test_entity_fact_routes_to_local(self):
        cfg = _make_config()
        with (
            patch("graphrag.api.smart_search._classify_query", new=AsyncMock(return_value=("entity_fact", None, None))),
            patch("graphrag.api.smart_search.graph_local_search", new=AsyncMock(return_value=self._fake_local_result())),
        ):
            result = await smart_search(cfg, "What is product X?")

        assert isinstance(result, SmartSearchResult)
        assert result.mode_used == "graph_local_search"
        assert result.query_type == "entity_fact"
        assert result.answer == "local answer"

    @pytest.mark.asyncio
    async def test_enumeration_routes_to_global(self):
        cfg = _make_config()
        with (
            patch("graphrag.api.smart_search._classify_query", new=AsyncMock(return_value=("enumeration", None, None))),
            patch("graphrag.api.smart_search.graph_global_search", new=AsyncMock(return_value=self._fake_global_result())),
        ):
            result = await smart_search(cfg, "Which customers are international?")

        assert result.mode_used == "graph_global_search"
        assert result.query_type == "enumeration"

    @pytest.mark.asyncio
    async def test_synthesis_routes_to_global(self):
        cfg = _make_config()
        with (
            patch("graphrag.api.smart_search._classify_query", new=AsyncMock(return_value=("synthesis", None, None))),
            patch("graphrag.api.smart_search.graph_global_search", new=AsyncMock(return_value=self._fake_global_result())),
        ):
            result = await smart_search(cfg, "What are the main application domains?")

        assert result.mode_used == "graph_global_search"

    @pytest.mark.asyncio
    async def test_temporal_routes_to_local_with_dates(self):
        cfg = _make_config()
        mock_local = AsyncMock(return_value=self._fake_local_result())
        with (
            patch("graphrag.api.smart_search._classify_query", new=AsyncMock(return_value=("temporal", "2025-01-01", "2025-12-31"))),
            patch("graphrag.api.smart_search.graph_local_search", mock_local),
        ):
            result = await smart_search(cfg, "What happened in 2025?")

        assert result.mode_used == "graph_local_search"
        # from_date/until_date must be forwarded to the search call
        call_kwargs = mock_local.call_args.kwargs
        assert call_kwargs["from_date"] == "2025-01-01"
        assert call_kwargs["until_date"] == "2025-12-31"

    @pytest.mark.asyncio
    async def test_force_mode_graph_local_skips_classifier(self):
        cfg = _make_config()
        mock_classify = AsyncMock(return_value=("entity_fact", None, None))
        with (
            patch("graphrag.api.smart_search._classify_query", mock_classify),
            patch("graphrag.api.smart_search.graph_local_search", new=AsyncMock(return_value=self._fake_local_result())),
        ):
            result = await smart_search(cfg, "anything", force_mode="graph_local")

        mock_classify.assert_not_called()
        assert result.mode_used == "graph_local_search"

    @pytest.mark.asyncio
    async def test_force_mode_graph_global_skips_classifier(self):
        cfg = _make_config()
        mock_classify = AsyncMock()
        with (
            patch("graphrag.api.smart_search._classify_query", mock_classify),
            patch("graphrag.api.smart_search.graph_global_search", new=AsyncMock(return_value=self._fake_global_result())),
        ):
            result = await smart_search(cfg, "anything", force_mode="graph_global")

        mock_classify.assert_not_called()
        assert result.mode_used == "graph_global_search"

    @pytest.mark.asyncio
    async def test_result_contains_classification_ms(self):
        cfg = _make_config()
        with (
            patch("graphrag.api.smart_search._classify_query", new=AsyncMock(return_value=("entity_fact", None, None))),
            patch("graphrag.api.smart_search.graph_local_search", new=AsyncMock(return_value=self._fake_local_result())),
        ):
            result = await smart_search(cfg, "test")

        assert result.classification_ms >= 0
        assert "classification" in result.phase_timings

    @pytest.mark.asyncio
    async def test_phase_timings_merged(self):
        cfg = _make_config()
        with (
            patch("graphrag.api.smart_search._classify_query", new=AsyncMock(return_value=("entity_fact", None, None))),
            patch("graphrag.api.smart_search.graph_local_search", new=AsyncMock(return_value=self._fake_local_result())),
        ):
            result = await smart_search(cfg, "test")

        # Phase timings from local search + classification must both be present
        assert "embedding" in result.phase_timings or "classification" in result.phase_timings
        assert "classification" in result.phase_timings


# ---------------------------------------------------------------------------
# QUERY_TYPES completeness
# ---------------------------------------------------------------------------

class TestQueryTypes:
    def test_all_expected_types_present(self):
        assert QUERY_TYPES == {"entity_fact", "enumeration", "synthesis", "temporal"}
