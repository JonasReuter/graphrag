# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Unit tests for context_only=True mode across search classes.

No LLM is instantiated — the context builder is mocked to return a fixed result.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from graphrag.query.context_builder.builders import ContextBuilderResult
from graphrag.query.structured_search.base import SearchResult
from graphrag.query.structured_search.basic_search.search import BasicSearch
from graphrag.query.structured_search.global_search.search import GlobalSearchResult
from graphrag.query.structured_search.local_search.search import LocalSearch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CONTEXT_CHUNKS = "-----Entities-----\nid|entity\ne1|Alice\n"
_CONTEXT_RECORDS = {"entities": MagicMock()}

_FIXED_CONTEXT = ContextBuilderResult(
    context_chunks=_CONTEXT_CHUNKS,
    context_records=_CONTEXT_RECORDS,
)


def _make_local_search() -> LocalSearch:
    model = MagicMock()
    context_builder = MagicMock()
    context_builder.build_context.return_value = _FIXED_CONTEXT
    return LocalSearch(
        model=model,
        context_builder=context_builder,
        context_builder_params={},
    )


def _make_basic_search() -> BasicSearch:
    model = MagicMock()
    context_builder = MagicMock()
    context_builder.build_context.return_value = _FIXED_CONTEXT
    return BasicSearch(
        model=model,
        context_builder=context_builder,
        context_builder_params={},
    )


# ---------------------------------------------------------------------------
# Tests: LocalSearch context_only
# ---------------------------------------------------------------------------

class TestLocalSearchContextOnly:
    def test_context_only_returns_empty_response(self):
        engine = _make_local_search()
        result: SearchResult = asyncio.get_event_loop().run_until_complete(
            engine.search("what is Alice?", context_only=True)
        )
        assert result.response == ""

    def test_context_only_preserves_context_data(self):
        engine = _make_local_search()
        result: SearchResult = asyncio.get_event_loop().run_until_complete(
            engine.search("what is Alice?", context_only=True)
        )
        assert result.context_data is _CONTEXT_RECORDS
        assert result.context_text == _CONTEXT_CHUNKS

    def test_context_only_skips_llm_call(self):
        engine = _make_local_search()
        asyncio.get_event_loop().run_until_complete(
            engine.search("what is Alice?", context_only=True)
        )
        engine.model.completion_async.assert_not_called()

    def test_context_only_false_calls_llm(self):
        """When context_only=False (default), the LLM is called."""
        engine = _make_local_search()
        # mock completion_async to avoid real network call
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = "answer"

        async def _stream():
            yield chunk

        engine.model.completion_async = AsyncMock(return_value=_stream())

        asyncio.get_event_loop().run_until_complete(
            engine.search("what is Alice?", context_only=False)
        )
        engine.model.completion_async.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: BasicSearch context_only
# ---------------------------------------------------------------------------

class TestBasicSearchContextOnly:
    def test_context_only_returns_empty_response(self):
        engine = _make_basic_search()
        result: SearchResult = asyncio.get_event_loop().run_until_complete(
            engine.search("what is Alice?", context_only=True)
        )
        assert result.response == ""

    def test_context_only_preserves_context_data(self):
        engine = _make_basic_search()
        result: SearchResult = asyncio.get_event_loop().run_until_complete(
            engine.search("what is Alice?", context_only=True)
        )
        assert result.context_data is _CONTEXT_RECORDS
        assert result.context_text == _CONTEXT_CHUNKS

    def test_context_only_skips_llm_call(self):
        engine = _make_basic_search()
        asyncio.get_event_loop().run_until_complete(
            engine.search("what is Alice?", context_only=True)
        )
        engine.model.completion_async.assert_not_called()
