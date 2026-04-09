# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Unit tests for EntityMergeExtractor (LLM wrapper)."""

from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock

import pytest

from graphrag.index.operations.resolve_entities.entity_merge_extractor import (
    EntityMergeExtractor,
    MergeDecision,
)
from graphrag.prompts.index.entity_resolution import ENTITY_RESOLUTION_PROMPT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class FakeLLMResponse:
    content: str


def _make_model(response_text: str) -> Any:
    """Return a mock LLMCompletion that returns response_text."""
    model = AsyncMock()
    model.completion_async = AsyncMock(
        return_value=FakeLLMResponse(content=response_text)
    )
    return model


_CANDIDATES = [
    {"title": "SEBASTIAN MUSTERMANN", "type": "PERSON", "description": "Sales agent at XYZ GmbH"},
    {"title": "HERR MUSTERMANN", "type": "PERSON", "description": "Agent at XYZ providing technical support"},
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEntityMergeExtractor:
    async def test_merge_true_returns_canonical(self):
        model = _make_model('{"merge": true, "canonical": "SEBASTIAN MUSTERMANN"}')
        extractor = EntityMergeExtractor(model=model, prompt=ENTITY_RESOLUTION_PROMPT)
        result = await extractor(_CANDIDATES)
        assert result.merge is True
        assert result.canonical == "SEBASTIAN MUSTERMANN"

    async def test_merge_false_returns_no_merge(self):
        model = _make_model('{"merge": false}')
        extractor = EntityMergeExtractor(model=model, prompt=ENTITY_RESOLUTION_PROMPT)
        result = await extractor(_CANDIDATES)
        assert result.merge is False
        assert result.canonical is None

    async def test_malformed_json_defaults_to_no_merge(self):
        """Any parse error must fall back to merge=False."""
        model = _make_model("Sorry, I cannot determine this.")
        extractor = EntityMergeExtractor(model=model, prompt=ENTITY_RESOLUTION_PROMPT)
        result = await extractor(_CANDIDATES)
        assert result.merge is False

    async def test_merge_true_missing_canonical_defaults_to_no_merge(self):
        """merge=true without a canonical title should fall back to merge=False."""
        model = _make_model('{"merge": true}')
        extractor = EntityMergeExtractor(model=model, prompt=ENTITY_RESOLUTION_PROMPT)
        result = await extractor(_CANDIDATES)
        assert result.merge is False

    async def test_markdown_fenced_json_parsed_correctly(self):
        """LLMs sometimes wrap JSON in ```json ... ``` fences."""
        model = _make_model('```json\n{"merge": true, "canonical": "SEBASTIAN MUSTERMANN"}\n```')
        extractor = EntityMergeExtractor(model=model, prompt=ENTITY_RESOLUTION_PROMPT)
        result = await extractor(_CANDIDATES)
        assert result.merge is True
        assert result.canonical == "SEBASTIAN MUSTERMANN"

    async def test_prompt_receives_candidate_count(self):
        """The formatted prompt must contain the candidate count."""
        captured: list[str] = []

        async def capture_call(messages: str):
            captured.append(messages)
            return FakeLLMResponse(content='{"merge": false}')

        model = AsyncMock()
        model.completion_async = capture_call
        extractor = EntityMergeExtractor(model=model, prompt=ENTITY_RESOLUTION_PROMPT)
        await extractor(_CANDIDATES)

        assert len(captured) == 1
        assert "2" in captured[0]  # n=2 candidates

    async def test_llm_called_exactly_once(self):
        model = _make_model('{"merge": false}')
        extractor = EntityMergeExtractor(model=model, prompt=ENTITY_RESOLUTION_PROMPT)
        await extractor(_CANDIDATES)
        model.completion_async.assert_awaited_once()

    async def test_empty_candidate_list_returns_no_merge(self):
        model = _make_model('{"merge": false}')
        extractor = EntityMergeExtractor(model=model, prompt=ENTITY_RESOLUTION_PROMPT)
        result = await extractor([])
        assert result.merge is False
