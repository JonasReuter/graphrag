# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Unit tests for EntityDisambiguationExtractor (LLM context-window resolver)."""

from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock

import pytest

from graphrag.index.operations.resolve_entities.entity_disambiguation_extractor import (
    DisambiguationGroup,
    EntityDisambiguationExtractor,
)
from graphrag.prompts.index.entity_disambiguation import ENTITY_DISAMBIGUATION_PROMPT


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


_ENTITIES = [
    {"title": "HEIKO DROGAN", "type": "person", "description": "Engineer at XYZ GmbH"},
    {"title": "HERR DROGAN", "type": "person", "description": "Contact person at XYZ"},
    {"title": "PATRICK TISMAR", "type": "person", "description": "Sales representative"},
    {"title": "PATRICK", "type": "person", "description": "Sales contact"},
    {"title": "SENSOR ABC", "type": "product", "description": "Force sensor"},
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEntityDisambiguationExtractor:
    async def test_returns_groups_for_valid_response(self):
        response = '[{"canonical": "HEIKO DROGAN", "aliases": ["HERR DROGAN"]}]'
        extractor = EntityDisambiguationExtractor(
            model=_make_model(response), prompt=ENTITY_DISAMBIGUATION_PROMPT
        )
        groups = await extractor(_ENTITIES)
        assert len(groups) == 1
        assert groups[0].canonical == "HEIKO DROGAN"
        assert groups[0].aliases == ["HERR DROGAN"]

    async def test_returns_multiple_groups(self):
        response = (
            '[{"canonical": "HEIKO DROGAN", "aliases": ["HERR DROGAN"]},'
            ' {"canonical": "PATRICK TISMAR", "aliases": ["PATRICK"]}]'
        )
        extractor = EntityDisambiguationExtractor(
            model=_make_model(response), prompt=ENTITY_DISAMBIGUATION_PROMPT
        )
        groups = await extractor(_ENTITIES)
        assert len(groups) == 2
        canonicals = {g.canonical for g in groups}
        assert canonicals == {"HEIKO DROGAN", "PATRICK TISMAR"}

    async def test_empty_array_returns_no_groups(self):
        extractor = EntityDisambiguationExtractor(
            model=_make_model("[]"), prompt=ENTITY_DISAMBIGUATION_PROMPT
        )
        groups = await extractor(_ENTITIES)
        assert groups == []

    async def test_malformed_json_returns_no_groups(self):
        extractor = EntityDisambiguationExtractor(
            model=_make_model("Sorry, I cannot determine this."),
            prompt=ENTITY_DISAMBIGUATION_PROMPT,
        )
        groups = await extractor(_ENTITIES)
        assert groups == []

    async def test_markdown_fenced_json_parsed_correctly(self):
        response = '```json\n[{"canonical": "HEIKO DROGAN", "aliases": ["HERR DROGAN"]}]\n```'
        extractor = EntityDisambiguationExtractor(
            model=_make_model(response), prompt=ENTITY_DISAMBIGUATION_PROMPT
        )
        groups = await extractor(_ENTITIES)
        assert len(groups) == 1
        assert groups[0].canonical == "HEIKO DROGAN"

    async def test_group_missing_canonical_is_skipped(self):
        """Items without a canonical title must be silently dropped."""
        response = '[{"aliases": ["HERR DROGAN"]}]'
        extractor = EntityDisambiguationExtractor(
            model=_make_model(response), prompt=ENTITY_DISAMBIGUATION_PROMPT
        )
        groups = await extractor(_ENTITIES)
        assert groups == []

    async def test_group_missing_aliases_is_skipped(self):
        """Items with no aliases (empty or missing) must be silently dropped."""
        response = '[{"canonical": "HEIKO DROGAN", "aliases": []}]'
        extractor = EntityDisambiguationExtractor(
            model=_make_model(response), prompt=ENTITY_DISAMBIGUATION_PROMPT
        )
        groups = await extractor(_ENTITIES)
        assert groups == []

    async def test_empty_entity_list_returns_no_groups(self):
        extractor = EntityDisambiguationExtractor(
            model=_make_model("[]"), prompt=ENTITY_DISAMBIGUATION_PROMPT
        )
        groups = await extractor([])
        assert groups == []

    async def test_llm_called_exactly_once(self):
        model = _make_model("[]")
        extractor = EntityDisambiguationExtractor(model=model, prompt=ENTITY_DISAMBIGUATION_PROMPT)
        await extractor(_ENTITIES)
        model.completion_async.assert_awaited_once()

    async def test_prompt_contains_entity_titles(self):
        """The formatted prompt must include entity titles from the input."""
        captured: list[str] = []

        async def capture_call(messages: str):
            captured.append(messages)
            return FakeLLMResponse(content="[]")

        model = AsyncMock()
        model.completion_async = capture_call
        extractor = EntityDisambiguationExtractor(model=model, prompt=ENTITY_DISAMBIGUATION_PROMPT)
        await extractor(_ENTITIES)

        assert len(captured) == 1
        assert "HEIKO DROGAN" in captured[0]
        assert "PATRICK TISMAR" in captured[0]

    async def test_returns_dataclass_instances(self):
        response = '[{"canonical": "HEIKO DROGAN", "aliases": ["HERR DROGAN"]}]'
        extractor = EntityDisambiguationExtractor(
            model=_make_model(response), prompt=ENTITY_DISAMBIGUATION_PROMPT
        )
        groups = await extractor(_ENTITIES)
        assert all(isinstance(g, DisambiguationGroup) for g in groups)
