# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""LLM-based entity disambiguation extractor for context-window resolution."""

import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from graphrag_llm.completion import LLMCompletion

logger = logging.getLogger(__name__)


@dataclass
class DisambiguationGroup:
    """A group of entity titles that refer to the same real-world entity."""

    canonical: str
    aliases: list[str] = field(default_factory=list)


class EntityDisambiguationExtractor:
    """Sends a window of entities to the LLM and returns duplicate groups."""

    def __init__(self, model: "LLMCompletion", prompt: str):
        self._model = model
        self._prompt = prompt

    async def __call__(
        self, entities: list[dict[str, str]]
    ) -> list[DisambiguationGroup]:
        """Ask the LLM to identify duplicate entity groups in the given list.

        Parameters
        ----------
        entities : list[dict[str, str]]
            Each dict has keys ``title``, ``type``, ``description``.

        Returns
        -------
        list[DisambiguationGroup]
            Groups of 2+ entities that refer to the same real-world entity.
        """
        formatted = self._prompt.format(
            entities_json=json.dumps(entities, ensure_ascii=False, indent=2)
        )

        try:
            response = await self._model.completion_async(messages=formatted)
            content = response.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            raw = json.loads(content)
            groups: list[DisambiguationGroup] = []
            for item in raw:
                canonical = item.get("canonical", "")
                aliases = item.get("aliases", [])
                if canonical and aliases:
                    groups.append(DisambiguationGroup(canonical=canonical, aliases=aliases))
            return groups
        except Exception:  # noqa: BLE001
            logger.debug(
                "EntityDisambiguationExtractor: failed to parse LLM response for %d entities",
                len(entities),
            )
            return []
