# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""LLM-based entity merge confirmation extractor."""

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from graphrag_llm.completion import LLMCompletion

logger = logging.getLogger(__name__)


@dataclass
class MergeDecision:
    """Result of an LLM merge confirmation call."""

    merge: bool
    canonical: str | None = None


class EntityMergeExtractor:
    """Calls the LLM to confirm whether a group of entities should be merged."""

    def __init__(self, model: "LLMCompletion", prompt: str):
        self._model = model
        self._prompt = prompt

    async def __call__(
        self, candidates: list[dict[str, str]]
    ) -> MergeDecision:
        """Ask the LLM whether all candidates refer to the same real-world entity.

        Parameters
        ----------
        candidates : list[dict[str, str]]
            Each dict has keys ``title``, ``type``, ``description``.

        Returns
        -------
        MergeDecision
            ``merge=True`` with the chosen canonical title, or ``merge=False``.
        """
        formatted = self._prompt.format(
            n=len(candidates),
            candidates_json=json.dumps(candidates, ensure_ascii=False, indent=2),
        )

        try:
            response = await self._model.completion_async(messages=formatted)
            content = response.content.strip()
            # Strip potential markdown code fences
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            result = json.loads(content)
            if result.get("merge") is True:
                canonical = result.get("canonical")
                if canonical:
                    return MergeDecision(merge=True, canonical=str(canonical))
        except Exception:  # noqa: BLE001
            logger.debug(
                "Entity merge extractor: failed to parse LLM response for candidates %s",
                [c["title"] for c in candidates],
            )

        return MergeDecision(merge=False)
