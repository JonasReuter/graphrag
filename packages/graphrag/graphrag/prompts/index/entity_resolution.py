# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Entity resolution prompt for knowledge graph deduplication."""

ENTITY_RESOLUTION_PROMPT = """\
You are an expert at entity resolution for knowledge graphs.

Below are {n} candidate entities that may refer to the same real-world entity.
Analyze whether they ALL refer to the same entity, considering name variations,
abbreviations, honorifics (Herr/Mr/Dr), role or company suffixes, and contextual descriptions.

Candidates:
{candidates_json}

Respond with valid JSON only — no text outside the JSON object:
- If ALL candidates refer to the same real-world entity:
  {{"merge": true, "canonical": "<the most complete and formal title from the list>"}}
- If they do NOT all refer to the same entity:
  {{"merge": false}}

When in doubt, prefer {{"merge": false}} to avoid incorrect merges.
"""
