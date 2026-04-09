# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Entity disambiguation prompt for LLM context-window-based entity resolution."""

ENTITY_DISAMBIGUATION_PROMPT = """\
You are an expert at entity resolution for knowledge graphs.

Below is a list of entities extracted from text documents. Your task is to identify \
groups of entities that refer to the same real-world entity. Consider name variations, \
abbreviations, honorifics (Herr/Mr/Dr/Prof), role or company suffixes, language \
variants, singular/plural forms, and contextual descriptions.

Entities:
{entities_json}

Return a JSON array of duplicate groups. Each group must have at least 2 members.
For each group, choose the most complete and formal title as the canonical name.
If no duplicates exist, return an empty JSON array: []

Format:
[
  {{"canonical": "<best complete title>", "aliases": ["<other title 1>", "<other title 2>"]}},
  ...
]

Rules:
- Only include groups where you are confident the entities refer to the same real-world thing.
- The canonical title must be one of the titles from the input list.
- Each title may appear in at most one group.
- Respond with valid JSON only — no text outside the JSON array.
"""
