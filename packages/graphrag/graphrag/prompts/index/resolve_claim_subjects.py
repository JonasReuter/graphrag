# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Prompt for LLM confirmation of claim subject / entity matches."""

RESOLVE_CLAIM_SUBJECT_PROMPT = """\
You are an expert at entity resolution in a knowledge graph.

A claim was extracted from a document. Its subject field is:
  "{subject_id}"

The most similar entity in the knowledge graph is:
  "{candidate_title}"
  (cosine similarity: {score:.3f})

Do these refer to the same real-world entity? Consider:
- Name variants, abbreviations, honorifics
- Company names with/without legal suffix (GmbH, AG, etc.)
- Person names in different formats (first-last vs last-first)
- Partial names (e.g. "Herr Agent Alpha" referring to "Agent Alpha")

When in doubt, prefer {{"match": false}} to avoid incorrect merging.

Respond with valid JSON only — no explanation, no markdown:
{{"match": true}}   if they refer to the same entity
{{"match": false}}  if they do NOT
"""
