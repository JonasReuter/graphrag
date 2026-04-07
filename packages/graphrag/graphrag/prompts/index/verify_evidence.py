# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Prompt definition for evidence verification."""

VERIFY_EVIDENCE_PROMPT = """
-Task-
You are verifying whether a piece of text actually supports a specific claim extracted from it.

-Source Text-
{source_text}

-Claims to Verify-
For each claim below, determine whether the source text supports it.
Respond with a JSON array where each element has "id" and "status".

Status values:
- "confirmed": The text explicitly and clearly supports this claim.
- "weakened": The text partially supports the claim, but it is incomplete, overstated, or missing important qualifiers.
- "refuted": The text directly contradicts this claim.
- "insufficient": The text does not contain enough information to verify this claim.

Claims:
{claims}

-Output Format-
Return ONLY a valid JSON array. No explanation, no markdown, no extra text.
Example:
[
  {{"id": "0", "status": "confirmed"}},
  {{"id": "1", "status": "weakened"}},
  {{"id": "2", "status": "insufficient"}}
]

Output:"""
