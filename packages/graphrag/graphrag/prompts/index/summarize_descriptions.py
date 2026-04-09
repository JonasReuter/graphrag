# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A file containing prompts definition."""

SUMMARIZE_PROMPT = """
You are a helpful assistant responsible for consolidating a list of descriptions into a single accurate summary.
Given one or more entities, and a list of descriptions, all related to the same entity or group of entities.

CRITICAL RULES — strictly follow these:
1. Only include facts that are explicitly stated in the provided descriptions. Do NOT infer, extrapolate, interpret, or add any information that is not directly present in the source descriptions.
2. If a fact appears in multiple descriptions, mention it once.
3. If descriptions contradict each other, note the contradiction briefly instead of resolving it with invented detail.
4. When in doubt, omit rather than infer.
5. Write in third person and include the entity names for context.
6. Limit the final description to {max_length} words.

#######
-Data-
Entities: {entity_name}
Description List: {description_list}
#######
Output:
"""
