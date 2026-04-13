# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""smart_search — automatic query routing across ArangoDB-native graph search modes.

Classifies the incoming query with a small LLM call, then dispatches to the
most suitable search mode with optimised per-call configuration overrides.
Existing search functions are unchanged (fully backward-compatible).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pandas as pd
from graphrag_llm.completion import create_completion
from graphrag_llm.utils import CompletionMessagesBuilder, gather_completion_response_async

from graphrag.callbacks.query_callbacks import QueryCallbacks
from graphrag.config.models.graph_rag_config import GraphRagConfig

from graphrag.api.query import (
    graph_global_search,
    graph_local_search,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Query types
# ---------------------------------------------------------------------------

QUERY_TYPES = frozenset({"entity_fact", "enumeration", "synthesis", "temporal"})

_CLASSIFIER_SYSTEM = """\
Classify the search query into exactly one category and extract optional date hints.

Categories — read the distinctions carefully:

- entity_fact: question about ONE specific, named entity, person, product, or fact.
  The subject is singular and already identified in the question.
  Examples: "What is product X?", "Who is person Y?",
  "What is the relationship between company A and company B?",
  "What spec does sensor Z have?"

- enumeration: asking to LIST or FIND ALL members of a structurally defined
  category — the answer is a discrete set of items. The category boundary is
  clear and structural (geography, type, organizational unit).
  Key signals: "which X", "what subsidiaries/products/customers",
  "list all", plural subjects without a named entity.
  Examples: "Which customers are from outside the country?",
  "What subsidiaries does company X have?"

- synthesis: requires COMBINING information across many sources to draw
  conclusions, describe patterns, identify roles, compare options, or give a
  broad overview. Also applies when "which X" asks about a qualitative state,
  condition, lifecycle stage, or transition — because the answer requires
  analysing the dataset rather than retrieving a structural membership list.
  Key signals: "what are the main", "what domains", "how does X compare",
  "describe the role of", "what responsibilities", "what patterns",
  "which X are/have [state or condition]".
  Examples: "What are the main application domains?",
  "Who are the sales agents and what are their responsibilities?",
  "How do product variants compare?"

- temporal: question explicitly involving SPECIFIC DATES or time periods.
  Examples: "What happened in February 2025?", "Events from 2024 to 2025."

Decision rule:
  ONE named thing                                    → entity_fact
  LIST with a clear structural category boundary     → enumeration
  LIST requiring state/condition/transition analysis → synthesis
  Patterns / roles / overview / comparison           → synthesis

Also extract any date hints (ISO-8601) if temporal.
Return ONLY valid JSON (no prose):
{"type": "<category>", "from_date": null, "until_date": null}

Date hints must be ISO-8601 strings (e.g. "2025-01-01") or null.
"""

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SmartSearchResult:
    """Result returned by smart_search(), augmented with routing metadata."""

    answer: str
    context_data: dict[str, pd.DataFrame]
    phase_timings: dict[str, float]
    query_type: str
    """Classifier output: entity_fact | enumeration | synthesis | temporal"""
    mode_used: str
    """Which search function was invoked: graph_local_search | graph_global_search | graph_drift_search"""
    classification_ms: float
    """Wall-clock time for the classifier LLM call in milliseconds."""


# ---------------------------------------------------------------------------
# Internal: classifier
# ---------------------------------------------------------------------------

async def _classify_query(
    query: str,
    config: GraphRagConfig,
) -> tuple[str, str | None, str | None]:
    """Return (query_type, from_date, until_date).

    Uses the default completion model at temperature=0 with max_tokens=60.
    Falls back to 'entity_fact' on any error so search always proceeds.
    """
    model_settings = config.get_completion_model_config(
        config.local_search.completion_model_id
    )
    model = create_completion(model_settings)

    messages = (
        CompletionMessagesBuilder()
        .add_system_message(_CLASSIFIER_SYSTEM)
        .add_user_message(query)
        .build()
    )

    try:
        raw = await model.completion_async(
            messages=messages,
            response_format_json_object=True,
            max_tokens=60,
            temperature=0,
        )
        text = await gather_completion_response_async(raw)
        parsed = json.loads(text)
        query_type = parsed.get("type", "entity_fact")
        if query_type not in QUERY_TYPES:
            logger.warning("Unknown query type '%s', falling back to entity_fact", query_type)
            query_type = "entity_fact"
        from_date = parsed.get("from_date") or None
        until_date = parsed.get("until_date") or None
        return query_type, from_date, until_date
    except Exception as exc:
        logger.warning("Query classification failed (%s), defaulting to entity_fact", exc)
        return "entity_fact", None, None


# ---------------------------------------------------------------------------
# Internal: config override per query type
# ---------------------------------------------------------------------------

def _build_call_config(config: GraphRagConfig, query_type: str) -> GraphRagConfig:
    """Return a deep-copied config with per-query-type overrides applied.

    The original config object is never mutated.
    """
    cfg = config.model_copy(deep=True)

    if query_type in ("entity_fact", "temporal"):
        # Precision-focused: more seed entities, reranker on
        cfg.local_search.rerank.enabled = True
        cfg.graph_store.top_k_seeds = max(cfg.graph_store.top_k_seeds, 20)

    # enumeration / synthesis go to graph_global_search — no local-search config needed

    return cfg


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def smart_search(
    config: GraphRagConfig,
    query: str,
    response_type: str = "Multiple Paragraphs",
    callbacks: list[QueryCallbacks] | None = None,
    verbose: bool = False,
    force_mode: str | None = None,
) -> SmartSearchResult:
    """Route a query to the best ArangoDB-native search mode automatically.

    Classification is performed by a fast LLM call (temperature=0, max_tokens=60).
    The routing decision and all timing data are included in the result.

    Parameters
    ----------
    config:
        GraphRAG configuration. Never mutated — a deep copy is used for overrides.
    query:
        The user's natural-language question.
    response_type:
        Passed through to the underlying search function.
    callbacks:
        Optional query callbacks forwarded to the search function.
    verbose:
        Enable debug logging in the underlying search function.
    force_mode:
        Skip classification and use this mode directly.
        Accepted values: ``"graph_local"``, ``"graph_global"``, ``"graph_drift"``.

    Returns
    -------
    SmartSearchResult
        Contains answer, context_data, phase_timings, query_type, mode_used,
        and classification_ms.
    """
    # --- Step 1: classify (or use forced mode) ---
    t_classify_start = time.perf_counter()

    if force_mode:
        _forced_to_type = {
            "graph_local": "entity_fact",
            "graph_global": "enumeration",
            "graph_drift": "synthesis",
        }
        query_type = _forced_to_type.get(force_mode, "entity_fact")
        from_date: str | None = None
        until_date: str | None = None
        logger.debug("smart_search: force_mode=%s → query_type=%s", force_mode, query_type)
    else:
        query_type, from_date, until_date = await _classify_query(query, config)
        logger.debug("smart_search: classified query as '%s'", query_type)

    classification_ms = round((time.perf_counter() - t_classify_start) * 1000, 1)

    # --- Step 2: build per-call config ---
    call_config = _build_call_config(config, query_type)

    # --- Step 3: determine mode ---
    if force_mode == "graph_drift":
        mode_used = "graph_drift_search"
    elif force_mode == "graph_local" or query_type in ("entity_fact", "temporal"):
        mode_used = "graph_local_search"
    else:
        # enumeration, synthesis
        mode_used = "graph_global_search"

    # --- Step 4: dispatch ---
    logger.debug("smart_search: dispatching to %s", mode_used)

    if mode_used == "graph_local_search":
        answer, context_data, phase_timings = await graph_local_search(
            config=call_config,
            response_type=response_type,
            query=query,
            callbacks=callbacks,
            verbose=verbose,
            from_date=from_date,
            until_date=until_date,
        )
    elif mode_used == "graph_global_search":
        answer, context_data, phase_timings = await graph_global_search(
            config=call_config,
            response_type=response_type,
            query=query,
            callbacks=callbacks,
            verbose=verbose,
        )
    else:
        # graph_drift_search — returns 2-tuple (no phase_timings yet)
        from graphrag.api.query import graph_drift_search
        result = await graph_drift_search(
            config=call_config,
            response_type=response_type,
            query=query,
            callbacks=callbacks,
            verbose=verbose,
        )
        answer, context_data = result[0], result[1]
        phase_timings = {}

    phase_timings["classification"] = classification_ms

    return SmartSearchResult(
        answer=answer,
        context_data=context_data,
        phase_timings=phase_timings,
        query_type=query_type,
        mode_used=mode_used,
        classification_ms=classification_ms,
    )
