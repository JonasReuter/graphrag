# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License
"""Algorithms to build context data for local search prompt."""

import logging
from copy import deepcopy
from typing import TYPE_CHECKING, Any

import pandas as pd
from graphrag_llm.tokenizer import Tokenizer
from graphrag_vectors import VectorStore

from graphrag.data_model.community_report import CommunityReport
from graphrag.data_model.covariate import Covariate
from graphrag.data_model.entity import Entity
from graphrag.data_model.evidence import Evidence
from graphrag.data_model.relationship import Relationship
from graphrag.data_model.text_unit import TextUnit
from graphrag.query.context_builder.builders import ContextBuilderResult
from graphrag.query.context_builder.community_context import (
    build_community_context,
)
from graphrag.query.context_builder.conversation_history import (
    ConversationHistory,
)
from graphrag.query.context_builder.entity_extraction import (
    EntityVectorStoreKey,
    map_query_to_entities,
    map_query_to_text_units,
)
from graphrag.query.context_builder.reranker import CohereReranker
from graphrag.query.context_builder.evidence_context import build_evidence_context
from graphrag.query.context_builder.local_context import (
    build_covariates_context,
    build_entity_context,
    build_relationship_context,
    get_candidate_context,
)
from graphrag.query.context_builder.source_context import (
    build_text_unit_context,
    count_relationships,
)
from graphrag.query.input.retrieval.community_reports import (
    get_candidate_communities,
)
from graphrag.query.input.retrieval.text_units import get_candidate_text_units
from graphrag.query.structured_search.base import LocalContextBuilder
from graphrag.tokenizer.get_tokenizer import get_tokenizer

if TYPE_CHECKING:
    from graphrag_llm.embedding import LLMEmbedding

logger = logging.getLogger(__name__)


class LocalSearchMixedContext(LocalContextBuilder):
    """Build data context for local search prompt combining community reports and entity/relationship/covariate tables."""

    def __init__(
        self,
        entities: list[Entity],
        entity_text_embeddings: VectorStore,
        text_embedder: "LLMEmbedding",
        text_units: list[TextUnit] | None = None,
        community_reports: list[CommunityReport] | None = None,
        relationships: list[Relationship] | None = None,
        covariates: dict[str, list[Covariate]] | None = None,
        evidence: list[Evidence] | None = None,
        tokenizer: Tokenizer | None = None,
        embedding_vectorstore_key: str = EntityVectorStoreKey.ID,
        text_unit_embeddings: VectorStore | None = None,
        reranker: CohereReranker | None = None,
        direct_text_unit_search_k: int = 50,
    ):
        if community_reports is None:
            community_reports = []
        if relationships is None:
            relationships = []
        if covariates is None:
            covariates = {}
        if text_units is None:
            text_units = []
        if evidence is None:
            evidence = []
        self.entities = {entity.id: entity for entity in entities}
        self.community_reports = {
            community.community_id: community for community in community_reports
        }
        self.text_units = {unit.id: unit for unit in text_units}
        self.relationships = {
            relationship.id: relationship for relationship in relationships
        }
        self.covariates = covariates
        self.evidence = evidence
        self.entity_text_embeddings = entity_text_embeddings
        self.text_unit_embeddings = text_unit_embeddings
        self.text_embedder = text_embedder
        self.tokenizer = tokenizer or get_tokenizer()
        self.embedding_vectorstore_key = embedding_vectorstore_key
        self.reranker = reranker
        self.direct_text_unit_search_k = direct_text_unit_search_k

    def build_context(
        self,
        query: str,
        conversation_history: ConversationHistory | None = None,
        include_entity_names: list[str] | None = None,
        exclude_entity_names: list[str] | None = None,
        conversation_history_max_turns: int | None = 5,
        conversation_history_user_turns_only: bool = True,
        max_context_tokens: int = 8000,
        text_unit_prop: float = 0.5,
        community_prop: float = 0.25,
        evidence_prop: float = 0.0,
        top_k_mapped_entities: int = 10,
        top_k_relationships: int = 10,
        include_community_rank: bool = False,
        include_entity_rank: bool = False,
        rank_description: str = "number of relationships",
        include_relationship_weight: bool = False,
        relationship_ranking_attribute: str = "rank",
        return_candidate_context: bool = False,
        use_community_summary: bool = False,
        min_community_rank: int = 0,
        community_context_name: str = "Reports",
        column_delimiter: str = "|",
        from_date: str | None = None,
        until_date: str | None = None,
        **kwargs: dict[str, Any],
    ) -> ContextBuilderResult:
        """
        Build data context for local search prompt.

        Build a context by combining community reports and entity/relationship/covariate tables, and text units using a predefined ratio set by summary_prop.
        """
        if include_entity_names is None:
            include_entity_names = []
        if exclude_entity_names is None:
            exclude_entity_names = []
        if community_prop + text_unit_prop + evidence_prop > 1:
            value_error = (
                "The sum of community_prop, text_unit_prop and evidence_prop should not exceed 1."
            )
            raise ValueError(value_error)

        # map user query to entities
        # if there is conversation history, attached the previous user questions to the current query
        if conversation_history:
            pre_user_questions = "\n".join(
                conversation_history.get_user_turns(conversation_history_max_turns)
            )
            query = f"{query}\n{pre_user_questions}"

        selected_entities = map_query_to_entities(
            query=query,
            text_embedding_vectorstore=self.entity_text_embeddings,
            text_embedder=self.text_embedder,
            all_entities_dict=self.entities,
            embedding_vectorstore_key=self.embedding_vectorstore_key,
            include_entity_names=include_entity_names,
            exclude_entity_names=exclude_entity_names,
            k=top_k_mapped_entities,
            oversample_scaler=2,
        )

        # Temporal filtering: restrict entities and relationships to a time window.
        # An entity passes if it was observed at any point within [from_date, until_date].
        # We keep entities without temporal data (None) so legacy data is not silently excluded.
        if from_date or until_date:
            selected_entities = [
                e for e in selected_entities
                if _entity_in_time_window(e, from_date, until_date)
            ]
            logger.debug(
                "Temporal filter [%s → %s]: %d entities remain",
                from_date, until_date, len(selected_entities),
            )

        # build context
        final_context = list[str]()
        final_context_data = dict[str, pd.DataFrame]()

        if conversation_history:
            # build conversation history context
            (
                conversation_history_context,
                conversation_history_context_data,
            ) = conversation_history.build_context(
                include_user_turns_only=conversation_history_user_turns_only,
                max_qa_turns=conversation_history_max_turns,
                column_delimiter=column_delimiter,
                max_context_tokens=max_context_tokens,
                recency_bias=False,
            )
            if conversation_history_context.strip() != "":
                final_context.append(conversation_history_context)
                final_context_data = conversation_history_context_data
                max_context_tokens = max_context_tokens - len(
                    self.tokenizer.encode(conversation_history_context)
                )

        # build community context
        community_tokens = max(int(max_context_tokens * community_prop), 0)
        community_context, community_context_data = self._build_community_context(
            query=query,
            selected_entities=selected_entities,
            max_context_tokens=community_tokens,
            use_community_summary=use_community_summary,
            column_delimiter=column_delimiter,
            include_community_rank=include_community_rank,
            min_community_rank=min_community_rank,
            return_candidate_context=return_candidate_context,
            context_name=community_context_name,
        )
        if community_context.strip() != "":
            final_context.append(community_context)
            final_context_data = {**final_context_data, **community_context_data}

        # build local (i.e. entity-relationship-covariate) context
        local_prop = 1 - community_prop - text_unit_prop - evidence_prop
        local_tokens = max(int(max_context_tokens * local_prop), 0)
        local_context, local_context_data = self._build_local_context(
            selected_entities=selected_entities,
            max_context_tokens=local_tokens,
            include_entity_rank=include_entity_rank,
            rank_description=rank_description,
            include_relationship_weight=include_relationship_weight,
            top_k_relationships=top_k_relationships,
            relationship_ranking_attribute=relationship_ranking_attribute,
            return_candidate_context=return_candidate_context,
            column_delimiter=column_delimiter,
            from_date=from_date,
            until_date=until_date,
        )
        if local_context.strip() != "":
            final_context.append(str(local_context))
            final_context_data = {**final_context_data, **local_context_data}

        text_unit_tokens = max(int(max_context_tokens * text_unit_prop), 0)
        text_unit_context, text_unit_context_data = self._build_text_unit_context(
            query=query,
            selected_entities=selected_entities,
            max_context_tokens=text_unit_tokens,
            return_candidate_context=return_candidate_context,
        )

        if text_unit_context.strip() != "":
            final_context.append(text_unit_context)
            final_context_data = {**final_context_data, **text_unit_context_data}

        # build evidence context (only when evidence data is available and evidence_prop > 0)
        if self.evidence and evidence_prop > 0:
            evidence_tokens = max(int(max_context_tokens * evidence_prop), 0)
            evidence_context, evidence_context_data = build_evidence_context(
                selected_entities=selected_entities,
                evidence=self.evidence,
                relationships=list(self.relationships.values()),
                tokenizer=self.tokenizer,
                max_context_tokens=evidence_tokens,
                column_delimiter=column_delimiter,
            )
            if evidence_context.strip() != "":
                final_context.append(evidence_context)
                final_context_data = {**final_context_data, **{"evidence": evidence_context_data}}

        return ContextBuilderResult(
            context_chunks="\n\n".join(final_context),
            context_records=final_context_data,
        )

    def _build_community_context(
        self,
        selected_entities: list[Entity],
        max_context_tokens: int = 4000,
        use_community_summary: bool = False,
        column_delimiter: str = "|",
        include_community_rank: bool = False,
        min_community_rank: int = 0,
        return_candidate_context: bool = False,
        context_name: str = "Reports",
        query: str = "",
    ) -> tuple[str, dict[str, pd.DataFrame]]:
        """Add community data to the context window until it hits the max_context_tokens limit."""
        if len(selected_entities) == 0 or len(self.community_reports) == 0:
            return ("", {context_name.lower(): pd.DataFrame()})

        community_matches = {}
        for entity in selected_entities:
            # increase count of the community that this entity belongs to
            if entity.community_ids:
                for community_id in entity.community_ids:
                    community_matches[community_id] = (
                        community_matches.get(community_id, 0) + 1
                    )

        # sort communities by number of matched entities and rank (no object mutation)
        selected_communities = [
            self.community_reports[community_id]
            for community_id in community_matches
            if community_id in self.community_reports
        ]
        selected_communities.sort(
            key=lambda c: (community_matches.get(c.community_id, 0), c.rank or 0),
            reverse=True,
        )

        # rerank community candidates by query relevance (replaces structural sort order)
        if self.reranker and query and selected_communities:
            docs = [
                c.summary if use_community_summary else (c.full_content or c.summary or "")
                for c in selected_communities
            ]
            selected_communities = self.reranker.rerank(
                query=query,
                documents=docs,
                metadata=selected_communities,
            )
            logger.debug(
                "Reranked %d community candidates with Cohere", len(selected_communities)
            )

        was_reranked = bool(self.reranker and query and selected_communities)
        context_text, context_data = build_community_context(
            community_reports=selected_communities,
            tokenizer=self.tokenizer,
            use_community_summary=use_community_summary,
            column_delimiter=column_delimiter,
            shuffle_data=False,
            include_community_rank=include_community_rank,
            min_community_rank=min_community_rank,
            max_context_tokens=max_context_tokens,
            single_batch=True,
            context_name=context_name,
            pre_sorted=was_reranked,
        )
        if isinstance(context_text, list) and len(context_text) > 0:
            context_text = "\n\n".join(context_text)

        if return_candidate_context:
            candidate_context_data = get_candidate_communities(
                selected_entities=selected_entities,
                community_reports=list(self.community_reports.values()),
                use_community_summary=use_community_summary,
                include_community_rank=include_community_rank,
            )
            context_key = context_name.lower()
            if context_key not in context_data:
                context_data[context_key] = candidate_context_data
                context_data[context_key]["in_context"] = False
            else:
                if (
                    "id" in candidate_context_data.columns
                    and "id" in context_data[context_key].columns
                ):
                    candidate_context_data["in_context"] = candidate_context_data[
                        "id"
                    ].isin(  # cspell:disable-line
                        context_data[context_key]["id"]
                    )
                    context_data[context_key] = candidate_context_data
                else:
                    context_data[context_key]["in_context"] = True
        return (str(context_text), context_data)

    def _build_text_unit_context(
        self,
        selected_entities: list[Entity],
        max_context_tokens: int = 8000,
        return_candidate_context: bool = False,
        column_delimiter: str = "|",
        context_name: str = "Sources",
        query: str = "",
    ) -> tuple[str, dict[str, pd.DataFrame]]:
        """Rank matching text units and add them to the context window until it hits the max_context_tokens limit.

        When a text_unit_embeddings store and reranker are configured, uses hybrid retrieval:
        - Path A (entity-derived): text units linked to selected entities
        - Path B (direct vector search): top-k text units by semantic similarity to the query
        The union of both paths is reranked by Cohere before token-budget fill.
        """
        if not selected_entities or not self.text_units:
            return ("", {context_name.lower(): pd.DataFrame()})

        text_unit_ids_set: set[str] = set()
        unit_info_list: list[tuple[TextUnit, int, int]] = []
        relationship_values = list(self.relationships.values())

        # Path A: entity-derived text units (existing logic)
        for index, entity in enumerate(selected_entities):
            entity_relationships = [
                rel
                for rel in relationship_values
                if rel.source == entity.title or rel.target == entity.title
            ]
            for text_id in entity.text_unit_ids or []:
                if text_id not in text_unit_ids_set and text_id in self.text_units:
                    selected_unit = deepcopy(self.text_units[text_id])
                    num_relationships = count_relationships(
                        entity_relationships, selected_unit
                    )
                    text_unit_ids_set.add(text_id)
                    unit_info_list.append((selected_unit, index, num_relationships))

        # Path B: direct text unit vector search (hybrid retrieval)
        if self.text_unit_embeddings and query:
            direct_units = map_query_to_text_units(
                query=query,
                text_unit_vectorstore=self.text_unit_embeddings,
                text_embedder=self.text_embedder,
                all_text_units=self.text_units,
                k=self.direct_text_unit_search_k,
            )
            for unit in direct_units:
                if unit.id not in text_unit_ids_set:
                    text_unit_ids_set.add(unit.id)
                    # index=len(selected_entities) places Path B units after Path A units
                    unit_info_list.append((deepcopy(unit), len(selected_entities), 0))
            logger.debug(
                "Hybrid text unit retrieval: %d Path-A + %d Path-B candidates",
                len(text_unit_ids_set) - len(direct_units),
                len(direct_units),
            )

        # sort by entity_order and the number of relationships desc (baseline ranking)
        unit_info_list.sort(key=lambda x: (x[1], -x[2]))
        selected_text_units = [unit[0] for unit in unit_info_list]

        # rerank the full candidate pool by query relevance
        if self.reranker and query and selected_text_units:
            selected_text_units = self.reranker.rerank(
                query=query,
                documents=[u.text for u in selected_text_units],
                metadata=selected_text_units,
            )
            logger.debug(
                "Reranked %d text unit candidates with Cohere", len(selected_text_units)
            )

        context_text, context_data = build_text_unit_context(
            text_units=selected_text_units,
            tokenizer=self.tokenizer,
            max_context_tokens=max_context_tokens,
            shuffle_data=False,
            context_name=context_name,
            column_delimiter=column_delimiter,
            relationships=self.relationships,
        )

        if return_candidate_context:
            candidate_context_data = get_candidate_text_units(
                selected_entities=selected_entities,
                text_units=list(self.text_units.values()),
            )
            context_key = context_name.lower()
            if context_key not in context_data:
                candidate_context_data["in_context"] = False
                context_data[context_key] = candidate_context_data
            else:
                if (
                    "id" in candidate_context_data.columns
                    and "id" in context_data[context_key].columns
                ):
                    candidate_context_data["in_context"] = candidate_context_data[
                        "id"
                    ].isin(context_data[context_key]["id"])
                    context_data[context_key] = candidate_context_data
                else:
                    context_data[context_key]["in_context"] = True

        return (str(context_text), context_data)

    def _build_local_context(
        self,
        selected_entities: list[Entity],
        max_context_tokens: int = 8000,
        include_entity_rank: bool = False,
        rank_description: str = "relationship count",
        include_relationship_weight: bool = False,
        top_k_relationships: int = 10,
        relationship_ranking_attribute: str = "rank",
        return_candidate_context: bool = False,
        column_delimiter: str = "|",
        from_date: str | None = None,
        until_date: str | None = None,
    ) -> tuple[str, dict[str, pd.DataFrame]]:
        """Build data context for local search prompt combining entity/relationship/covariate tables.

        When ``from_date`` or ``until_date`` are set, only relationships observed
        within the given time window are included.  Relationships without temporal
        data (``observed_at`` is None) are always kept so legacy data is not
        silently excluded.
        """
        # Apply temporal filter to relationships when a time window is requested
        all_relationships = list(self.relationships.values())
        if from_date or until_date:
            all_relationships = [
                r for r in all_relationships
                if _relationship_in_time_window(r, from_date, until_date)
            ]

        # build entity context
        entity_context, entity_context_data = build_entity_context(
            selected_entities=selected_entities,
            tokenizer=self.tokenizer,
            max_context_tokens=max_context_tokens,
            column_delimiter=column_delimiter,
            include_entity_rank=include_entity_rank,
            rank_description=rank_description,
            context_name="Entities",
        )
        entity_tokens = len(self.tokenizer.encode(entity_context))

        # build relationship-covariate context
        added_entities = []
        final_context = []
        final_context_data = {}

        # gradually add entities and associated metadata to the context until we reach limit
        for entity in selected_entities:
            current_context = []
            current_context_data = {}
            added_entities.append(entity)

            # build relationship context (with source provenance)
            (
                relationship_context,
                relationship_context_data,
            ) = build_relationship_context(
                selected_entities=added_entities,
                relationships=all_relationships,
                tokenizer=self.tokenizer,
                max_context_tokens=max_context_tokens,
                column_delimiter=column_delimiter,
                top_k_relationships=top_k_relationships,
                include_relationship_weight=include_relationship_weight,
                relationship_ranking_attribute=relationship_ranking_attribute,
                context_name="Relationships",
                text_units=self.text_units,
            )
            current_context.append(relationship_context)
            current_context_data["relationships"] = relationship_context_data
            total_tokens = entity_tokens + len(
                self.tokenizer.encode(relationship_context)
            )

            # build covariate context
            for covariate in self.covariates:
                covariate_context, covariate_context_data = build_covariates_context(
                    selected_entities=added_entities,
                    covariates=self.covariates[covariate],
                    tokenizer=self.tokenizer,
                    max_context_tokens=max_context_tokens,
                    column_delimiter=column_delimiter,
                    context_name=covariate,
                )
                total_tokens += len(self.tokenizer.encode(covariate_context))
                current_context.append(covariate_context)
                current_context_data[covariate.lower()] = covariate_context_data

            if total_tokens > max_context_tokens:
                logger.warning(
                    "Reached token limit - reverting to previous context state"
                )
                break

            final_context = current_context
            final_context_data = current_context_data

        # attach entity context to final context
        final_context_text = entity_context + "\n\n" + "\n\n".join(final_context)
        final_context_data["entities"] = entity_context_data

        if return_candidate_context:
            # we return all the candidate entities/relationships/covariates (not only those that were fitted into the context window)
            # and add a tag to indicate which records were included in the context window
            candidate_context_data = get_candidate_context(
                selected_entities=selected_entities,
                entities=list(self.entities.values()),
                relationships=all_relationships,
                covariates=self.covariates,
                include_entity_rank=include_entity_rank,
                entity_rank_description=rank_description,
                include_relationship_weight=include_relationship_weight,
            )
            for key in candidate_context_data:
                candidate_df = candidate_context_data[key]
                if key not in final_context_data:
                    final_context_data[key] = candidate_df
                    final_context_data[key]["in_context"] = False
                else:
                    in_context_df = final_context_data[key]

                    if "id" in in_context_df.columns and "id" in candidate_df.columns:
                        candidate_df["in_context"] = candidate_df[
                            "id"
                        ].isin(  # cspell:disable-line
                            in_context_df["id"]
                        )
                        final_context_data[key] = candidate_df
                    else:
                        final_context_data[key]["in_context"] = True
        else:
            for key in final_context_data:
                final_context_data[key]["in_context"] = True
        return (final_context_text, final_context_data)


# ---------------------------------------------------------------------------
# Temporal filter helpers
# ---------------------------------------------------------------------------

def _entity_in_time_window(
    entity: "Entity",
    from_date: str | None,
    until_date: str | None,
) -> bool:
    """Return True if the entity was observed at any point within the window.

    Uses ``observed_at`` (first seen) and ``last_observed_at`` (last seen).
    If both are None the entity passes (legacy / no temporal data).
    """
    obs = entity.observed_at
    last = entity.last_observed_at
    # Entity with no temporal data: always include
    if obs is None and last is None:
        return True
    # Entity passes if *any* of its observation window overlaps with [from_date, until_date]
    # i.e. last_observed >= from_date  AND  observed_at <= until_date
    if from_date and last is not None and last < from_date:
        return False
    if until_date and obs is not None and obs > until_date:
        return False
    return True


def _relationship_in_time_window(
    rel: "Relationship",
    from_date: str | None,
    until_date: str | None,
) -> bool:
    """Return True if the relationship was observed within the window.

    Falls back to ``valid_from``/``valid_until`` when ``observed_at`` is absent.
    Relationships without any temporal data always pass.
    """
    # Prefer document-level timestamps; fall back to LLM-extracted validity
    obs = rel.observed_at or rel.valid_from
    last = rel.last_observed_at or rel.valid_until
    if obs is None and last is None:
        return True
    if from_date and last is not None and last < from_date:
        return False
    if until_date and obs is not None and obs > until_date:
        return False
    return True

