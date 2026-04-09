# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""ArangoDB-backed local search context builder using native graph traversal."""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import TYPE_CHECKING, Any

import pandas as pd
from graphrag_llm.tokenizer import Tokenizer
from graphrag_vectors import VectorStore

from graphrag.data_model.community_report import CommunityReport
from graphrag.data_model.covariate import Covariate
from graphrag.data_model.entity import Entity
from graphrag.data_model.relationship import Relationship
from graphrag.data_model.text_unit import TextUnit
from graphrag.query.structured_search.local_search.mixed_context import (
    _entity_in_time_window,
    _relationship_in_time_window,
)
from graphrag.query.context_builder.builders import ContextBuilderResult
from graphrag.query.context_builder.community_context import build_community_context
from graphrag.query.context_builder.conversation_history import ConversationHistory
from graphrag.query.context_builder.entity_extraction import (
    EntityVectorStoreKey,
    map_query_to_entities,
    map_query_to_text_units,
)
from graphrag.query.context_builder.reranker import CohereReranker
from graphrag.query.context_builder.local_context import (
    build_covariates_context,
    build_entity_context,
    build_relationship_context,
)
from graphrag.query.context_builder.source_context import (
    build_text_unit_context,
    count_relationships,
)
from graphrag.query.input.retrieval.arangodb_graph_retriever import (
    ArangoDBGraphRetriever,
)
from graphrag.query.structured_search.base import LocalContextBuilder
from graphrag.tokenizer.get_tokenizer import get_tokenizer

if TYPE_CHECKING:
    from graphrag_llm.embedding import LLMEmbedding
    from graphrag_vectors.arangodb_graph import ArangoDBGraphStore

logger = logging.getLogger(__name__)


class ArangoDBGraphContextBuilder(LocalContextBuilder):
    """LocalContextBuilder backed by ArangoDB native graph traversal and hybrid search.

    Replaces the in-memory entity/relationship filtering of LocalSearchMixedContext
    with AQL graph traversal. Supports:
    - Hybrid search: APPROX_NEAR_COSINE vector seed + k-hop graph expansion
    - Pure graph traversal from vector-seeded entities
    - k-Hop neighborhood expansion (configurable depth)
    - Community context retrieved via edge traversal
    - Text unit context via optional pre-loaded text units
    """

    def __init__(
        self,
        graph_store: ArangoDBGraphStore,
        entity_text_embeddings: VectorStore,
        text_embedder: LLMEmbedding,
        community_reports: list[CommunityReport] | None = None,
        text_units: list[TextUnit] | None = None,
        covariates: dict[str, list[Covariate]] | None = None,
        tokenizer: Tokenizer | None = None,
        embedding_vectorstore_key: str = EntityVectorStoreKey.ID,
        traversal_depth: int = 2,
        top_k_seeds: int = 10,
        use_hybrid_search: bool = True,
        reranker: CohereReranker | None = None,
        text_unit_embeddings: VectorStore | None = None,
        direct_text_unit_search_k: int = 50,
    ) -> None:
        """Initialize the graph context builder.

        Parameters
        ----------
        graph_store : ArangoDBGraphStore
            Connected ArangoDB graph store instance.
        entity_text_embeddings : VectorStore
            Vector store containing entity description embeddings.
            Used as fallback seed source when use_hybrid_search=False.
        text_embedder : LLMEmbedding
            Embedding model for encoding the query.
        community_reports : list[CommunityReport] | None
            Pre-loaded community reports (used as fallback if graph retrieval fails).
        text_units : list[TextUnit] | None
            Pre-loaded text units for source context. If None, text unit context
            is omitted from the built context.
        covariates : dict[str, list[Covariate]] | None
            Pre-loaded covariates keyed by covariate type.
        tokenizer : Tokenizer | None
            Tokenizer for token budget management. Uses default if None.
        embedding_vectorstore_key : str
            The field name used as ID in the entity embedding vector store.
        traversal_depth : int
            Default k-hop depth for graph traversal neighbor expansion.
        top_k_seeds : int
            Number of top-ranked seed entities from vector search.
        use_hybrid_search : bool
            If True, uses a single combined AQL query (APPROX_NEAR_COSINE + traversal).
            If False, seeds from the entity vector store and expands via separate traversal.
        """
        self.graph_store = graph_store
        self.retriever = ArangoDBGraphRetriever(graph_store)
        self.entity_text_embeddings = entity_text_embeddings
        self.text_unit_embeddings = text_unit_embeddings
        self.text_embedder = text_embedder
        self.community_reports_list = community_reports or []
        self.text_units = {unit.id: unit for unit in (text_units or [])}
        self.covariates = covariates or {}
        self.tokenizer = tokenizer or get_tokenizer()
        self.embedding_vectorstore_key = embedding_vectorstore_key
        self.traversal_depth = traversal_depth
        self.top_k_seeds = top_k_seeds
        self.use_hybrid_search = use_hybrid_search
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
        traversal_depth: int | None = None,
        **kwargs: dict[str, Any],
    ) -> ContextBuilderResult:
        """Build local search context using ArangoDB graph traversal.

        Uses hybrid vector+graph search to find relevant entities, then expands
        their neighborhood via k-hop traversal. Context is assembled with the
        same token-budget logic as LocalSearchMixedContext.
        """
        if text_unit_prop + community_prop > 1:
            msg = "text_unit_prop + community_prop must not exceed 1."
            raise ValueError(msg)

        depth = traversal_depth if traversal_depth is not None else self.traversal_depth
        final_query = query
        if conversation_history:
            pre_user_questions = "\n".join(
                conversation_history.get_user_turns(conversation_history_max_turns)
            )
            final_query = f"{query}\n{pre_user_questions}"

        # --- Step 1: Entity retrieval via ArangoDB ---
        entities, relationships = self._retrieve_entities_and_relationships(
            query=final_query,
            depth=depth,
            k=self.top_k_seeds,
        )

        # Apply temporal filters when a time window is requested
        from_date: str | None = kwargs.get("from_date")  # type: ignore[assignment]
        until_date: str | None = kwargs.get("until_date")  # type: ignore[assignment]
        if from_date or until_date:
            entities = [
                e for e in entities
                if _entity_in_time_window(e, from_date, until_date)
            ]
            relationships = [
                r for r in relationships
                if _relationship_in_time_window(r, from_date, until_date)
            ]
            logger.debug(
                "Temporal filter [%s → %s]: %d entities, %d relationships remain",
                from_date, until_date, len(entities), len(relationships),
            )

        # Apply include/exclude name filters
        if include_entity_names:
            include_set = set(include_entity_names)
            entities = [e for e in entities if e.title in include_set] or entities
        if exclude_entity_names:
            exclude_set = set(exclude_entity_names)
            entities = [e for e in entities if e.title not in exclude_set]

        # Rank and limit to top_k
        selected_entities = sorted(
            entities, key=lambda e: e.rank or 0, reverse=True
        )[:top_k_mapped_entities]

        # --- Assemble context ---
        final_context: list[str] = []
        final_context_data: dict[str, pd.DataFrame] = {}

        # Conversation history
        if conversation_history:
            (
                conv_context,
                conv_context_data,
            ) = conversation_history.build_context(
                include_user_turns_only=conversation_history_user_turns_only,
                max_qa_turns=conversation_history_max_turns,
                column_delimiter=column_delimiter,
                max_context_tokens=max_context_tokens,
                recency_bias=False,
            )
            if conv_context.strip():
                final_context.append(conv_context)
                final_context_data = conv_context_data
                max_context_tokens -= len(self.tokenizer.encode(conv_context))

        # Community context
        community_tokens = max(int(max_context_tokens * community_prop), 0)
        community_context, community_context_data = self._build_community_context(
            selected_entities=selected_entities,
            max_context_tokens=community_tokens,
            use_community_summary=use_community_summary,
            column_delimiter=column_delimiter,
            include_community_rank=include_community_rank,
            min_community_rank=min_community_rank,
            context_name=community_context_name,
            query=query,
        )
        if community_context.strip():
            final_context.append(community_context)
            final_context_data = {**final_context_data, **community_context_data}

        # Local (entity + relationship) context
        local_prop = 1.0 - community_prop - text_unit_prop
        local_tokens = max(int(max_context_tokens * local_prop), 0)
        local_context, local_context_data = self._build_local_context(
            selected_entities=selected_entities,
            relationships=relationships,
            max_context_tokens=local_tokens,
            include_entity_rank=include_entity_rank,
            rank_description=rank_description,
            include_relationship_weight=include_relationship_weight,
            top_k_relationships=top_k_relationships,
            relationship_ranking_attribute=relationship_ranking_attribute,
            column_delimiter=column_delimiter,
        )
        if local_context.strip():
            final_context.append(local_context)
            final_context_data = {**final_context_data, **local_context_data}

        # Text unit context — fetched from ArangoDB via entity_text_unit edges
        text_unit_tokens = max(int(max_context_tokens * text_unit_prop), 0)
        text_unit_context, text_unit_context_data = self._build_text_unit_context(
            selected_entities=selected_entities,
            relationships=relationships,
            max_context_tokens=text_unit_tokens,
            column_delimiter=column_delimiter,
            query=query,
        )
        if text_unit_context.strip():
            final_context.append(text_unit_context)
            final_context_data = {**final_context_data, **text_unit_context_data}

        # Covariate context — fetched from ArangoDB via entity_covariate edges
        covariates = self._fetch_covariates_from_graph(selected_entities)
        if covariates:
            for cov_type, cov_list in covariates.items():
                cov_context, cov_context_data = build_covariates_context(
                    selected_entities=selected_entities,
                    covariates=cov_list,
                    tokenizer=self.tokenizer,
                    max_context_tokens=local_tokens,
                    context_name=cov_type,
                )
                if cov_context.strip():
                    final_context.append(cov_context)
                    final_context_data = {**final_context_data, **cov_context_data}

        return ContextBuilderResult(
            context_chunks="\n\n".join(final_context),
            context_records=final_context_data,
        )

    # ------------------------------------------------------------------
    # Internal retrieval helpers
    # ------------------------------------------------------------------

    def _retrieve_entities_and_relationships(
        self,
        query: str,
        depth: int,
        k: int,
    ) -> tuple[list[Entity], list[Relationship]]:
        """Retrieve entities and relationships from ArangoDB."""
        if self.use_hybrid_search:
            # Combined vector + graph in one AQL query
            try:
                emb_response = self.text_embedder.embedding(input=[query])
                raw = emb_response.first_embedding
                query_embedding = raw.tolist() if hasattr(raw, "tolist") else list(raw)
                return self.retriever.hybrid_retrieve(
                    query_vector=query_embedding,
                    depth=depth,
                    k=k,
                )
            except Exception as exc:
                logger.warning(
                    "Hybrid search failed: %s. Falling back to vector seed + traversal.",
                    exc,
                )

        # Fallback: seed from entity vector store, expand via graph traversal
        seed_entities = map_query_to_entities(
            query=query,
            text_embedding_vectorstore=self.entity_text_embeddings,
            text_embedder=self.text_embedder,
            all_entities_dict={},  # empty — we just want the vector search results
            embedding_vectorstore_key=self.embedding_vectorstore_key,
            k=k,
            oversample_scaler=2,
        )
        if not seed_entities:
            return [], []
        return self.retriever.get_neighbors(seed_entities, depth=depth)

    def _build_community_context(
        self,
        selected_entities: list[Entity],
        max_context_tokens: int = 4000,
        use_community_summary: bool = False,
        column_delimiter: str = "|",
        include_community_rank: bool = False,
        min_community_rank: int = 0,
        context_name: str = "Reports",
        query: str = "",
    ) -> tuple[str, dict[str, pd.DataFrame]]:
        """Retrieve and format community reports for selected entities."""
        if not selected_entities:
            return ("", {context_name.lower(): pd.DataFrame()})

        # Get reports from ArangoDB graph
        reports = self.retriever.get_community_reports(selected_entities)

        # Fall back to pre-loaded reports if graph query returned nothing
        if not reports and self.community_reports_list:
            entity_community_ids: set[str] = set()
            for entity in selected_entities:
                for cid in (entity.community_ids or []):
                    entity_community_ids.add(str(cid))
            reports = [
                r for r in self.community_reports_list
                if r.community_id in entity_community_ids
            ]

        if not reports:
            return ("", {context_name.lower(): pd.DataFrame()})

        reports.sort(key=lambda r: r.rank or 0, reverse=True)

        was_reranked = False
        if self.reranker and query and reports:
            docs = [
                r.summary if use_community_summary else (r.full_content or r.summary or "")
                for r in reports
            ]
            reports = self.reranker.rerank(query=query, documents=docs, metadata=reports)
            was_reranked = True
            logger.debug("Reranked %d community candidates with Cohere", len(reports))

        context_text, context_data = build_community_context(
            community_reports=reports,
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
        if isinstance(context_text, list) and context_text:
            context_text = "\n\n".join(context_text)
        return (str(context_text), context_data)

    def _build_local_context(
        self,
        selected_entities: list[Entity],
        relationships: list[Relationship],
        max_context_tokens: int = 8000,
        include_entity_rank: bool = False,
        rank_description: str = "number of relationships",
        include_relationship_weight: bool = False,
        top_k_relationships: int = 10,
        relationship_ranking_attribute: str = "rank",
        column_delimiter: str = "|",
    ) -> tuple[str, dict[str, pd.DataFrame]]:
        """Format entity and relationship tables for local context."""
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

        # Pre-filter: only keep relationships where at least one endpoint is a selected entity.
        # Graph traversal may return edges between non-selected neighbors — those are noise.
        selected_titles = {e.title for e in selected_entities}
        relationships = [
            r for r in relationships
            if r.source in selected_titles or r.target in selected_titles
        ]

        added_entities: list[Entity] = []
        final_rel_context = ""
        final_context_data: dict[str, pd.DataFrame] = {}

        for entity in selected_entities:
            added_entities.append(entity)
            rel_context, rel_context_data = build_relationship_context(
                selected_entities=added_entities,
                relationships=relationships,
                tokenizer=self.tokenizer,
                max_context_tokens=max_context_tokens,
                column_delimiter=column_delimiter,
                top_k_relationships=top_k_relationships,
                include_relationship_weight=include_relationship_weight,
                relationship_ranking_attribute=relationship_ranking_attribute,
                context_name="Relationships",
                text_units=self.text_units,
            )
            total_tokens = entity_tokens + len(self.tokenizer.encode(rel_context))
            if total_tokens > max_context_tokens:
                logger.debug("Reached token limit in local context — stopping entity expansion.")
                break
            final_rel_context = rel_context
            final_context_data["relationships"] = rel_context_data
            final_context_data["entities"] = entity_context_data

        full_text = entity_context + "\n\n" + final_rel_context
        final_context_data["entities"] = entity_context_data
        return (full_text, final_context_data)

    def _fetch_text_units_from_graph(self, selected_entities: list[Entity]) -> dict[str, TextUnit]:
        """Fetch text units from ArangoDB via entity_text_unit edges, return id→TextUnit dict."""
        entity_ids = [e.id for e in selected_entities if e.id]
        docs = self.graph_store.get_text_units_for_entities(entity_ids)
        result = {}
        for doc in docs:
            unit_id = str(doc.get("id", doc.get("_key", "")))
            raw_short = doc.get("human_readable_id")
            if unit_id and doc.get("text"):
                result[unit_id] = TextUnit(
                    id=unit_id,
                    short_id=str(raw_short) if raw_short is not None else None,
                    text=str(doc["text"]),
                    n_tokens=doc.get("n_tokens"),
                    document_id=doc.get("document_id"),
                )
        return result

    def _fetch_covariates_from_graph(
        self, selected_entities: list[Entity]
    ) -> dict[str, list[Covariate]]:
        """Fetch covariates from ArangoDB via entity_covariate edges, grouped by type."""
        entity_ids = [e.id for e in selected_entities if e.id]
        docs = self.graph_store.get_covariates_for_entities(entity_ids)
        result: dict[str, list[Covariate]] = {}
        for doc in docs:
            cov_type = str(doc.get("covariate_type", "claim"))
            raw_short = doc.get("human_readable_id")
            cov = Covariate(
                id=str(doc.get("id", doc.get("_key", ""))),
                short_id=str(raw_short) if raw_short is not None else None,
                subject_id=str(doc.get("subject_id", "")),
                subject_type=str(doc.get("subject_type", "entity")),
                covariate_type=cov_type,
                text_unit_ids=doc.get("text_unit_ids"),
                attributes=doc.get("attributes"),
            )
            result.setdefault(cov_type, []).append(cov)
        return result

    def _build_text_unit_context(
        self,
        selected_entities: list[Entity],
        relationships: list[Relationship],
        max_context_tokens: int = 8000,
        column_delimiter: str = "|",
        context_name: str = "Sources",
        query: str = "",
    ) -> tuple[str, dict[str, pd.DataFrame]]:
        """Format text units associated with selected entities.

        Uses ArangoDB graph traversal when text_units were not pre-loaded at construction.
        When a text_unit_embeddings store and reranker are configured, applies hybrid
        retrieval (Path B) and reranking before token-budget fill.
        """
        if not selected_entities:
            return ("", {context_name.lower(): pd.DataFrame()})

        # Prefer pre-loaded dict; fall back to ArangoDB graph traversal
        text_units = self.text_units or self._fetch_text_units_from_graph(selected_entities)
        if not text_units:
            return ("", {context_name.lower(): pd.DataFrame()})

        unit_info_list: list[tuple] = []
        text_unit_ids_seen: set[str] = set()

        # Path A: entity-derived text units
        for index, entity in enumerate(selected_entities):
            entity_relationships = [
                rel for rel in relationships
                if rel.source == entity.title or rel.target == entity.title
            ]
            for text_id in entity.text_unit_ids or []:
                if text_id not in text_unit_ids_seen and text_id in text_units:
                    selected_unit = deepcopy(text_units[text_id])
                    num_relationships = count_relationships(entity_relationships, selected_unit)
                    text_unit_ids_seen.add(text_id)
                    unit_info_list.append((selected_unit, index, num_relationships))

        # Path B: direct text unit vector search (hybrid retrieval)
        if self.text_unit_embeddings and query:
            direct_units = map_query_to_text_units(
                query=query,
                text_unit_vectorstore=self.text_unit_embeddings,
                text_embedder=self.text_embedder,
                all_text_units=text_units,
                k=self.direct_text_unit_search_k,
            )
            for unit in direct_units:
                if unit.id not in text_unit_ids_seen:
                    text_unit_ids_seen.add(unit.id)
                    unit_info_list.append((deepcopy(unit), len(selected_entities), 0))

        unit_info_list.sort(key=lambda x: (x[1], -x[2]))
        selected_text_units = [item[0] for item in unit_info_list]

        if self.reranker and query and selected_text_units:
            selected_text_units = self.reranker.rerank(
                query=query,
                documents=[u.text for u in selected_text_units],
                metadata=selected_text_units,
            )
            logger.debug("Reranked %d text unit candidates with Cohere", len(selected_text_units))

        relationships_dict = {r.id: r for r in relationships if r.id}
        context_text, context_data = build_text_unit_context(
            text_units=selected_text_units,
            tokenizer=self.tokenizer,
            max_context_tokens=max_context_tokens,
            shuffle_data=False,
            context_name=context_name,
            column_delimiter=column_delimiter,
            relationships=relationships_dict,
        )
        return (str(context_text), context_data)
