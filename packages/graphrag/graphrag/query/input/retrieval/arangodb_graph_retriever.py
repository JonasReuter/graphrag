# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""ArangoDB graph retriever: converts AQL results to GraphRAG domain objects."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from graphrag.data_model.community_report import CommunityReport
from graphrag.data_model.entity import Entity
from graphrag.data_model.relationship import Relationship

if TYPE_CHECKING:
    from graphrag_vectors.arangodb_graph import ArangoDBGraphStore

logger = logging.getLogger(__name__)


class ArangoDBGraphRetriever:
    """Retrieves entities and relationships using ArangoDB graph traversal.

    Converts raw ArangoDB document dicts (from AQL queries) into the domain
    objects that the existing context builders consume.
    """

    def __init__(self, graph_store: ArangoDBGraphStore) -> None:
        self.graph_store = graph_store

    def get_neighbors(
        self,
        seed_entities: list[Entity],
        depth: int = 2,
    ) -> tuple[list[Entity], list[Relationship]]:
        """K-hop traversal from multiple seed entities.

        Returns deduplicated (entities, relationships) as domain objects.
        """
        seen_entity_ids: set[str] = set()
        seen_rel_ids: set[str] = set()
        entities: list[Entity] = []
        relationships: list[Relationship] = []

        for seed in seed_entities:
            verts, edges = self.graph_store.traverse_neighbors(seed.id, depth=depth)
            for v in verts:
                eid = str(v.get("id", v.get("_key", "")))
                if eid and eid not in seen_entity_ids:
                    seen_entity_ids.add(eid)
                    entities.append(_doc_to_entity(v))
            for e in edges:
                if e is None:
                    continue
                rid = str(e.get("id", e.get("_key", "")))
                if rid and rid not in seen_rel_ids:
                    seen_rel_ids.add(rid)
                    relationships.append(_doc_to_relationship(e))

        return entities, relationships

    def hybrid_retrieve(
        self,
        query_vector: list[float],
        depth: int = 2,
        k: int = 10,
    ) -> tuple[list[Entity], list[Relationship]]:
        """Combined vector seed + graph traversal.

        Returns deduplicated (entities, relationships) as domain objects.
        """
        verts, edges = self.graph_store.hybrid_search(
            query_vector, depth=depth, k=k
        )
        seen_entity_ids: set[str] = set()
        seen_rel_ids: set[str] = set()
        entities: list[Entity] = []
        relationships: list[Relationship] = []

        for v in verts:
            eid = str(v.get("id", v.get("_key", "")))
            if eid and eid not in seen_entity_ids:
                seen_entity_ids.add(eid)
                entities.append(_doc_to_entity(v))

        for e in edges:
            if e is None:
                continue
            rid = str(e.get("id", e.get("_key", "")))
            if rid and rid not in seen_rel_ids:
                seen_rel_ids.add(rid)
                relationships.append(_doc_to_relationship(e))

        return entities, relationships

    def get_community_reports(
        self, entities: list[Entity]
    ) -> list[CommunityReport]:
        """Retrieve community reports for a list of entities via graph edges."""
        entity_ids = [e.id for e in entities]
        docs = self.graph_store.get_community_reports_for_entities(entity_ids)
        return _docs_to_community_reports(docs)

    def get_all_community_reports(self) -> list[CommunityReport]:
        """Retrieve all community reports from ArangoDB (for DRIFT primer phase etc.)."""
        docs = self.graph_store.get_all_community_reports()
        return _docs_to_community_reports(docs)


# ---------------------------------------------------------------------------
# Document → domain object converters
# ---------------------------------------------------------------------------

def _docs_to_community_reports(docs: list[dict]) -> list[CommunityReport]:
    """Convert a list of raw ArangoDB community report docs, deduplicating by id."""
    seen: set[str] = set()
    reports: list[CommunityReport] = []
    for doc in docs:
        doc_id = str(doc.get("id", doc.get("_key", "")))
        if doc_id and doc_id not in seen:
            seen.add(doc_id)
            try:
                reports.append(_doc_to_community_report(doc))
            except (KeyError, TypeError) as exc:
                logger.debug("Skipping malformed community report doc: %s", exc)
    return reports

def _doc_to_entity(doc: dict[str, Any]) -> Entity:
    """Convert a raw ArangoDB entity document to an Entity domain object."""
    raw_community_ids = doc.get("community_ids") or []
    raw_text_unit_ids = doc.get("text_unit_ids") or []
    raw_short_id = doc.get("human_readable_id")
    return Entity(
        id=str(doc.get("id", doc.get("_key", ""))),
        short_id=str(raw_short_id) if raw_short_id is not None else None,
        title=str(doc.get("title", "")),
        type=doc.get("type"),
        description=doc.get("description"),
        community_ids=[str(c) for c in raw_community_ids],
        text_unit_ids=[str(t) for t in raw_text_unit_ids],
        rank=int(doc["degree"]) if doc.get("degree") is not None else 1,
    )


def _doc_to_relationship(doc: dict[str, Any]) -> Relationship:
    """Convert a raw ArangoDB edge document to a Relationship domain object.

    The edge document stores the original source/target entity titles
    (copied from the relationship row during indexing), so they map
    correctly to the Relationship.source/target fields.
    """
    raw_short_id_rel = doc.get("human_readable_id")
    return Relationship(
        id=str(doc.get("id", doc.get("_key", ""))),
        short_id=str(raw_short_id_rel) if raw_short_id_rel is not None else None,
        source=str(doc.get("source", "")),
        target=str(doc.get("target", "")),
        weight=float(doc["weight"]) if doc.get("weight") is not None else 1.0,
        description=doc.get("description"),
        rank=int(doc["combined_degree"]) if doc.get("combined_degree") is not None else 1,
        text_unit_ids=doc.get("text_unit_ids"),
    )


def _doc_to_community_report(doc: dict[str, Any]) -> CommunityReport:
    """Convert a raw ArangoDB community report document to a CommunityReport domain object."""
    raw_short_id_cr = doc.get("human_readable_id")
    return CommunityReport(
        id=str(doc.get("id", doc.get("_key", ""))),
        short_id=str(raw_short_id_cr) if raw_short_id_cr is not None else None,
        title=str(doc.get("title", "")),
        community_id=str(doc.get("community", "")),
        summary=str(doc.get("summary", "")),
        full_content=str(doc.get("full_content", "")),
        rank=float(doc["rank"]) if doc.get("rank") is not None else 1.0,
        size=doc.get("size"),
        period=doc.get("period"),
    )
