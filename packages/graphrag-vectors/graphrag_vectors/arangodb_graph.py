# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""ArangoDB native graph store for GraphRAG knowledge graphs.

This module provides ArangoDBGraphStore, which stores GraphRAG pipeline output
(entities, relationships, communities, text_units) as a proper ArangoDB named
graph with edge collections, enabling native graph traversal, shortest-path
queries, and combined vector+graph (hybrid) AQL queries.

Collections created:
  Document: entities, communities, community_reports, text_units
  Edge:     relationships (entities→entities)
            entity_community_membership (entities→communities)
            entity_text_unit (entities→text_units)

Named graph: <graph_name> (default: "knowledge_graph")
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_BATCH_SIZE = 500

# ArangoDB error code for "index still building"
_ERR_INDEX_BUILDING = 1555

_DOCUMENT_COLLECTIONS = ["entities", "communities", "community_reports", "text_units"]
_EDGE_COLLECTIONS = [
    "relationships",
    "entity_community_membership",
    "entity_text_unit",
]


class ArangoDBGraphStore:
    """Manages GraphRAG knowledge graph structure in ArangoDB.

    Stores entities as documents, relationships as edges between entity
    documents, and provides AQL-based graph traversal and hybrid
    vector+graph search.
    """

    def __init__(
        self,
        url: str = "http://localhost:8529",
        username: str = "root",
        password: str = "",
        db_name: str = "graphrag",
        graph_name: str = "knowledge_graph",
        batch_size: int = _BATCH_SIZE,
        vector_size: int = 3072,
    ) -> None:
        self.url = url
        self.username = username
        self.password = password
        self.db_name = db_name
        self.graph_name = graph_name
        self.batch_size = batch_size
        self.vector_size = vector_size
        self._db: Any = None

    # ------------------------------------------------------------------
    # Connection & Schema
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Connect to ArangoDB and ensure the target database exists."""
        from arango import ArangoClient

        client = ArangoClient(hosts=self.url)
        sys_db = client.db("_system", username=self.username, password=self.password)
        if not sys_db.has_database(self.db_name):
            sys_db.create_database(self.db_name)
        self._db = client.db(
            self.db_name, username=self.username, password=self.password
        )

    def setup_graph(self) -> None:
        """Create all collections and the named graph (idempotent)."""
        for name in _DOCUMENT_COLLECTIONS:
            if not self._db.has_collection(name):
                self._db.create_collection(name)

        for name in _EDGE_COLLECTIONS:
            if not self._db.has_collection(name):
                self._db.create_collection(name, edge=True)

        if not self._db.has_graph(self.graph_name):
            self._db.create_graph(
                self.graph_name,
                edge_definitions=[
                    {
                        "edge_collection": "relationships",
                        "from_vertex_collections": ["entities"],
                        "to_vertex_collections": ["entities"],
                    },
                    {
                        "edge_collection": "entity_community_membership",
                        "from_vertex_collections": ["entities"],
                        "to_vertex_collections": ["communities"],
                    },
                    {
                        "edge_collection": "entity_text_unit",
                        "from_vertex_collections": ["entities"],
                        "to_vertex_collections": ["text_units"],
                    },
                ],
            )

    def _ensure_entity_vector_index(self) -> None:
        """Add a sparse HNSW vector index on entities.vector.

        Sparse means documents without a 'vector' field are still accepted —
        they are simply not included in ANN queries.  If an existing index
        has the wrong dimension, it is dropped and recreated.

        Non-fatal: any remaining failure is logged as a warning and hybrid
        search falls back to local cosine similarity.
        """
        coll = self._db.collection("entities")
        existing = next(
            (
                idx for idx in coll.indexes()
                if idx.get("type") == "vector" and "vector" in idx.get("fields", [])
            ),
            None,
        )

        if existing:
            existing_dim = existing.get("params", {}).get("dimension")
            already_sparse = existing.get("sparse", False)
            if existing_dim == self.vector_size and already_sparse:
                return  # index is correct
            # Drop mismatched or non-sparse index and recreate
            try:
                coll.delete_index(existing["id"])
                logger.info(
                    "Dropped entity vector index (dim=%s, sparse=%s) to recreate.",
                    existing_dim, already_sparse,
                )
            except Exception as exc:
                logger.warning("Could not drop old entity vector index: %s", exc)
                return

        try:
            coll.add_index({
                "type": "vector",
                "fields": ["vector"],
                "sparse": True,   # allow documents without vector field
                "params": {
                    "metric": "cosine",
                    "dimension": self.vector_size,
                    "nLists": 2,  # required by ArangoDB vector index
                },
            })
            logger.info(
                "Entity vector index created (dimension=%d, sparse=True).",
                self.vector_size,
            )
        except Exception as exc:
            logger.warning(
                "Could not create entity vector index (dimension=%d): %s. "
                "Hybrid search will fall back to local cosine similarity.",
                self.vector_size,
                exc,
            )

    # ------------------------------------------------------------------
    # Upsert helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sanitize(value: Any) -> Any:
        """Recursively convert numpy types to JSON-serializable Python types.

        pandas.DataFrame.to_dict("records") can produce numpy int64, float64,
        ndarray, and NaN values that ArangoDB's import_bulk cannot serialize.
        """
        import math

        import numpy as np

        if isinstance(value, np.ndarray):
            return [ArangoDBGraphStore._sanitize(v) for v in value.tolist()]
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            v = float(value)
            return None if math.isnan(v) else v
        if isinstance(value, float) and math.isnan(value):
            return None
        if isinstance(value, list):
            return [ArangoDBGraphStore._sanitize(v) for v in value]
        if isinstance(value, dict):
            return {k: ArangoDBGraphStore._sanitize(v) for k, v in value.items()}
        return value

    def _sanitize_doc(self, doc: dict) -> dict:
        """Return a copy of doc with all numpy types converted."""
        return {k: self._sanitize(v) for k, v in doc.items()}

    def _bulk_upsert(self, collection_name: str, docs: list[dict]) -> None:
        """Batch-upsert documents into a collection."""
        if not docs:
            return
        coll = self._db.collection(collection_name)
        for i in range(0, len(docs), self.batch_size):
            batch = [self._sanitize_doc(d) for d in docs[i : i + self.batch_size]]
            try:
                coll.import_bulk(batch, on_duplicate="replace", halt_on_error=True)
            except Exception as exc:
                logger.error(
                    "import_bulk failed for collection '%s' (batch %d–%d): %s",
                    collection_name, i, i + len(batch) - 1, exc,
                )
                raise

    # ------------------------------------------------------------------
    # Upsert methods
    # ------------------------------------------------------------------

    def upsert_entities(
        self,
        rows: list[dict[str, Any]],
        vector_map: dict[str, list[float]] | None = None,
    ) -> int:
        """Upsert entity documents. Optionally embed vectors for hybrid search."""
        docs = []
        for row in rows:
            entity_id = str(row["id"])
            doc: dict[str, Any] = {
                "_key": entity_id,
                **{k: v for k, v in row.items() if not k.startswith("_")},
            }
            if vector_map and entity_id in vector_map:
                doc["vector"] = vector_map[entity_id]
            docs.append(doc)
        self._bulk_upsert("entities", docs)
        logger.info("Upserted %d entity documents.", len(docs))
        return len(docs)

    def upsert_relationships(
        self,
        rows: list[dict[str, Any]],
        title_to_id: dict[str, str],
    ) -> int:
        """Upsert relationship edges. Resolves source/target entity titles to UUIDs."""
        docs = []
        skipped = 0
        for row in rows:
            src_id = title_to_id.get(str(row.get("source", "")))
            tgt_id = title_to_id.get(str(row.get("target", "")))
            if not src_id or not tgt_id:
                skipped += 1
                continue
            doc: dict[str, Any] = {
                "_key": str(row["id"]),
                "_from": f"entities/{src_id}",
                "_to": f"entities/{tgt_id}",
                **{k: v for k, v in row.items() if not k.startswith("_")},
            }
            docs.append(doc)
        if skipped:
            logger.warning(
                "Skipped %d relationships with unresolvable entity titles.", skipped
            )
        self._bulk_upsert("relationships", docs)
        logger.info("Upserted %d relationship edges.", len(docs))
        return len(docs)

    def upsert_communities(
        self, rows: list[dict[str, Any]]
    ) -> tuple[int, dict[str, str]]:
        """Upsert community documents.

        Returns (count, community_int_to_uuid) map needed for membership edges.
        The 'community' column is the integer cluster ID; 'id' is the UUID.
        """
        docs = []
        community_int_to_uuid: dict[str, str] = {}
        for row in rows:
            community_uuid = str(row["id"])
            # 'community' column is the integer cluster ID
            community_int_id = str(row.get("community", ""))
            if community_int_id:
                community_int_to_uuid[community_int_id] = community_uuid
            doc: dict[str, Any] = {
                "_key": community_uuid,
                **{k: v for k, v in row.items() if not k.startswith("_")},
            }
            docs.append(doc)
        self._bulk_upsert("communities", docs)
        logger.info("Upserted %d community documents.", len(docs))
        return len(docs), community_int_to_uuid

    def upsert_community_reports(self, rows: list[dict[str, Any]]) -> int:
        """Upsert community report documents."""
        docs = [
            {"_key": str(row["id"]), **{k: v for k, v in row.items() if not k.startswith("_")}}
            for row in rows
        ]
        self._bulk_upsert("community_reports", docs)
        logger.info("Upserted %d community report documents.", len(docs))
        return len(docs)

    def upsert_text_units(self, rows: list[dict[str, Any]]) -> int:
        """Upsert text unit documents."""
        docs = [
            {"_key": str(row["id"]), **{k: v for k, v in row.items() if not k.startswith("_")}}
            for row in rows
        ]
        self._bulk_upsert("text_units", docs)
        logger.info("Upserted %d text unit documents.", len(docs))
        return len(docs)

    def upsert_entity_community_edges(
        self,
        entity_rows: list[dict[str, Any]],
        community_int_to_uuid: dict[str, str],
    ) -> int:
        """Upsert entity→community membership edges.

        IMPORTANT: community_ids in entities are integer cluster IDs, not UUIDs.
        Uses community_int_to_uuid map to resolve the correct community document key.
        """
        import json

        docs = []
        for row in entity_rows:
            entity_uuid = str(row["id"])
            community_ids = row.get("community_ids")
            # numpy arrays raise "ambiguous truth value" with `or []`; use explicit check
            if community_ids is None:
                community_ids = []
            elif isinstance(community_ids, str):
                try:
                    community_ids = json.loads(community_ids)
                except (ValueError, TypeError):
                    community_ids = []
            else:
                community_ids = list(community_ids)  # convert ndarray → list
            for comm_int_id in community_ids:
                comm_uuid = community_int_to_uuid.get(str(comm_int_id))
                if not comm_uuid:
                    continue
                docs.append({
                    "_key": f"{entity_uuid}--{comm_int_id}",
                    "_from": f"entities/{entity_uuid}",
                    "_to": f"communities/{comm_uuid}",
                })
        self._bulk_upsert("entity_community_membership", docs)
        logger.info("Upserted %d entity-community membership edges.", len(docs))
        return len(docs)

    def upsert_entity_community_edges_from_communities(
        self,
        community_rows: list[dict[str, Any]],
    ) -> int:
        """Upsert entity→community membership edges from the communities table.

        Uses community.entity_ids (list of entity UUIDs) rather than
        entity.community_ids, which is not present in the final entities table.
        """
        import json

        docs = []
        for row in community_rows:
            comm_uuid = str(row["id"])
            community_int_id = str(row.get("community", ""))
            entity_ids = row.get("entity_ids")
            if entity_ids is None:
                entity_ids = []
            elif isinstance(entity_ids, str):
                try:
                    entity_ids = json.loads(entity_ids)
                except (ValueError, TypeError):
                    entity_ids = []
            else:
                entity_ids = list(entity_ids)
            for entity_uuid in entity_ids:
                entity_uuid = str(entity_uuid)
                docs.append({
                    "_key": f"{entity_uuid}--{community_int_id}",
                    "_from": f"entities/{entity_uuid}",
                    "_to": f"communities/{comm_uuid}",
                })
        self._bulk_upsert("entity_community_membership", docs)
        logger.info("Upserted %d entity-community membership edges.", len(docs))
        return len(docs)

    def upsert_entity_text_unit_edges(
        self,
        entity_rows: list[dict[str, Any]],
    ) -> int:
        """Upsert entity→text_unit edges."""
        import json

        docs = []
        for row in entity_rows:
            entity_uuid = str(row["id"])
            text_unit_ids = row.get("text_unit_ids")
            if text_unit_ids is None:
                text_unit_ids = []
            elif isinstance(text_unit_ids, str):
                try:
                    text_unit_ids = json.loads(text_unit_ids)
                except (ValueError, TypeError):
                    text_unit_ids = []
            else:
                text_unit_ids = list(text_unit_ids)  # convert ndarray → list
            for tu_id in text_unit_ids:
                tu_str = str(tu_id)
                docs.append({
                    "_key": f"{entity_uuid}--{tu_str}",
                    "_from": f"entities/{entity_uuid}",
                    "_to": f"text_units/{tu_str}",
                })
        self._bulk_upsert("entity_text_unit", docs)
        logger.info("Upserted %d entity-text_unit edges.", len(docs))
        return len(docs)

    # ------------------------------------------------------------------
    # AQL Graph Queries
    # ------------------------------------------------------------------

    def traverse_neighbors(
        self,
        entity_id: str,
        depth: int = 2,
        direction: str = "ANY",
    ) -> tuple[list[dict], list[dict]]:
        """K-hop graph traversal from a single entity.

        Returns (vertex_docs, edge_docs). Only entity vertices are returned
        (communities and text_units are filtered out).
        """
        aql = f"""
            FOR v, e, p IN 1..@depth {direction} @start
              GRAPH @graph
              OPTIONS {{bfs: true, uniqueVertices: "global"}}
              FILTER IS_SAME_COLLECTION("entities", v)
              RETURN DISTINCT {{vertex: v, edge: e}}
        """
        cursor = self._db.aql.execute(
            aql,
            bind_vars={
                "depth": depth,
                "start": f"entities/{entity_id}",
                "graph": self.graph_name,
            },
        )
        vertices, edges = [], []
        for row in cursor:
            if row.get("vertex"):
                vertices.append(row["vertex"])
            if row.get("edge"):
                edges.append(row["edge"])
        return vertices, edges

    def shortest_path(
        self,
        src_entity_id: str,
        dst_entity_id: str,
    ) -> list[dict]:
        """Return vertex documents along the shortest path between two entities."""
        aql = """
            FOR v IN ANY SHORTEST_PATH @src TO @dst
              GRAPH @graph
              RETURN v
        """
        cursor = self._db.aql.execute(
            aql,
            bind_vars={
                "src": f"entities/{src_entity_id}",
                "dst": f"entities/{dst_entity_id}",
                "graph": self.graph_name,
            },
        )
        return list(cursor)

    def hybrid_search(
        self,
        query_vector: list[float],
        depth: int = 2,
        k: int = 10,
    ) -> tuple[list[dict], list[dict]]:
        """Combined vector seed + graph traversal in a single AQL query.

        Uses APPROX_NEAR_COSINE to find seed entities, then expands k hops.
        Falls back to separate vector search + get_neighbors if the vector
        index is not yet ready (ArangoDB error 1555).

        Returns (vertex_docs, edge_docs).
        """
        aql = """
            LET seeds = (
                FOR doc IN entities
                    LET score = APPROX_NEAR_COSINE(doc.`vector`, @qvec)
                    SORT score DESC
                    LIMIT @k
                    RETURN doc
            )
            FOR seed IN seeds
                FOR v, e IN 1..@depth ANY seed._id
                  GRAPH @graph
                  OPTIONS {bfs: true, uniqueVertices: "global"}
                  FILTER IS_SAME_COLLECTION("entities", v)
                  RETURN DISTINCT {vertex: v, edge: e}
        """
        try:
            cursor = self._db.aql.execute(
                aql,
                bind_vars={
                    "qvec": query_vector,
                    "k": k,
                    "depth": depth,
                    "graph": self.graph_name,
                },
            )
            vertices, edges = [], []
            for row in cursor:
                if row.get("vertex"):
                    vertices.append(row["vertex"])
                if row.get("edge"):
                    edges.append(row["edge"])
            return vertices, edges
        except Exception as exc:
            if str(_ERR_INDEX_BUILDING) in str(exc) or "APPROX_NEAR_COSINE" in str(exc):
                logger.warning(
                    "Hybrid AQL failed (%s). Vector index may still be building. "
                    "Falling back to in-memory similarity + graph traversal.",
                    exc,
                )
                return self._hybrid_search_fallback(query_vector, depth, k)
            raise

    def _hybrid_search_fallback(
        self,
        query_vector: list[float],
        depth: int,
        k: int,
    ) -> tuple[list[dict], list[dict]]:
        """Fallback: fetch all entities, rank by cosine similarity, then traverse."""
        import numpy as np

        q = np.array(query_vector, dtype=float)
        q_norm = np.linalg.norm(q)

        all_docs = list(self._db.aql.execute("FOR doc IN entities RETURN doc"))
        scored = []
        for doc in all_docs:
            vec = doc.get("vector")
            if not vec:
                continue
            d = np.array(vec, dtype=float)
            d_norm = np.linalg.norm(d)
            if q_norm == 0 or d_norm == 0:
                continue
            score = float(np.dot(q, d) / (q_norm * d_norm))
            scored.append((score, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        seeds = [doc for _, doc in scored[:k]]

        all_vertices: dict[str, dict] = {}
        all_edges: dict[str, dict] = {}
        for seed in seeds:
            seed_id = seed.get("id", seed.get("_key", ""))
            verts, edges = self.traverse_neighbors(seed_id, depth=depth)
            for v in verts:
                vid = v.get("_id", "")
                if vid:
                    all_vertices[vid] = v
            for e in edges:
                eid = e.get("_id", "")
                if eid:
                    all_edges[eid] = e

        return list(all_vertices.values()), list(all_edges.values())

    def get_community_reports_for_entities(
        self,
        entity_ids: list[str],
    ) -> list[dict]:
        """Retrieve community reports for a list of entity UUIDs via graph edges."""
        if not entity_ids:
            return []
        aql = """
            FOR entity_id IN @entity_ids
                FOR e IN entity_community_membership
                    FILTER e._from == CONCAT("entities/", entity_id)
                    LET community = DOCUMENT(e._to)
                    FOR report IN community_reports
                        FILTER report.community == community.community
                        RETURN DISTINCT report
        """
        cursor = self._db.aql.execute(
            aql, bind_vars={"entity_ids": entity_ids}
        )
        return list(cursor)
