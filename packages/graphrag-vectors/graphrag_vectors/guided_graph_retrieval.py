# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Query-guided, vector-first graph retrieval for ArangoDB.

This module is the scalable replacement for broad k-hop retrieval. It does not
materialize all paths up to a hop depth. Instead it:

1. retrieves a small set of semantic seed entities via the Arango vector index;
2. expands the graph with a best-first priority queue;
3. asks Arango for only the top-M edges per expanded node;
4. applies depth decay, hub penalties, type/relation hints, and hard budgets;
5. optionally returns community reports for the best result region.

The runtime is bounded primarily by ``seed_k + max_expansions * max_edges_per_node``
instead of by graph degree^depth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import heapq
import logging
import time
from typing import Any, Literal

logger = logging.getLogger(__name__)

_ERR_INDEX_BUILDING = 1555
Direction = Literal["ANY", "OUTBOUND", "INBOUND"]


@dataclass(frozen=True)
class QueryGraphPlan:
    """Optional query hints for guided graph expansion.

    These hints are intentionally soft by default: matching relation/entity types
    receive a boost, while non-matching candidates are still allowed unless the
    caller sets ``hard_entity_type_filter`` or ``hard_relation_type_filter``.
    """

    entity_types: tuple[str, ...] = ()
    relation_types: tuple[str, ...] = ()
    relation_terms: tuple[str, ...] = ()
    direction: Direction = "ANY"
    hard_entity_type_filter: bool = False
    hard_relation_type_filter: bool = False

    @classmethod
    def from_query_text(
        cls,
        query: str,
        *,
        entity_types: list[str] | tuple[str, ...] | None = None,
        relation_types: list[str] | tuple[str, ...] | None = None,
        direction: Direction = "ANY",
        hard_entity_type_filter: bool = False,
        hard_relation_type_filter: bool = False,
    ) -> "QueryGraphPlan":
        """Create a lightweight plan from caller-provided hints and query text."""
        return cls(
            entity_types=tuple(_normalise_terms(entity_types or [])),
            relation_types=tuple(_normalise_terms(relation_types or [])),
            relation_terms=tuple(_normalise_terms(_query_terms(query))),
            direction=direction,
            hard_entity_type_filter=hard_entity_type_filter,
            hard_relation_type_filter=hard_relation_type_filter,
        )


@dataclass(frozen=True)
class GuidedGraphRetrievalConfig:
    """Runtime budgets and scoring weights for guided graph retrieval."""

    seed_k: int = 12
    max_depth: int = 3
    max_edges_per_node: int = 8
    max_expansions: int = 128
    max_frontier_size: int = 256
    max_results: int = 80
    min_path_score: float = 0.05
    depth_decay: float = 0.72
    seed_vector_weight: float = 0.80
    seed_type_weight: float = 0.10
    seed_hub_penalty_weight: float = 0.10
    edge_vector_weight: float = 0.35
    edge_weight_weight: float = 0.20
    relation_match_weight: float = 0.20
    target_type_weight: float = 0.10
    hub_penalty_weight: float = 0.15
    ppr_boost_weight: float = 0.10
    community_report_limit: int = 8
    allow_vector_scan_fallback: bool = False


@dataclass(order=True)
class _FrontierItem:
    priority: float
    counter: int
    node_id: str = field(compare=False)
    depth: int = field(compare=False)
    score: float = field(compare=False)
    path_edges: tuple[str, ...] = field(compare=False)
    path_nodes: tuple[str, ...] = field(compare=False)


class GuidedArangoGraphRetriever:
    """Vector-first best-first retriever for ArangoDB GraphRAG graphs.

    Use this with the existing ``ArangoDBGraphStore`` connection::

        retriever = GuidedArangoGraphRetriever(graph_store._db, graph_store.graph_name)
        result = retriever.retrieve(query_vector, query="...", config=...)

    The result contains deduplicated vertices, edges, ranked path records,
    optional community reports, and timing/budget stats.
    """

    def __init__(self, db: Any, graph_name: str) -> None:
        self._db = db
        self._graph_name = graph_name

    def retrieve(
        self,
        query_vector: list[float],
        *,
        query: str = "",
        plan: QueryGraphPlan | None = None,
        config: GuidedGraphRetrievalConfig | None = None,
    ) -> dict[str, Any]:
        """Retrieve a bounded, ranked graph region for a query vector.

        Returns a dictionary with ``vertices``, ``edges``, ``paths``,
        ``community_reports`` and ``stats``.
        """
        start = time.perf_counter()
        cfg = config or GuidedGraphRetrievalConfig()
        graph_plan = plan or QueryGraphPlan.from_query_text(query)
        direction = _safe_direction(graph_plan.direction)

        seeds = self._vector_seed_entities(query_vector, graph_plan, cfg)
        if not seeds:
            return {
                "vertices": [],
                "edges": [],
                "paths": [],
                "community_reports": [],
                "stats": {
                    "seed_count": 0,
                    "expansions": 0,
                    "duration_ms": round((time.perf_counter() - start) * 1000),
                },
            }

        vertices: dict[str, dict] = {}
        edges: dict[str, dict] = {}
        paths: list[dict] = []
        ppr_mass: dict[str, float] = {}
        best_node_score: dict[str, float] = {}
        frontier: list[_FrontierItem] = []
        counter = 0

        for seed in seeds:
            node_id = seed["_id"]
            seed_score = float(seed.get("_score", 0.0))
            vertices[node_id] = seed
            ppr_mass[node_id] = ppr_mass.get(node_id, 0.0) + seed_score
            best_node_score[node_id] = seed_score
            heapq.heappush(
                frontier,
                _FrontierItem(
                    priority=-seed_score,
                    counter=counter,
                    node_id=node_id,
                    depth=0,
                    score=seed_score,
                    path_edges=(),
                    path_nodes=(node_id,),
                ),
            )
            counter += 1

        expansions = 0
        while frontier and expansions < cfg.max_expansions and len(paths) < cfg.max_results:
            current = heapq.heappop(frontier)
            if current.depth >= cfg.max_depth:
                continue

            candidates = self._top_edges_for_node(
                node_id=current.node_id,
                query_vector=query_vector,
                plan=graph_plan,
                config=cfg,
                direction=direction,
            )
            expansions += 1
            if not candidates:
                continue

            # Local PPR-like mass distribution over only the bounded candidates.
            outgoing_mass = current.score / len(candidates)

            for candidate in candidates:
                vertex = candidate.get("vertex") or {}
                edge = candidate.get("edge") or {}
                vertex_id = vertex.get("_id")
                edge_id = edge.get("_id")
                if not vertex_id or not edge_id:
                    continue
                if vertex_id in current.path_nodes:
                    continue

                edge_score = float(candidate.get("score") or 0.0)
                ppr_mass[vertex_id] = ppr_mass.get(vertex_id, 0.0) + outgoing_mass
                ppr_boost = cfg.ppr_boost_weight * ppr_mass[vertex_id]
                path_score = current.score * edge_score * cfg.depth_decay + ppr_boost
                if path_score < cfg.min_path_score:
                    continue

                vertices[vertex_id] = vertex
                edges[edge_id] = edge
                next_edges = (*current.path_edges, edge_id)
                next_nodes = (*current.path_nodes, vertex_id)
                paths.append(
                    {
                        "score": path_score,
                        "depth": current.depth + 1,
                        "nodes": list(next_nodes),
                        "edges": list(next_edges),
                        "last_vertex": vertex,
                        "last_edge": edge,
                        "edge_scores": {
                            "combined": edge_score,
                            "vector": candidate.get("vector_score", 0.0),
                            "edge_weight": candidate.get("edge_weight_score", 0.0),
                            "relation_match": candidate.get("relation_match_score", 0.0),
                            "type": candidate.get("type_score", 0.0),
                            "hub": candidate.get("hub_score", 0.0),
                        },
                    }
                )

                if path_score <= best_node_score.get(vertex_id, -1.0):
                    continue
                best_node_score[vertex_id] = path_score
                heapq.heappush(
                    frontier,
                    _FrontierItem(
                        priority=-path_score,
                        counter=counter,
                        node_id=vertex_id,
                        depth=current.depth + 1,
                        score=path_score,
                        path_edges=next_edges,
                        path_nodes=next_nodes,
                    ),
                )
                counter += 1

            if len(frontier) > cfg.max_frontier_size:
                frontier = heapq.nsmallest(cfg.max_frontier_size, frontier)
                heapq.heapify(frontier)

        paths.sort(key=lambda path: path["score"], reverse=True)
        paths = paths[: cfg.max_results]
        community_reports = self._community_reports_for_vertices(
            vertices=list(vertices.values()),
            limit=cfg.community_report_limit,
        )

        return {
            "vertices": list(vertices.values()),
            "edges": list(edges.values()),
            "paths": paths,
            "community_reports": community_reports,
            "stats": {
                "seed_count": len(seeds),
                "vertex_count": len(vertices),
                "edge_count": len(edges),
                "path_count": len(paths),
                "expansions": expansions,
                "max_possible_edge_reads": cfg.max_expansions * cfg.max_edges_per_node,
                "duration_ms": round((time.perf_counter() - start) * 1000),
            },
        }

    def _vector_seed_entities(
        self,
        query_vector: list[float],
        plan: QueryGraphPlan,
        config: GuidedGraphRetrievalConfig,
    ) -> list[dict]:
        hard_filter = ""
        if plan.hard_entity_type_filter and plan.entity_types:
            hard_filter = "FILTER LOWER(doc.type) IN @entity_types"

        aql = f"""
            FOR doc IN entities
                FILTER HAS(doc, "vector")
                {hard_filter}
                LET vector_score = APPROX_NEAR_COSINE(doc.`vector`, @qvec)
                LET type_score = LENGTH(@entity_types) == 0 ? 0.5 : (LOWER(NOT_NULL(doc.type, "")) IN @entity_types ? 1 : 0.25)
                LET degree = TO_NUMBER(NOT_NULL(doc.rank, LENGTH(NOT_NULL(doc.text_unit_ids, [])), 1))
                LET hub_score = 1 / LOG(2 + MAX([degree, 1]))
                LET score = @seed_vector_weight * vector_score + @seed_type_weight * type_score + @seed_hub_penalty_weight * hub_score
                SORT score DESC
                LIMIT @seed_k
                RETURN MERGE(doc, {{_score: score, _vector_score: vector_score, _type_score: type_score, _hub_score: hub_score}})
        """
        try:
            cursor = self._db.aql.execute(
                aql,
                bind_vars={
                    "qvec": query_vector,
                    "seed_k": config.seed_k,
                    "entity_types": list(plan.entity_types),
                    "seed_vector_weight": config.seed_vector_weight,
                    "seed_type_weight": config.seed_type_weight,
                    "seed_hub_penalty_weight": config.seed_hub_penalty_weight,
                },
            )
            return list(cursor)
        except Exception as exc:
            if str(_ERR_INDEX_BUILDING) in str(exc) or "APPROX_NEAR_COSINE" in str(exc):
                logger.warning(
                    "Vector seed retrieval failed (%s). Full vector scan fallback is disabled unless explicitly enabled.",
                    exc,
                )
                if config.allow_vector_scan_fallback:
                    return self._vector_seed_entities_scan_fallback(query_vector, plan, config)
                return []
            raise

    def _vector_seed_entities_scan_fallback(
        self,
        query_vector: list[float],
        plan: QueryGraphPlan,
        config: GuidedGraphRetrievalConfig,
    ) -> list[dict]:
        """Small-data fallback. Disabled by default to protect production latency."""
        import numpy as np

        q = np.array(query_vector, dtype=float)
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return []

        cursor = self._db.aql.execute(
            "FOR doc IN entities FILTER HAS(doc, 'vector') RETURN doc"
        )
        scored: list[tuple[float, dict]] = []
        entity_types = set(plan.entity_types)
        for doc in cursor:
            if plan.hard_entity_type_filter and entity_types and str(doc.get("type", "")).casefold() not in entity_types:
                continue
            vec = doc.get("vector")
            if not vec:
                continue
            d = np.array(vec, dtype=float)
            d_norm = np.linalg.norm(d)
            if d_norm == 0:
                continue
            vector_score = float(np.dot(q, d) / (q_norm * d_norm))
            type_score = 0.5 if not entity_types else (1.0 if str(doc.get("type", "")).casefold() in entity_types else 0.25)
            degree = float(doc.get("rank") or 1)
            hub_score = 1.0 / max(1.0, float(__import__("math").log(2 + max(degree, 1))))
            score = (
                config.seed_vector_weight * vector_score
                + config.seed_type_weight * type_score
                + config.seed_hub_penalty_weight * hub_score
            )
            doc["_score"] = score
            scored.append((score, doc))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [doc for _, doc in scored[: config.seed_k]]

    def _top_edges_for_node(
        self,
        *,
        node_id: str,
        query_vector: list[float],
        plan: QueryGraphPlan,
        config: GuidedGraphRetrievalConfig,
        direction: Direction,
    ) -> list[dict]:
        hard_entity_filter = ""
        if plan.hard_entity_type_filter and plan.entity_types:
            hard_entity_filter = "FILTER LOWER(NOT_NULL(v.type, '')) IN @entity_types"

        hard_relation_filter = ""
        if plan.hard_relation_type_filter and plan.relation_types:
            hard_relation_filter = "FILTER LOWER(NOT_NULL(e.type, NOT_NULL(e.predicate, ''))) IN @relation_types"

        aql = f"""
            FOR v, e IN 1..1 {direction} @node_id
              GRAPH @graph
              OPTIONS {{bfs: true, uniqueVertices: "path"}}
              FILTER IS_SAME_COLLECTION("entities", v)
              {hard_entity_filter}
              {hard_relation_filter}
              LET relation_type = LOWER(NOT_NULL(e.type, NOT_NULL(e.predicate, "")))
              LET relation_text = LOWER(CONCAT_SEPARATOR(" ", relation_type, NOT_NULL(e.description, "")))
              LET relation_type_score = LENGTH(@relation_types) == 0 ? 0.5 : (relation_type IN @relation_types ? 1 : 0.25)
              LET relation_term_hits = LENGTH(FOR term IN @relation_terms FILTER CONTAINS(relation_text, term) RETURN 1)
              LET relation_term_score = LENGTH(@relation_terms) == 0 ? 0.5 : relation_term_hits / LENGTH(@relation_terms)
              LET relation_match_score = MAX([relation_type_score, relation_term_score])
              LET vector_score = HAS(v, "vector") ? APPROX_NEAR_COSINE(v.`vector`, @qvec) : 0
              LET raw_weight = TO_NUMBER(NOT_NULL(e.weight, NOT_NULL(e.rank, 1)))
              LET edge_weight_score = raw_weight / (raw_weight + 1)
              LET type_score = LENGTH(@entity_types) == 0 ? 0.5 : (LOWER(NOT_NULL(v.type, "")) IN @entity_types ? 1 : 0.25)
              LET degree = TO_NUMBER(NOT_NULL(v.rank, LENGTH(NOT_NULL(v.text_unit_ids, [])), 1))
              LET hub_score = 1 / LOG(2 + MAX([degree, 1]))
              LET score = @edge_vector_weight * vector_score + @edge_weight_weight * edge_weight_score + @relation_match_weight * relation_match_score + @target_type_weight * type_score + @hub_penalty_weight * hub_score
              SORT score DESC
              LIMIT @limit
              RETURN {{
                vertex: v,
                edge: e,
                score: score,
                vector_score: vector_score,
                edge_weight_score: edge_weight_score,
                relation_match_score: relation_match_score,
                type_score: type_score,
                hub_score: hub_score
              }}
        """
        try:
            cursor = self._db.aql.execute(
                aql,
                bind_vars={
                    "node_id": node_id,
                    "graph": self._graph_name,
                    "qvec": query_vector,
                    "limit": config.max_edges_per_node,
                    "entity_types": list(plan.entity_types),
                    "relation_types": list(plan.relation_types),
                    "relation_terms": list(plan.relation_terms),
                    "edge_vector_weight": config.edge_vector_weight,
                    "edge_weight_weight": config.edge_weight_weight,
                    "relation_match_weight": config.relation_match_weight,
                    "target_type_weight": config.target_type_weight,
                    "hub_penalty_weight": config.hub_penalty_weight,
                },
            )
            return list(cursor)
        except Exception as exc:
            if str(_ERR_INDEX_BUILDING) in str(exc) or "APPROX_NEAR_COSINE" in str(exc):
                logger.warning(
                    "Vector-aware edge expansion failed (%s). Falling back to bounded edge-only expansion.",
                    exc,
                )
                return self._top_edges_for_node_without_vectors(
                    node_id=node_id,
                    plan=plan,
                    config=config,
                    direction=direction,
                )
            raise

    def _top_edges_for_node_without_vectors(
        self,
        *,
        node_id: str,
        plan: QueryGraphPlan,
        config: GuidedGraphRetrievalConfig,
        direction: Direction,
    ) -> list[dict]:
        hard_entity_filter = ""
        if plan.hard_entity_type_filter and plan.entity_types:
            hard_entity_filter = "FILTER LOWER(NOT_NULL(v.type, '')) IN @entity_types"

        hard_relation_filter = ""
        if plan.hard_relation_type_filter and plan.relation_types:
            hard_relation_filter = "FILTER LOWER(NOT_NULL(e.type, NOT_NULL(e.predicate, ''))) IN @relation_types"

        aql = f"""
            FOR v, e IN 1..1 {direction} @node_id
              GRAPH @graph
              OPTIONS {{bfs: true, uniqueVertices: "path"}}
              FILTER IS_SAME_COLLECTION("entities", v)
              {hard_entity_filter}
              {hard_relation_filter}
              LET relation_type = LOWER(NOT_NULL(e.type, NOT_NULL(e.predicate, "")))
              LET relation_text = LOWER(CONCAT_SEPARATOR(" ", relation_type, NOT_NULL(e.description, "")))
              LET relation_type_score = LENGTH(@relation_types) == 0 ? 0.5 : (relation_type IN @relation_types ? 1 : 0.25)
              LET relation_term_hits = LENGTH(FOR term IN @relation_terms FILTER CONTAINS(relation_text, term) RETURN 1)
              LET relation_term_score = LENGTH(@relation_terms) == 0 ? 0.5 : relation_term_hits / LENGTH(@relation_terms)
              LET relation_match_score = MAX([relation_type_score, relation_term_score])
              LET raw_weight = TO_NUMBER(NOT_NULL(e.weight, NOT_NULL(e.rank, 1)))
              LET edge_weight_score = raw_weight / (raw_weight + 1)
              LET type_score = LENGTH(@entity_types) == 0 ? 0.5 : (LOWER(NOT_NULL(v.type, "")) IN @entity_types ? 1 : 0.25)
              LET degree = TO_NUMBER(NOT_NULL(v.rank, LENGTH(NOT_NULL(v.text_unit_ids, [])), 1))
              LET hub_score = 1 / LOG(2 + MAX([degree, 1]))
              LET score = @edge_weight_weight * edge_weight_score + @relation_match_weight * relation_match_score + @target_type_weight * type_score + @hub_penalty_weight * hub_score
              SORT score DESC
              LIMIT @limit
              RETURN {{
                vertex: v,
                edge: e,
                score: score,
                vector_score: 0,
                edge_weight_score: edge_weight_score,
                relation_match_score: relation_match_score,
                type_score: type_score,
                hub_score: hub_score
              }}
        """
        cursor = self._db.aql.execute(
            aql,
            bind_vars={
                "node_id": node_id,
                "graph": self._graph_name,
                "limit": config.max_edges_per_node,
                "entity_types": list(plan.entity_types),
                "relation_types": list(plan.relation_types),
                "relation_terms": list(plan.relation_terms),
                "edge_weight_weight": config.edge_weight_weight,
                "relation_match_weight": config.relation_match_weight,
                "target_type_weight": config.target_type_weight,
                "hub_penalty_weight": config.hub_penalty_weight,
            },
        )
        return list(cursor)

    def _community_reports_for_vertices(
        self,
        *,
        vertices: list[dict],
        limit: int,
    ) -> list[dict]:
        if limit <= 0 or not vertices:
            return []
        entity_keys = [str(v.get("_key") or v.get("id") or "") for v in vertices]
        entity_keys = [key for key in entity_keys if key]
        if not entity_keys:
            return []

        aql = """
            FOR entity_key IN @entity_keys
                FOR community IN 1..1 OUTBOUND CONCAT("entities/", entity_key)
                  GRAPH @graph
                  OPTIONS {bfs: true}
                  FILTER IS_SAME_COLLECTION("communities", community)
                  FOR report IN community_reports
                      FILTER report.community == community.community
                      COLLECT report_id = report._id INTO grouped = report
                      LET report_doc = FIRST(grouped)
                      LIMIT @limit
                      RETURN report_doc
        """
        try:
            cursor = self._db.aql.execute(
                aql,
                bind_vars={
                    "entity_keys": entity_keys,
                    "graph": self._graph_name,
                    "limit": limit,
                },
            )
            return list(cursor)
        except Exception as exc:
            logger.warning("Could not retrieve community reports for guided graph result: %s", exc)
            return []


def _safe_direction(direction: str) -> Direction:
    upper = direction.upper()
    if upper not in {"ANY", "OUTBOUND", "INBOUND"}:
        raise ValueError("direction must be one of ANY, OUTBOUND, or INBOUND")
    return upper  # type: ignore[return-value]


def _normalise_terms(terms: list[str] | tuple[str, ...]) -> list[str]:
    return [term.strip().casefold() for term in terms if term and term.strip()]


def _query_terms(query: str) -> list[str]:
    # Lightweight lexical hints only. The vector index remains the primary signal.
    cleaned = "".join(ch.casefold() if ch.isalnum() else " " for ch in query)
    terms = [term for term in cleaned.split() if len(term) >= 4]
    return terms[:12]
