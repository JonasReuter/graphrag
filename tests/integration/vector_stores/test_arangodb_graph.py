# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Integration tests for ArangoDBGraphStore.

Requires a running ArangoDB 3.12+ instance with vector index support enabled.
Start one locally with:

    docker compose up -d

Tests are automatically skipped when ArangoDB is not reachable.
"""

from __future__ import annotations

import pytest

try:
    from arango import ArangoClient

    _ARANGO_AVAILABLE = True
except ImportError:
    _ARANGO_AVAILABLE = False

from graphrag_vectors.arangodb_graph import ArangoDBGraphStore

pytestmark = pytest.mark.skipif(
    not _ARANGO_AVAILABLE, reason="python-arango not installed"
)

_URL = "http://localhost:8529"
_USERNAME = "root"
_PASSWORD = ""
_DB = "graphrag_graph_test"
_GRAPH = "test_knowledge_graph"
_VECTOR_SIZE = 4


def _is_arangodb_reachable() -> bool:
    if not _ARANGO_AVAILABLE:
        return False
    try:
        client = ArangoClient(hosts=_URL)
        sys_db = client.db("_system", username=_USERNAME, password=_PASSWORD)
        sys_db.version()
        return True
    except Exception:  # noqa: BLE001
        return False


skip_if_no_arango = pytest.mark.skipif(
    not _is_arangodb_reachable(),
    reason="ArangoDB not reachable at localhost:8529 — start with `docker compose up -d`",
)


@pytest.fixture
def store():
    """Create a connected ArangoDBGraphStore and tear down test database afterward."""
    s = ArangoDBGraphStore(
        url=_URL,
        username=_USERNAME,
        password=_PASSWORD,
        db_name=_DB,
        graph_name=_GRAPH,
        batch_size=10,
        vector_size=_VECTOR_SIZE,
    )
    s.connect()
    s.setup_graph()
    yield s
    # Teardown: drop the entire test database
    try:
        client = ArangoClient(hosts=_URL)
        sys_db = client.db("_system", username=_USERNAME, password=_PASSWORD)
        if sys_db.has_database(_DB):
            sys_db.delete_database(_DB)
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

@skip_if_no_arango
class TestSetupGraph:
    def test_named_graph_created(self, store):
        assert store._db.has_graph(_GRAPH)

    def test_entity_collection_exists(self, store):
        assert store._db.has_collection("entities")

    def test_edge_collections_exist(self, store):
        for name in ["relationships", "entity_community_membership", "entity_text_unit"]:
            assert store._db.has_collection(name), f"Missing edge collection: {name}"

    def test_setup_graph_idempotent(self, store):
        """Calling setup_graph() a second time should not raise."""
        store.setup_graph()
        assert store._db.has_graph(_GRAPH)


# ---------------------------------------------------------------------------
# Upsert entities
# ---------------------------------------------------------------------------

@skip_if_no_arango
class TestUpsertEntities:
    def test_entities_stored_as_documents(self, store):
        rows = [
            {"id": "ent-1", "title": "Alice", "type": "person", "degree": 3},
            {"id": "ent-2", "title": "Bob",   "type": "person", "degree": 2},
        ]
        store.upsert_entities(rows)

        coll = store._db.collection("entities")
        assert coll.count() == 2

    def test_entity_key_is_uuid(self, store):
        rows = [{"id": "my-uuid", "title": "Charlie", "degree": 1}]
        store.upsert_entities(rows)

        doc = store._db.collection("entities").get("my-uuid")
        assert doc is not None
        assert doc["title"] == "Charlie"

    def test_entity_upsert_is_idempotent(self, store):
        rows = [{"id": "ent-1", "title": "Alice", "degree": 3}]
        store.upsert_entities(rows)
        store.upsert_entities(rows)

        assert store._db.collection("entities").count() == 1

    def test_vector_stored_in_entity_document(self, store):
        rows = [{"id": "ent-v", "title": "Vec Entity", "degree": 1}]
        vector_map = {"ent-v": [0.1, 0.2, 0.3, 0.4]}
        store.upsert_entities(rows, vector_map=vector_map)

        doc = store._db.collection("entities").get("ent-v")
        assert "vector" in doc
        assert len(doc["vector"]) == 4


# ---------------------------------------------------------------------------
# Upsert relationships as edges
# ---------------------------------------------------------------------------

@skip_if_no_arango
class TestUpsertRelationships:
    def _seed_entities(self, store):
        rows = [
            {"id": "ent-a", "title": "Alice", "degree": 2},
            {"id": "ent-b", "title": "Bob",   "degree": 1},
            {"id": "ent-c", "title": "Carol", "degree": 3},
        ]
        store.upsert_entities(rows)
        return {"Alice": "ent-a", "Bob": "ent-b", "Carol": "ent-c"}

    def test_relationship_edge_stored(self, store):
        title_to_id = self._seed_entities(store)
        rows = [{"id": "rel-0", "source": "Alice", "target": "Bob", "weight": 1.0}]
        store.upsert_relationships(rows, title_to_id=title_to_id)

        assert store._db.collection("relationships").count() == 1

    def test_from_to_point_to_entity_documents(self, store):
        title_to_id = self._seed_entities(store)
        rows = [{"id": "rel-0", "source": "Alice", "target": "Bob", "weight": 1.0}]
        store.upsert_relationships(rows, title_to_id=title_to_id)

        edge = store._db.collection("relationships").get("rel-0")
        assert edge["_from"] == "entities/ent-a"
        assert edge["_to"] == "entities/ent-b"

    def test_skips_unresolvable_titles(self, store):
        title_to_id = self._seed_entities(store)
        rows = [
            {"id": "rel-ok",  "source": "Alice",   "target": "Bob",    "weight": 1.0},
            {"id": "rel-bad", "source": "UNKNOWN", "target": "Bob",    "weight": 1.0},
        ]
        count = store.upsert_relationships(rows, title_to_id=title_to_id)
        assert count == 1
        assert store._db.collection("relationships").count() == 1


# ---------------------------------------------------------------------------
# Communities + membership edges
# ---------------------------------------------------------------------------

@skip_if_no_arango
class TestUpsertCommunities:
    def test_community_documents_stored(self, store):
        rows = [
            {"id": "c-uuid-0", "community": "0", "title": "C0"},
            {"id": "c-uuid-1", "community": "1", "title": "C1"},
        ]
        count, mapping = store.upsert_communities(rows)

        assert count == 2
        assert store._db.collection("communities").count() == 2

    def test_int_to_uuid_map_correct(self, store):
        rows = [{"id": "c-uuid-42", "community": "42", "title": "C42"}]
        _, mapping = store.upsert_communities(rows)

        assert mapping.get("42") == "c-uuid-42"

    def test_membership_edges_connect_entity_to_community(self, store):
        # Seed entities
        entity_rows = [{"id": "ent-0", "title": "E0", "degree": 1, "community_ids": ["0"], "text_unit_ids": []}]
        store.upsert_entities(entity_rows)

        # Seed communities
        community_rows = [{"id": "c-uuid-0", "community": "0", "title": "C0"}]
        _, mapping = store.upsert_communities(community_rows)

        # Create membership edges
        store.upsert_entity_community_edges(entity_rows, mapping)

        edges = list(store._db.collection("entity_community_membership").all())
        assert len(edges) == 1
        assert edges[0]["_from"] == "entities/ent-0"
        assert edges[0]["_to"] == "communities/c-uuid-0"


# ---------------------------------------------------------------------------
# Entity-TextUnit edges
# ---------------------------------------------------------------------------

@skip_if_no_arango
class TestUpsertEntityTextUnitEdges:
    def test_edges_connect_entity_to_text_unit(self, store):
        entity_rows = [{"id": "ent-0", "title": "E0", "degree": 1, "community_ids": [], "text_unit_ids": ["tu-0", "tu-1"]}]
        store.upsert_entities(entity_rows)

        text_unit_rows = [
            {"id": "tu-0", "text": "Text 0"},
            {"id": "tu-1", "text": "Text 1"},
        ]
        store.upsert_text_units(text_unit_rows)
        store.upsert_entity_text_unit_edges(entity_rows)

        edges = list(store._db.collection("entity_text_unit").all())
        assert len(edges) == 2
        tos = {e["_to"] for e in edges}
        assert "text_units/tu-0" in tos
        assert "text_units/tu-1" in tos


# ---------------------------------------------------------------------------
# Graph traversal
# ---------------------------------------------------------------------------

@skip_if_no_arango
class TestTraverseNeighbors:
    def _build_small_graph(self, store) -> dict[str, str]:
        """Create 3 entities connected in a chain: A→B→C. Returns title_to_id."""
        entity_rows = [
            {"id": "ent-a", "title": "Alice", "degree": 2},
            {"id": "ent-b", "title": "Bob",   "degree": 1},
            {"id": "ent-c", "title": "Carol", "degree": 3},
        ]
        store.upsert_entities(entity_rows)
        title_to_id = {"Alice": "ent-a", "Bob": "ent-b", "Carol": "ent-c"}

        rel_rows = [
            {"id": "rel-ab", "source": "Alice", "target": "Bob",   "weight": 1.0},
            {"id": "rel-bc", "source": "Bob",   "target": "Carol", "weight": 1.0},
        ]
        store.upsert_relationships(rel_rows, title_to_id=title_to_id)
        return title_to_id

    def test_1hop_from_alice_returns_bob(self, store):
        self._build_small_graph(store)
        verts, edges = store.traverse_neighbors("ent-a", depth=1)

        titles = {v["title"] for v in verts}
        assert "Bob" in titles
        assert "Carol" not in titles    # 2 hops away

    def test_2hop_from_alice_returns_carol(self, store):
        self._build_small_graph(store)
        verts, edges = store.traverse_neighbors("ent-a", depth=2)

        titles = {v["title"] for v in verts}
        assert "Carol" in titles

    def test_returns_edges_along_with_vertices(self, store):
        self._build_small_graph(store)
        verts, edges = store.traverse_neighbors("ent-a", depth=2)

        assert len(edges) >= 1

    def test_traversal_does_not_include_community_vertices(self, store):
        """Traversal should be filtered to entities collection only."""
        self._build_small_graph(store)
        # Add community + membership edge
        store.upsert_communities([{"id": "c-0", "community": "0", "title": "C0"}])
        entity_rows = [{"id": "ent-a", "community_ids": ["0"], "text_unit_ids": []}]
        store.upsert_entity_community_edges(entity_rows, {"0": "c-0"})

        verts, _ = store.traverse_neighbors("ent-a", depth=1)
        for v in verts:
            assert not v.get("_id", "").startswith("communities/")


# ---------------------------------------------------------------------------
# Shortest path
# ---------------------------------------------------------------------------

@skip_if_no_arango
class TestShortestPath:
    def test_finds_direct_path(self, store):
        entity_rows = [
            {"id": "ent-a", "title": "Alice", "degree": 1},
            {"id": "ent-b", "title": "Bob",   "degree": 1},
        ]
        store.upsert_entities(entity_rows)
        store.upsert_relationships(
            [{"id": "rel-0", "source": "Alice", "target": "Bob", "weight": 1.0}],
            title_to_id={"Alice": "ent-a", "Bob": "ent-b"},
        )

        path = store.shortest_path("ent-a", "ent-b")
        titles = [v["title"] for v in path]
        assert "Alice" in titles
        assert "Bob" in titles

    def test_returns_empty_for_disconnected_nodes(self, store):
        entity_rows = [
            {"id": "ent-x", "title": "X", "degree": 1},
            {"id": "ent-y", "title": "Y", "degree": 1},
        ]
        store.upsert_entities(entity_rows)
        # No relationships — nodes are disconnected

        path = store.shortest_path("ent-x", "ent-y")
        assert path == []


# ---------------------------------------------------------------------------
# Community report retrieval
# ---------------------------------------------------------------------------

@skip_if_no_arango
class TestGetCommunityReportsForEntities:
    def test_retrieves_report_via_membership_edge(self, store):
        # Entity
        entity_rows = [{"id": "ent-0", "title": "E0", "degree": 1, "community_ids": ["0"], "text_unit_ids": []}]
        store.upsert_entities(entity_rows)

        # Community (integer id = "0", uuid = "c-uuid-0")
        community_rows = [{"id": "c-uuid-0", "community": "0", "title": "C0"}]
        _, mapping = store.upsert_communities(community_rows)
        store.upsert_entity_community_edges(entity_rows, mapping)

        # Community report (references community integer id)
        report_rows = [{"id": "rep-0", "community": "0", "title": "Report", "summary": "s", "full_content": "fc", "rank": 1.0}]
        store.upsert_community_reports(report_rows)

        reports = store.get_community_reports_for_entities(["ent-0"])
        assert len(reports) == 1
        assert reports[0]["title"] == "Report"

    def test_returns_empty_for_entity_without_community(self, store):
        entity_rows = [{"id": "ent-z", "title": "Z", "degree": 1}]
        store.upsert_entities(entity_rows)

        reports = store.get_community_reports_for_entities(["ent-z"])
        assert reports == []
