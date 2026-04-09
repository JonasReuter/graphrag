# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Unit tests for ArangoDBGraphStore.

All ArangoDB I/O is mocked — no running database required.
"""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

from graphrag_vectors.arangodb_graph import ArangoDBGraphStore


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_store(**kwargs) -> ArangoDBGraphStore:
    return ArangoDBGraphStore(
        url="http://localhost:8529",
        username="root",
        password="",
        db_name="graphrag_test",
        graph_name="knowledge_graph",
        batch_size=3,      # small batch to exercise batching logic
        vector_size=4,
        **kwargs,
    )


def _attach_mock_db(store: ArangoDBGraphStore) -> MagicMock:
    """Attach a MagicMock as _db and return it."""
    db = MagicMock()
    store._db = db
    return db


def _make_entity_rows(n: int = 3) -> list[dict]:
    return [
        {
            "id": f"ent-{i}",
            "title": f"Entity{i}",
            "type": "person",
            "description": f"Description {i}",
            "degree": i + 1,
            "community_ids": [str(i)],
            "text_unit_ids": [f"tu-{i}"],
        }
        for i in range(n)
    ]


def _make_relationship_rows(n: int = 2) -> list[dict]:
    return [
        {
            "id": f"rel-{i}",
            "source": f"Entity{i}",
            "target": f"Entity{i+1}",
            "weight": 1.0,
            "description": f"Rel {i}",
            "combined_degree": i + 2,
        }
        for i in range(n)
    ]


def _make_community_rows(n: int = 2) -> list[dict]:
    return [
        {
            "id": f"comm-uuid-{i}",
            "community": str(i),   # integer cluster ID
            "title": f"Community {i}",
            "level": "0",
        }
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# connect()
# ---------------------------------------------------------------------------

class TestConnect:
    def test_creates_database_if_missing(self):
        store = _make_store()
        mock_client = MagicMock()
        mock_sys_db = MagicMock()
        mock_sys_db.has_database.return_value = False
        mock_client.db.return_value = mock_sys_db

        # ArangoClient is imported inside connect(), so patch arango directly
        with patch("arango.ArangoClient", return_value=mock_client):
            store.connect()

        mock_sys_db.create_database.assert_called_once_with("graphrag_test")

    def test_skips_create_if_database_exists(self):
        store = _make_store()
        mock_client = MagicMock()
        mock_sys_db = MagicMock()
        mock_sys_db.has_database.return_value = True
        mock_client.db.return_value = mock_sys_db

        with patch("arango.ArangoClient", return_value=mock_client):
            store.connect()

        mock_sys_db.create_database.assert_not_called()


# ---------------------------------------------------------------------------
# setup_graph()
# ---------------------------------------------------------------------------

class TestSetupGraph:
    def test_creates_missing_collections_and_graph(self):
        store = _make_store()
        db = _attach_mock_db(store)
        db.has_collection.return_value = False
        db.has_graph.return_value = False

        store.setup_graph()

        # 5 document + 4 edge collections
        assert db.create_collection.call_count == 9
        db.create_graph.assert_called_once()

    def test_idempotent_when_all_exist(self):
        store = _make_store()
        db = _attach_mock_db(store)
        db.has_collection.return_value = True
        db.has_graph.return_value = True

        store.setup_graph()

        db.create_collection.assert_not_called()
        db.create_graph.assert_not_called()

    def test_edge_definitions_include_all_three_edge_collections(self):
        store = _make_store()
        db = _attach_mock_db(store)
        db.has_collection.return_value = False
        db.has_graph.return_value = False

        store.setup_graph()

        call_kwargs = db.create_graph.call_args
        edge_defs = call_kwargs[1]["edge_definitions"] if call_kwargs[1] else call_kwargs[0][1]
        edge_collections = {ed["edge_collection"] for ed in edge_defs}
        assert edge_collections == {
            "relationships",
            "entity_community_membership",
            "entity_text_unit",
            "entity_covariate",
        }


# ---------------------------------------------------------------------------
# upsert_entities()
# ---------------------------------------------------------------------------

class TestUpsertEntities:
    def test_uses_id_as_key(self):
        store = _make_store()
        db = _attach_mock_db(store)
        coll = MagicMock()
        db.collection.return_value = coll

        rows = [{"id": "abc-123", "title": "Foo", "degree": 2}]
        store.upsert_entities(rows)

        coll.import_bulk.assert_called_once()
        batch = coll.import_bulk.call_args[0][0]
        assert batch[0]["_key"] == "abc-123"
        assert batch[0]["title"] == "Foo"

    def test_attaches_vector_when_provided(self):
        store = _make_store()
        db = _attach_mock_db(store)
        coll = MagicMock()
        db.collection.return_value = coll

        rows = [{"id": "ent-1", "title": "Bar", "degree": 1}]
        vector_map = {"ent-1": [0.1, 0.2, 0.3, 0.4]}
        store.upsert_entities(rows, vector_map=vector_map)

        batch = coll.import_bulk.call_args[0][0]
        assert batch[0]["vector"] == [0.1, 0.2, 0.3, 0.4]

    def test_omits_vector_when_id_not_in_map(self):
        store = _make_store()
        db = _attach_mock_db(store)
        coll = MagicMock()
        db.collection.return_value = coll

        rows = [{"id": "ent-1", "title": "Bar", "degree": 1}]
        store.upsert_entities(rows, vector_map={"other-id": [0.1, 0.2, 0.3, 0.4]})

        batch = coll.import_bulk.call_args[0][0]
        assert "vector" not in batch[0]

    def test_batches_by_batch_size(self):
        store = _make_store()   # batch_size=3
        db = _attach_mock_db(store)
        coll = MagicMock()
        db.collection.return_value = coll

        rows = _make_entity_rows(7)
        store.upsert_entities(rows)

        # ceil(7 / 3) = 3 calls
        assert coll.import_bulk.call_count == 3

    def test_returns_count(self):
        store = _make_store()
        db = _attach_mock_db(store)
        db.collection.return_value = MagicMock()

        rows = _make_entity_rows(5)
        count = store.upsert_entities(rows)
        assert count == 5


# ---------------------------------------------------------------------------
# upsert_relationships()
# ---------------------------------------------------------------------------

class TestUpsertRelationships:
    def _title_to_id(self) -> dict:
        return {f"Entity{i}": f"ent-{i}" for i in range(4)}

    def test_sets_from_and_to(self):
        store = _make_store()
        db = _attach_mock_db(store)
        coll = MagicMock()
        db.collection.return_value = coll

        rows = [{"id": "rel-0", "source": "Entity0", "target": "Entity1", "weight": 1.0}]
        store.upsert_relationships(rows, title_to_id=self._title_to_id())

        batch = coll.import_bulk.call_args[0][0]
        assert batch[0]["_from"] == "entities/ent-0"
        assert batch[0]["_to"] == "entities/ent-1"

    def test_uses_relationship_uuid_as_key(self):
        store = _make_store()
        db = _attach_mock_db(store)
        coll = MagicMock()
        db.collection.return_value = coll

        rows = [{"id": "rel-uuid-999", "source": "Entity0", "target": "Entity1", "weight": 1.0}]
        store.upsert_relationships(rows, title_to_id=self._title_to_id())

        batch = coll.import_bulk.call_args[0][0]
        assert batch[0]["_key"] == "rel-uuid-999"

    def test_skips_unresolvable_source(self):
        store = _make_store()
        db = _attach_mock_db(store)
        coll = MagicMock()
        db.collection.return_value = coll

        rows = [{"id": "rel-0", "source": "UNKNOWN", "target": "Entity1", "weight": 1.0}]
        count = store.upsert_relationships(rows, title_to_id=self._title_to_id())

        assert count == 0
        coll.import_bulk.assert_not_called()

    def test_skips_unresolvable_target(self):
        store = _make_store()
        db = _attach_mock_db(store)
        coll = MagicMock()
        db.collection.return_value = coll

        rows = [{"id": "rel-0", "source": "Entity0", "target": "UNKNOWN", "weight": 1.0}]
        count = store.upsert_relationships(rows, title_to_id=self._title_to_id())

        assert count == 0

    def test_partial_success_skips_only_bad_rows(self):
        store = _make_store()
        db = _attach_mock_db(store)
        coll = MagicMock()
        db.collection.return_value = coll

        rows = [
            {"id": "rel-0", "source": "Entity0", "target": "Entity1", "weight": 1.0},
            {"id": "rel-1", "source": "MISSING", "target": "Entity2", "weight": 1.0},
            {"id": "rel-2", "source": "Entity2", "target": "Entity3", "weight": 1.0},
        ]
        count = store.upsert_relationships(rows, title_to_id=self._title_to_id())

        assert count == 2
        upserted_keys = {d["_key"] for d in coll.import_bulk.call_args[0][0]}
        assert "rel-0" in upserted_keys
        assert "rel-2" in upserted_keys
        assert "rel-1" not in upserted_keys


# ---------------------------------------------------------------------------
# upsert_communities()
# ---------------------------------------------------------------------------

class TestUpsertCommunities:
    def test_returns_int_to_uuid_map(self):
        store = _make_store()
        db = _attach_mock_db(store)
        db.collection.return_value = MagicMock()

        rows = _make_community_rows(3)
        count, mapping = store.upsert_communities(rows)

        assert count == 3
        assert mapping == {
            "0": "comm-uuid-0",
            "1": "comm-uuid-1",
            "2": "comm-uuid-2",
        }

    def test_uses_uuid_as_key_not_int_id(self):
        store = _make_store()
        db = _attach_mock_db(store)
        coll = MagicMock()
        db.collection.return_value = coll

        rows = [{"id": "comm-uuid-99", "community": "42", "title": "C"}]
        store.upsert_communities(rows)

        batch = coll.import_bulk.call_args[0][0]
        assert batch[0]["_key"] == "comm-uuid-99"


# ---------------------------------------------------------------------------
# upsert_entity_community_edges()
# ---------------------------------------------------------------------------

class TestUpsertEntityCommunityEdges:
    def test_resolves_integer_ids_to_uuid(self):
        store = _make_store()
        db = _attach_mock_db(store)
        coll = MagicMock()
        db.collection.return_value = coll

        entity_rows = [{"id": "ent-0", "community_ids": ["5"], "text_unit_ids": []}]
        community_int_to_uuid = {"5": "comm-uuid-5"}

        store.upsert_entity_community_edges(entity_rows, community_int_to_uuid)

        batch = coll.import_bulk.call_args[0][0]
        assert len(batch) == 1
        assert batch[0]["_from"] == "entities/ent-0"
        assert batch[0]["_to"] == "communities/comm-uuid-5"

    def test_skips_unresolvable_community_id(self):
        store = _make_store()
        db = _attach_mock_db(store)
        coll = MagicMock()
        db.collection.return_value = coll

        entity_rows = [{"id": "ent-0", "community_ids": ["99"], "text_unit_ids": []}]
        store.upsert_entity_community_edges(entity_rows, community_int_to_uuid={})

        # No edges created — batch never sent
        coll.import_bulk.assert_not_called()

    def test_edge_key_is_deterministic(self):
        store = _make_store()
        db = _attach_mock_db(store)
        coll = MagicMock()
        db.collection.return_value = coll

        entity_rows = [{"id": "ent-A", "community_ids": ["7"], "text_unit_ids": []}]
        store.upsert_entity_community_edges(entity_rows, {"7": "comm-uuid-7"})

        batch = coll.import_bulk.call_args[0][0]
        assert batch[0]["_key"] == "ent-A--7"

    def test_handles_list_community_ids(self):
        store = _make_store()
        db = _attach_mock_db(store)
        coll = MagicMock()
        db.collection.return_value = coll

        entity_rows = [{"id": "ent-0", "community_ids": ["0", "1"], "text_unit_ids": []}]
        mapping = {"0": "comm-uuid-0", "1": "comm-uuid-1"}
        store.upsert_entity_community_edges(entity_rows, mapping)

        batch = coll.import_bulk.call_args[0][0]
        assert len(batch) == 2


# ---------------------------------------------------------------------------
# upsert_entity_text_unit_edges()
# ---------------------------------------------------------------------------

class TestUpsertEntityTextUnitEdges:
    def test_creates_edge_per_text_unit(self):
        store = _make_store()
        db = _attach_mock_db(store)
        coll = MagicMock()
        db.collection.return_value = coll

        entity_rows = [{"id": "ent-X", "community_ids": [], "text_unit_ids": ["tu-1", "tu-2"]}]
        store.upsert_entity_text_unit_edges(entity_rows)

        batch = coll.import_bulk.call_args[0][0]
        froms = {e["_from"] for e in batch}
        tos = {e["_to"] for e in batch}
        assert froms == {"entities/ent-X"}
        assert tos == {"text_units/tu-1", "text_units/tu-2"}

    def test_empty_text_unit_ids_produces_no_edges(self):
        store = _make_store()
        db = _attach_mock_db(store)
        coll = MagicMock()
        db.collection.return_value = coll

        entity_rows = [{"id": "ent-X", "community_ids": [], "text_unit_ids": []}]
        count = store.upsert_entity_text_unit_edges(entity_rows)

        assert count == 0
        coll.import_bulk.assert_not_called()


# ---------------------------------------------------------------------------
# traverse_neighbors() — AQL is mocked via _db.aql.execute
# ---------------------------------------------------------------------------

class TestTraverseNeighbors:
    def test_returns_vertices_and_edges(self):
        store = _make_store()
        db = _attach_mock_db(store)
        db.aql.execute.return_value = iter([
            {"vertex": {"id": "v1", "title": "A"}, "edge": {"id": "e1", "source": "A", "target": "B"}},
            {"vertex": {"id": "v2", "title": "B"}, "edge": None},
        ])

        verts, edges = store.traverse_neighbors("ent-0", depth=2)

        assert len(verts) == 2
        assert len(edges) == 1   # None edges filtered out
        assert verts[0]["title"] == "A"

    def test_passes_depth_and_start_to_aql(self):
        store = _make_store()
        db = _attach_mock_db(store)
        db.aql.execute.return_value = iter([])

        store.traverse_neighbors("my-entity-id", depth=3, direction="OUTBOUND")

        call_kwargs = db.aql.execute.call_args[1]
        bind_vars = call_kwargs["bind_vars"]
        assert bind_vars["depth"] == 3
        assert bind_vars["start"] == "entities/my-entity-id"
        assert bind_vars["graph"] == "knowledge_graph"


# ---------------------------------------------------------------------------
# shortest_path()
# ---------------------------------------------------------------------------

class TestShortestPath:
    def test_returns_vertex_list(self):
        store = _make_store()
        db = _attach_mock_db(store)
        db.aql.execute.return_value = iter([
            {"id": "v1", "title": "A"},
            {"id": "v2", "title": "B"},
        ])

        path = store.shortest_path("src-id", "dst-id")

        assert len(path) == 2
        assert path[0]["title"] == "A"

    def test_passes_correct_bind_vars(self):
        store = _make_store()
        db = _attach_mock_db(store)
        db.aql.execute.return_value = iter([])

        store.shortest_path("src-123", "dst-456")

        bind_vars = db.aql.execute.call_args[1]["bind_vars"]
        assert bind_vars["src"] == "entities/src-123"
        assert bind_vars["dst"] == "entities/dst-456"
        assert bind_vars["graph"] == "knowledge_graph"


# ---------------------------------------------------------------------------
# hybrid_search() — fallback on error 1555
# ---------------------------------------------------------------------------

class TestHybridSearch:
    def test_returns_vertices_and_edges_on_success(self):
        store = _make_store()
        db = _attach_mock_db(store)
        db.aql.execute.return_value = iter([
            {"vertex": {"id": "v1", "title": "A"}, "edge": {"id": "e1"}},
        ])

        verts, edges = store.hybrid_search([0.1, 0.2, 0.3, 0.4], depth=2, k=5)

        assert len(verts) == 1
        assert len(edges) == 1

    def test_falls_back_on_approx_near_error(self):
        store = _make_store()
        db = _attach_mock_db(store)

        # First call (hybrid AQL) raises error 1555, second call (fallback traverse) succeeds
        db.aql.execute.side_effect = [
            Exception("Error 1555: vector index not ready"),
            iter([{"id": "v1", "title": "A", "vector": [0.1, 0.2, 0.3, 0.4]}]),   # fetch all
            iter([]),    # traverse from seed
        ]

        verts, edges = store.hybrid_search([0.1, 0.2, 0.3, 0.4], depth=1, k=1)

        # Fallback executed without raising
        assert isinstance(verts, list)
        assert isinstance(edges, list)

    def test_hybrid_search_includes_seeds(self):
        """Seed entities must be included in the returned vertices, not just their neighbors."""
        store = _make_store()
        db = _attach_mock_db(store)
        db.aql.execute.return_value = iter([
            {"vertex": {"id": "seed-1", "title": "Seed"}, "edge": None},
            {"vertex": {"id": "neighbor-1", "title": "Neighbor"}, "edge": {"id": "e1"}},
        ])

        verts, edges = store.hybrid_search([0.1, 0.2, 0.3, 0.4], depth=1, k=1)

        titles = [v.get("title") for v in verts]
        assert "Seed" in titles
        assert "Neighbor" in titles
        assert len(edges) == 1  # null edge not counted


# ---------------------------------------------------------------------------
# get_text_units_for_entities() — AQL traversal
# ---------------------------------------------------------------------------

class TestGetTextUnitsForEntities:
    def test_returns_empty_list_for_empty_input(self):
        store = _make_store()
        db = _attach_mock_db(store)
        result = store.get_text_units_for_entities([])
        db.aql.execute.assert_not_called()
        assert result == []

    def test_returns_text_unit_docs(self):
        store = _make_store()
        db = _attach_mock_db(store)
        db.aql.execute.return_value = iter([
            {"id": "tu-1", "text": "hello world"},
            {"id": "tu-2", "text": "second unit"},
        ])
        result = store.get_text_units_for_entities(["ent-1"])
        assert len(result) == 2
        assert result[0]["id"] == "tu-1"

    def test_passes_entity_ids_and_graph_name(self):
        store = _make_store()
        db = _attach_mock_db(store)
        db.aql.execute.return_value = iter([])
        store.get_text_units_for_entities(["abc", "def"])
        _, kwargs = db.aql.execute.call_args
        bind_vars = kwargs.get("bind_vars") or db.aql.execute.call_args[0][1]
        assert bind_vars["entity_ids"] == ["abc", "def"]
        assert bind_vars["graph"] == store.graph_name


# ---------------------------------------------------------------------------
# get_covariates_for_entities() — AQL traversal
# ---------------------------------------------------------------------------

class TestGetCovariatesForEntities:
    def test_returns_empty_list_for_empty_input(self):
        store = _make_store()
        db = _attach_mock_db(store)
        result = store.get_covariates_for_entities([])
        db.aql.execute.assert_not_called()
        assert result == []

    def test_returns_covariate_docs(self):
        store = _make_store()
        db = _attach_mock_db(store)
        db.aql.execute.return_value = iter([
            {"id": "cov-1", "subject_id": "Entity A", "covariate_type": "claim"},
        ])
        result = store.get_covariates_for_entities(["ent-1"])
        assert len(result) == 1
        assert result[0]["covariate_type"] == "claim"


# ---------------------------------------------------------------------------
# upsert_covariates() — documents + edges
# ---------------------------------------------------------------------------

class TestUpsertCovariates:
    def test_upserts_docs_and_edges(self):
        store = _make_store()
        db = _attach_mock_db(store)
        coll = MagicMock()
        db.collection.return_value = coll
        title_to_id = {"Entity A": "uuid-a"}
        rows = [{"id": "cov-1", "subject_id": "Entity A", "covariate_type": "claim"}]

        count = store.upsert_covariates(rows, title_to_id)

        assert count == 1
        # Two _bulk_upsert calls: one for docs, one for edges
        assert coll.import_bulk.call_count == 2

    def test_skips_edge_when_entity_not_found(self):
        store = _make_store()
        db = _attach_mock_db(store)
        coll = MagicMock()
        db.collection.return_value = coll
        rows = [{"id": "cov-1", "subject_id": "Unknown Entity", "covariate_type": "claim"}]

        count = store.upsert_covariates(rows, title_to_id={})

        assert count == 1
        # Only one _bulk_upsert call: docs only, no edges
        assert coll.import_bulk.call_count == 1

    def test_returns_zero_for_empty_rows(self):
        store = _make_store()
        _attach_mock_db(store)
        assert store.upsert_covariates([], {}) == 0

    def test_edge_uses_entity_and_covariate_ids(self):
        store = _make_store()
        db = _attach_mock_db(store)
        captured_batches = []
        coll = MagicMock()
        coll.import_bulk.side_effect = lambda batch, **kw: captured_batches.append(batch)
        db.collection.return_value = coll

        store.upsert_covariates(
            [{"id": "cov-1", "subject_id": "Entity A", "covariate_type": "claim"}],
            title_to_id={"Entity A": "uuid-a"},
        )

        # Second batch is the edge batch
        edge_batch = captured_batches[1]
        assert edge_batch[0]["_from"] == "entities/uuid-a"
        assert edge_batch[0]["_to"] == "covariates/cov-1"


# ---------------------------------------------------------------------------
# get_all_covariates() — direct collection scan with optional filters
# ---------------------------------------------------------------------------

class TestGetAllCovariates:
    def _make_aql_result(self, rows):
        return iter(rows)

    def test_returns_all_when_no_filters(self):
        store = _make_store()
        db = _attach_mock_db(store)
        db.aql.execute.return_value = self._make_aql_result([
            {"id": "c1", "customer": "CUSTOMER A", "type": "SUPPORT_CASE"},
            {"id": "c2", "customer": "CUSTOMER B", "type": "PURCHASE"},
        ])
        result = store.get_all_covariates()
        assert len(result) == 2
        _, kwargs = db.aql.execute.call_args
        bind_vars = kwargs.get("bind_vars") or db.aql.execute.call_args[0][1]
        assert bind_vars["subject"] is None
        assert bind_vars["covariate_type"] is None
        assert bind_vars["status"] is None

    def test_subject_filter_passed_as_bind_var(self):
        store = _make_store()
        db = _attach_mock_db(store)
        db.aql.execute.return_value = iter([])
        store.get_all_covariates(subject="CUSTOMER A")
        _, kwargs = db.aql.execute.call_args
        bind_vars = kwargs.get("bind_vars") or db.aql.execute.call_args[0][1]
        assert bind_vars["subject"] == "CUSTOMER A"

    def test_covariate_type_filter_passed_as_bind_var(self):
        store = _make_store()
        db = _attach_mock_db(store)
        db.aql.execute.return_value = iter([])
        store.get_all_covariates(covariate_type="PURCHASE")
        _, kwargs = db.aql.execute.call_args
        bind_vars = kwargs.get("bind_vars") or db.aql.execute.call_args[0][1]
        assert bind_vars["covariate_type"] == "PURCHASE"

    def test_status_filter_passed_as_bind_var(self):
        store = _make_store()
        db = _attach_mock_db(store)
        db.aql.execute.return_value = iter([])
        store.get_all_covariates(status="CONFIRMED")
        _, kwargs = db.aql.execute.call_args
        bind_vars = kwargs.get("bind_vars") or db.aql.execute.call_args[0][1]
        assert bind_vars["status"] == "CONFIRMED"

    def test_returns_list_of_dicts(self):
        store = _make_store()
        db = _attach_mock_db(store)
        row = {
            "id": "c1", "customer": "CUSTOMER B", "type": "PURCHASE",
            "status": "CONFIRMED", "start_date": "2025-01-24T00:00:00",
            "end_date": "2025-01-24T00:00:00", "description": "Purchased K6D40",
            "source_text": "...",
        }
        db.aql.execute.return_value = iter([row])
        result = store.get_all_covariates()
        assert result == [row]
