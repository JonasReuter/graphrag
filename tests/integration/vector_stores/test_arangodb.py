# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Integration tests for ArangoDB vector store implementation.

Requires a running ArangoDB 3.12+ instance. Start one locally with:

    docker compose up -d

The default connection (http://localhost:8529, root, no password) is used.
Tests are skipped automatically when ArangoDB is not reachable.
"""

import numpy as np
import pytest

try:
    from arango import ArangoClient
    from arango.exceptions import ServerConnectionError

    _ARANGO_AVAILABLE = True
except ImportError:
    _ARANGO_AVAILABLE = False

from graphrag_vectors import VectorStoreDocument
from graphrag_vectors.arangodb import ArangoDBVectorStore

pytestmark = pytest.mark.skipif(
    not _ARANGO_AVAILABLE, reason="python-arango not installed"
)

_URL = "http://localhost:8529"
_USERNAME = "root"
_PASSWORD = ""
_DB = "graphrag_test"
_VECTOR_SIZE = 5


def _is_arangodb_reachable() -> bool:
    """Return True if a local ArangoDB instance is reachable."""
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
    """Create a fresh ArangoDBVectorStore for each test."""
    s = ArangoDBVectorStore(
        url=_URL,
        username=_USERNAME,
        password=_PASSWORD,
        db_name=_DB,
        index_name="test_vectors",
        vector_size=_VECTOR_SIZE,
        fields={"title": "str"},
    )
    s.connect()
    s.create_index()
    yield s
    # Teardown: drop the test collection
    try:
        if s._db.has_collection("test_vectors"):
            s._db.delete_collection("test_vectors")
        if s._db.has_database(_DB):
            client = ArangoClient(hosts=_URL)
            sys_db = client.db("_system", username=_USERNAME, password=_PASSWORD)
            sys_db.delete_database(_DB)
    except Exception:  # noqa: BLE001
        pass


@skip_if_no_arango
class TestArangoDBVectorStoreIntegration:
    def test_connect_creates_database_and_collection(self, store):
        assert store._db is not None
        assert store._collection is not None

    def test_count_empty_after_create_index(self, store):
        assert store.count() == 0

    def test_load_documents_and_count(self, store):
        docs = [
            VectorStoreDocument(id="1", vector=[0.1, 0.2, 0.3, 0.4, 0.5]),
            VectorStoreDocument(id="2", vector=[0.2, 0.3, 0.4, 0.5, 0.6]),
        ]
        store.load_documents(docs)
        assert store.count() == 2

    def test_search_by_id_returns_document(self, store):
        store.load_documents([
            VectorStoreDocument(
                id="doc1",
                vector=[0.1, 0.2, 0.3, 0.4, 0.5],
                data={"title": "Test Entity"},
            )
        ])
        result = store.search_by_id("doc1")
        assert result.id == "doc1"
        assert result.vector is not None
        assert np.allclose(result.vector, [0.1, 0.2, 0.3, 0.4, 0.5], atol=1e-5)

    def test_search_by_id_returns_metadata(self, store):
        store.load_documents([
            VectorStoreDocument(
                id="meta1",
                vector=[0.1, 0.2, 0.3, 0.4, 0.5],
                data={"title": "Entity With Meta"},
            )
        ])
        result = store.search_by_id("meta1")
        assert result.data.get("title") == "Entity With Meta"

    def test_similarity_search_returns_results(self, store):
        docs = [
            VectorStoreDocument(id="1", vector=[1.0, 0.0, 0.0, 0.0, 0.0]),
            VectorStoreDocument(id="2", vector=[0.0, 1.0, 0.0, 0.0, 0.0]),
            VectorStoreDocument(id="3", vector=[0.0, 0.0, 1.0, 0.0, 0.0]),
        ]
        store.load_documents(docs)
        results = store.similarity_search_by_vector(
            query_embedding=[1.0, 0.0, 0.0, 0.0, 0.0], k=3
        )
        assert len(results) >= 1
        assert results[0].document.id == "1"

    def test_similarity_search_score_between_minus_one_and_one(self, store):
        store.load_documents([
            VectorStoreDocument(id="1", vector=[0.1, 0.2, 0.3, 0.4, 0.5]),
        ])
        results = store.similarity_search_by_vector(
            query_embedding=[0.1, 0.2, 0.3, 0.4, 0.5], k=1
        )
        assert len(results) == 1
        assert -1.0 <= results[0].score <= 1.0

    def test_remove_deletes_documents(self, store):
        store.load_documents([
            VectorStoreDocument(id="del1", vector=[0.1, 0.2, 0.3, 0.4, 0.5]),
            VectorStoreDocument(id="del2", vector=[0.2, 0.3, 0.4, 0.5, 0.6]),
        ])
        assert store.count() == 2
        store.remove(["del1"])
        assert store.count() == 1

    def test_upsert_via_load_documents_is_idempotent(self, store):
        """Loading the same document twice must not create duplicates."""
        doc = VectorStoreDocument(id="upsert1", vector=[0.1, 0.2, 0.3, 0.4, 0.5])
        store.load_documents([doc])
        store.load_documents([doc])
        assert store.count() == 1

    def test_create_index_resets_collection(self, store):
        """create_index() must drop and recreate the collection."""
        store.load_documents([
            VectorStoreDocument(id="x", vector=[0.1, 0.2, 0.3, 0.4, 0.5]),
        ])
        assert store.count() == 1
        store.create_index()
        assert store.count() == 0

    def test_insert_single_document(self, store):
        store.insert(VectorStoreDocument(id="single", vector=[0.5, 0.5, 0.0, 0.0, 0.0]))
        assert store.count() == 1

    def test_similarity_search_without_vectors(self, store):
        """include_vectors=False must not return vectors in results."""
        store.load_documents([
            VectorStoreDocument(id="1", vector=[0.1, 0.2, 0.3, 0.4, 0.5]),
        ])
        results = store.similarity_search_by_vector(
            [0.1, 0.2, 0.3, 0.4, 0.5], k=1, include_vectors=False
        )
        assert results[0].document.vector is None
