# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Unit tests for ArangoDBTableProvider and ArangoDBTable.

All ArangoDB I/O is mocked — no running database required.
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock, call, patch

import pandas as pd
import pytest

from graphrag_storage.tables.arango_table import (
    ArangoDBTable,
    _sanitize,
    _sanitize_doc,
    _strip_meta,
)
from graphrag_storage.tables.arango_table_provider import ArangoDBTableProvider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_provider(prefix: str = "pipeline_") -> tuple[ArangoDBTableProvider, MagicMock]:
    """Return a provider with a fully mocked ArangoDB database."""
    mock_db = MagicMock()
    mock_sys_db = MagicMock()
    mock_client = MagicMock()
    mock_sys_db.has_database.return_value = True
    mock_client.db.return_value = mock_sys_db

    with patch("arango.ArangoClient", return_value=mock_client):
        provider = ArangoDBTableProvider(
            url="http://localhost:8529",
            username="root",
            password="",
            db_name="graphrag_test",
            collection_prefix=prefix,
        )

    # Override the internal _db with a fresh mock so tests are isolated
    provider._db = mock_db
    return provider, mock_db


def _make_table(db: MagicMock | None = None, *, truncate: bool = True) -> tuple[ArangoDBTable, MagicMock]:
    if db is None:
        db = MagicMock()
    table = ArangoDBTable(db, "pipeline_entities", truncate=truncate)
    return table, db


# ---------------------------------------------------------------------------
# _sanitize helpers
# ---------------------------------------------------------------------------

class TestSanitize:
    def test_nan_float_becomes_none(self):
        assert _sanitize(float("nan")) is None

    def test_nested_nan_in_list(self):
        result = _sanitize([1.0, float("nan"), 3.0])
        assert result[1] is None

    def test_nested_nan_in_dict(self):
        result = _sanitize({"a": float("nan"), "b": 42})
        assert result["a"] is None
        assert result["b"] == 42

    def test_plain_types_unchanged(self):
        assert _sanitize(42) == 42
        assert _sanitize("hello") == "hello"
        assert _sanitize(True) is True

    def test_numpy_int_converted(self):
        try:
            import numpy as np
            assert isinstance(_sanitize(np.int64(7)), int)
            assert _sanitize(np.int64(7)) == 7
        except ImportError:
            pytest.skip("numpy not available")

    def test_numpy_float_nan_becomes_none(self):
        try:
            import numpy as np
            assert _sanitize(np.float64("nan")) is None
        except ImportError:
            pytest.skip("numpy not available")

    def test_numpy_array_converted_to_list(self):
        try:
            import numpy as np
            result = _sanitize(np.array([1, 2, 3]))
            assert isinstance(result, list)
            assert result == [1, 2, 3]
        except ImportError:
            pytest.skip("numpy not available")


class TestSanitizeDoc:
    def test_sets_key_from_id(self):
        doc = {"id": "abc-123", "title": "Foo"}
        result = _sanitize_doc(doc)
        assert result["_key"] == "abc-123"

    def test_strips_existing_arango_meta_fields(self):
        doc = {"_id": "col/key", "_rev": "xxx", "_key": "key", "id": "uuid", "title": "X"}
        result = _sanitize_doc(doc)
        assert "_id" not in result
        assert "_rev" not in result
        # _key is SET from id, not copied from input
        assert result["_key"] == "uuid"

    def test_no_id_field_produces_no_key(self):
        doc = {"name": "no-id-doc"}
        result = _sanitize_doc(doc)
        assert "_key" not in result


class TestStripMeta:
    def test_removes_arango_system_fields(self):
        doc = {"_id": "x/y", "_rev": "r", "_key": "k", "id": "uuid", "title": "T"}
        result = _strip_meta(doc)
        assert "_id" not in result
        assert "_rev" not in result
        assert "_key" not in result
        assert result["id"] == "uuid"
        assert result["title"] == "T"

    def test_passthrough_when_no_meta(self):
        doc = {"id": "abc", "value": 42}
        assert _strip_meta(doc) == doc


# ---------------------------------------------------------------------------
# ArangoDBTableProvider — __init__
# ---------------------------------------------------------------------------

class TestArangoDBTableProviderInit:
    def test_creates_database_when_missing(self):
        mock_client = MagicMock()
        mock_sys_db = MagicMock()
        mock_sys_db.has_database.return_value = False
        mock_client.db.return_value = mock_sys_db

        with patch("arango.ArangoClient", return_value=mock_client):
            ArangoDBTableProvider(db_name="new_db")

        mock_sys_db.create_database.assert_called_once_with("new_db")

    def test_skips_create_when_database_exists(self):
        mock_client = MagicMock()
        mock_sys_db = MagicMock()
        mock_sys_db.has_database.return_value = True
        mock_client.db.return_value = mock_sys_db

        with patch("arango.ArangoClient", return_value=mock_client):
            ArangoDBTableProvider()

        mock_sys_db.create_database.assert_not_called()

    def test_custom_prefix_stored(self):
        provider, _ = _make_provider(prefix="myapp_")
        assert provider._prefix == "myapp_"

    def test_empty_prefix_accepted(self):
        provider, _ = _make_provider(prefix="")
        assert provider._prefix == ""


# ---------------------------------------------------------------------------
# ArangoDBTableProvider — read_dataframe
# ---------------------------------------------------------------------------

class TestReadDataframe:
    def test_strips_arango_meta_and_returns_dataframe(self):
        provider, db = _make_provider()
        db.has_collection.return_value = True
        db.aql.execute.return_value = iter([
            {"_key": "uuid-1", "_id": "pipeline_entities/uuid-1", "_rev": "r1",
             "id": "uuid-1", "title": "Entity A", "degree": 3},
        ])

        df = provider.read_dataframe.__wrapped__(provider, "entities") if hasattr(
            provider.read_dataframe, "__wrapped__") else None

        # Use sync call via asyncio since the method is async
        import asyncio
        df = asyncio.get_event_loop().run_until_complete(provider.read_dataframe("entities"))

        assert "_key" not in df.columns
        assert "_id" not in df.columns
        assert "id" in df.columns
        assert df.loc[0, "title"] == "Entity A"

    def test_raises_when_collection_missing(self):
        import asyncio

        provider, db = _make_provider()
        db.has_collection.return_value = False

        with pytest.raises(ValueError, match="does not exist"):
            asyncio.get_event_loop().run_until_complete(
                provider.read_dataframe("entities")
            )

    def test_returns_empty_dataframe_when_collection_is_empty(self):
        import asyncio

        provider, db = _make_provider()
        db.has_collection.return_value = True
        db.aql.execute.return_value = iter([])

        df = asyncio.get_event_loop().run_until_complete(provider.read_dataframe("entities"))

        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_queries_correct_collection_name(self):
        import asyncio

        provider, db = _make_provider(prefix="p_")
        db.has_collection.return_value = True
        db.aql.execute.return_value = iter([])

        asyncio.get_event_loop().run_until_complete(provider.read_dataframe("entities"))

        call_kwargs = db.aql.execute.call_args[1]
        assert call_kwargs["bind_vars"]["@col"] == "p_entities"


# ---------------------------------------------------------------------------
# ArangoDBTableProvider — write_dataframe
# ---------------------------------------------------------------------------

class TestWriteDataframe:
    def test_truncates_and_bulk_inserts(self):
        import asyncio

        provider, db = _make_provider()
        coll = MagicMock()
        db.has_collection.return_value = True
        db.collection.return_value = coll

        df = pd.DataFrame({"id": ["a", "b"], "title": ["A", "B"]})
        asyncio.get_event_loop().run_until_complete(provider.write_dataframe("entities", df))

        coll.truncate.assert_called_once()
        coll.import_bulk.assert_called_once()

    def test_creates_collection_when_missing(self):
        import asyncio

        provider, db = _make_provider()
        db.has_collection.return_value = False
        coll = MagicMock()
        db.collection.return_value = coll

        df = pd.DataFrame({"id": ["a"]})
        asyncio.get_event_loop().run_until_complete(provider.write_dataframe("entities", df))

        db.create_collection.assert_called_once_with("pipeline_entities")

    def test_skips_import_for_empty_dataframe(self):
        import asyncio

        provider, db = _make_provider()
        coll = MagicMock()
        db.has_collection.return_value = True
        db.collection.return_value = coll

        asyncio.get_event_loop().run_until_complete(
            provider.write_dataframe("entities", pd.DataFrame())
        )

        coll.import_bulk.assert_not_called()

    def test_sets_key_from_id_in_inserted_docs(self):
        import asyncio

        provider, db = _make_provider()
        coll = MagicMock()
        db.has_collection.return_value = True
        db.collection.return_value = coll

        df = pd.DataFrame({"id": ["uuid-42"], "title": ["Foo"]})
        asyncio.get_event_loop().run_until_complete(provider.write_dataframe("entities", df))

        inserted_batch = coll.import_bulk.call_args[0][0]
        assert inserted_batch[0]["_key"] == "uuid-42"

    def test_large_df_batched(self):
        import asyncio
        from graphrag_storage.tables.arango_table_provider import _BATCH_SIZE

        provider, db = _make_provider()
        coll = MagicMock()
        db.has_collection.return_value = True
        db.collection.return_value = coll

        n = _BATCH_SIZE + 1
        df = pd.DataFrame({"id": [f"id-{i}" for i in range(n)]})
        asyncio.get_event_loop().run_until_complete(provider.write_dataframe("entities", df))

        assert coll.import_bulk.call_count == 2


# ---------------------------------------------------------------------------
# ArangoDBTableProvider — has
# ---------------------------------------------------------------------------

class TestHas:
    def test_returns_false_when_collection_missing(self):
        import asyncio

        provider, db = _make_provider()
        db.has_collection.return_value = False

        result = asyncio.get_event_loop().run_until_complete(provider.has("entities"))
        assert result is False

    def test_returns_false_when_collection_empty(self):
        import asyncio

        provider, db = _make_provider()
        db.has_collection.return_value = True
        db.aql.execute.return_value = iter([False])

        result = asyncio.get_event_loop().run_until_complete(provider.has("entities"))
        assert result is False

    def test_returns_true_when_collection_exists_and_nonempty(self):
        import asyncio

        provider, db = _make_provider()
        db.has_collection.return_value = True
        db.aql.execute.return_value = iter([True])

        result = asyncio.get_event_loop().run_until_complete(provider.has("entities"))
        assert result is True


# ---------------------------------------------------------------------------
# ArangoDBTableProvider — list
# ---------------------------------------------------------------------------

class TestList:
    def test_lists_only_prefixed_collections(self):
        provider, db = _make_provider(prefix="pipeline_")
        db.collections.return_value = [
            {"name": "pipeline_entities", "system": False},
            {"name": "pipeline_communities", "system": False},
            {"name": "unrelated_collection", "system": False},
            {"name": "_system_internal", "system": True},
        ]

        result = provider.list()
        assert set(result) == {"entities", "communities"}

    def test_empty_prefix_returns_all_non_system(self):
        provider, db = _make_provider(prefix="")
        db.collections.return_value = [
            {"name": "entities", "system": False},
            {"name": "_internal", "system": True},
        ]

        result = provider.list()
        assert result == ["entities"]


# ---------------------------------------------------------------------------
# ArangoDBTableProvider — open
# ---------------------------------------------------------------------------

class TestOpen:
    def test_returns_arango_table_instance(self):
        provider, db = _make_provider()
        db.has_collection.return_value = True

        table = provider.open("entities")
        assert isinstance(table, ArangoDBTable)

    def test_creates_collection_if_missing(self):
        provider, db = _make_provider()
        db.has_collection.return_value = False

        provider.open("entities")
        db.create_collection.assert_called_once_with("pipeline_entities")

    def test_passes_truncate_flag_to_table(self):
        provider, db = _make_provider()
        db.has_collection.return_value = True

        table_truncate = provider.open("entities", truncate=True)
        table_append = provider.open("entities", truncate=False)

        assert table_truncate._truncate is True
        assert table_append._truncate is False


# ---------------------------------------------------------------------------
# ArangoDBTable — __aiter__
# ---------------------------------------------------------------------------

class TestArangoDBTableIter:
    def test_strips_meta_fields_on_iteration(self):
        import asyncio

        table, db = _make_table()
        db.aql.execute.return_value = iter([
            {"_key": "k1", "_id": "col/k1", "_rev": "r1", "id": "uuid-1", "title": "A"},
        ])

        async def collect():
            return [row async for row in table]

        rows = asyncio.get_event_loop().run_until_complete(collect())
        assert len(rows) == 1
        assert "_key" not in rows[0]
        assert rows[0]["title"] == "A"

    def test_applies_transformer(self):
        import asyncio

        class EntityModel:
            def __init__(self, id, title, **kwargs):
                self.id = id
                self.title = title

        table, db = _make_table()
        table._transformer = EntityModel
        db.aql.execute.return_value = iter([
            {"id": "uuid-1", "title": "Test"},
        ])

        async def collect():
            return [row async for row in table]

        rows = asyncio.get_event_loop().run_until_complete(collect())
        assert isinstance(rows[0], EntityModel)
        assert rows[0].title == "Test"

    def test_queries_correct_collection(self):
        import asyncio

        table, db = _make_table()
        db.aql.execute.return_value = iter([])

        async def drain():
            async for _ in table:
                pass

        asyncio.get_event_loop().run_until_complete(drain())

        call_kwargs = db.aql.execute.call_args[1]
        assert call_kwargs["bind_vars"]["@col"] == "pipeline_entities"


# ---------------------------------------------------------------------------
# ArangoDBTable — length
# ---------------------------------------------------------------------------

class TestArangoDBTableLength:
    def test_returns_collection_count(self):
        import asyncio

        table, db = _make_table()
        db.aql.execute.return_value = iter([42])

        length = asyncio.get_event_loop().run_until_complete(table.length())
        assert length == 42

    def test_returns_zero_on_empty_cursor(self):
        import asyncio

        table, db = _make_table()
        db.aql.execute.return_value = iter([])

        length = asyncio.get_event_loop().run_until_complete(table.length())
        assert length == 0


# ---------------------------------------------------------------------------
# ArangoDBTable — has
# ---------------------------------------------------------------------------

class TestArangoDBTableHas:
    def test_returns_true_when_row_exists(self):
        import asyncio

        table, db = _make_table()
        db.aql.execute.return_value = iter([True])

        result = asyncio.get_event_loop().run_until_complete(table.has("uuid-1"))
        assert result is True

    def test_returns_false_when_row_missing(self):
        import asyncio

        table, db = _make_table()
        db.aql.execute.return_value = iter([False])

        result = asyncio.get_event_loop().run_until_complete(table.has("no-such-id"))
        assert result is False


# ---------------------------------------------------------------------------
# ArangoDBTable — write + close (truncate=True)
# ---------------------------------------------------------------------------

class TestArangoDBTableWriteTruncate:
    def test_truncates_collection_on_first_write(self):
        import asyncio

        table, db = _make_table(truncate=True)
        coll = MagicMock()
        db.collection.return_value = coll

        async def run():
            await table.write({"id": "row-1", "title": "A"})

        asyncio.get_event_loop().run_until_complete(run())
        coll.truncate.assert_called_once()

    def test_truncates_only_once_for_multiple_writes(self):
        import asyncio

        table, db = _make_table(truncate=True)
        coll = MagicMock()
        db.collection.return_value = coll

        async def run():
            await table.write({"id": "row-1"})
            await table.write({"id": "row-2"})
            await table.close()

        asyncio.get_event_loop().run_until_complete(run())
        coll.truncate.assert_called_once()

    def test_close_flushes_pending_rows(self):
        import asyncio

        table, db = _make_table(truncate=True)
        coll = MagicMock()
        db.collection.return_value = coll

        async def run():
            await table.write({"id": "row-1", "title": "A"})
            await table.write({"id": "row-2", "title": "B"})
            await table.close()

        asyncio.get_event_loop().run_until_complete(run())
        coll.import_bulk.assert_called_once()
        batch = coll.import_bulk.call_args[0][0]
        assert len(batch) == 2

    def test_keys_set_from_id_in_written_rows(self):
        import asyncio

        table, db = _make_table(truncate=True)
        coll = MagicMock()
        db.collection.return_value = coll

        async def run():
            await table.write({"id": "uuid-99", "title": "X"})
            await table.close()

        asyncio.get_event_loop().run_until_complete(run())
        batch = coll.import_bulk.call_args[0][0]
        assert batch[0]["_key"] == "uuid-99"


# ---------------------------------------------------------------------------
# ArangoDBTable — write + close (truncate=False / append mode)
# ---------------------------------------------------------------------------

class TestArangoDBTableWriteAppend:
    def test_does_not_truncate_collection(self):
        import asyncio

        table, db = _make_table(truncate=False)
        coll = MagicMock()
        db.collection.return_value = coll

        async def run():
            await table.write({"id": "row-1"})
            await table.close()

        asyncio.get_event_loop().run_until_complete(run())
        coll.truncate.assert_not_called()


# ---------------------------------------------------------------------------
# ArangoDBTable — batching
# ---------------------------------------------------------------------------

class TestArangoDBTableBatching:
    def test_auto_flushes_at_batch_size(self):
        import asyncio
        from graphrag_storage.tables.arango_table import _BATCH_SIZE

        table, db = _make_table(truncate=False)
        coll = MagicMock()
        db.collection.return_value = coll

        async def run():
            for i in range(_BATCH_SIZE + 1):
                await table.write({"id": f"row-{i}"})
            # At this point the first batch should have been flushed
            # but the +1 row is still pending

        asyncio.get_event_loop().run_until_complete(run())
        # One batch flushed mid-stream, one pending (not yet closed)
        assert coll.import_bulk.call_count == 1

    def test_close_flushes_remainder(self):
        import asyncio
        from graphrag_storage.tables.arango_table import _BATCH_SIZE

        table, db = _make_table(truncate=False)
        coll = MagicMock()
        db.collection.return_value = coll

        async def run():
            for i in range(_BATCH_SIZE + 1):
                await table.write({"id": f"row-{i}"})
            await table.close()

        asyncio.get_event_loop().run_until_complete(run())
        assert coll.import_bulk.call_count == 2


# ---------------------------------------------------------------------------
# ArangoDBTable — context manager
# ---------------------------------------------------------------------------

class TestArangoDBTableContextManager:
    def test_close_called_on_exit(self):
        import asyncio

        table, db = _make_table(truncate=False)
        coll = MagicMock()
        db.collection.return_value = coll

        async def run():
            async with table:
                await table.write({"id": "row-1"})

        asyncio.get_event_loop().run_until_complete(run())
        coll.import_bulk.assert_called_once()


# ---------------------------------------------------------------------------
# TableProviderFactory integration
# ---------------------------------------------------------------------------

class TestFactoryIntegration:
    def test_factory_registers_and_creates_arangodb_provider(self):
        from graphrag_storage.tables.table_provider_factory import create_table_provider
        from graphrag_storage.tables.table_provider_config import TableProviderConfig
        from graphrag_storage.tables.table_type import TableType

        mock_client = MagicMock()
        mock_sys_db = MagicMock()
        mock_sys_db.has_database.return_value = True
        mock_client.db.return_value = mock_sys_db

        with patch("arango.ArangoClient", return_value=mock_client):
            provider = create_table_provider(
                TableProviderConfig(
                    type=TableType.ArangoDB,
                    url="http://localhost:8529",
                    username="root",
                    password="",
                    db_name="graphrag_test",
                )
            )

        assert isinstance(provider, ArangoDBTableProvider)

    def test_factory_raises_helpful_error_when_arango_missing(self):
        """When python-arango is not installed, a clear ImportError is raised."""
        from graphrag_storage.tables.table_provider_factory import (
            TableProviderFactory,
            create_table_provider,
            table_provider_factory,
        )
        from graphrag_storage.tables.table_provider_config import TableProviderConfig
        from graphrag_storage.tables.table_type import TableType

        # Deregister arangodb so the factory tries to import it fresh
        if TableType.ArangoDB in table_provider_factory:
            del table_provider_factory[TableType.ArangoDB]

        with patch.dict("sys.modules", {"arango": None}):
            with pytest.raises((ImportError, ModuleNotFoundError)):
                create_table_provider(
                    TableProviderConfig(type=TableType.ArangoDB)
                )
