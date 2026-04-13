# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""ArangoDB-backed TableProvider implementation.

Stores each pipeline table as a document collection in ArangoDB.
Collection names are prefixed (default: ``pipeline_``) to avoid conflicts
with the graph store collections (entities, relationships, etc.) which use
proper edge collections and UUID-keyed documents.

Usage in graphrag_config.yaml::

    output:
      type: arangodb
      url: http://localhost:8529
      username: root
      password: ""
      db_name: graphrag
      collection_prefix: "pipeline_"   # optional, default shown
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from graphrag_storage.tables.arango_table import (
    ArangoDBTable,
    _sanitize_doc,
    _strip_meta,
)
from graphrag_storage.tables.table import RowTransformer, Table
from graphrag_storage.tables.table_provider import TableProvider

logger = logging.getLogger(__name__)

_BATCH_SIZE = 500


class ArangoDBTableProvider(TableProvider):
    """TableProvider that persists pipeline tables as ArangoDB document collections.

    Each logical table name (e.g. ``entities``) maps to a collection named
    ``{collection_prefix}{table_name}`` (e.g. ``pipeline_entities``).

    Parameters
    ----------
    url:
        ArangoDB server URL (default: ``http://localhost:8529``).
    username:
        Database username (default: ``root``).
    password:
        Database password (default: ``""``).
    db_name:
        Target database name (default: ``graphrag``).
    collection_prefix:
        Prefix applied to every collection name (default: ``pipeline_``).
        Set to ``""`` to disable prefixing — only do this if you are sure
        there are no naming conflicts with graph store collections.
    **kwargs:
        Accepts and ignores ``storage``, ``type``, and other factory
        pass-throughs so the provider is compatible with ``create_table_provider``.
    """

    def __init__(
        self,
        url: str = "http://localhost:8529",
        username: str = "root",
        password: str = "",
        db_name: str = "graphrag",
        collection_prefix: str = "pipeline_",
        **kwargs: Any,
    ) -> None:
        from arango import ArangoClient

        self._prefix = collection_prefix
        client = ArangoClient(hosts=url)
        sys_db = client.db("_system", username=username, password=password)
        if not sys_db.has_database(db_name):
            sys_db.create_database(db_name)
        self._db = client.db(db_name, username=username, password=password)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _coll_name(self, table_name: str) -> str:
        return f"{self._prefix}{table_name}"

    def _ensure_collection(self, coll_name: str) -> None:
        if not self._db.has_collection(coll_name):
            self._db.create_collection(coll_name)

    def _bulk_write(self, coll_name: str, docs: list[dict]) -> None:
        coll = self._db.collection(coll_name)
        for i in range(0, len(docs), _BATCH_SIZE):
            coll.import_bulk(
                docs[i : i + _BATCH_SIZE],
                on_duplicate="replace",
                halt_on_error=True,
            )

    # ------------------------------------------------------------------
    # TableProvider interface
    # ------------------------------------------------------------------

    async def read_dataframe(self, table_name: str) -> pd.DataFrame:
        """Read a collection into a DataFrame, stripping ArangoDB metadata."""
        coll_name = self._coll_name(table_name)
        if not self._db.has_collection(coll_name):
            msg = f"ArangoDB collection '{coll_name}' does not exist."
            raise ValueError(msg)
        logger.info("reading table from ArangoDB: %s", coll_name)
        cursor = self._db.aql.execute(
            "FOR doc IN @@col RETURN doc",
            bind_vars={"@col": coll_name},
        )
        rows = [_strip_meta(doc) for doc in cursor]
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    async def write_dataframe(self, table_name: str, df: pd.DataFrame) -> None:
        """Truncate the collection and write the DataFrame as documents."""
        coll_name = self._coll_name(table_name)
        self._ensure_collection(coll_name)
        self._db.collection(coll_name).truncate()
        if df.empty:
            return
        logger.info(
            "writing table to ArangoDB: %s (%d rows)", coll_name, len(df)
        )
        docs = [_sanitize_doc(row) for row in df.to_dict("records")]
        self._bulk_write(coll_name, docs)

    async def has(self, table_name: str) -> bool:
        """Return True if the corresponding collection exists (and is non-empty)."""
        coll_name = self._coll_name(table_name)
        if not self._db.has_collection(coll_name):
            return False
        cursor = self._db.aql.execute(
            "RETURN LENGTH(@@col) > 0",
            bind_vars={"@col": coll_name},
        )
        return next(iter(cursor), False)

    def list(self) -> list[str]:
        """List table names (without prefix) for all pipeline collections."""
        prefix = self._prefix
        return [
            c["name"][len(prefix):]
            for c in self._db.collections()
            if not c["system"] and c["name"].startswith(prefix)
        ]

    def open(
        self,
        table_name: str,
        transformer: RowTransformer | None = None,
        truncate: bool = True,
    ) -> Table:
        """Open a collection for streaming row-by-row access."""
        coll_name = self._coll_name(table_name)
        self._ensure_collection(coll_name)
        return ArangoDBTable(self._db, coll_name, transformer, truncate)
