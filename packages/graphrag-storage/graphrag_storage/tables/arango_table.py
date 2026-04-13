# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""ArangoDB-backed Table implementation for streaming row access."""

from __future__ import annotations

import inspect
import math
import logging
from typing import TYPE_CHECKING, Any

import pandas as pd

from graphrag_storage.tables.table import RowTransformer, Table

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

logger = logging.getLogger(__name__)

_BATCH_SIZE = 500
_ARANGO_META_FIELDS = {"_id", "_rev", "_key", "_from", "_to"}


def _identity(row: dict[str, Any]) -> Any:
    return row


def _apply_transformer(transformer: RowTransformer, row: dict[str, Any]) -> Any:
    if inspect.isclass(transformer):
        return transformer(**row)
    return transformer(row)


def _strip_meta(doc: dict[str, Any]) -> dict[str, Any]:
    """Remove ArangoDB internal metadata fields from a document."""
    return {k: v for k, v in doc.items() if k not in _ARANGO_META_FIELDS}


def _sanitize(value: Any) -> Any:
    """Recursively convert numpy types and NaN to JSON-serializable Python types."""
    try:
        import numpy as np
        if isinstance(value, np.ndarray):
            return [_sanitize(v) for v in value.tolist()]
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            v = float(value)
            return None if math.isnan(v) else v
    except ImportError:
        pass
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, list):
        return [_sanitize(v) for v in value]
    if isinstance(value, dict):
        return {k: _sanitize(v) for k, v in value.items()}
    return value


def _sanitize_doc(doc: dict[str, Any]) -> dict[str, Any]:
    """Sanitize a document dict, setting _key from id if available."""
    sanitized = {k: _sanitize(v) for k, v in doc.items() if k not in _ARANGO_META_FIELDS}
    if "id" in sanitized:
        sanitized["_key"] = str(sanitized["id"])
    return sanitized


class ArangoDBTable(Table):
    """ArangoDB collection-backed Table for row-by-row streaming.

    Reads iterate the collection via AQL cursor.
    Writes accumulate rows in memory and batch-upsert on close().
    When truncate=True, the collection is cleared before the first write.
    """

    def __init__(
        self,
        db: Any,
        collection_name: str,
        transformer: RowTransformer | None = None,
        truncate: bool = True,
    ) -> None:
        self._db = db
        self._collection_name = collection_name
        self._transformer = transformer or _identity
        self._truncate = truncate
        self._pending: list[dict[str, Any]] = []
        self._truncated = False

    def __aiter__(self) -> AsyncIterator[Any]:
        return self._aiter_impl()

    async def _aiter_impl(self) -> AsyncIterator[Any]:
        cursor = self._db.aql.execute(
            "FOR doc IN @@col RETURN doc",
            bind_vars={"@col": self._collection_name},
        )
        for doc in cursor:
            row = _strip_meta(doc)
            yield _apply_transformer(self._transformer, row)

    async def length(self) -> int:
        cursor = self._db.aql.execute(
            "RETURN LENGTH(@@col)",
            bind_vars={"@col": self._collection_name},
        )
        return next(iter(cursor), 0)

    async def has(self, row_id: str) -> bool:
        cursor = self._db.aql.execute(
            "RETURN LENGTH(FOR doc IN @@col FILTER doc.id == @id LIMIT 1 RETURN 1) > 0",
            bind_vars={"@col": self._collection_name, "id": row_id},
        )
        return next(iter(cursor), False)

    async def write(self, row: dict[str, Any]) -> None:
        if self._truncate and not self._truncated:
            self._db.collection(self._collection_name).truncate()
            self._truncated = True
        self._pending.append(row)
        if len(self._pending) >= _BATCH_SIZE:
            await self._flush()

    async def _flush(self) -> None:
        if not self._pending:
            return
        docs = [_sanitize_doc(r) for r in self._pending]
        self._db.collection(self._collection_name).import_bulk(
            docs, on_duplicate="replace", halt_on_error=True
        )
        self._pending = []

    async def close(self) -> None:
        await self._flush()
