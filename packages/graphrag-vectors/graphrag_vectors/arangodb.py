# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""ArangoDB vector store implementation."""

from typing import Any

from graphrag_vectors.filtering import (
    AndExpr,
    Condition,
    FilterExpr,
    NotExpr,
    Operator,
    OrExpr,
)
from graphrag_vectors.vector_store import (
    VectorStore,
    VectorStoreDocument,
    VectorStoreSearchResult,
)

_BATCH_SIZE = 500
# ArangoDB 3.12.0+ vector AQL function name
_AQL_APPROX_NEAR = "APPROX_NEAR_COSINE"


class ArangoDBVectorStore(VectorStore):
    """ArangoDB vector storage implementation using HNSW vector index."""

    def __init__(
        self,
        url: str = "http://localhost:8529",
        username: str = "root",
        password: str = "",
        db_name: str = "graphrag",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.url = url
        self.username = username
        self.password = password
        self.db_name = db_name
        self._db: Any = None
        self._collection: Any = None
        self._aql_near_fn: str = _AQL_APPROX_NEAR

    def connect(self) -> None:
        """Connect to ArangoDB and ensure database and collection exist."""
        from arango import ArangoClient

        client = ArangoClient(hosts=self.url)
        sys_db = client.db("_system", username=self.username, password=self.password)

        if not sys_db.has_database(self.db_name):
            sys_db.create_database(self.db_name)

        self._db = client.db(
            self.db_name, username=self.username, password=self.password
        )

        if not self._db.has_collection(self.index_name):
            self._db.create_collection(self.index_name)

        self._collection = self._db.collection(self.index_name)
        self._ensure_vector_index()

    def _ensure_vector_index(self) -> None:
        """Create the HNSW vector index if it does not already exist."""
        existing = [
            idx
            for idx in self._collection.indexes()
            if idx.get("type") == "vector"
            and self.vector_field in idx.get("fields", [])
        ]
        if existing:
            return

        try:
            self._collection.add_index(
                {
                    "type": "vector",
                    "fields": [self.vector_field],
                    "params": {
                        "metric": "cosine",
                        "dimension": self.vector_size,
                        "nLists": 2,
                    },
                }
            )
        except Exception as exc:  # noqa: BLE001
            msg = (
                f"Failed to create vector index on collection '{self.index_name}'. "
                "Ensure ArangoDB >= 3.12 with vector search enabled. "
                f"Original error: {exc}"
            )
            raise RuntimeError(msg) from exc

    def create_index(self) -> None:
        """Drop and recreate the collection with a fresh vector index."""
        if self._db.has_collection(self.index_name):
            self._db.delete_collection(self.index_name)
        self._db.create_collection(self.index_name)
        self._collection = self._db.collection(self.index_name)
        self._ensure_vector_index()

    def load_documents(self, documents: list[VectorStoreDocument]) -> None:
        """Upsert documents in batches."""
        batch: list[dict[str, Any]] = []
        for document in documents:
            self._prepare_document(document)
            if document.vector is None:
                continue
            doc = self._to_arango_doc(document)
            batch.append(doc)
            if len(batch) >= _BATCH_SIZE:
                self._upsert_batch(batch)
                batch = []
        if batch:
            self._upsert_batch(batch)

    def _upsert_batch(self, batch: list[dict[str, Any]]) -> None:
        self._collection.import_bulk(batch, on_duplicate="replace", complete=True)

    def _to_arango_doc(self, document: VectorStoreDocument) -> dict[str, Any]:
        """Convert a VectorStoreDocument to an ArangoDB document dict."""
        doc: dict[str, Any] = {
            "_key": str(document.id),
            self.id_field: document.id,
            self.vector_field: document.vector,
            self.create_date_field: document.create_date,
            self.update_date_field: document.update_date,
        }
        if document.data:
            for field_name in self.fields:
                if field_name in document.data:
                    doc[field_name] = document.data[field_name]
        return doc

    def similarity_search_by_vector(
        self,
        query_embedding: list[float],
        k: int = 10,
        select: list[str] | None = None,
        filters: FilterExpr | None = None,
        include_vectors: bool = True,
    ) -> list[VectorStoreSearchResult]:
        """Perform approximate nearest-neighbour search by vector."""
        filter_clause = ""
        bind_vars: dict[str, Any] = {
            "@collection": self.index_name,
            "query_vector": query_embedding,
            "k": k,
        }
        if filters is not None:
            filter_clause = f"FILTER {self._compile_filter(filters)}"

        aql = f"""
            FOR doc IN @@collection
              {filter_clause}
              LET score = {self._aql_near_fn}(doc.`{self.vector_field}`, @query_vector)
              SORT score DESC
              LIMIT @k
              RETURN {{doc: doc, score: score}}
        """

        try:
            cursor = self._db.aql.execute(aql, bind_vars=bind_vars)
            rows = list(cursor)
        except Exception:  # noqa: BLE001
            # Fallback: fetch all and compute cosine locally
            rows = self._local_similarity_search(query_embedding, k, filters)

        return [
            VectorStoreSearchResult(
                document=VectorStoreDocument(
                    id=row["doc"].get(self.id_field, ""),
                    vector=row["doc"].get(self.vector_field) if include_vectors else None,
                    data=self._extract_data(row["doc"], select),
                    create_date=row["doc"].get(self.create_date_field),
                    update_date=row["doc"].get(self.update_date_field),
                ),
                score=float(row["score"] or 0.0),
            )
            for row in rows
        ]

    def _local_similarity_search(
        self,
        query_embedding: list[float],
        k: int,
        filters: FilterExpr | None,
    ) -> list[dict[str, Any]]:
        """Fallback: fetch all documents and compute cosine similarity locally."""
        from numpy import array, dot
        from numpy.linalg import norm

        filter_clause = ""
        bind_vars: dict[str, Any] = {"@collection": self.index_name}
        if filters is not None:
            filter_clause = f"FILTER {self._compile_filter(filters)}"

        aql = f"FOR doc IN @@collection {filter_clause} RETURN doc"
        docs = list(self._db.aql.execute(aql, bind_vars=bind_vars))

        q = array(query_embedding, dtype="float32")
        q_norm = norm(q)

        scored = []
        for doc in docs:
            v = doc.get(self.vector_field)
            if not v:
                continue
            d = array(v, dtype="float32")
            d_norm = norm(d)
            score = float(dot(q, d) / (q_norm * d_norm)) if q_norm * d_norm > 0 else 0.0
            scored.append({"doc": doc, "score": score})

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:k]

    def search_by_id(
        self,
        id: str,
        select: list[str] | None = None,
        include_vectors: bool = True,
    ) -> VectorStoreDocument:
        """Retrieve a document by its ID."""
        aql = """
            FOR doc IN @@collection
              FILTER doc.@id_field == @id
              LIMIT 1
              RETURN doc
        """
        cursor = self._db.aql.execute(
            aql,
            bind_vars={
                "@collection": self.index_name,
                "id_field": self.id_field,
                "id": id,
            },
        )
        results = list(cursor)
        if not results:
            return VectorStoreDocument(id=id, vector=None, data={})

        doc = results[0]
        return VectorStoreDocument(
            id=doc.get(self.id_field, id),
            vector=doc.get(self.vector_field) if include_vectors else None,
            data=self._extract_data(doc, select),
            create_date=doc.get(self.create_date_field),
            update_date=doc.get(self.update_date_field),
        )

    def count(self) -> int:
        """Return total document count."""
        return self._collection.count()

    def remove(self, ids: list[str]) -> None:
        """Remove documents by ID."""
        aql = """
            FOR doc IN @@collection
              FILTER doc.@id_field IN @ids
              REMOVE doc IN @@collection
        """
        self._db.aql.execute(
            aql,
            bind_vars={
                "@collection": self.index_name,
                "id_field": self.id_field,
                "ids": ids,
            },
        )

    def update(self, document: VectorStoreDocument) -> None:
        """Update an existing document."""
        self._prepare_update(document)
        doc = self._to_arango_doc(document)
        self._collection.update_match(
            {self.id_field: document.id},
            doc,
        )

    def _compile_filter(self, expr: FilterExpr) -> str:
        """Compile a FilterExpr into an AQL FILTER expression."""
        match expr:
            case Condition():
                return self._compile_condition(expr)
            case AndExpr():
                parts = [self._compile_filter(e) for e in expr.and_]
                return " AND ".join(f"({p})" for p in parts)
            case OrExpr():
                parts = [self._compile_filter(e) for e in expr.or_]
                return " OR ".join(f"({p})" for p in parts)
            case NotExpr():
                inner = self._compile_filter(expr.not_)
                return f"NOT ({inner})"
            case _:
                msg = f"Unsupported filter expression type: {type(expr)}"
                raise ValueError(msg)

    def _compile_condition(self, cond: Condition) -> str:
        """Compile a single Condition to AQL syntax."""
        field = f"doc.`{cond.field}`"
        value = cond.value

        def quote(v: Any) -> str:
            return f'"{v}"' if isinstance(v, str) else str(v)

        match cond.operator:
            case Operator.eq:
                return f"{field} == {quote(value)}"
            case Operator.ne:
                return f"{field} != {quote(value)}"
            case Operator.gt:
                return f"{field} > {quote(value)}"
            case Operator.gte:
                return f"{field} >= {quote(value)}"
            case Operator.lt:
                return f"{field} < {quote(value)}"
            case Operator.lte:
                return f"{field} <= {quote(value)}"
            case Operator.in_:
                items = ", ".join(quote(v) for v in value)
                return f"{field} IN [{items}]"
            case Operator.not_in:
                items = ", ".join(quote(v) for v in value)
                return f"{field} NOT IN [{items}]"
            case Operator.contains:
                return f'CONTAINS({field}, "{value}")'
            case Operator.startswith:
                return f'STARTS_WITH({field}, "{value}")'
            case Operator.endswith:
                return f'ENDS_WITH({field}, "{value}")'  # no native AQL ENDS_WITH; handled via LIKE
            case Operator.exists:
                return f"HAS(doc, `{cond.field}`)" if value else f"NOT HAS(doc, `{cond.field}`)"
            case _:
                msg = f"Unsupported operator for ArangoDB: {cond.operator}"
                raise ValueError(msg)

    def _extract_data(
        self, doc: dict[str, Any], select: list[str] | None = None
    ) -> dict[str, Any]:
        """Extract configured field data from a document."""
        fields_to_extract = select if select is not None else list(self.fields.keys())
        return {f: doc[f] for f in fields_to_extract if f in doc}
