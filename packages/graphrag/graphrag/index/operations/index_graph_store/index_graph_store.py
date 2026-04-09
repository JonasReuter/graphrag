# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Core indexing operation: write GraphRAG pipeline output into ArangoDB graph store."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from graphrag.config.embeddings import entity_description_embedding as _ENTITY_EMBED_NAME
from graphrag.data_model.data_reader import DataReader

if TYPE_CHECKING:
    from graphrag_storage.tables import TableProvider
    from graphrag_vectors.arangodb_graph import ArangoDBGraphStore

logger = logging.getLogger(__name__)

# ArangoDB collection name for entity description embeddings.
# Matches the index_name used by generate_text_embeddings.
_ENTITY_VECTOR_COLLECTION = _ENTITY_EMBED_NAME  # "entity_description"


async def index_graph_store(
    graph_store: ArangoDBGraphStore,
    table_provider: TableProvider,
    store_vectors: bool = True,
    db: Any = None,
    entity_vector_collection: str = _ENTITY_VECTOR_COLLECTION,
) -> dict[str, int]:
    """Index all pipeline output tables into ArangoDB as a named graph.

    Execution order is important:
    1. Entities must be upserted before relationships (edges need _from/_to)
    2. Communities must be upserted before membership edges
    3. Text units must be upserted before entity-text_unit edges

    Parameters
    ----------
    graph_store : ArangoDBGraphStore
        Connected graph store instance.
    table_provider : TableProvider
        Pipeline output table provider (reads entities, relationships, etc.)
    store_vectors : bool
        Whether to copy entity description vectors into entity documents.
        Only works when the vector store type is also ArangoDB (db must be provided).
    db : Any
        The python-arango StandardDatabase connection, used to read existing
        entity_description_embedding vectors. Pass graph_store._db.
    entity_vector_collection : str
        Name of the ArangoDB collection holding entity description embeddings.
    """
    counts: dict[str, int] = {}
    reader = DataReader(table_provider)

    # --- Setup schema ---
    graph_store.setup_graph()
    graph_store._ensure_entity_vector_index()

    # --- Phase 1: Load entities ---
    logger.info("Loading entities...")
    entities_df = await reader.entities()
    entity_rows = entities_df.to_dict("records")

    # Build lookup maps needed later
    title_to_id: dict[str, str] = {}
    for row in entity_rows:
        title_to_id[str(row.get("title", ""))] = str(row["id"])

    # --- Phase 2: Optionally read vectors from existing ArangoDB vector collection ---
    vector_map: dict[str, list[float]] = {}
    if store_vectors and db is not None:
        logger.info("Reading entity vectors from '%s'...", entity_vector_collection)
        try:
            cursor = db.aql.execute(
                "FOR doc IN @@coll RETURN {id: doc.id, vector: doc.vector}",
                bind_vars={"@coll": entity_vector_collection},
            )
            for row in cursor:
                if row.get("vector") and row.get("id"):
                    vector_map[str(row["id"])] = row["vector"]
            logger.info("Loaded %d entity vectors.", len(vector_map))
        except Exception as exc:
            logger.warning(
                "Could not read entity vectors from '%s': %s. "
                "Entities will be indexed without vectors.",
                entity_vector_collection,
                exc,
            )

    # --- Phase 3: Upsert entities ---
    logger.info("Upserting %d entities...", len(entity_rows))
    counts["entities"] = graph_store.upsert_entities(
        entity_rows, vector_map=vector_map if vector_map else None
    )

    # --- Phase 4: Communities (must come before membership edges) ---
    logger.info("Loading and upserting communities...")
    try:
        communities_df = await reader.communities()
        community_rows = communities_df.to_dict("records")
        community_count, community_int_to_uuid = graph_store.upsert_communities(
            community_rows
        )
        counts["communities"] = community_count
    except Exception as exc:
        logger.warning("Could not load communities: %s", exc)
        community_rows = []
        community_int_to_uuid = {}

    # --- Phase 5: Community reports ---
    logger.info("Loading and upserting community reports...")
    try:
        reports_df = await reader.community_reports()
        report_rows = reports_df.to_dict("records")
        counts["community_reports"] = graph_store.upsert_community_reports(report_rows)
    except Exception as exc:
        logger.warning("Could not load community reports: %s", exc)

    # --- Phase 6: Text units ---
    logger.info("Loading and upserting text units...")
    try:
        text_units_df = await reader.text_units()
        text_unit_rows = text_units_df.to_dict("records")
        counts["text_units"] = graph_store.upsert_text_units(text_unit_rows)
    except Exception as exc:
        logger.warning("Could not load text units: %s", exc)
        text_unit_rows = []

    # --- Phase 6b: Covariates (optional — extract_claims must be enabled) ---
    logger.info("Loading and upserting covariates...")
    try:
        covariates_df = await reader.covariates()
        if covariates_df is not None and len(covariates_df) > 0:
            covariate_rows = covariates_df.to_dict("records")
            counts["covariates"] = graph_store.upsert_covariates(
                covariate_rows, title_to_id=title_to_id
            )
        else:
            counts["covariates"] = 0
    except Exception as exc:
        logger.info("No covariates to index (extract_claims likely disabled): %s", exc)
        counts["covariates"] = 0

    # --- Phase 7: Relationships as edges ---
    logger.info("Loading and upserting relationships as graph edges...")
    relationships_df = await reader.relationships()
    rel_rows = relationships_df.to_dict("records")
    counts["relationships"] = graph_store.upsert_relationships(
        rel_rows, title_to_id=title_to_id
    )

    # --- Phase 8: Entity→Community membership edges ---
    # Built from communities.entity_ids (not entities.community_ids which is absent
    # from ENTITIES_FINAL_COLUMNS in the final parquet output).
    logger.info("Upserting entity-community membership edges...")
    counts["entity_community_membership"] = (
        graph_store.upsert_entity_community_edges_from_communities(community_rows)
        if community_rows
        else 0
    )

    # --- Phase 9: Entity→TextUnit edges ---
    logger.info("Upserting entity-text_unit edges...")
    counts["entity_text_unit"] = graph_store.upsert_entity_text_unit_edges(entity_rows)

    logger.info("Graph store indexing complete: %s", counts)
    return counts
