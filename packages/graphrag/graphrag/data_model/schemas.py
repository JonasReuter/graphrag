# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License
"""Common field name definitions for data frames."""

ID = "id"
SHORT_ID = "human_readable_id"
TITLE = "title"
DESCRIPTION = "description"

TYPE = "type"

# POST-PREP NODE TABLE SCHEMA
NODE_DEGREE = "degree"
NODE_FREQUENCY = "frequency"
NODE_DETAILS = "node_details"

# POST-PREP EDGE TABLE SCHEMA
EDGE_SOURCE = "source"
EDGE_TARGET = "target"
EDGE_DEGREE = "combined_degree"
EDGE_DETAILS = "edge_details"
EDGE_WEIGHT = "weight"

# POST-PREP CLAIM TABLE SCHEMA
CLAIM_SUBJECT = "subject_id"
CLAIM_STATUS = "status"
CLAIM_DETAILS = "claim_details"

# COMMUNITY HIERARCHY TABLE SCHEMA
SUB_COMMUNITY = "sub_community"

# COMMUNITY CONTEXT TABLE SCHEMA
ALL_CONTEXT = "all_context"
CONTEXT_STRING = "context_string"
CONTEXT_SIZE = "context_size"
CONTEXT_EXCEED_FLAG = "context_exceed_limit"

# COMMUNITY REPORT TABLE SCHEMA
COMMUNITY_ID = "community"
COMMUNITY_LEVEL = "level"
COMMUNITY_PARENT = "parent"
COMMUNITY_CHILDREN = "children"
TITLE = "title"
SUMMARY = "summary"
FINDINGS = "findings"
RATING = "rank"
EXPLANATION = "rating_explanation"
FULL_CONTENT = "full_content"
FULL_CONTENT_JSON = "full_content_json"

ENTITY_IDS = "entity_ids"
RELATIONSHIP_IDS = "relationship_ids"
TEXT_UNIT_IDS = "text_unit_ids"
COVARIATE_IDS = "covariate_ids"
DOCUMENT_ID = "document_id"

PERIOD = "period"
SIZE = "size"

# text units
ENTITY_DEGREE = "entity_degree"
ALL_DETAILS = "all_details"
TEXT = "text"
N_TOKENS = "n_tokens"

CREATION_DATE = "creation_date"
RAW_DATA = "raw_data"

# TEMPORAL FIELDS
OBSERVED_AT = "observed_at"
LAST_OBSERVED_AT = "last_observed_at"
VALID_FROM = "valid_from"
VALID_UNTIL = "valid_until"
CREATED_AT = "created_at"

# the following lists define the final content and ordering of columns in the data model parquet outputs
ENTITIES_FINAL_COLUMNS = [
    ID,
    SHORT_ID,
    TITLE,
    TYPE,
    DESCRIPTION,
    TEXT_UNIT_IDS,
    NODE_FREQUENCY,
    NODE_DEGREE,
    OBSERVED_AT,
    LAST_OBSERVED_AT,
    VALID_FROM,
    VALID_UNTIL,
]

RELATIONSHIPS_FINAL_COLUMNS = [
    ID,
    SHORT_ID,
    EDGE_SOURCE,
    EDGE_TARGET,
    DESCRIPTION,
    EDGE_WEIGHT,
    EDGE_DEGREE,
    TEXT_UNIT_IDS,
    OBSERVED_AT,
    LAST_OBSERVED_AT,
    VALID_FROM,
    VALID_UNTIL,
]

COMMUNITIES_FINAL_COLUMNS = [
    ID,
    SHORT_ID,
    COMMUNITY_ID,
    COMMUNITY_LEVEL,
    COMMUNITY_PARENT,
    COMMUNITY_CHILDREN,
    TITLE,
    ENTITY_IDS,
    RELATIONSHIP_IDS,
    TEXT_UNIT_IDS,
    PERIOD,
    SIZE,
]

COMMUNITY_REPORTS_FINAL_COLUMNS = [
    ID,
    SHORT_ID,
    COMMUNITY_ID,
    COMMUNITY_LEVEL,
    COMMUNITY_PARENT,
    COMMUNITY_CHILDREN,
    TITLE,
    SUMMARY,
    FULL_CONTENT,
    RATING,
    EXPLANATION,
    FINDINGS,
    FULL_CONTENT_JSON,
    PERIOD,
    SIZE,
]

COVARIATES_FINAL_COLUMNS = [
    ID,
    SHORT_ID,
    "covariate_type",
    TYPE,
    DESCRIPTION,
    "subject_id",
    "object_id",
    "status",
    "start_date",
    "end_date",
    "source_text",
    "text_unit_id",
]

TEXT_UNITS_FINAL_COLUMNS = [
    ID,
    SHORT_ID,
    TEXT,
    N_TOKENS,
    DOCUMENT_ID,
    ENTITY_IDS,
    RELATIONSHIP_IDS,
    COVARIATE_IDS,
    CREATED_AT,
]

DOCUMENTS_FINAL_COLUMNS = [
    ID,
    SHORT_ID,
    TITLE,
    TEXT,
    TEXT_UNIT_IDS,
    CREATION_DATE,
    RAW_DATA,
]

# EVIDENCE TABLE SCHEMA
EVIDENCE_SUBJECT_TYPE = "subject_type"
EVIDENCE_SUBJECT_ID = "subject_id"
EVIDENCE_TEXT_UNIT_ID = "text_unit_id"
EVIDENCE_SOURCE_SPAN = "source_span"
EVIDENCE_EXTRACTION_CONFIDENCE = "extraction_confidence"
EVIDENCE_COMPLETENESS_STATUS = "completeness_status"
EVIDENCE_TIME_SCOPE = "time_scope"
EVIDENCE_QUALIFIER = "qualifier"
EVIDENCE_VERIFICATION_STATUS = "verification_status"
EVIDENCE_VERIFICATION_METHOD = "verification_method"

# CONTRADICTION TABLE SCHEMA
CONTRADICTION_RELATION_TYPE = "relation_type"
CONTRADICTION_SUBJECT_A_TYPE = "subject_a_type"
CONTRADICTION_SUBJECT_A_ID = "subject_a_id"
CONTRADICTION_SUBJECT_B_TYPE = "subject_b_type"
CONTRADICTION_SUBJECT_B_ID = "subject_b_id"
CONTRADICTION_CONFIDENCE = "confidence"
CONTRADICTION_DETECTION_METHOD = "detection_method"

CONTRADICTIONS_FINAL_COLUMNS = [
    ID,
    SHORT_ID,
    CONTRADICTION_RELATION_TYPE,
    CONTRADICTION_SUBJECT_A_TYPE,
    CONTRADICTION_SUBJECT_A_ID,
    CONTRADICTION_SUBJECT_B_TYPE,
    CONTRADICTION_SUBJECT_B_ID,
    DESCRIPTION,
    CONTRADICTION_CONFIDENCE,
    CONTRADICTION_DETECTION_METHOD,
]

EVIDENCE_FINAL_COLUMNS = [
    ID,
    SHORT_ID,
    EVIDENCE_SUBJECT_TYPE,
    EVIDENCE_SUBJECT_ID,
    EVIDENCE_TEXT_UNIT_ID,
    EVIDENCE_SOURCE_SPAN,
    EVIDENCE_EXTRACTION_CONFIDENCE,
    EVIDENCE_COMPLETENESS_STATUS,
    EVIDENCE_TIME_SCOPE,
    EVIDENCE_QUALIFIER,
    EVIDENCE_VERIFICATION_STATUS,
    EVIDENCE_VERIFICATION_METHOD,
]
