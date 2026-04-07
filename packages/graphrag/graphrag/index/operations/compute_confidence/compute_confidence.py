# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Multi-factor confidence scoring for entities and relationships.

All computations are pure pandas/numpy operations — no LLM calls.
"""

import logging
import uuid

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_entity_relationship_confidence(
    entities_df: pd.DataFrame,
    relationships_df: pd.DataFrame,
    evidence_df: pd.DataFrame,
    text_units_df: pd.DataFrame,
    weights: dict[str, float] | None = None,
    cross_doc_similarity_threshold: float = 0.6,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute multi-factor confidence scores for entities and relationships.

    Formula:
        confidence = (
            w_extraction * avg(evidence.extraction_confidence)
          + w_agreement  * source_agreement_score
          + w_crossdoc   * normalized(distinct_document_count)
          - w_contradict * normalized(contradiction_count)
        )

    Enriches entity and relationship attributes dicts with:
        - confidence (float)
        - evidence_count (int)
        - source_count (int)
        - contradiction_count (int)
        - is_partial (bool)
        - has_temporal_scope (bool)

    Also detects cross-document contradictions and returns a contradictions DataFrame.

    Returns:
        (enriched_entities_df, enriched_relationships_df, contradictions_df)
    """
    if weights is None:
        weights = {
            "extraction": 0.3,
            "source_agreement": 0.3,
            "cross_doc": 0.25,
            "contradiction_penalty": 0.15,
        }

    # Normalize weights to sum to 1.0
    total = sum(weights.values())
    if total > 0:
        weights = {k: v / total for k, v in weights.items()}

    if evidence_df is None or len(evidence_df) == 0:
        return entities_df, relationships_df, pd.DataFrame(
            columns=["id", "human_readable_id", "relation_type",
                     "subject_a_type", "subject_a_id",
                     "subject_b_type", "subject_b_id",
                     "description", "confidence", "detection_method"]
        )

    # Build doc_id lookup: text_unit_id -> document_id
    doc_lookup: dict[str, str] = {}
    if text_units_df is not None and len(text_units_df) > 0:
        for _, row in text_units_df.iterrows():
            doc_lookup[str(row["id"])] = str(row.get("document_id", ""))

    contradictions = []

    # --- Process entities ---
    entities_df = entities_df.copy()
    if "attributes" not in entities_df.columns:
        entities_df["attributes"] = None

    entity_evidence = evidence_df[evidence_df["subject_type"] == "entity"]
    entity_groups = entity_evidence.groupby("subject_id", sort=False)

    for subject_id, group in entity_groups:
        metrics = _compute_group_metrics(group, doc_lookup, cross_doc_similarity_threshold)

        # Find matching rows in entities_df by title
        mask = entities_df["title"] == subject_id
        for idx in entities_df.index[mask]:
            attrs = entities_df.at[idx, "attributes"] or {}
            attrs = dict(attrs)  # ensure mutable copy
            confidence = _compute_confidence(metrics, weights)
            attrs.update({
                "confidence": confidence,
                "evidence_count": metrics["evidence_count"],
                "source_count": metrics["source_count"],
                "contradiction_count": metrics["contradiction_count"],
                "is_partial": metrics["is_partial"],
                "has_temporal_scope": metrics["has_temporal_scope"],
            })
            entities_df.at[idx, "attributes"] = attrs

        # Collect contradictions for this entity
        contradictions.extend(metrics["contradictions"])

    # --- Process relationships ---
    relationships_df = relationships_df.copy()
    if "attributes" not in relationships_df.columns:
        relationships_df["attributes"] = None

    rel_evidence = evidence_df[evidence_df["subject_type"] == "relationship"]
    rel_groups = rel_evidence.groupby("subject_id", sort=False)

    for subject_id, group in rel_groups:
        metrics = _compute_group_metrics(group, doc_lookup, cross_doc_similarity_threshold)

        # subject_id for relationships is "SOURCE::TARGET"
        if "::" in subject_id:
            source, target = subject_id.split("::", 1)
        else:
            continue

        mask = (relationships_df["source"] == source) & (relationships_df["target"] == target)
        for idx in relationships_df.index[mask]:
            attrs = relationships_df.at[idx, "attributes"] or {}
            attrs = dict(attrs)
            confidence = _compute_confidence(metrics, weights)
            attrs.update({
                "confidence": confidence,
                "evidence_count": metrics["evidence_count"],
                "source_count": metrics["source_count"],
                "contradiction_count": metrics["contradiction_count"],
                "is_partial": metrics["is_partial"],
                "has_temporal_scope": metrics["has_temporal_scope"],
            })
            relationships_df.at[idx, "attributes"] = attrs

        contradictions.extend(metrics["contradictions"])

    # Build contradictions DataFrame
    contradictions_df = _build_contradictions_df(contradictions)

    return entities_df, relationships_df, contradictions_df


def _compute_group_metrics(
    group: pd.DataFrame,
    doc_lookup: dict[str, str],
    cross_doc_threshold: float,
) -> dict:
    """Compute metrics for a group of evidence records for the same subject."""
    evidence_count = len(group)

    # Extraction confidence: average
    if "extraction_confidence" in group.columns:
        avg_extraction_confidence = float(group["extraction_confidence"].mean())
    else:
        avg_extraction_confidence = 0.5

    # Source count: distinct documents
    text_unit_ids = group["text_unit_id"].tolist()
    doc_ids = {doc_lookup.get(str(tid), tid) for tid in text_unit_ids}
    source_count = len(doc_ids)

    # Normalized cross-doc support: log scale, capped at 1.0
    cross_doc_score = min(1.0, np.log1p(source_count) / np.log1p(10))

    # Contradiction count: refuted or weakened evidence
    contradiction_count = 0
    if "verification_status" in group.columns:
        contradiction_count = int(
            group["verification_status"].isin(["refuted", "weakened"]).sum()
        )

    # Source agreement: based on verification status
    # confirmed evidence increases agreement, refuted/weakened decreases
    source_agreement_score = 0.5
    if "verification_status" in group.columns and len(group) > 0:
        confirmed = int((group["verification_status"] == "confirmed").sum())
        total = len(group)
        source_agreement_score = confirmed / total

    # Partiality
    is_partial = False
    if "completeness_status" in group.columns:
        is_partial = bool(group["completeness_status"].isin(["partial", "inferred"]).any())

    # Temporal scope
    has_temporal_scope = False
    if "time_scope" in group.columns:
        has_temporal_scope = bool(group["time_scope"].notna().any())

    # Contradiction penalty: normalized 0.0-1.0
    contradiction_penalty = min(1.0, contradiction_count / max(evidence_count, 1))

    # Cross-document contradictions detection
    contradictions = []
    if source_count > 1 and "verification_status" in group.columns:
        refuted_rows = group[group["verification_status"] == "refuted"]
        for _, row in refuted_rows.iterrows():
            # Find a confirmed row from a different document to pair with
            confirmed_rows = group[group["verification_status"] == "confirmed"]
            for _, confirmed_row in confirmed_rows.iterrows():
                a_doc = doc_lookup.get(str(row.get("text_unit_id", "")), "")
                b_doc = doc_lookup.get(str(confirmed_row.get("text_unit_id", "")), "")
                if a_doc != b_doc:
                    contradictions.append({
                        "relation_type": "contradicts",
                        "subject_a_type": row.get("subject_type", "entity"),
                        "subject_a_id": str(row.get("subject_id", "")),
                        "subject_b_type": confirmed_row.get("subject_type", "entity"),
                        "subject_b_id": str(confirmed_row.get("subject_id", "")),
                        "description": (
                            f"Evidence from {a_doc} is refuted while evidence from {b_doc} is confirmed"
                        ),
                        "confidence": 0.7,
                        "detection_method": "llm_verified",
                    })
                    break  # One contradiction record per refuted item is enough

    return {
        "evidence_count": evidence_count,
        "avg_extraction_confidence": avg_extraction_confidence,
        "source_agreement_score": source_agreement_score,
        "cross_doc_score": cross_doc_score,
        "source_count": source_count,
        "contradiction_count": contradiction_count,
        "contradiction_penalty": contradiction_penalty,
        "is_partial": is_partial,
        "has_temporal_scope": has_temporal_scope,
        "contradictions": contradictions,
    }


def _compute_confidence(metrics: dict, weights: dict) -> float:
    """Compute the final confidence score from metrics and weights."""
    score = (
        weights.get("extraction", 0.3) * metrics["avg_extraction_confidence"]
        + weights.get("source_agreement", 0.3) * metrics["source_agreement_score"]
        + weights.get("cross_doc", 0.25) * metrics["cross_doc_score"]
        - weights.get("contradiction_penalty", 0.15) * metrics["contradiction_penalty"]
    )
    return max(0.0, min(1.0, score))


def _build_contradictions_df(contradictions: list[dict]) -> pd.DataFrame:
    """Build a contradictions DataFrame from collected contradiction records."""
    if not contradictions:
        return pd.DataFrame(
            columns=[
                "id", "human_readable_id", "relation_type",
                "subject_a_type", "subject_a_id",
                "subject_b_type", "subject_b_id",
                "description", "confidence", "detection_method",
            ]
        )

    rows = []
    for i, c in enumerate(contradictions):
        rows.append({
            "id": str(uuid.uuid4()),
            "human_readable_id": str(i),
            "relation_type": c.get("relation_type", "contradicts"),
            "subject_a_type": c.get("subject_a_type", "entity"),
            "subject_a_id": c.get("subject_a_id", ""),
            "subject_b_type": c.get("subject_b_type", "entity"),
            "subject_b_id": c.get("subject_b_id", ""),
            "description": c.get("description"),
            "confidence": c.get("confidence", 0.5),
            "detection_method": c.get("detection_method", "embedding_similarity"),
        })
    return pd.DataFrame(rows)
