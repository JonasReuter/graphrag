# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Quality metrics computation for the knowledge graph.

All computations are pure pandas/numpy operations — no LLM calls.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_quality_metrics(
    entities_df: pd.DataFrame,
    relationships_df: pd.DataFrame,
    evidence_df: pd.DataFrame | None,
    contradictions_df: pd.DataFrame | None,
    low_confidence_threshold: float = 0.3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute knowledge graph quality metrics.

    Returns:
        (quality_metrics_df, quality_details_df)

        quality_metrics_df: Single-row DataFrame with all aggregated metrics.
        quality_details_df: Per-entity/relationship quality scores.
    """
    metrics: dict[str, float | int] = {}
    details_rows = []

    # --- Entity metrics ---
    n_entities = len(entities_df) if entities_df is not None else 0
    n_relationships = len(relationships_df) if relationships_df is not None else 0
    n_evidence = len(evidence_df) if evidence_df is not None else 0
    n_contradictions = len(contradictions_df) if contradictions_df is not None else 0

    metrics["total_entities"] = n_entities
    metrics["total_relationships"] = n_relationships
    metrics["total_evidence"] = n_evidence
    metrics["total_contradictions"] = n_contradictions

    if evidence_df is not None and len(evidence_df) > 0:
        # Evidence coverage
        entity_evidence = evidence_df[evidence_df["subject_type"] == "entity"]
        rel_evidence = evidence_df[evidence_df["subject_type"] == "relationship"]

        # Entities with evidence
        entities_with_evidence = entity_evidence["subject_id"].nunique()
        metrics["pct_entities_with_evidence"] = (
            entities_with_evidence / n_entities if n_entities > 0 else 0.0
        )
        metrics["pct_entities_without_evidence"] = 1.0 - metrics["pct_entities_with_evidence"]

        # Relationships with evidence
        rels_with_evidence = rel_evidence["subject_id"].nunique()
        metrics["pct_relationships_with_evidence"] = (
            rels_with_evidence / n_relationships if n_relationships > 0 else 0.0
        )

        # Single-source entities (only one text_unit_id in evidence)
        entity_source_counts = entity_evidence.groupby("subject_id")["text_unit_id"].nunique()
        single_source_entities = int((entity_source_counts == 1).sum())
        metrics["pct_single_source_entities"] = (
            single_source_entities / n_entities if n_entities > 0 else 0.0
        )

        # Verification status distribution
        if "verification_status" in evidence_df.columns:
            status_counts = evidence_df["verification_status"].value_counts(normalize=True)
            metrics["pct_evidence_confirmed"] = float(status_counts.get("confirmed", 0.0))
            metrics["pct_evidence_weakened"] = float(status_counts.get("weakened", 0.0))
            metrics["pct_evidence_refuted"] = float(status_counts.get("refuted", 0.0))
            metrics["pct_evidence_insufficient"] = float(status_counts.get("insufficient", 0.0))
            metrics["pct_evidence_unverified"] = float(status_counts.get("unverified", 0.0))
            metrics["pct_evidence_verified"] = 1.0 - metrics["pct_evidence_unverified"]
        else:
            metrics["pct_evidence_confirmed"] = 0.0
            metrics["pct_evidence_verified"] = 0.0

        # Completeness distribution
        if "completeness_status" in evidence_df.columns:
            comp_counts = evidence_df["completeness_status"].value_counts(normalize=True)
            metrics["pct_evidence_complete"] = float(comp_counts.get("complete", 0.0))
            metrics["pct_evidence_partial"] = float(comp_counts.get("partial", 0.0))
            metrics["pct_evidence_inferred"] = float(comp_counts.get("inferred", 0.0))

        # Extraction confidence distribution
        if "extraction_confidence" in evidence_df.columns:
            conf = evidence_df["extraction_confidence"]
            metrics["avg_extraction_confidence"] = float(conf.mean())
            metrics["extraction_confidence_p25"] = float(conf.quantile(0.25))
            metrics["extraction_confidence_p50"] = float(conf.quantile(0.50))
            metrics["extraction_confidence_p75"] = float(conf.quantile(0.75))

    else:
        metrics["pct_entities_without_evidence"] = 1.0
        metrics["pct_entities_with_evidence"] = 0.0
        metrics["pct_single_source_entities"] = 0.0
        metrics["pct_evidence_confirmed"] = 0.0
        metrics["pct_evidence_verified"] = 0.0

    # --- Confidence distribution (from entity/relationship attributes) ---
    all_confidences = []

    if entities_df is not None and len(entities_df) > 0 and "attributes" in entities_df.columns:
        for _, row in entities_df.iterrows():
            attrs = row.get("attributes") or {}
            if isinstance(attrs, dict) and "confidence" in attrs:
                conf = float(attrs["confidence"])
                all_confidences.append(conf)
                details_rows.append({
                    "subject_type": "entity",
                    "subject_id": row.get("title", row.get("id", "")),
                    "confidence": conf,
                    "evidence_count": attrs.get("evidence_count", 0),
                    "source_count": attrs.get("source_count", 0),
                    "contradiction_count": attrs.get("contradiction_count", 0),
                    "is_partial": attrs.get("is_partial", False),
                    "has_temporal_scope": attrs.get("has_temporal_scope", False),
                })

    if relationships_df is not None and len(relationships_df) > 0 and "attributes" in relationships_df.columns:
        for _, row in relationships_df.iterrows():
            attrs = row.get("attributes") or {}
            if isinstance(attrs, dict) and "confidence" in attrs:
                conf = float(attrs["confidence"])
                all_confidences.append(conf)
                details_rows.append({
                    "subject_type": "relationship",
                    "subject_id": f"{row.get('source', '')}::{row.get('target', '')}",
                    "confidence": conf,
                    "evidence_count": attrs.get("evidence_count", 0),
                    "source_count": attrs.get("source_count", 0),
                    "contradiction_count": attrs.get("contradiction_count", 0),
                    "is_partial": attrs.get("is_partial", False),
                    "has_temporal_scope": attrs.get("has_temporal_scope", False),
                })

    if all_confidences:
        conf_arr = np.array(all_confidences)
        metrics["avg_confidence"] = float(conf_arr.mean())
        metrics["confidence_p25"] = float(np.percentile(conf_arr, 25))
        metrics["confidence_p50"] = float(np.percentile(conf_arr, 50))
        metrics["confidence_p75"] = float(np.percentile(conf_arr, 75))
        metrics["pct_low_confidence"] = float(
            (conf_arr < low_confidence_threshold).mean()
        )
    else:
        metrics["avg_confidence"] = 0.0
        metrics["pct_low_confidence"] = 0.0

    # --- Contradiction metrics ---
    if contradictions_df is not None and len(contradictions_df) > 0:
        total_subjects = max(n_entities + n_relationships, 1)
        unique_contradicted = len(
            set(contradictions_df["subject_a_id"].tolist())
            | set(contradictions_df["subject_b_id"].tolist())
        )
        metrics["pct_contradicted"] = unique_contradicted / total_subjects
    else:
        metrics["pct_contradicted"] = 0.0

    quality_metrics_df = pd.DataFrame([metrics])
    quality_details_df = pd.DataFrame(details_rows) if details_rows else pd.DataFrame(
        columns=["subject_type", "subject_id", "confidence", "evidence_count",
                 "source_count", "contradiction_count", "is_partial", "has_temporal_scope"]
    )

    return quality_metrics_df, quality_details_df
