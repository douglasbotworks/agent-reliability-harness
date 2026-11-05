"""Local shadow replay and input-drift analysis for opt-in production adapters.

This module has no network, storage, scheduling, or telemetry behavior. Callers
must explicitly load inputs and supply both predictors.
"""

from __future__ import annotations

import copy
from collections import Counter
from typing import Any, Literal

from pydantic import Field

from .models import GoldRecord, StrictModel, TraceRecord
from .providers import Predictor


class ProductionInput(StrictModel):
    schema_version: int = Field(ge=1, le=1)
    record_id: str = Field(min_length=1)
    input: Any
    # Only present in the synthetic reference pack or after human labeling.
    expected_label: str | None = None


class ReviewCandidate(StrictModel):
    schema_version: Literal[1] = 1
    record_id: str = Field(min_length=1)
    input: Any
    baseline_prediction: str
    candidate_prediction: str
    baseline_confidence: float | None = Field(default=None, ge=0, le=1)
    candidate_confidence: float | None = Field(default=None, ge=0, le=1)
    reasons: list[str] = Field(min_length=1)
    human_label: str = ""


class WeakObservation(StrictModel):
    schema_version: Literal[1] = 1
    record_id: str = Field(min_length=1)
    signal: Literal[
        "reopened_ticket", "corrected_answer", "thumbs_down", "complaint", "escalation",
        "thumbs_up", "fixed_confirmation", "no_follow_up", "reaction_positive", "reaction_negative",
    ]
    source_says_positive: bool
    confidence_weight: float = Field(ge=0, le=1)


class KeyedFixturePredictor:
    """Synthetic-only predictor that reads a caller-selected label from fixture input."""

    def __init__(self, name: str, label_key: str):
        self.name, self.label_key = name, label_key

    def predict(self, item: GoldRecord) -> tuple[Any, str, float | None, float | None, float | None]:
        if not isinstance(item.input, dict) or not isinstance(item.input.get(self.label_key), str):
            raise RuntimeError(f"synthetic fixture input requires string {self.label_key!r}")
        return {"synthetic_fixture": True}, item.input[self.label_key], 0.0, 0.0, None


def shadow_replay(
    records: list[ProductionInput], baseline: Predictor, candidate: Predictor, *, task_name: str
) -> dict[str, list[TraceRecord]]:
    """Run both implementations against exactly the same inputs, serving neither output."""
    if not records:
        raise ValueError("production input contains no records")
    ids = [record.record_id for record in records]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate production input record IDs")
    paired: dict[str, list[TraceRecord]] = {"baseline": [], "candidate": []}
    for record in records:
        item = GoldRecord(schema_version=1, id=record.record_id, input=record.input,
                          expected_label=record.expected_label or "unlabeled")
        for key, predictor in (("baseline", baseline), ("candidate", candidate)):
            # Adapters are independent: mutation by one implementation cannot change
            # the other implementation's view of the shared source record.
            isolated = item.model_copy(deep=True)
            output, label, latency, cost, confidence = predictor.predict(isolated)
            paired[key].append(TraceRecord(
                schema_version=1, record_id=record.record_id, task_name=task_name,
                input=record.input, output=output, predicted_label=label,
                model_version=predictor.name, prompt_version="unspecified",
                latency_ms=latency, cost_usd=cost, confidence=confidence,
            ))
    return paired


def _bucket(value: Any) -> str | None:
    if isinstance(value, str):
        return "empty" if not value else "1-32" if len(value) <= 32 else "33-128" if len(value) <= 128 else "129+"
    return None


def _features(value: Any) -> dict[str, str]:
    """Extract low-cardinality, non-content features suitable for a drift summary."""
    features: dict[str, str] = {}
    if isinstance(value, dict):
        for key in ("intent", "topic", "language"):
            category = value.get(key)
            if isinstance(category, str) and category:
                features[key] = category
        for key in ("text", "message", "prompt"):
            length_bucket = _bucket(value.get(key))
            if length_bucket is not None:
                features[f"{key}_length"] = length_bucket
    elif isinstance(value, str):
        features["input_length"] = _bucket(value) or "empty"
    return features


def drift_report(gold: list[GoldRecord], incoming: list[ProductionInput]) -> dict:
    """Compare feature distributions without serializing caller-provided category values."""
    if not gold or not incoming:
        raise ValueError("gold and incoming inputs must both contain records")
    gold_counts: dict[str, Counter[str]] = {}
    incoming_counts: dict[str, Counter[str]] = {}
    for row, target in [(row, gold_counts) for row in gold] + [(row, incoming_counts) for row in incoming]:
        for feature, value in _features(row.input).items():
            target.setdefault(feature, Counter())[value] += 1
    features = sorted(set(gold_counts) | set(incoming_counts))
    result = {}
    for feature in features:
        base, current = gold_counts.get(feature, Counter()), incoming_counts.get(feature, Counter())
        if feature in {"intent", "topic", "language"}:
            novel = set(current) - set(base)
            novel_records = sum(count for value, count in current.items() if value in novel)
            result[feature] = {
                "gold_distinct_value_count": len(base),
                "incoming_distinct_value_count": len(current),
                "incoming_novel_value_count": len(novel),
                "incoming_novel_record_count": novel_records,
                "incoming_novel_record_rate": round(novel_records / len(incoming), 6),
                "incoming_novel_record_rate_denominator": "all incoming records",
            }
        else:
            # Numeric length buckets are fixed, low-risk labels rather than caller text.
            result[feature] = {
                "gold_distribution": dict(sorted(base.items())),
                "incoming_distribution": dict(sorted(current.items())),
                "novel_values": sorted(set(current) - set(base)),
            }
    return {
        "schema_version": 1,
        "gold_count": len(gold),
        "incoming_count": len(incoming),
        "features": result,
        "novel_feature_values": sum(value.get("incoming_novel_value_count", len(value.get("novel_values", []))) for value in result.values()),
        "scope_note": "Descriptive feature drift only; not a semantic coverage or quality estimate. Caller-provided category values and raw input are omitted.",
    }


def mine_review_candidates(
    gold: list[GoldRecord], incoming: list[ProductionInput],
    baseline: list[TraceRecord], candidate: list[TraceRecord], *, confidence_below: float = 0.7,
) -> list[ReviewCandidate]:
    """Queue novel, uncertain, or disagreeing records; labels remain human-owned."""
    if not 0 <= confidence_below <= 1:
        raise ValueError("confidence threshold must be between 0 and 1")
    gold_values: dict[str, set[str]] = {}
    for record in gold:
        for feature, value in _features(record.input).items():
            gold_values.setdefault(feature, set()).add(value)
    def by_id(rows: list[TraceRecord], name: str) -> dict[str, TraceRecord]:
        result = {row.record_id: row for row in rows}
        if len(result) != len(rows):
            raise ValueError(f"duplicate {name} trace IDs")
        if set(result) != {row.record_id for row in incoming}:
            raise ValueError(f"{name} traces do not match incoming record IDs")
        return result
    base_by_id, candidate_by_id = by_id(baseline, "baseline"), by_id(candidate, "candidate")
    if len({row.record_id for row in incoming}) != len(incoming):
        raise ValueError("duplicate production input record IDs")
    queue = []
    for record in incoming:
        left, right = base_by_id[record.record_id], candidate_by_id[record.record_id]
        if left.input != record.input or right.input != record.input:
            raise ValueError(f"trace input mismatch for record {record.record_id!r}")
        reasons = []
        if any(value not in gold_values.get(feature, set()) for feature, value in _features(record.input).items()):
            reasons.append("novel_input_feature")
        if left.predicted_label != right.predicted_label:
            reasons.append("predictor_disagreement")
        if any(score is not None and score < confidence_below for score in (left.confidence, right.confidence)):
            reasons.append("low_confidence")
        if reasons:
            queue.append(ReviewCandidate(
                record_id=record.record_id, input=copy.deepcopy(record.input),
                baseline_prediction=left.predicted_label, candidate_prediction=right.predicted_label,
                baseline_confidence=left.confidence, candidate_confidence=right.confidence,
                reasons=reasons,
            ))
    return queue


def fold_review_labels(gold: list[GoldRecord], reviewed: list[ReviewCandidate]) -> list[GoldRecord]:
    """Append only explicitly human-labeled queue rows; never infer or overwrite labels."""
    existing = {row.id for row in gold}
    reviewed_ids = [row.record_id for row in reviewed]
    if len(set(reviewed_ids)) != len(reviewed_ids):
        raise ValueError("duplicate review candidate IDs")
    additions = []
    for row in reviewed:
        if not row.human_label.strip():
            raise ValueError(f"review candidate {row.record_id!r} has no human label")
        if row.record_id in existing:
            raise ValueError(f"review candidate {row.record_id!r} already exists in gold")
        existing.add(row.record_id)
        additions.append(GoldRecord(schema_version=1, id=row.record_id, input=row.input,
                                    expected_label=row.human_label.strip(),
                                    metadata={"source": "human_review_foldback"}))
    return [*gold, *additions]


def weak_observation_report(observations: list[WeakObservation]) -> dict:
    """Summarize caller-weighted reactions as weak signals, separate from gold metrics."""
    if not observations:
        raise ValueError("weak observation dataset contains no records")
    if len({row.record_id for row in observations}) != len(observations):
        raise ValueError("duplicate weak observation record IDs")
    total_weight = sum(row.confidence_weight for row in observations)
    positive_weight = sum(row.confidence_weight for row in observations if row.source_says_positive)
    by_signal: dict[str, dict[str, float | int]] = {}
    for signal in sorted({row.signal for row in observations}):
        rows = [row for row in observations if row.signal == signal]
        weight = sum(row.confidence_weight for row in rows)
        by_signal[signal] = {
            "count": len(rows), "total_weight": round(weight, 6),
            "positive_weight": round(sum(row.confidence_weight for row in rows if row.source_says_positive), 6),
        }
    return {
        "schema_version": 1, "kind": "weak_observation_summary",
        "observation_count": len(observations), "positive_weight": round(positive_weight, 6),
        "total_weight": round(total_weight, 6),
        "weighted_positive_rate": round(positive_weight / total_weight, 6) if total_weight else None,
        "by_signal": by_signal,
        "weighting": "confidence_weight supplied by the source adapter; no default signal weights are assumed",
        "scope_note": "Weak reactions are lower-confidence observations, not gold labels or ground truth; never combine silently with labeled evaluation metrics.",
    }
