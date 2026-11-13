"""Validated interchange models for gold records, traces, and reports."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GoldRecord(StrictModel):
    schema_version: Literal[1]
    id: str = Field(min_length=1)
    input: Any
    expected_label: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TraceRecord(StrictModel):
    """One observed execution; labels stay open-ended and dataset-defined."""

    schema_version: Literal[1]
    record_id: str = Field(min_length=1)
    task_name: str = Field(min_length=1)
    input: Any
    output: Any
    predicted_label: str = Field(min_length=1)
    model_version: str = "unknown"
    prompt_version: str = "unknown"
    latency_ms: float | None = Field(default=None, ge=0)
    cost_usd: float | None = Field(default=None, ge=0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    confidence: float | None = Field(default=None, ge=0, le=1)
    judge_label: str | None = None
    judge_score: float | None = Field(default=None, ge=0, le=1)


class MetricThresholds(StrictModel):
    min_accuracy: float = Field(default=0.0, ge=0, le=1)
    min_recall: float = Field(default=0.0, ge=0, le=1)
    min_support: int = Field(default=1, ge=1)


class ComparisonThresholds(StrictModel):
    max_accuracy_drop: float = Field(default=0.02, ge=0, le=1)
    max_recall_drop: float = Field(default=0.05, ge=0, le=1)
    max_latency_increase_pct: float | None = Field(default=None, ge=0)
    max_cost_increase_pct: float | None = Field(default=None, ge=0)
    max_latency_increase_ms: float | None = Field(default=None, ge=0)
    max_cost_increase_usd: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def reject_ambiguous_metric_limits(self) -> "ComparisonThresholds":
        if self.max_latency_increase_pct is not None and self.max_latency_increase_ms is not None:
            raise ValueError("choose one latency increase limit: percent or milliseconds")
        if self.max_cost_increase_pct is not None and self.max_cost_increase_usd is not None:
            raise ValueError("choose one cost increase limit: percent or USD")
        return self


class LabelMetric(StrictModel):
    support: int = Field(ge=0)
    predicted: int = Field(ge=0)
    precision: float = Field(ge=0, le=1)
    recall: float = Field(ge=0, le=1)
    f1: float = Field(ge=0, le=1)


class EvaluationReport(StrictModel):
    schema_version: Literal[1] = 1
    dataset: str
    run_name: str
    record_count: int = Field(ge=1)
    accuracy: float = Field(ge=0, le=1)
    labels: list[str]
    per_label: dict[str, LabelMetric]
    confusion: dict[str, dict[str, int]]
    mean_latency_ms: float | None
    mean_cost_usd: float | None
    p50_latency_ms: float | None = Field(default=None, ge=0)
    p95_latency_ms: float | None = Field(default=None, ge=0)
    p50_cost_usd: float | None = Field(default=None, ge=0)
    p95_cost_usd: float | None = Field(default=None, ge=0)
    thresholds: MetricThresholds
    passed: bool
    provenance: dict[str, str] | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def validate_provenance(self) -> "EvaluationReport":
        if self.provenance is not None:
            if set(self.provenance) != {"gold_sha256", "traces_sha256"}:
                raise ValueError("provenance must contain gold_sha256 and traces_sha256")
            for digest in self.provenance.values():
                if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                    raise ValueError("provenance digests must be lowercase SHA-256 hex")
        return self

    @model_validator(mode="after")
    def validate_report_consistency(self) -> "EvaluationReport":
        labels = set(self.labels)
        if len(labels) != len(self.labels) or not labels:
            raise ValueError("report labels must be non-empty and unique")
        if set(self.per_label) != labels or set(self.confusion) != labels:
            raise ValueError("report metric and confusion keys must match labels")
        total = 0
        correct = 0
        for expected in self.labels:
            row = self.confusion[expected]
            if set(row) != labels or any(not isinstance(count, int) or count < 0 for count in row.values()):
                raise ValueError("confusion rows must contain nonnegative counts for every label")
            support = sum(row.values())
            predicted = sum(self.confusion[label][expected] for label in self.labels)
            metric = self.per_label[expected]
            if metric.support != support or metric.predicted != predicted:
                raise ValueError(f"report counts do not match confusion matrix for label {expected!r}")
            true_positive = row[expected]
            precision = true_positive / predicted if predicted else 0.0
            recall = true_positive / support if support else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            if any(abs(actual - expected_value) > 1e-5 for actual, expected_value in (
                (metric.precision, precision), (metric.recall, recall), (metric.f1, f1),
            )):
                raise ValueError(f"report metrics do not match confusion matrix for label {expected!r}")
            total += support
            correct += row[expected]
        if total != self.record_count or abs(correct / total - self.accuracy) > 1e-5:
            raise ValueError("report record count or accuracy does not match confusion matrix")
        expected_pass = self.accuracy >= self.thresholds.min_accuracy and all(
            metric.recall >= self.thresholds.min_recall
            for metric in self.per_label.values()
            if metric.support >= self.thresholds.min_support
        )
        if self.passed != expected_pass:
            raise ValueError("report pass status does not match metrics and thresholds")
        return self


def ensure_compatible_reports(base: EvaluationReport, candidate: EvaluationReport) -> None:
    if base.dataset != candidate.dataset:
        raise ValueError("reports use different datasets")
    if base.record_count != candidate.record_count:
        raise ValueError("reports use different record counts")
    if set(base.labels) != set(candidate.labels):
        raise ValueError("reports use incompatible label sets")
    if (base.provenance is not None and candidate.provenance is not None
            and base.provenance["gold_sha256"] != candidate.provenance["gold_sha256"]):
        raise ValueError("reports use different gold file contents")
