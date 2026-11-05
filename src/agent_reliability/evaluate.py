"""Open-label classification metrics, reports, judge agreement, and comparisons."""

from __future__ import annotations

import csv
import io
from collections import Counter
from statistics import fmean
from typing import Iterable

from .models import (
    ComparisonThresholds,
    EvaluationReport,
    GoldRecord,
    LabelMetric,
    MetricThresholds,
    TraceRecord,
    ensure_compatible_reports,
)


def evaluate(
    gold: list[GoldRecord], traces: list[TraceRecord], *, dataset: str, run_name: str,
    thresholds: MetricThresholds | None = None,
) -> EvaluationReport:
    thresholds = thresholds or MetricThresholds()
    if not gold:
        raise ValueError("gold dataset contains no records")
    gold_by_id = {item.id: item for item in gold}
    trace_by_id = {item.record_id: item for item in traces}
    if len(gold_by_id) != len(gold) or len(trace_by_id) != len(traces):
        raise ValueError("duplicate IDs in gold records or traces")
    missing = sorted(gold_by_id.keys() - trace_by_id.keys())
    extra = sorted(trace_by_id.keys() - gold_by_id.keys())
    if missing or extra:
        raise ValueError(f"record coverage mismatch: missing traces={missing}; unexpected traces={extra}")

    labels = sorted(
        {item.expected_label for item in gold} | {item.predicted_label for item in traces}
    )
    matrix: dict[str, Counter[str]] = {label: Counter() for label in labels}
    support: Counter[str] = Counter()
    predicted: Counter[str] = Counter()
    correct = 0
    for record_id, item in gold_by_id.items():
        guess = trace_by_id[record_id].predicted_label
        target = item.expected_label
        matrix[target][guess] += 1
        support[target] += 1
        predicted[guess] += 1
        correct += target == guess

    per_label: dict[str, LabelMetric] = {}
    for label in labels:
        tp = matrix[label][label]
        precision = tp / predicted[label] if predicted[label] else 0.0
        recall = tp / support[label] if support[label] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label[label] = LabelMetric(
            support=support[label], predicted=predicted[label],
            precision=round(precision, 6), recall=round(recall, 6), f1=round(f1, 6),
        )
    accuracy = correct / len(gold)
    passed = accuracy >= thresholds.min_accuracy and all(
        metric.recall >= thresholds.min_recall
        for metric in per_label.values()
        if metric.support >= thresholds.min_support
    )
    latencies = [trace.latency_ms for trace in traces if trace.latency_ms is not None]
    costs = [trace.cost_usd for trace in traces if trace.cost_usd is not None]
    confusion = {label: {pred: matrix[label][pred] for pred in labels} for label in labels}
    return EvaluationReport(
        dataset=dataset, run_name=run_name, record_count=len(gold), accuracy=round(accuracy, 6),
        labels=labels, per_label=per_label, confusion=confusion,
        mean_latency_ms=round(fmean(latencies), 6) if latencies else None,
        mean_cost_usd=round(fmean(costs), 8) if costs else None,
        thresholds=thresholds, passed=passed,
    )


def report_markdown(report: EvaluationReport) -> str:
    status = "PASS" if report.passed else "FAIL"
    lines = [
        f"# Evaluation: {report.run_name}", "", f"- Dataset: `{report.dataset}`",
        f"- Records: {report.record_count}", f"- Accuracy: {report.accuracy:.1%}",
        f"- Threshold result: **{status}** (minimum accuracy {report.thresholds.min_accuracy:.1%}; "
        f"minimum recall {report.thresholds.min_recall:.1%} for support ≥ {report.thresholds.min_support})",
        "", "| Label | Support | Predicted | Precision | Recall | F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label in report.labels:
        metric = report.per_label[label]
        lines.append(
            f"| {_md_cell(label)} | {metric.support} | {metric.predicted} | {metric.precision:.3f} | "
            f"{metric.recall:.3f} | {metric.f1:.3f} |"
        )
    safe_labels = [_md_cell(label) for label in report.labels]
    lines += ["", "## Confusion matrix (expected → predicted)", "", "| Expected \\ predicted | " +
              " | ".join(safe_labels) + " |", "|" + "---|" * (len(report.labels) + 1)]
    for expected in report.labels:
        lines.append("| " + _md_cell(expected) + " | " + " | ".join(
            str(report.confusion[expected][pred]) for pred in report.labels
        ) + " |")
    lines += ["", f"- Mean latency: {_fmt(report.mean_latency_ms, ' ms')}",
              f"- Mean cost: {_fmt(report.mean_cost_usd, ' USD')}", "",
              "Metrics describe only this dataset and run; passing thresholds do not establish production reliability.", ""]
    return "\n".join(lines)


def _fmt(value: float | None, suffix: str) -> str:
    return "not reported" if value is None else f"{value:.4f}{suffix}"


def _md_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def _validate_trace_coverage(gold: list[GoldRecord], traces: list[TraceRecord]) -> dict[str, GoldRecord]:
    if not gold:
        raise ValueError("gold dataset contains no records")
    gold_by_id = {record.id: record for record in gold}
    trace_ids = [trace.record_id for trace in traces]
    if len(gold_by_id) != len(gold):
        raise ValueError("duplicate IDs in gold records")
    if len(set(trace_ids)) != len(trace_ids):
        raise ValueError("duplicate IDs in traces")
    missing, extra = sorted(gold_by_id.keys() - set(trace_ids)), sorted(set(trace_ids) - gold_by_id.keys())
    if missing or extra:
        raise ValueError(f"record coverage mismatch: missing traces={missing}; unexpected traces={extra}")
    return gold_by_id


def judge_calibration(gold: list[GoldRecord], traces: list[TraceRecord]) -> dict:
    gold_by_id = _validate_trace_coverage(gold, traces)
    covered = [trace for trace in traces if trace.judge_label is not None]
    labels = sorted({record.expected_label for record in gold} | {t.judge_label for t in covered})
    hits = sum(gold_by_id[trace.record_id].expected_label == trace.judge_label for trace in covered)
    per_label: dict[str, dict[str, int | float]] = {}
    for label in labels:
        covered_ids = {t.record_id for t in covered}
        support_count = sum(r.expected_label == label and r.id in covered_ids for r in gold)
        predicted_count = sum(t.judge_label == label for t in covered)
        true_positive = sum(t.judge_label == label and gold_by_id[t.record_id].expected_label == label for t in covered)
        per_label[label] = {
            "support": support_count, "predicted": predicted_count,
            "precision": round(true_positive / predicted_count, 6) if predicted_count else 0.0,
            "recall": round(true_positive / support_count, 6) if support_count else 0.0,
        }
    return {
        "schema_version": 1, "total": len(traces), "covered": len(covered),
        "uncovered": len(traces) - len(covered), "agreement": round(hits / len(covered), 6) if covered else None,
        "per_label": per_label,
        "scope_note": "Agreement is measured only on these labeled records; it does not establish general judge quality.",
    }


def compare_reports(base: EvaluationReport, candidate: EvaluationReport, limits: ComparisonThresholds) -> dict:
    ensure_compatible_reports(base, candidate)
    regressions: list[str] = []
    acc_drop = base.accuracy - candidate.accuracy
    if acc_drop > limits.max_accuracy_drop:
        regressions.append(f"accuracy dropped by {acc_drop:.3%} (limit {limits.max_accuracy_drop:.3%})")
    for label in base.labels:
        before, after = base.per_label[label].recall, candidate.per_label[label].recall
        drop = before - after
        if drop > limits.max_recall_drop:
            regressions.append(
                f"{label} recall dropped by {drop:.3%} (limit {limits.max_recall_drop:.3%})"
            )
    _compare_resource("latency", base.mean_latency_ms, candidate.mean_latency_ms,
                      limits.max_latency_increase_pct, limits.max_latency_increase_ms, regressions)
    _compare_resource("cost", base.mean_cost_usd, candidate.mean_cost_usd,
                      limits.max_cost_increase_pct, limits.max_cost_increase_usd, regressions)
    return {
        "schema_version": 1, "baseline": base.run_name, "candidate": candidate.run_name,
        "passed": not regressions, "regressions": regressions,
        "accuracy_drop": round(acc_drop, 6), "limits": limits.model_dump(mode="json"),
    }


def _compare_resource(name: str, before: float | None, after: float | None,
                      pct_limit: float | None, absolute_limit: float | None,
                      regressions: list[str]) -> None:
    if pct_limit is None and absolute_limit is None:
        return
    if before is None or after is None:
        regressions.append(f"{name} comparison requested but a report is missing the metric")
    elif pct_limit is not None:
        increase = (after - before) / before if before > 0 else (0.0 if after == 0 else float("inf"))
        if increase > pct_limit:
            regressions.append(f"{name} increased by {increase:.3%} (limit {pct_limit:.3%})")
    elif absolute_limit is not None and after - before > absolute_limit:
        regressions.append(f"{name} increased by {after - before:.6g} (limit {absolute_limit:.6g})")


def comparison_markdown(result: dict) -> str:
    lines = [f"# Comparison: {result['baseline']} → {result['candidate']}", "",
             f"- Result: **{'PASS' if result['passed'] else 'FAIL'}**", ""]
    lines.extend(f"- Regression: {reason}" for reason in result["regressions"])
    if not result["regressions"]:
        lines.append("No configured regression threshold was exceeded.")
    lines += ["", "This comparison is limited to the saved reports and configured thresholds.", ""]
    return "\n".join(lines)


def review_rows(gold: list[GoldRecord], traces: list[TraceRecord], confidence_below: float) -> list[dict]:
    if not 0 <= confidence_below <= 1:
        raise ValueError("confidence threshold must be between 0 and 1")
    gold_by_id = _validate_trace_coverage(gold, traces)
    rows: list[dict] = []
    for trace in traces:
        expected = gold_by_id[trace.record_id].expected_label
        reasons: list[str] = []
        if trace.confidence is not None and trace.confidence < confidence_below:
            reasons.append("low_confidence")
        if trace.judge_label is not None and trace.judge_label != expected:
            reasons.append("judge_disagrees_with_gold")
        if trace.predicted_label != expected:
            reasons.append("prediction_disagrees_with_gold")
        if reasons:
            rows.append({
                "schema_version": 1,
                "record_id": trace.record_id, "task_name": trace.task_name,
                "expected_label": expected, "predicted_label": trace.predicted_label,
                "judge_label": trace.judge_label, "confidence": trace.confidence,
                "reasons": reasons, "human_label": "",
            })
    return rows


def review_csv(rows: list[dict]) -> str:
    stream = io.StringIO(newline="")
    fields = ["schema_version", "record_id", "task_name", "expected_label", "predicted_label", "judge_label",
              "confidence", "reasons", "human_label"]
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({**row, "reasons": ";".join(row["reasons"])})
    return stream.getvalue()
