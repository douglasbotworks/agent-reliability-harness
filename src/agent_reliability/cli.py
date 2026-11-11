"""Command line interface for local, labeled agent evaluations."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

from pydantic import ValidationError

from .evaluate import (
    compare_reports, comparison_markdown, evaluate, judge_calibration,
    report_markdown, review_csv, review_rows,
)
from .io import load_gold, load_jsonl, load_traces, write_json, write_jsonl
from .models import ComparisonThresholds, EvaluationReport, MetricThresholds
from .providers import require_provider, run_predictor
from .production import (
    KeyedFixturePredictor, ProductionInput, ReviewCandidate, WeakObservation,
    drift_report, fold_review_labels, mine_review_candidates, shadow_replay,
    weak_observation_report,
)


def _thresholds(args: argparse.Namespace) -> MetricThresholds:
    return MetricThresholds(
        min_accuracy=args.min_accuracy, min_recall=args.min_recall,
        min_support=args.min_support,
    )


def _report_paths(output: Path, name: str) -> tuple[Path, Path]:
    _validate_name(name)
    output.mkdir(parents=True, exist_ok=True)
    return output / f"{name}.json", output / f"{name}.md"


def _validate_name(name: str) -> None:
    if name in {".", ".."} or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name):
        raise ValueError("name must contain only letters, digits, dot, underscore, or hyphen and start alphanumeric")


def _save_report(report: EvaluationReport, output: Path) -> tuple[Path, Path]:
    json_path, md_path = _report_paths(output, report.run_name)
    write_json(json_path, report)
    md_path.write_text(report_markdown(report), encoding="utf-8")
    return json_path, md_path


def _evaluate_command(args: argparse.Namespace) -> int:
    gold = load_gold(args.gold)
    traces = load_traces(args.traces)
    report = evaluate(gold, traces, dataset=args.dataset, run_name=args.name,
                      thresholds=_thresholds(args))
    report = _with_provenance(report, _input_provenance(args.gold, args.traces))
    json_path, md_path = _save_report(report, args.output)
    print(report_markdown(report))
    print(f"JSON report: {json_path}\nMarkdown report: {md_path}")
    return 0 if report.passed else 1


def _run_command(args: argparse.Namespace) -> int:
    gold = load_gold(args.gold)
    predictor = require_provider(args.provider)
    traces = run_predictor(gold, predictor, args.task)
    if args.traces_out:
        write_jsonl(args.traces_out, traces)
    report = evaluate(gold, traces, dataset=args.dataset, run_name=args.name,
                      thresholds=_thresholds(args))
    report = _with_provenance(report, {
        "gold_sha256": _sha256(args.gold),
        "traces_sha256": _sha256(args.traces_out) if args.traces_out else _sha256_records(traces),
    })
    json_path, md_path = _save_report(report, args.output)
    print(report_markdown(report))
    if args.traces_out:
        print(f"Traces: {args.traces_out}")
    print(f"JSON report: {json_path}\nMarkdown report: {md_path}")
    return 0 if report.passed else 1


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_records(records: list) -> str:
    payload = "".join(json.dumps(record.model_dump(mode="json"), sort_keys=True) + "\n"
                      for record in records).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _input_provenance(gold_path: Path, traces_path: Path) -> dict[str, str]:
    return {"gold_sha256": _sha256(gold_path), "traces_sha256": _sha256(traces_path)}


def _with_provenance(report: EvaluationReport, provenance: dict[str, str]) -> EvaluationReport:
    return EvaluationReport.model_validate({**report.model_dump(), "provenance": provenance})


def _compare_command(args: argparse.Namespace) -> int:
    _validate_name(args.name)
    base = EvaluationReport.model_validate_json(args.baseline.read_text(encoding="utf-8"))
    candidate = EvaluationReport.model_validate_json(args.candidate.read_text(encoding="utf-8"))
    limits = ComparisonThresholds(
        max_accuracy_drop=args.max_accuracy_drop,
        max_recall_drop=args.max_recall_drop,
        max_latency_increase_pct=args.max_latency_increase_pct,
        max_cost_increase_pct=args.max_cost_increase_pct,
        max_latency_increase_ms=args.max_latency_increase_ms,
        max_cost_increase_usd=args.max_cost_increase_usd,
    )
    result = compare_reports(base, candidate, limits)
    args.output.mkdir(parents=True, exist_ok=True)
    json_path, md_path = args.output / f"{args.name}.json", args.output / f"{args.name}.md"
    write_json(json_path, result)
    md_path.write_text(comparison_markdown(result), encoding="utf-8")
    print(comparison_markdown(result))
    print(f"JSON comparison: {json_path}\nMarkdown comparison: {md_path}")
    return 0 if result["passed"] else 1


def _calibrate_command(args: argparse.Namespace) -> int:
    result = judge_calibration(load_gold(args.gold), load_traces(args.traces))
    write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"Calibration report: {args.output}")
    return 0


def _review_command(args: argparse.Namespace) -> int:
    rows = review_rows(load_gold(args.gold), load_traces(args.traces), args.confidence_below)
    args.output.mkdir(parents=True, exist_ok=True)
    jsonl_path, csv_path = args.output / "review.jsonl", args.output / "review.csv"
    # JSONL rows are ordinary dictionaries rather than model instances.
    jsonl_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    csv_path.write_text(review_csv(rows), encoding="utf-8", newline="")
    print(f"Selected {len(rows)} records for human review.\nJSONL: {jsonl_path}\nCSV: {csv_path}")
    return 0


def _shadow_command(args: argparse.Namespace) -> int:
    gold = load_gold(args.gold)
    incoming = load_jsonl(args.inputs, ProductionInput)
    pair = shadow_replay(incoming,
        KeyedFixturePredictor("synthetic-baseline", args.baseline_key),
        KeyedFixturePredictor("synthetic-candidate", args.candidate_key), task_name=args.task)
    expected = {row.record_id: row.expected_label for row in incoming}
    labeled = all(label is not None for label in expected.values())
    result: dict = {
        "schema_version": 1, "mode": "shadow_replay", "served_candidate_output": False,
        "same_input_count": len(incoming), "prediction_disagreements": sum(
            left.predicted_label != right.predicted_label for left, right in zip(pair["baseline"], pair["candidate"])),
        "drift": drift_report(gold, incoming),
        "scope_note": "Synthetic fixture workflow. Prediction disagreement is not a quality estimate without human-validated labels.",
    }
    if labeled:
        labeled_gold = [type(gold[0])(schema_version=1, id=row.record_id, input=row.input,
                                     expected_label=row.expected_label or "") for row in incoming]
        result["baseline_evaluation"] = evaluate(labeled_gold, pair["baseline"], dataset=args.dataset,
            run_name="shadow-baseline", thresholds=MetricThresholds(min_accuracy=0, min_recall=0)).model_dump(mode="json")
        result["candidate_evaluation"] = evaluate(labeled_gold, pair["candidate"], dataset=args.dataset,
            run_name="shadow-candidate", thresholds=MetricThresholds(min_accuracy=0, min_recall=0)).model_dump(mode="json")
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "shadow-report.json", result)
    write_jsonl(args.output / "baseline-traces.jsonl", pair["baseline"])
    write_jsonl(args.output / "candidate-traces.jsonl", pair["candidate"])
    print(json.dumps({k: v for k, v in result.items() if k not in {"baseline_evaluation", "candidate_evaluation"}}, indent=2))
    print(f"Shadow report and paired traces: {args.output}")
    return 0


def _mine_review_command(args: argparse.Namespace) -> int:
    rows = mine_review_candidates(load_gold(args.gold), load_jsonl(args.inputs, ProductionInput),
        load_traces(args.baseline_traces), load_traces(args.candidate_traces),
        confidence_below=args.confidence_below)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, rows)
    print(f"Selected {len(rows)} records for human review: {args.output}")
    return 0


def _fold_review_command(args: argparse.Namespace) -> int:
    updated = fold_review_labels(load_gold(args.gold), load_jsonl(args.reviewed, ReviewCandidate))
    write_jsonl(args.output, updated)
    print(f"Wrote gold set with {len(updated)} records: {args.output}")
    return 0


def _weak_observations_command(args: argparse.Namespace) -> int:
    result = weak_observation_report(load_jsonl(args.observations, WeakObservation))
    write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"Weak observation report: {args.output}")
    return 0


def _add_eval_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--name", default="run")
    parser.add_argument("--output", type=Path, default=Path("reports"))
    parser.add_argument("--min-accuracy", type=float, default=0.90)
    parser.add_argument("--min-recall", type=float, default=0.80)
    parser.add_argument("--min-support", type=int, default=5)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agent-reliability", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="execute an explicitly configured predictor and score it")
    run.add_argument("--gold", type=Path, required=True)
    run.add_argument("--provider", choices=("echo",), required=True,
                     help="echo only uses input.baseline_label; no network provider is bundled")
    run.add_argument("--task", default="classification")
    run.add_argument("--traces-out", type=Path)
    _add_eval_options(run)
    run.set_defaults(handler=_run_command)

    ev = commands.add_parser("evaluate", help="score existing trace JSONL against gold JSONL")
    ev.add_argument("--gold", type=Path, required=True)
    ev.add_argument("--traces", type=Path, required=True)
    _add_eval_options(ev)
    ev.set_defaults(handler=_evaluate_command)

    cmp = commands.add_parser("compare", help="compare saved evaluation report JSON files")
    cmp.add_argument("--baseline", type=Path, required=True)
    cmp.add_argument("--candidate", type=Path, required=True)
    cmp.add_argument("--name", default="comparison")
    cmp.add_argument("--output", type=Path, default=Path("reports"))
    cmp.add_argument("--max-accuracy-drop", type=float, default=0.02)
    cmp.add_argument("--max-recall-drop", type=float, default=0.05)
    cmp.add_argument("--max-latency-increase-pct", type=float)
    cmp.add_argument("--max-latency-increase-ms", type=float)
    cmp.add_argument("--max-cost-increase-pct", type=float)
    cmp.add_argument("--max-cost-increase-usd", type=float)
    cmp.set_defaults(handler=_compare_command)

    cal = commands.add_parser("calibrate", help="measure judge/gold agreement and coverage")
    cal.add_argument("--gold", type=Path, required=True)
    cal.add_argument("--traces", type=Path, required=True)
    cal.add_argument("--output", type=Path, default=Path("reports/calibration.json"))
    cal.set_defaults(handler=_calibrate_command)

    review = commands.add_parser("review-export", help="export low-confidence and disagreement rows")
    review.add_argument("--gold", type=Path, required=True)
    review.add_argument("--traces", type=Path, required=True)
    review.add_argument("--confidence-below", type=float, default=0.7)
    review.add_argument("--output", type=Path, default=Path("review-export"))
    review.set_defaults(handler=_review_command)

    shadow = commands.add_parser("shadow-replay", help="synthetic reference: compare two predictors on identical local inputs")
    shadow.add_argument("--gold", type=Path, required=True)
    shadow.add_argument("--inputs", type=Path, required=True)
    shadow.add_argument("--baseline-key", default="baseline_label")
    shadow.add_argument("--candidate-key", default="candidate_label")
    shadow.add_argument("--task", default="synthetic-production")
    shadow.add_argument("--dataset", default="synthetic-reference")
    shadow.add_argument("--output", type=Path, default=Path("reports/shadow"))
    shadow.set_defaults(handler=_shadow_command)

    mine = commands.add_parser("mine-review", help="select novel/uncertain synthetic reference cases for human labeling")
    mine.add_argument("--gold", type=Path, required=True)
    mine.add_argument("--inputs", type=Path, required=True)
    mine.add_argument("--baseline-traces", type=Path, required=True)
    mine.add_argument("--candidate-traces", type=Path, required=True)
    mine.add_argument("--confidence-below", type=float, default=0.7)
    mine.add_argument("--output", type=Path, default=Path("review-export/production-review.jsonl"))
    mine.set_defaults(handler=_mine_review_command)

    fold = commands.add_parser("fold-review", help="fold only explicitly human-labeled review rows into a new gold JSONL")
    fold.add_argument("--gold", type=Path, required=True)
    fold.add_argument("--reviewed", type=Path, required=True)
    fold.add_argument("--output", type=Path, required=True)
    fold.set_defaults(handler=_fold_review_command)

    weak = commands.add_parser("weak-observations", help="summarize confidence-weighted reactions separately from gold metrics")
    weak.add_argument("--observations", type=Path, required=True)
    weak.add_argument("--output", type=Path, default=Path("reports/weak-observations.json"))
    weak.set_defaults(handler=_weak_observations_command)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except (OSError, ValueError, ValidationError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
