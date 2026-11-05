from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from agent_reliability.evaluate import compare_reports, evaluate, judge_calibration, review_csv, review_rows
from agent_reliability.io import load_gold, load_traces
from agent_reliability.models import ComparisonThresholds, GoldRecord, MetricThresholds, TraceRecord
from agent_reliability.providers import EchoBaseline, MissingProviderError, require_provider, run_predictor
from agent_reliability.production import (
    KeyedFixturePredictor, ProductionInput, ReviewCandidate, WeakObservation,
    drift_report, fold_review_labels, mine_review_candidates, shadow_replay,
    weak_observation_report,
)

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "examples/toy_gold.jsonl"
TRACES = ROOT / "examples/toy_traces.jsonl"


def dataset():
    return load_gold(GOLD), load_traces(TRACES)


def test_fixture_scores_open_labels_and_passes_configured_thresholds():
    gold, traces = dataset()
    report = evaluate(gold, traces, dataset="toy", run_name="baseline",
                      thresholds=MetricThresholds(min_accuracy=1, min_recall=1, min_support=2))
    assert report.accuracy == 1
    assert report.passed
    assert report.labels == ["fail", "pass", "review"]
    assert report.confusion["review"]["review"] == 2


def test_false_positive_open_label_appears_in_confusion_and_fails():
    gold, traces = dataset()
    traces[0] = traces[0].model_copy(update={"predicted_label": "novel-label"})
    report = evaluate(gold, traces, dataset="toy", run_name="open-label",
                      thresholds=MetricThresholds(min_accuracy=1, min_recall=1, min_support=1))
    assert "novel-label" in report.labels
    assert report.confusion["pass"]["novel-label"] == 1
    assert not report.passed


def test_rejects_coverage_mismatch():
    gold, traces = dataset()
    with pytest.raises(ValueError, match="coverage mismatch"):
        evaluate(gold, traces[:-1], dataset="toy", run_name="short")


def test_calibration_and_review_reject_incomplete_coverage():
    gold, traces = dataset()
    with pytest.raises(ValueError, match="coverage mismatch"):
        judge_calibration(gold, traces[:-1])
    with pytest.raises(ValueError, match="coverage mismatch"):
        review_rows(gold, traces[:-1], confidence_below=0.7)


def test_jsonl_errors_name_line_and_reject_blank(tmp_path):
    malformed = tmp_path / "bad.jsonl"
    malformed.write_text('{"schema_version":1,"id":"a","input":{},"expected_label":"ok"}\n{oops}\n')
    with pytest.raises(ValueError, match=r"bad.jsonl:2: invalid GoldRecord"):
        load_gold(malformed)
    malformed.write_text('{"schema_version":1,"id":"a","input":{},"expected_label":"ok"}\n\n')
    with pytest.raises(ValueError, match=r"bad.jsonl:2: blank lines"):
        load_gold(malformed)


def test_jsonl_rejects_duplicate_record_ids(tmp_path):
    path = tmp_path / "duplicate.jsonl"
    row = '{"schema_version":1,"id":"same","input":{},"expected_label":"ok"}\n'
    path.write_text(row + row)
    with pytest.raises(ValueError, match="duplicate record id"):
        load_gold(path)


def test_jsonl_rejects_unsupported_schema_version(tmp_path):
    path = tmp_path / "future.jsonl"
    path.write_text('{"schema_version":2,"id":"x","input":{},"expected_label":"ok"}\n')
    with pytest.raises(ValueError, match=r"future.jsonl:1: invalid GoldRecord"):
        load_gold(path)


def test_compare_detects_accuracy_recall_and_absolute_resource_regressions():
    gold, traces = dataset()
    baseline = evaluate(gold, traces, dataset="toy", run_name="base")
    candidate_traces = [t.model_copy(update={"predicted_label": "fail", "latency_ms": 30})
                        if t.record_id == "toy-001" else t for t in traces]
    candidate = evaluate(gold, candidate_traces, dataset="toy", run_name="candidate")
    result = compare_reports(
        baseline, candidate,
        ComparisonThresholds(max_accuracy_drop=0, max_recall_drop=0,
                             max_latency_increase_ms=2),
    )
    assert not result["passed"]
    assert any("accuracy" in item for item in result["regressions"])
    assert any("pass recall" in item for item in result["regressions"])
    assert any("latency" in item for item in result["regressions"])


def test_compare_fails_closed_when_enabled_resource_metric_is_missing():
    gold, traces = dataset()
    baseline = evaluate(gold, traces, dataset="toy", run_name="base")
    no_latency = [t.model_copy(update={"latency_ms": None}) for t in traces]
    candidate = evaluate(gold, no_latency, dataset="toy", run_name="candidate")
    result = compare_reports(
        baseline, candidate,
        ComparisonThresholds(max_latency_increase_pct=0.1),
    )
    assert not result["passed"]
    assert any("missing the metric" in item for item in result["regressions"])


def test_compare_rejects_incompatible_labels():
    gold, traces = dataset()
    base = evaluate(gold, traces, dataset="toy", run_name="base")
    traces[0] = traces[0].model_copy(update={"predicted_label": "new"})
    candidate = evaluate(gold, traces, dataset="toy", run_name="candidate")
    with pytest.raises(ValueError, match="incompatible label sets"):
        compare_reports(base, candidate, ComparisonThresholds())


def test_compare_rejects_reports_from_different_datasets():
    gold, traces = dataset()
    base = evaluate(gold, traces, dataset="toy-v1", run_name="base")
    candidate = evaluate(gold, traces, dataset="toy-v2", run_name="candidate")
    with pytest.raises(ValueError, match="different datasets"):
        compare_reports(base, candidate, ComparisonThresholds())


def test_judge_calibration_counts_coverage_without_overclaiming():
    gold, traces = dataset()
    report = judge_calibration(gold, traces)
    assert report["covered"] == 5
    assert report["uncovered"] == 1
    assert report["agreement"] == 0.8
    assert "does not establish" in report["scope_note"]


def test_review_export_selects_uncertainty_and_disagreements():
    gold, traces = dataset()
    rows = review_rows(gold, traces, confidence_below=0.6)
    assert [row["record_id"] for row in rows] == ["toy-005", "toy-006"]
    assert "judge_disagrees_with_gold" in rows[0]["reasons"]
    assert rows[0]["human_label"] == ""
    assert "human_label" in review_csv(rows)
    with pytest.raises(ValueError, match="between 0 and 1"):
        review_rows(gold, traces, confidence_below=2)


def test_echo_provider_is_explicit_and_does_not_copy_gold_label():
    item = GoldRecord(schema_version=1, id="x", input={"baseline_label": "actual-prediction"}, expected_label="gold")
    traces = run_predictor([item], EchoBaseline(), "toy")
    assert traces[0].predicted_label == "actual-prediction"
    with pytest.raises(MissingProviderError, match="no prediction provider configured"):
        require_provider(None)
    with pytest.raises(RuntimeError, match="requires input.baseline_label"):
        EchoBaseline().predict(GoldRecord(schema_version=1, id="y", input={"text": "unknown"}, expected_label="gold"))


def invoke_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, "-m", "agent_reliability.cli", *args], cwd=ROOT,
                          capture_output=True, text=True, check=False)


def test_cli_run_writes_trace_and_both_reports(tmp_path):
    traces_out = tmp_path / "run.jsonl"
    result = invoke_cli("run", "--gold", str(GOLD), "--provider", "echo",
                        "--traces-out", str(traces_out), "--name", "run-test",
                        "--output", str(tmp_path), "--min-accuracy", "1",
                        "--min-recall", "1", "--min-support", "2")
    assert result.returncode == 0, result.stderr
    assert traces_out.exists()
    assert (tmp_path / "run-test.json").exists()
    assert (tmp_path / "run-test.md").exists()


def test_cli_missing_provider_and_failed_eval_return_distinct_nonzero_codes(tmp_path):
    missing = invoke_cli("run", "--gold", str(GOLD), "--output", str(tmp_path))
    assert missing.returncode == 2
    assert "required: --provider" in missing.stderr
    no_baseline = tmp_path / "no-baseline.jsonl"
    no_baseline.write_text('{"schema_version":1,"id":"x","input":{"text":"unknown"},"expected_label":"gold"}\n')
    invalid_echo = invoke_cli("run", "--gold", str(no_baseline), "--provider", "echo",
                              "--output", str(tmp_path))
    assert invalid_echo.returncode == 2
    assert "requires input.baseline_label" in invalid_echo.stderr
    assert "Traceback" not in invalid_echo.stderr
    bad = [json.loads(line) for line in TRACES.read_text().splitlines()]
    bad[0]["predicted_label"] = "fail"
    trace_path = tmp_path / "bad-traces.jsonl"
    trace_path.write_text("".join(json.dumps(row) + "\n" for row in bad))
    failed = invoke_cli("evaluate", "--gold", str(GOLD), "--traces", str(trace_path),
                        "--min-accuracy", "1", "--output", str(tmp_path), "--name", "bad")
    assert failed.returncode == 1


def test_cli_calibration_and_review_export(tmp_path):
    calibrated = invoke_cli("calibrate", "--gold", str(GOLD), "--traces", str(TRACES),
                            "--output", str(tmp_path / "calibration.json"))
    assert calibrated.returncode == 0, calibrated.stderr
    assert json.loads((tmp_path / "calibration.json").read_text())["covered"] == 5
    exported = invoke_cli("review-export", "--gold", str(GOLD), "--traces", str(TRACES),
                          "--confidence-below", "0.6", "--output", str(tmp_path / "review"))
    assert exported.returncode == 0, exported.stderr
    assert (tmp_path / "review/review.jsonl").read_text().count("\n") == 2
    assert "human_label" in (tmp_path / "review/review.csv").read_text()


def test_cli_compare_writes_report_and_returns_failure_for_regression(tmp_path):
    gold, traces = dataset()
    baseline = evaluate(gold, traces, dataset="toy", run_name="baseline")
    worse_traces = [t.model_copy(update={"predicted_label": "fail"})
                    if t.record_id == "toy-001" else t for t in traces]
    candidate = evaluate(gold, worse_traces, dataset="toy", run_name="candidate")
    base_path, candidate_path = tmp_path / "baseline.json", tmp_path / "candidate.json"
    base_path.write_text(baseline.model_dump_json())
    candidate_path.write_text(candidate.model_dump_json())
    result = invoke_cli("compare", "--baseline", str(base_path), "--candidate", str(candidate_path),
                        "--max-accuracy-drop", "0", "--max-recall-drop", "0",
                        "--output", str(tmp_path), "--name", "comparison")
    assert result.returncode == 1
    assert (tmp_path / "comparison.json").exists()
    assert "FAIL" in (tmp_path / "comparison.md").read_text()


def test_cli_compare_rejects_internally_inconsistent_report_without_traceback(tmp_path):
    gold, traces = dataset()
    report = evaluate(gold, traces, dataset="toy", run_name="baseline")
    raw = report.model_dump(mode="json")
    raw["per_label"] = {}
    base_path, candidate_path = tmp_path / "bad.json", tmp_path / "candidate.json"
    base_path.write_text(json.dumps(raw))
    candidate_path.write_text(report.model_dump_json())
    result = invoke_cli("compare", "--baseline", str(base_path), "--candidate", str(candidate_path),
                        "--output", str(tmp_path))
    assert result.returncode == 2
    assert "Traceback" not in result.stderr
    assert "keys must match labels" in result.stderr


def test_saved_report_rejects_metric_values_that_disagree_with_confusion():
    gold, traces = dataset()
    report = evaluate(gold, traces, dataset="toy", run_name="baseline")
    raw = report.model_dump(mode="json")
    raw["per_label"]["review"]["recall"] = 0.0
    raw["per_label"]["review"]["f1"] = 0.0
    with pytest.raises(ValueError, match="metrics do not match confusion matrix"):
        type(report).model_validate(raw)


def test_cli_rejects_path_like_report_names(tmp_path):
    result = invoke_cli("evaluate", "--gold", str(GOLD), "--traces", str(TRACES),
                        "--name", "../escape", "--output", str(tmp_path))
    assert result.returncode == 2
    assert "name must contain" in result.stderr


def test_shadow_replay_pairs_identical_inputs_and_drift_omits_raw_content():
    gold, _ = dataset()
    incoming = [ProductionInput(schema_version=1, record_id="p1", input={
        "baseline_label": "pass", "candidate_label": "fail", "intent": "novel", "text": "private synthetic string"},
        expected_label="pass")]
    pair = shadow_replay(incoming, KeyedFixturePredictor("base", "baseline_label"),
                         KeyedFixturePredictor("candidate", "candidate_label"), task_name="toy")
    assert pair["baseline"][0].input == pair["candidate"][0].input
    assert pair["baseline"][0].predicted_label == "pass"
    assert pair["candidate"][0].predicted_label == "fail"
    drift = drift_report(gold, incoming)
    assert drift["features"]["intent"]["incoming_novel_value_count"] == 1
    assert "private synthetic string" not in json.dumps(drift)


def test_drift_never_serializes_caller_category_values():
    gold, _ = dataset()
    sensitive = "person@example.test and free form private note " + ("x" * 200)
    incoming = [ProductionInput(schema_version=1, record_id="p1", input={"intent": sensitive})]
    report = json.dumps(drift_report(gold, incoming))
    assert sensitive not in report
    assert "person@example.test" not in report
    assert "incoming_novel_value_count" in report


def test_shadow_replay_rejects_duplicate_ids_and_missing_fixture_key():
    record = ProductionInput(schema_version=1, record_id="x", input={"baseline_label": "ok"})
    with pytest.raises(ValueError, match="duplicate"):
        shadow_replay([record, record], KeyedFixturePredictor("a", "baseline_label"),
                      KeyedFixturePredictor("b", "baseline_label"), task_name="toy")
    with pytest.raises(RuntimeError, match="candidate_label"):
        shadow_replay([record], KeyedFixturePredictor("a", "baseline_label"),
                      KeyedFixturePredictor("b", "candidate_label"), task_name="toy")


def test_shadow_replay_isolates_predictor_input_mutations():
    class Mutator:
        name = "mutator"

        def predict(self, item):
            item.input["baseline_label"] = "corrupted"
            return {}, "pass", 0, 0, None

    incoming = [ProductionInput(schema_version=1, record_id="x", input={
        "baseline_label": "pass", "candidate_label": "fail"})]
    pair = shadow_replay(incoming, Mutator(), KeyedFixturePredictor("candidate", "candidate_label"), task_name="toy")
    assert pair["candidate"][0].predicted_label == "fail"


def test_review_mining_rejects_trace_input_mismatch():
    gold, _ = dataset()
    incoming = [ProductionInput(schema_version=1, record_id="x", input={"intent": "new"})]
    mismatched = [TraceRecord(schema_version=1, record_id="x", task_name="test", input={"intent": "other"},
        output={}, predicted_label="pass")]
    with pytest.raises(ValueError, match="input mismatch"):
        mine_review_candidates(gold, incoming, mismatched, mismatched)


def test_gold_mining_queues_disagreements_and_foldback_requires_human_labels():
    gold, _ = dataset()
    incoming = [ProductionInput(schema_version=1, record_id="new-1", input={
        "baseline_label": "pass", "candidate_label": "fail", "intent": "novel"})]
    baseline = run_predictor([GoldRecord(schema_version=1, id="new-1", input=incoming[0].input,
        expected_label="unlabeled")], KeyedFixturePredictor("base", "baseline_label"), "toy")
    candidate = run_predictor([GoldRecord(schema_version=1, id="new-1", input=incoming[0].input,
        expected_label="unlabeled")], KeyedFixturePredictor("candidate", "candidate_label"), "toy")
    queue = mine_review_candidates(gold, incoming, baseline, candidate)
    assert queue[0].human_label == ""
    assert "novel_input_feature" in queue[0].reasons
    assert "predictor_disagreement" in queue[0].reasons
    with pytest.raises(ValueError, match="no human label"):
        fold_review_labels(gold, queue)
    labeled = [queue[0].model_copy(update={"human_label": "fail"})]
    folded = fold_review_labels(gold, labeled)
    assert len(folded) == len(gold) + 1
    assert folded[-1].expected_label == "fail"


def test_gold_foldback_rejects_overwrite_and_duplicate_review_ids():
    gold, _ = dataset()
    known = ReviewCandidate(record_id=gold[0].id, input=gold[0].input,
        baseline_prediction="pass", candidate_prediction="pass", reasons=["low_confidence"], human_label="fail")
    with pytest.raises(ValueError, match="already exists"):
        fold_review_labels(gold, [known])


def test_weak_observations_use_caller_weights_and_remain_separate_from_gold():
    rows = [WeakObservation(schema_version=1, record_id="a", signal="thumbs_up", source_says_positive=True, confidence_weight=0.5),
            WeakObservation(schema_version=1, record_id="b", signal="complaint", source_says_positive=False, confidence_weight=1)]
    report = weak_observation_report(rows)
    assert report["weighted_positive_rate"] == pytest.approx(1 / 3)
    assert "not gold labels" in report["scope_note"]
    assert "no default signal weights" in report["weighting"]
    with pytest.raises(ValueError, match="duplicate"):
        weak_observation_report([rows[0], rows[0]])
    with pytest.raises(ValueError):
        WeakObservation(schema_version=1, record_id="pii", signal="person@example.test", source_says_positive=False, confidence_weight=1)


def test_cli_mines_review_and_folds_back_only_labeled_rows(tmp_path):
    shadow = invoke_cli("shadow-replay", "--gold", str(GOLD),
        "--inputs", str(ROOT / "examples/toy_production_inputs.jsonl"),
        "--output", str(tmp_path / "shadow"))
    assert shadow.returncode == 0, shadow.stderr
    queue_path = tmp_path / "queue.jsonl"
    mined = invoke_cli("mine-review", "--gold", str(GOLD),
        "--inputs", str(ROOT / "examples/toy_production_inputs.jsonl"),
        "--baseline-traces", str(tmp_path / "shadow/baseline-traces.jsonl"),
        "--candidate-traces", str(tmp_path / "shadow/candidate-traces.jsonl"),
        "--output", str(queue_path))
    assert mined.returncode == 0, mined.stderr
    queue = [json.loads(line) for line in queue_path.read_text().splitlines()]
    assert queue and all(row["human_label"] == "" for row in queue)
    for row in queue:
        row["human_label"] = "review"
    reviewed = tmp_path / "reviewed.jsonl"
    reviewed.write_text("".join(json.dumps(row) + "\n" for row in queue))
    folded_path = tmp_path / "expanded-gold.jsonl"
    folded = invoke_cli("fold-review", "--gold", str(GOLD), "--reviewed", str(reviewed),
                        "--output", str(folded_path))
    assert folded.returncode == 0, folded.stderr
    assert len(load_gold(folded_path)) == len(load_gold(GOLD)) + len(queue)


def test_cli_writes_weak_observation_summary(tmp_path):
    result = invoke_cli("weak-observations", "--observations",
        str(ROOT / "examples/toy_weak_observations.jsonl"), "--output", str(tmp_path / "weak.json"))
    assert result.returncode == 0, result.stderr
    report = json.loads((tmp_path / "weak.json").read_text())
    assert report["observation_count"] == 3
    assert "not gold labels" in report["scope_note"]


def test_cli_shadow_replay_writes_report_and_paired_traces(tmp_path):
    inputs = tmp_path / "incoming.jsonl"
    inputs.write_text(json.dumps({"schema_version": 1, "record_id": "x", "input": {
        "baseline_label": "pass", "candidate_label": "fail", "intent": "novel", "text": "fixture"},
        "expected_label": "pass"}) + "\n")
    result = invoke_cli("shadow-replay", "--gold", str(GOLD), "--inputs", str(inputs),
                        "--output", str(tmp_path / "out"))
    assert result.returncode == 0, result.stderr
    report = json.loads((tmp_path / "out/shadow-report.json").read_text())
    assert report["served_candidate_output"] is False
    assert report["prediction_disagreements"] == 1
    assert report["candidate_evaluation"]["accuracy"] == 0
    assert (tmp_path / "out/baseline-traces.jsonl").exists()
