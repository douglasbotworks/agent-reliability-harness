from pathlib import Path

from agent_reliability.evaluate import evaluate
from agent_reliability.io import load_gold, load_traces
from agent_reliability.models import MetricThresholds

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples/support-routing"


def test_synthetic_support_routing_baseline_passes_and_candidate_regresses():
    gold = load_gold(EXAMPLE / "gold.jsonl")
    baseline = load_traces(EXAMPLE / "baseline-traces.jsonl")
    candidate = load_traces(EXAMPLE / "candidate-traces.jsonl")
    assert len(gold) == len(baseline) == len(candidate) == 20
    assert {row.expected_label for row in gold} == {"billing", "account", "technical", "refund"}
    assert all(row.metadata["kind"] == "synthetic" for row in gold)
    assert all("SYNTHETIC EXAMPLE:" in row.input["text"] for row in gold)
    thresholds = MetricThresholds(min_accuracy=0.96, min_recall=0.80, min_support=5)
    baseline_report = evaluate(gold, baseline, dataset="synthetic-support-routing-v1",
                               run_name="baseline", thresholds=thresholds)
    candidate_report = evaluate(gold, candidate, dataset="synthetic-support-routing-v1",
                                run_name="candidate", thresholds=thresholds)
    assert baseline_report.passed and baseline_report.accuracy == 1
    assert not candidate_report.passed and candidate_report.accuracy == 0.95
    assert candidate_report.per_label["technical"].recall == 0.8
    assert candidate_report.confusion["technical"]["billing"] == 1
