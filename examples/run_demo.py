"""Run a passing toy evaluation and a deliberately failing regression comparison."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "examples/toy_gold.jsonl"
TRACES = ROOT / "examples/toy_traces.jsonl"


def cli(*args: str, expected: int = 0) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "agent_reliability.cli", *args],
        cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        check=False,
    )
    print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    if result.returncode != expected:
        raise SystemExit(f"command returned {result.returncode}, expected {expected}")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="agent-reliability-demo-") as temp:
        output = Path(temp)
        cli("evaluate", "--gold", str(GOLD), "--traces", str(TRACES),
            "--dataset", "synthetic-toy-v1", "--name", "baseline",
            "--output", str(output), "--min-accuracy", "1", "--min-recall", "1",
            "--min-support", "2")

        raw = [json.loads(line) for line in TRACES.read_text(encoding="utf-8").splitlines()]
        raw[-1]["predicted_label"] = "fail"
        candidate_traces = output / "candidate.jsonl"
        candidate_traces.write_text("".join(json.dumps(row) + "\n" for row in raw), encoding="utf-8")
        cli("evaluate", "--gold", str(GOLD), "--traces", str(candidate_traces),
            "--dataset", "synthetic-toy-v1", "--name", "candidate", "--output", str(output),
            "--min-accuracy", "1", "--min-recall", "1", "--min-support", "2", expected=1)
        cli("compare", "--baseline", str(output / "baseline.json"),
            "--candidate", str(output / "candidate.json"), "--name", "regression-demo",
            "--output", str(output), "--max-accuracy-drop", "0", "--max-recall-drop", "0", expected=1)
        print("Demo complete: the synthetic baseline passed and the deliberate regression failed.")


if __name__ == "__main__":
    main()
