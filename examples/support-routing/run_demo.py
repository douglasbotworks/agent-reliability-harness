"""Run a fully synthetic support-routing baseline and regression comparison."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = Path(__file__).resolve().parent


def cli(*args: str, expected: int = 0) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "agent_reliability.cli", *args],
        cwd=ROOT, text=True, check=False,
    )
    if result.returncode != expected:
        raise SystemExit(f"command returned {result.returncode}; expected {expected}")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="agent-reliability-routing-") as temp:
        output = Path(temp)
        common = ("--gold", str(EXAMPLE / "gold.jsonl"),
                  "--dataset", "synthetic-support-routing-v1",
                  "--output", str(output), "--min-accuracy", "0.96",
                  "--min-recall", "0.80", "--min-support", "5")
        cli("evaluate", *common, "--traces", str(EXAMPLE / "baseline-traces.jsonl"),
            "--name", "synthetic-baseline")
        cli("evaluate", *common, "--traces", str(EXAMPLE / "candidate-traces.jsonl"),
            "--name", "synthetic-candidate", expected=1)
        cli("compare", "--baseline", str(output / "synthetic-baseline.json"),
            "--candidate", str(output / "synthetic-candidate.json"),
            "--name", "synthetic-routing-comparison", "--output", str(output),
            "--max-accuracy-drop", "0.02", "--max-recall-drop", "0.05", expected=1)
        print("Synthetic demo complete: baseline PASS; deliberate candidate regression FAIL.")


if __name__ == "__main__":
    main()
