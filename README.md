# Agent Reliability Harness

A local Python CLI for scoring labeled agent runs, comparing candidate runs with saved baselines, checking judge agreement, and preparing human-review queues. It is a general-purpose evaluation utility; it does not call a model, collect telemetry, or claim production reliability.

## Install

Requires Python 3.10 or newer.

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
pytest
```

The only runtime dependency is Pydantic. Pytest is a development extra. The package makes no network requests. `run` requires an explicit provider; the bundled `echo` example predicts only from `input.baseline_label` and refuses to infer a prediction from `expected_label`. A real application can implement the small `Predictor` protocol and call the package locally.

## Commands

```sh
# Score recorded traces. A failed configured threshold exits 1.
agent-reliability evaluate --gold examples/toy_gold.jsonl \
  --traces examples/toy_traces.jsonl --dataset synthetic-toy-v1 --name baseline

# Run the explicit deterministic example provider and save traces.
agent-reliability run --gold examples/toy_gold.jsonl --provider echo \
  --traces-out reports/toy-run.jsonl --name echo-example

# Compare saved evaluation JSON reports. Limits are maximum absolute drops.
agent-reliability compare --baseline reports/baseline.json \
  --candidate reports/candidate.json --max-accuracy-drop 0.02 \
  --max-recall-drop 0.05 --max-latency-increase-ms 100

# Inspect judge coverage/agreement and export uncertain cases for a person.
agent-reliability calibrate --gold examples/toy_gold.jsonl \
  --traces examples/toy_traces.jsonl
agent-reliability review-export --gold examples/toy_gold.jsonl \
  --traces examples/toy_traces.jsonl --confidence-below 0.7

# Run a complete temporary-directory demo: baseline PASS, regression FAIL.
python examples/run_demo.py
```

Evaluation writes JSON and Markdown. Comparison does the same. CLI exit codes are `0` for a passing evaluation/comparison, `1` for a measured threshold failure, and `2` for invalid input, missing configuration, or operational errors. Judge agreement and review exports are descriptive and exit `0` when inputs are valid; they do not represent a quality threshold.

## Data formats

Each gold JSONL row starts with `schema_version: 1` and contains a unique non-empty `id`, arbitrary JSON `input`, non-empty `expected_label`, and optional object `metadata`. Each trace row also carries `schema_version: 1`, a matching `record_id`, `task_name`, arbitrary JSON `input` and `output`, `predicted_label`, model/prompt versions, and optional nonnegative latency/cost, UTC timestamp, confidence, judge label, and judge score. Unknown fields and unsupported schema versions are rejected. Blank lines, malformed rows, duplicate IDs, empty files, missing predictions, and extra predictions are errors with file and line context where available.

Labels are dataset-defined strings, not a built-in enum. Reports include the union of expected and predicted labels so unexpected labels remain visible in precision/recall and confusion results. Accuracy and per-label metrics describe only the supplied labeled records. Comparison requires the same label set and dataset; configured resource thresholds fail closed if either report has no corresponding measurement. One relative or absolute increase limit may be selected per resource.

## Privacy and limits

The tool operates only on files the caller names. It does not read email, access private datasets, send data to a provider, or upload reports. The included fixtures contain synthetic task examples only. Keep private inputs, traces, and generated reports out of source control. Review exports are inert templates; a blank `human_label` is intentionally left for a person.

Passing synthetic tests or thresholds does not establish model quality, calibrated confidence, customer acceptance, or production reliability. Results depend on representative labels, correct annotation, matching record coverage, and task-specific thresholds. A judge's agreement with gold is not proof of the judge's general quality.

## Development

```sh
pytest
python examples/run_demo.py
```
