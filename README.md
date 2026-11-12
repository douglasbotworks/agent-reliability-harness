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

# Walk through a four-queue support-routing regression example (synthetic only).
python examples/support-routing/run_demo.py

# Synthetic post-launch reference: paired predictors see identical inputs;
# candidate output is recorded for evaluation but is never served.
agent-reliability shadow-replay --gold examples/toy_gold.jsonl \
  --inputs examples/toy_production_inputs.jsonl --output reports/shadow

# Select uncertain/novel/disagreeing cases, then a person fills human_label.
agent-reliability mine-review --gold examples/toy_gold.jsonl \
  --inputs examples/toy_production_inputs.jsonl \
  --baseline-traces reports/shadow/baseline-traces.jsonl \
  --candidate-traces reports/shadow/candidate-traces.jsonl \
  --output reports/review/production-review.jsonl

# Fold explicitly labeled rows into a separate expanded gold file.
agent-reliability fold-review --gold examples/toy_gold.jsonl \
  --reviewed reports/review/production-review-labeled.jsonl \
  --output reports/gold-v2.jsonl

# Summarize weak reactions using source-supplied confidence weights.
agent-reliability weak-observations \
  --observations examples/toy_weak_observations.jsonl
```

The support-routing example is a candidate first pilot workflow, not a validated
customer use case. Its balanced 20-row fixture is invented and demonstrates how
a deliberately misrouted ticket can fail configured gates; it makes no claim
about real ticket data or model behavior. See
[`examples/support-routing/README.md`](examples/support-routing/README.md).

Evaluation writes JSON and Markdown. Comparison does the same. CLI exit codes are `0` for a passing evaluation/comparison, `1` for a measured threshold failure, and `2` for invalid input, missing configuration, or operational errors. Judge agreement and review exports are descriptive and exit `0` when inputs are valid; they do not represent a quality threshold.

Comparison reports show baseline and candidate mean, nearest-rank p50, and
nearest-rank p95 latency and cost when those measurements are present. These
percentiles are descriptive only; configured latency and cost regression gates
continue to use means. Missing measurements appear as `not reported` in
Markdown and `null` in JSON. Small samples can make p95 equal the maximum.

## CLI defaults and report scope

`evaluate` and `run` default to dataset ID `dataset`, report name `run`, output
directory `reports/`, minimum accuracy `0.90`, minimum per-label recall `0.80`,
and minimum label support `5`. Accuracy covers every supplied row; the recall
gate applies only to labels with at least five gold examples. These values are
convenience defaults, not recommended acceptance limits. Always provide a
stable, workflow-specific `--dataset` ID for a real comparison: compatibility
checks compare the ID, row count, and label set, not the dataset's contents.

`compare` defaults to allowing up to a 2 percentage-point accuracy drop and a 5
point recall drop for each label. Latency and cost are not gated unless one
absolute or percentage limit is set for each metric. `review-export` and
`mine-review` select confidence below `0.7` by default; calibrate this threshold
to the predictor, because confidence values need not be comparable across
models.

Evaluation reports contain accuracy, per-label support/predicted counts,
precision/recall/F1, the confusion matrix, and mean latency/cost when present.
CLI evaluation reports include SHA-256 digests of the exact gold and trace input
files. For `run` without `--traces-out`, the trace digest covers the canonical
JSONL serialization of the generated trace records. `compare` rejects
reports with different gold-file digests when both reports include provenance.
Older reports without provenance remain comparable by dataset ID, row count, and
label set only. Digests identify bytes; they do not prove who created a file or
that it represents the target population. Reports include mean and nearest-rank
p50/p95 latency and cost for records with measurements, but do not summarize
model/prompt versions from traces. Comparison resource gates still use the mean
values.
Preserve the matching local inputs and trace files with each report, and choose
the dataset, labels, sample, and gates with the team running the pilot.

## Data formats

Each gold JSONL row starts with `schema_version: 1` and contains a unique non-empty `id`, arbitrary JSON `input`, non-empty `expected_label`, and optional object `metadata`. Each trace row also carries `schema_version: 1`, a matching `record_id`, `task_name`, arbitrary JSON `input` and `output`, `predicted_label`, model/prompt versions, and optional nonnegative latency/cost, UTC timestamp, confidence, judge label, and judge score. Unknown fields and unsupported schema versions are rejected. Blank lines, malformed rows, duplicate IDs, empty files, missing predictions, and extra predictions are errors with file and line context where available.

Labels are dataset-defined strings, not a built-in enum. Reports include the union of expected and predicted labels so unexpected labels remain visible in precision/recall and confusion results. Accuracy and per-label metrics describe only the supplied labeled records. Comparison requires the same label set and dataset; configured resource thresholds fail closed if either report has no corresponding measurement. One relative or absolute increase limit may be selected per resource.

## Privacy and limits

The tool operates only on files the caller names. It does not read email, access private datasets, send data to a provider, or upload reports. The included fixtures contain synthetic task examples only. Keep private inputs, traces, and generated reports out of source control. Review exports are inert templates; a blank `human_label` is intentionally left for a person.

Passing synthetic tests or thresholds does not establish model quality, calibrated confidence, customer acceptance, or production reliability. Results depend on representative labels, correct annotation, matching record coverage, and task-specific thresholds. A judge's agreement with gold is not proof of the judge's general quality.

## Production evaluation reference (Phase 2a)

`shadow-replay` is a local, synthetic reference pack. It requires explicitly supplied gold and incoming JSONL files, runs both fixture predictors on each identical input, records paired traces, reports disagreements and coarse input-feature drift, and evaluates quality only when every incoming row has an expected label. Candidate output is never served. The fixture predictors read explicit `baseline_label` and `candidate_label` fields; they are not real models. Drift output contains counts of distinct and novel categorical values plus coarse text-length buckets; it omits caller-provided category values and raw text. The novel-record rate denominator is all incoming records, including rows without that feature. Novel-value counts prompt investigation, not a semantic drift score.

The two paired trace JSONL files intentionally include each supplied input and predictor output. The drift summary omits raw content, but the trace artifacts are not privacy-filtered: a real source adapter must apply its deployment's retention, access, and PII controls to the input and output files. The reference command itself does not upload data or enforce retention.

`mine-review` selects records with a novel coarse feature, baseline/candidate disagreement, or a confidence value below the threshold. The exported queue intentionally includes the original input and should receive the same storage controls as traces. It leaves `human_label` blank. `fold-review` creates a new gold JSONL only when every queued row has a nonempty human label, and rejects duplicate IDs or overwriting existing gold records. It never infers labels.

`weak-observations` summarizes positive/negative reactions with caller-provided `[0, 1]` confidence weights. Signal names must use the bounded schema taxonomy (`reopened_ticket`, `corrected_answer`, `thumbs_down`, `complaint`, `escalation`, `thumbs_up`, `fixed_confirmation`, `no_follow_up`, `reaction_positive`, `reaction_negative`); arbitrary strings are rejected and never copied into report keys. It does not define universal signal weights, infer truth, or merge weak observations into evaluation metrics. Reports label this as weak evidence. The included reactions and weights are synthetic examples.

The library exposes `ProductionInput`, `shadow_replay(records, baseline, candidate, ...)`, and `drift_report(gold, incoming)` for local source adapters. A deployment must supply its own source, retention, PII handling, observation and reporting adapters. This repository does not retain or transmit production data, implement continuous polling, or choose customer sampling, retention, labeling ownership, or hosting. Gold-set mining and weak-label scoring remain later phases.

## Development

```sh
pytest
python examples/run_demo.py
```
