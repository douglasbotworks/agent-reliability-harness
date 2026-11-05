# Market validation and first paid pilot

## Assessment

There is a real problem category: teams need repeatable evaluations when they change prompts, models, or agent code. Established products already teach and sell this workflow. For example, Braintrust describes evaluations as datasets, tasks, and scorers, with saved experiments and comparisons. That confirms demand for the category, while also showing that a general-purpose eval platform is a crowded position ([Braintrust eval guide](https://www.braintrust.dev/articles/how-to-eval), [Braintrust eval course](https://www.braintrust.dev/foundations/build-a-simple-eval-in-braintrust-ui)).

The current harness is not yet differentiated enough to sell as standalone software. It is a small, local-first Python CLI for labeled classification traces: strict JSONL input, metrics and confusion reporting, threshold failures, saved-report comparison, judge coverage/agreement, and human-review exports. It has no production provider integrations, trace ingestion, dashboard, or proven customer outcome. The bundled Echo provider is only a deterministic demo.

The strongest near-term commercial use is a fixed-scope implementation pilot. Sell the result—an agreed regression suite for one narrow, existing AI workflow—using the harness as an inspectable local tool. Keep the open-source repo as proof and a no-cost trial path. Do not claim production reliability, universal agent coverage, or time savings before measuring them.

## Best first customer

A small product team already shipping a classification or routing feature powered by an LLM, and changing its prompt/model without a representative regression set. Examples include support-ticket routing, lead categorization, or document-type labeling. Avoid starting with open-ended autonomous agents: today's label-based metrics do not measure tool use, multi-step completion, factual grounding, or safety adequately.

## Pilot offer hypothesis

**AI change-regression setup — one workflow, fixed scope**

- One existing classification workflow and one baseline/candidate comparison.
- Up to 50 customer-approved, de-identified examples; the customer supplies expected labels or approves proposed labels.
- A local JSONL fixture, one adapter example, threshold configuration, and a reproducible report.
- A short findings memo with per-label misses and a human-review queue.
- Data stays in the customer's environment; no provider call is added by this package. If a customer chooses to run its own model, that execution remains under its own controls.
- Completion criterion: the customer can rerun the evaluation on its own machine and see a deliberately introduced regression fail the agreed gate.

Test an introductory fixed price of **$300–$500** for this tightly bounded pilot. This is a pricing hypothesis, not validated market pricing. Cap delivery at roughly one working day after receiving usable examples. Quote separately if annotation, app integration, CI wiring, privacy review, or more than one task is required. Don't promise business lift; report measured results on the agreed sample only.

## Validation sequence

1. Ask 5–8 qualified builders whether they changed a prompt/model in the last month, how they checked for regressions, and what a missed classification costs them. Do not lead with the tool.
2. Offer one paid pilot to a team with an active, narrow classification workflow. A free short discovery call is fine; do not do the implementation for free.
3. Record baseline and candidate metrics, annotation effort, setup time, false-positive/negative examples, and whether the customer reruns it after handoff.
4. Ask for permission separately before using anonymized results as a case study or testimonial. No customer data or claims go into the public repo without explicit authorization.
5. Continue the service only if a buyer pays and reruns the gate. Consider a hosted product only if multiple customers independently ask for shared datasets, team review, integrations, or dashboards.

## Current proof and gaps

The v0.1.0 release has synthetic fixtures and a passing Python 3.12 test/demo/package verification record in the workspace handoff. That proves the local mechanics, not customer demand or model quality. Before describing it as a customer-ready solution, complete one real, permissioned pilot and capture a rerunnable before/after report. The next product improvement should be driven by that pilot's integration friction rather than guessed feature breadth.

## Ethical selling language

> I help small teams put a repeatable regression check around one labeled AI workflow. We agree on representative examples and pass limits, run the check locally, and hand over the fixture and report so your team can rerun it when prompts or models change. The check measures only the supplied cases; it does not certify your system or guarantee production behavior.

