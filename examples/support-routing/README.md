# Synthetic support ticket routing evaluation

This is a workflow hypothesis for the first narrow pilot: route short support
messages into `billing`, `account`, `technical`, or `refund`. Every record here
is invented test text. The 20-row set is balanced (5 per queue), contains no
customer information, and is not representative evidence about any real model
or support team.

The baseline fixture routes all 20 examples correctly. The candidate fixture
contains one deliberate `technical` → `billing` regression. Both are static,
hand-authored synthetic traces: no model is called and no production traffic is
served. The example demonstrates evaluation mechanics and the effect of
reviewable thresholds only; it does not establish model quality, production
reliability, or demand for this workflow.

From the repository root, with the package installed (see the main README), run:

```sh
python examples/support-routing/run_demo.py
```

The demo writes reports to a temporary directory. Baseline evaluation passes
at the configured 96% accuracy / 80% per-label recall floor. Candidate evaluation
fails the same floor and the comparison fails a 2 percentage-point maximum
accuracy-drop gate. Those are demonstration thresholds, not recommended
operational limits; a pilot would set limits with the customer before running
their approved data locally.
