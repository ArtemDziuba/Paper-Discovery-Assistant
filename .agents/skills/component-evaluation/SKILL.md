---
name: component-evaluation
description: Evaluate PDA component quality against independent judgments and baselines, keeping software tests separate from NLP quality results.
---

Read docs/evaluation.md and the affected component before choosing metrics. Identify inputs,
outputs, ground truth, and the population the evaluation covers. Define criteria before
examining scores. Investigate existing datasets before constructing synthetic data.

Run `uv run pytest` for correctness. For judged retrieval files, run
`uv run python scripts/evaluate.py --data PATH --output PATH --method lexical`
and repeat with `--method semantic` using the same corpus and judgments.
Save configuration, dataset hash, per-query scores, model revision, and aggregate results.
Never present unjudged candidates as known irrelevant without stating that convention.

Review query interpretation, explanations, and overview with the human rubric in the
evaluation document. Do not use source membership as a substitute for usefulness or factual
support. Report failures, domain mismatch, missing judgments, and unexecuted evaluations.
Compare baselines and record limitations before recommending model changes.
