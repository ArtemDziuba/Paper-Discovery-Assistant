# Full and truncated ranking evaluation

This page records the three-model full run from 2026-11-07. The four-model comparison
including v2, its training history and plots is in [finetuning-v2.md](finetuning-v2.md),
with raw results in evaluation/results/ranking-comparison-full-v2.json.

The full evaluation was executed on 2026-11-07 with local CUDA inference, comparing TF-IDF,
Semantic, and syntetic finetune v1. All three methods received identical frozen candidate
pools from evaluation/annotations/prepared.json. Human and synthetic results are kept separate.
The full report is evaluation/results/ranking-comparison-full.json, with per-query metrics,
complete scored rankings, mode/cutoff definitions, input and model hashes, versions, timings,
metadata exclusions and original training exposure. Earlier reports remain unchanged.

## Modes and commands

Both scripts/evaluate_annotations.py and scripts/evaluate.py accept --mode truncated or
--mode full. The annotation runner defaults to truncated and uses a separate output filename
for each mode: evaluation/results/ranking-comparison-MODE.json. Existing reports cannot be
overwritten; pass a new --output path to repeat a run.

| Mode | Metrics |
| --- | --- |
| truncated | Primary: nDCG@10 and Precision@5. Additional: MAP@10 and full-pool pairwise accuracy. |
| full | nDCG@5/10/20/50; Precision@5/10/20; MAP@10/20/50; pairwise accuracy@50; MRR@50. |

Run from the repository root:

```powershell
uv run --extra cuda python scripts/evaluate_annotations.py --mode truncated --device cuda
uv run --extra cuda python scripts/evaluate_annotations.py --mode full --device cuda --output evaluation/results/reproduced-full.json
```

Truncated mode selects fewer metrics while retaining all candidate scoring needed to rank
the papers. It does not reduce the encoder inputs or change the retrieval/evaluation pools.
The shared Python interfaces metrics(..., mode="full") and evaluate(..., mode="full") also
select the full configuration; their default is truncated. Training checkpoint selection
continues to use nDCG@10. No training was performed in this run.

## Protocol and cutoff rules

There are 12 source batches: two human queries with 65 supplied judgments and ten synthetic
queries with 500 judgments. The existing title-agreement audit excludes one human and 16
synthetic judgments, leaving 64 and 484 evaluated judgments respectively. No labels changed
and no metadata was fetched again. All 45 new training batches remain unused.

- nDCG@k uses gains 2^grade - 1, logarithmic discounts, and the ideal top-k ordering from
  the complete verified judgment pool.
- Precision@k uses grade >=2 as relevant and always divides by k, even if fewer papers exist.
- AP@k sums precision at relevant hits through rank k and divides by min(k, R), with R the
  total number of grade >=2 judgments in the complete verified pool; no relevant judgments
  gives zero. MAP@k is the equal-weight mean of AP@k across queries within each source group.
- Pairwise accuracy@50 compares every unordered, unequal-grade judged pair whose papers
  both appear within the first 50 ranked results. Correct order scores 1, reversal 0, and an
  exact raw-score tie 0.5. Equal-grade pairs and pairs involving a paper below rank 50 are
  omitted. No comparable pairs yields null, omitted from the mean with valid counts reported.
- MRR@50 is the mean reciprocal rank of the first grade >=2 paper through rank 50; a first
  relevant result at rank 51 or later gives zero.

Short pools are not padded with fabricated papers. Stable score ties retain annotation order,
including at the rank-50 boundary. Every current pool has at most 50 papers, so pairwise@50
equals the earlier full-pool measure here, and MRR@50 equals the earlier unbounded MRR.
Software tests separately cover larger pools where those measures differ.

## Human results: 2 queries, 64 judgments

| Metric | TF-IDF | Semantic | syntetic finetune v1 |
| --- | ---: | ---: | ---: |
| nDCG@5 | 0.7065 | 0.7997 | 0.7128 |
| nDCG@10 | 0.7558 | 0.8168 | 0.7764 |
| nDCG@20 | 0.8093 | 0.8183 | 0.7716 |
| nDCG@50 | 0.9056 | 0.9311 | 0.8889 |
| Precision@5 | 0.9000 | 0.9000 | 0.9000 |
| Precision@10 | 0.9000 | 0.9500 | 0.9500 |
| Precision@20 | 0.8000 | 0.8000 | 0.8000 |
| MAP@10 | 0.8521 | 0.8952 | 0.8786 |
| MAP@20 | 0.7257 | 0.7393 | 0.7272 |
| MAP@50 | 0.8733 | 0.8842 | 0.8749 |
| Pairwise accuracy@50 | 0.6587 | 0.6863 | 0.6644 |
| MRR@50 | 1.0000 | 1.0000 | 1.0000 |

## Synthetic results: 10 queries, 484 judgments

| Metric | TF-IDF | Semantic | syntetic finetune v1 |
| --- | ---: | ---: | ---: |
| nDCG@5 | 0.7898 | 0.8616 | 0.8972 |
| nDCG@10 | 0.7953 | 0.8349 | 0.8645 |
| nDCG@20 | 0.7697 | 0.8227 | 0.8532 |
| nDCG@50 | 0.9168 | 0.9418 | 0.9518 |
| Precision@5 | 0.8600 | 0.9600 | 0.9800 |
| Precision@10 | 0.9000 | 0.9600 | 0.9600 |
| Precision@20 | 0.8650 | 0.9450 | 0.9400 |
| MAP@10 | 0.8302 | 0.9427 | 0.9433 |
| MAP@20 | 0.7833 | 0.9165 | 0.9158 |
| MAP@50 | 0.8675 | 0.9456 | 0.9349 |
| Pairwise accuracy@50 | 0.6905 | 0.7824 | 0.7869 |
| MRR@50 | 0.9500 | 1.0000 | 1.0000 |

## Interpretation and verification

Semantic leads the human macro nDCG, MAP and pairwise measures. V1 leads synthetic nDCG
at every cutoff and slightly leads pairwise accuracy, but Semantic leads synthetic MAP@20,
MAP@50 and Precision@20. Stronger early graded ordering therefore does not establish better
binary-relevant ordering throughout the entire pool. AP normalization changes with k, so
MAP@10, MAP@20 and MAP@50 need not increase monotonically.

These measures cover the small explicitly judged candidate pools, not live arXiv recall or
probability calibration. Synthetic labels are not human ground truth. Two human queries,
previously used during application development, limit generalization claims. No significance
test or causal explanation for individual failures is claimed. Existing title-mismatch and
model-pretraining-overlap limitations in ranking-evaluation.md still apply.

Verification: 79 pytest tests passed, including both modes in the CLI runner, precise cutoff
boundaries, short pools, raw-score ties, undefined pairwise pools, and exclusion of rank-51
papers from pairwise@50/MRR@50. Ruff lint and formatting checks passed. All 36 query/model
outputs were independently checked using scikit-learn nDCG, grade-pair-weighted ROC AUC,
and direct precision/AP/MRR calculations. Scores, rankings, annotations, model hashes and
the frozen snapshot agree with the earlier extended run. README was not edited.
