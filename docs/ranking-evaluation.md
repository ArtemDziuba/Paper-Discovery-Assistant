# Ranking evaluation on the revised annotation sets

This page records the earlier five-metric extended run. The current selectable truncated/full
modes and the latest executed full results are in [full-ranking-evaluation.md](full-ranking-evaluation.md).

Executed on 2026-11-07 using CUDA. The fixed evaluation comprises the two JSON batches in
evaluation/human_eval and ten in evaluation/syntetic_eval. The 45 JSON batches under
finetune_data were not read, trained on, or included in these metrics.

The existing fine-tuned model is now named **syntetic finetune v1** in the interface and
exported model identity (`pda/syntetic-finetune-v1`). Its weights and training manifest are
unchanged at models/finetuned-minilm; the original training run remains ec2dc397f1446876.
Semantic remains the application's default.

## Saved artifacts and inputs

- Frozen evaluation metadata: evaluation/annotations/prepared.json.
- Complete results with five metrics: evaluation/results/ranking-comparison-extended-metrics.json.
- Previous three-metric run: evaluation/results/ranking-comparison.json, preserved unchanged.
- Historical comparison: evaluation/ranking-comparison.json, preserved unchanged.
- Original v1 training provenance: evaluation/finetuning/prepared.json and training-results.json.

The new report records source-file hashes, frozen-data hash, model revisions, v1 weight and
manifest hashes, library versions, stage timings, every query's full ranking and raw cosine
scores, score means by grade, metadata exclusions, and original training/development overlap.

Metadata for 559 distinct arXiv IDs was requested through the existing rate-limited client.
Source annotations contain 65 human judgments and 500 synthetic judgments. The existing
title-agreement policy (Unicode/case/punctuation normalization, SequenceMatcher >=0.85)
excluded one human and 16 synthetic judgments. All exclusions were title mismatches; none
was silently relabeled or replaced. The evaluated pools contain 64 human and 484 synthetic
judgments. Exclusions, supplied titles, returned titles, and agreement values are saved in
both the snapshot and report.

This automated policy can exclude a valid paper whose title changed or was paraphrased.
For example, arXiv:2406.08039 has title agreement 0.8138, below the fixed threshold.
The excluded human item remains arXiv:2605.03953. Review excluded ID/title pairs before
changing any judgments; a mismatch alone does not establish that a paper is fabricated.

## Protocol

Each query is ranked against its own explicitly judged, verified papers, using the retrieved
title plus abstract. Every method receives identical candidates. No unjudged cross-query
paper is assigned a zero label. This comparison excludes live topic retrieval, query
interpretation, and the application's exact-title priority rule.

nDCG@10 uses grades 0–3, gains 2^grade - 1, and log2(rank + 1) discounts. Precision@5 and
MRR treat grades >=2 as relevant. Precision@5 always divides by five. Ties preserve the
annotation order. Macro averages give each query equal weight, separately within each
source group. Metrics and the 0.85 metadata threshold retain the earlier protocol; changing
the source batches changes the evaluation population, so the old synthetic aggregate is
not a direct measure of improvement over time. The extended run adds the following metrics
without changing earlier metric definitions, metadata, annotations, or model weights.

For each query, AP@10 = sum over ranks 1–10 of (precision at that rank * binary relevance),
divided by min(10, R), where R counts all grade >=2 judgments in that query's verified pool.
If R=0, AP@10 is zero. MAP@10 is the source-group mean of these AP values. AP is saved as
`ap@10` per query and its macro average as `map@10`. AP uses the same stable tie order as
the ranked list. This top-k normalization follows the
[evaluation playbook](https://playbooks.aip.gov.sg/rag-playbook/evaluation/); uncapped-R
normalization used by some packages is a different convention.

Pairwise accuracy covers the full judged pool, including papers below rank 10. It compares
every unordered pair of scored papers with unequal relevance grades: correct score order
receives 1, reversed order 0, and an exact raw-score tie 0.5. Equal-grade pairs are excluded.
The measure therefore distinguishes grades 2 and 3 even though both count as binary relevant.
Queries without comparable pairs receive null and are omitted from the pairwise macro mean;
the valid query count is exported. All two human and ten synthetic queries have comparable
pairs in this run. Explicitly unjudged or unscored papers create no pairs.

## Aggregate results

| Source | Model | nDCG@10 | Precision@5 | MRR | MAP@10 | Pairwise accuracy |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Human: 2 queries, 64 judgments | TF-IDF | 0.7558 | 0.9000 | 1.0000 | 0.8521 | 0.6587 |
| Human | Semantic | 0.8168 | 0.9000 | 1.0000 | 0.8952 | 0.6863 |
| Human | syntetic finetune v1 | 0.7764 | 0.9000 | 1.0000 | 0.8786 | 0.6644 |
| Synthetic: 10 queries, 484 judgments | TF-IDF | 0.7953 | 0.8600 | 0.9500 | 0.8302 | 0.6905 |
| Synthetic | Semantic | 0.8349 | 0.9600 | 1.0000 | 0.9427 | 0.7824 |
| Synthetic | syntetic finetune v1 | 0.8645 | 0.9800 | 1.0000 | 0.9433 | 0.7869 |

## Per-batch nDCG@10

Batch numbers refer to human_evaluation_batch_NN.json or synthetic_evaluation_batch_NN.json.
Full Precision@5, MRR, grades and ranking positions for each batch are in the JSON report.

| Source / batch | Query | Verified papers | TF-IDF | Semantic | syntetic finetune v1 |
| --- | --- | ---: | ---: | ---: | ---: |
| Human 01 | transformer architecture | 31 | 0.9748 | 0.8820 | 0.8376 |
| Human 02 | cnn image recognition | 33 | 0.5368 | 0.7516 | 0.7151 |
| Synthetic 01 | Contrastive self-supervised learning | 50 | 1.0000 | 0.9552 | 0.8993 |
| Synthetic 02 | Differential privacy in deep learning | 48 | 0.5355 | 0.7501 | 0.8193 |
| Synthetic 03 | Learning with noisy labels | 48 | 0.9621 | 0.8526 | 0.8810 |
| Synthetic 04 | Neural radiance fields | 49 | 0.5074 | 0.8264 | 0.8156 |
| Synthetic 05 | Knowledge graph completion | 46 | 0.9603 | 0.9513 | 1.0000 |
| Synthetic 06 | ECG arrhythmia detection | 49 | 0.6545 | 0.5875 | 0.6517 |
| Synthetic 07 | Reinforcement learning navigation | 48 | 0.7726 | 0.7165 | 0.7673 |
| Synthetic 08 | Multimodal emotion recognition | 48 | 0.8742 | 0.8323 | 0.8742 |
| Synthetic 09 | Satellite change detection | 48 | 0.8137 | 0.9636 | 1.0000 |
| Synthetic 10 | Deep learning image inpainting | 50 | 0.8725 | 0.9133 | 0.9371 |

## Per-batch AP@10 and pairwise accuracy

Each model's cell shows **AP@10 / pairwise accuracy**; batch-to-query mapping is above.

| Source / batch | TF-IDF | Semantic | syntetic finetune v1 |
| --- | ---: | ---: | ---: |
| Human 01 | 1.0000 / 0.8071 | 0.7904 / 0.7878 | 0.7571 / 0.7235 |
| Human 02 | 0.7042 / 0.5102 | 1.0000 / 0.5848 | 1.0000 / 0.6053 |
| Synthetic 01 | 1.0000 / 0.9111 | 1.0000 / 0.9233 | 0.8664 / 0.8300 |
| Synthetic 02 | 0.3963 / 0.3797 | 0.8664 / 0.6648 | 0.8354 / 0.6579 |
| Synthetic 03 | 1.0000 / 0.7113 | 1.0000 / 0.7648 | 1.0000 / 0.7535 |
| Synthetic 04 | 0.5975 / 0.5562 | 1.0000 / 0.7834 | 1.0000 / 0.7527 |
| Synthetic 05 | 1.0000 / 0.7669 | 1.0000 / 0.8187 | 1.0000 / 0.8763 |
| Synthetic 06 | 0.8789 / 0.7344 | 0.7454 / 0.7173 | 0.8521 / 0.7062 |
| Synthetic 07 | 1.0000 / 0.7258 | 0.8154 / 0.7736 | 0.8789 / 0.7919 |
| Synthetic 08 | 1.0000 / 0.7738 | 1.0000 / 0.7617 | 1.0000 / 0.8260 |
| Synthetic 09 | 0.6842 / 0.6799 | 1.0000 / 0.7884 | 1.0000 / 0.8849 |
| Synthetic 10 | 0.7454 / 0.6662 | 1.0000 / 0.8279 | 1.0000 / 0.7898 |

## Interpretation and limitations

V1's synthetic macro nDCG@10 exceeds Semantic by 0.0297. It improves over Semantic on eight
of ten synthetic queries, but declines for contrastive self-supervised learning and neural
radiance fields. TF-IDF is stronger than both dense methods on several individual cases;
for example, noisy-label learning scores 0.9621 with TF-IDF versus 0.8526 with Semantic.
For neural radiance fields, TF-IDF scores only 0.5074 versus Semantic's 0.8264. These describe
ranking failures under the supplied labels; no causal diagnosis of individual errors is claimed.

Semantic remains stronger on both human queries. All models' human macro Precision@5 and
MRR are equal, while graded nDCG distinguishes their ordering. MRR is already 1.0 for both
dense methods on every query, so it cannot distinguish their top-ten ordering here.

The extended metrics also favor Semantic on the human macro averages. On synthetic queries,
v1's MAP@10 exceeds Semantic by only 0.0006 and pairwise accuracy by 0.0045; these small
aggregate gaps do not establish statistical significance. MAP can saturate when all top-ten
papers are binary relevant even if their grades are ordered poorly. For the CNN human query,
both dense methods have AP@10=1.0 while their pairwise accuracies remain below 0.61, showing
that relevant top results can coexist with substantial graded ordering errors across the pool.

An audit against v1's original training/development splits found zero matching query IDs,
exact normalized query texts, or version-normalized paper IDs among the evaluated candidates.
This does not audit pretraining overlap or semantic similarity between topics. The human
queries were used in earlier application development, so they are not an untouched test of
all system design choices. Synthetic results measure agreement with synthetic labels, whose
generation provenance is not recorded in these files. Two human queries cannot establish
general usefulness or statistical significance. Title exclusions further restrict the measured
population. No significance test, live retrieval recall evaluation, or score calibration was
performed; these scores are not relevance probabilities.

## Reproduction and verification

From the repository root:

```powershell
uv run --extra cuda python scripts/evaluate_annotations.py --device cuda --output evaluation/results/reproduced-v1-extended.json
uv run --extra cuda python -m pytest -q
uv run --extra cuda python -m streamlit run app.py
```

The evaluation reuses evaluation/annotations/prepared.json without network requests, and
requires the complete current annotation file set and its hashes to match. Missing, renamed,
added, or edited files require a new snapshot. To prepare revised annotations:

```powershell
uv run --extra cuda python scripts/prepare_evaluation.py --output evaluation/annotations/revised.json
uv run --extra cuda python scripts/evaluate_annotations.py --prepared evaluation/annotations/revised.json --output evaluation/results/revised.json --device cuda
```

Preparation uses arXiv. Both scripts refuse to overwrite existing output files, preserving
earlier runs. Retain the original snapshot to reproduce this exact comparison; refetching
can change metadata versions and exclusions. Model inference runs locally. Keep --extra cuda
when using uv, and use --device cpu if desired without changing the installed environment.

Executed software verification: **68 pytest tests passed**; Ruff lint and formatting checks
passed. The additional checks cover hand-calculated AP, cutoff/denominator behavior, raw-score
ties, equal grades, undefined pairwise cases, full-pool comparisons, macro averages, and metric
propagation through the runner. All 36 query/model results were also checked independently
using grade-pair-weighted scikit-learn ROC AUC for pairwise accuracy and a vectorized AP
calculation. The rerun's raw scores, rankings and earlier per-query metrics match the prior
run. Software verification remains separate from the quality results above.
