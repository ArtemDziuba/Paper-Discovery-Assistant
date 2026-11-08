# Evaluation protocol and current limitations

Criteria are defined here before reporting quality scores. Unit tests use explicitly
synthetic metadata solely to test software behavior. Passing tests does not establish NLP
quality. No human relevance judgments or quality scores have been fabricated.

Current fixed evaluation (2026-11-07): the two batches in evaluation/human_eval and ten
batches in evaluation/syntetic_eval are reserved for evaluation. All three rankers are
compared on both groups using the existing graded/binary metric definitions. New metadata
is frozen separately in evaluation/annotations/prepared.json; the latest full evaluation
and title exclusions are in evaluation/results/ranking-comparison-full.json. See
full-ranking-evaluation.md for all 12 metrics. The earlier five-metric run is preserved as
evaluation/results/ranking-comparison-extended-metrics.json. The prior three-metric run is
preserved as evaluation/results/ranking-comparison.json. The 45 batches in finetune_data
are not part of this run. See ranking-evaluation.md for the complete protocol and results.
The older fine-tuning and SciFact artifacts remain historical experiments.

V2 (2026-11-08) explicitly changes the role of these evaluation queries during training:
the ten synthetic and two human query groups form one combined development mean, used for
checkpoint selection. Human judgments do not supply gradients, but do guide selection. The
post-training comparison keeps human and synthetic results separate for all four models and
is saved in evaluation/results/ranking-comparison-full-v2.json. These v2 scores are development
performance. Training/development paper overlap is retained and reported, following the
requested no-deduplication policy. See finetuning-v2.md for the audit, parameters and epoch plots.

Two selectable modes now supersede the single extended metric set. Use --mode truncated
(the default) for primary nDCG@10 and Precision@5, plus MAP@10 and full-pool pairwise accuracy.
Use --mode full for nDCG@5/10/20/50, Precision@5/10/20, MAP@10/20/50,
pairwise_accuracy@50 and mrr@50. The report saves its mode, cutoff configuration, metric
definitions and primary-metric names. Human and synthetic macro averages remain separate.
Both modes score the complete fixed candidate pools; truncated reduces the metric set,
not encoder inference or paper retrieval.

For full mode, pairwise accuracy@50 compares unequal-grade judged pairs wholly within the
first 50 ranked papers, preserving half credit for raw-score ties. It ignores pairs involving
papers below rank 50. MRR@50 is zero if the first relevant paper is below rank 50. MAP@k
uses the same min(k, R) normalization at each cutoff, based on all relevant judgments in
the complete verified pool. nDCG's ideal ordering also uses the complete judged pool.
Precision@k divides by k even when fewer papers are available; no candidates or judgments
are invented to reach a cutoff. Score ties retain annotation order, including at rank 50.
The paragraph below records the definitions used by the earlier extended run; full mode
restricts pairwise scope to the top 50 and bounds reciprocal rank at 50 as specified above.

The extended evaluation adds MAP@10 and graded pairwise accuracy. For each query, AP@10
sums precision at each relevant hit in ranks 1–10 and divides by min(10, R), where R is
the number of relevant judgments in its complete verified pool. Relevant means grade >=2
for the fixed annotation comparison; AP is zero when R=0. MAP@10 is the unweighted mean
of AP@10 within each human/synthetic group. This uses the top-k normalization described in
the [evaluation playbook](https://playbooks.aip.gov.sg/rag-playbook/evaluation/); implementations
that divide by R without a cap can give different values and must not be compared directly.

Pairwise accuracy compares every scored, explicitly judged unordered pair with unequal
grades over the full candidate pool, including beyond rank 10. A correctly ordered pair
scores 1, an inversion 0, and an exactly equal raw-score pair 0.5. Equal-grade pairs are
excluded. A query without comparable pairs has null accuracy; source-group averages omit
nulls and report pairwise_valid_query_count. Unscored or unjudged papers do not create
artificial pairs. The generic fixed-corpus runner retains its grade >0 binary default and
unjudged-as-zero convention for other metrics; its pairwise measure uses explicit judgments.

## Existing data investigation

BEIR SciFact supplies scientific claims, abstracts and relevance judgments in a compact
retrieval benchmark: https://huggingface.co/datasets/BeIR/scifact and
https://github.com/allenai/scifact . It is a useful candidate for embedding/ranking comparison,
but biomedical claim-to-evidence retrieval differs from user questions on arXiv; its IDs
are not arXiv IDs. Check the original data terms before redistributing data. No benchmark
data is bundled. Do not use it to claim recall of live arXiv retrieval.

The evaluation script accepts a JSON object with `documents` (id, title, abstract), `queries`
(id, text), and `qrels` mapping query IDs to document IDs and nonnegative graded relevance.
Use a fixed corpus and independent human judgments. Keep training/tuning separate from the
final test set. Record dataset provenance and license alongside the file. For sparse
judgments, computed metrics conventionally treat unjudged documents as zero; report the
resulting uncertainty. Precision@5 divides by five even for fewer returned documents.

```powershell
uv run python scripts/evaluate.py --data evaluation/data.json --output evaluation/lexical.json --method lexical
uv run python scripts/evaluate.py --data evaluation/data.json --output evaluation/semantic.json --method semantic
```

## Component criteria

Known-title regression protocol: check the complete lowercase unquoted title, an ASCII-quoted
variant, and a smart-quoted variant against real arXiv records. Require the actual known ID
to be returned first, without assigning a fabricated relevance probability. Separately test
that near-title variants are filtered, auxiliary records merge/deduplicate under the budget,
and either retrieval path can fail with visible diagnostics. This is a development regression
for one known item, not evidence of general title-retrieval recall or held-out quality.
The live title path now uses an auxiliary title-field query and exact returned-title priority
with or without quotes; earlier quoted-only diagnostic results describe the earlier behavior.

Candidate-scaling protocol: retrieve one real snapshot up to 10,000 entries, then compare
the first 50, 100, 1,000 and 10,000 usable candidates under the same query and encoder. Record
retrieval and ranking time separately, actual candidate counts, per-page provenance, peak
process working set, and CUDA allocated/reserved memory. Check whether the user's known
relevant paper is retrieved and reaches the top 100. Treat this as a development diagnostic,
not a held-out quality benchmark; higher best cosine alone is not proof of improved relevance.
Synthetic pagination and batching tests establish correctness separately. Corpus/model
scoring is otherwise unchanged; chunked scores should agree with the previous encoder path
within floating-point tolerance. Changes to the live API across calls can affect candidates.

| Component | Measurement and interpretation |
| --- | --- |
| Query interpretation | Human-marked required concepts: recall and intent preservation (0 lost, 1 partial, 2 preserved); inspect negation and phrases |
| Live retrieval | Known-relevant candidate coverage on a frozen query set; call it recall only relative to specified judgments, never all arXiv |
| Preprocessing | Required-field completeness, rejected records, duplicate false merges/misses on manually inspected records |
| Embeddings | Relevant/irrelevant pair score distributions; ranking comparison to TF-IDF under identical corpus and qrels |
| Ranking | Truncated/full modes with configured nDCG, Precision, MAP, pairwise and reciprocal-rank cutoffs; per-query failures and baseline comparison |
| Explanations | Human ratings: support (0 unsupported, 1 partial, 2 fully grounded), relevance (0 unrelated, 1 partial, 2 directly relevant) |
| Overview | Source-ID validity plus human claim support and coverage of relevant top-paper concepts; assess each separately |
| End to end | User task success (0 failed, 1 partial, 2 useful), timings, measured peak RAM/VRAM, network/error handling |

For explanations and overview, annotate individual claims as supported, unsupported, or
unverifiable from the abstract. Unsupported-claim rate = unsupported / reviewed claims;
report unverifiable claims separately. Citation validity does not prove claim support.
Use at least two human reviewers for a subset when feasible and report disagreements.
Use the same frozen cases across changes. Include narrow, broad, ambiguous, negated,
missing-data, and no-result cases. Do not select only successful cases.

Executed baseline results are in implementation-results.md and evaluation/*.json.
Human ranking judgments have now been evaluated for the fine-tuning experiment; see
finetuning.md and evaluation/finetuning/training-results.json. The comparison uses per-query
judged pools, grades 0–3 for nDCG and grade >=2 for binary metrics. Synthetic labels are kept
distinct from human labels, and human papers are excluded from training. Other human quality
criteria remain pending; structural checks do not complete explanation/overview evaluation.

## CRISP-DM evidence

Business understanding: arXiv discovery with local inference and constrained laptop resources.
Data understanding: arXiv metadata and SciFact investigation above.
Data preparation: Atom normalization, required-field checks, identifier-preserving deduplication.
Modeling: lexical baseline and local semantic encoder; extractive evidence and overview.
Evaluation: automated correctness tests and judged-data runner; human quality review remains
required before making effectiveness claims.
Deployment: local Streamlit run instructions, source export, and demo outline in setup.md.

## Sources for implementation decisions

- arXiv API protocol and rate guidance: https://info.arxiv.org/help/api/user-manual.html
- Encoder model card and input limitations: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

The earlier overview was a collection of evidence excerpts and was removed at the user's
request. Current evaluation scope covers ranked papers and their individual evidence, rather
than literature-overview generation. Earlier overview criteria above describe the original
prototype plan. The initial OR retrieval produced broad/noisy candidates; topic search
now uses AND after conversational-prefix cleanup. This can reduce recall when a required
term is absent; it does not guarantee retrieval of foundational papers. Quoted titles retain
their full phrase for known-item search. Fixed-corpus SciFact results remain the earlier
encoder/ranking comparison, not an evaluation of this query-interpretation change.
Quoted queries now prioritize exact returned titles independently of similarity, with raw
scores preserved. evaluation/retrieval-diagnosis.json records the reported development case
and failed query variants; evaluation/cuda-verification.json records actual CPU/GPU agreement
and final title-query rank. This is a targeted regression, not an independent held-out
evaluation or evidence of generally improved candidate recall. No paper IDs are injected.
Token concepts lack
phrase disambiguation, and exclusions are warned about rather than enforced. These are
baseline limitations to examine in evaluation, not claims of satisfactory quality.
