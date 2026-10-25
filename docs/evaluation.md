# Evaluation protocol and current limitations

Criteria are defined here before reporting quality scores. Unit tests use explicitly
synthetic metadata solely to test software behavior. Passing tests does not establish NLP
quality. No human relevance judgments or quality scores have been fabricated.

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
| Ranking | nDCG@10, Precision@5, MRR; per-query failures and comparison to baseline |
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
The human quality criteria above remain pending; structural source checks alone are
insufficient to complete explanation and overview quality evaluation.

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

The current overview is a collection of selected evidence excerpts, not a generated
cross-paper synthesis. The initial OR retrieval produced broad/noisy candidates; topic search
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
