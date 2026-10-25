# First prototype: implementation and verification

The implemented baseline runs as a local Streamlit application. It retrieves real arXiv
metadata, ranks title/abstract candidates using a local encoder or TF-IDF, displays raw
and transformed scores, selects source evidence, and produces an extractive overview with
links to the retrieved papers. All inference is local. README was left unchanged.

## Verification performed

- 38 pytest tests passed, including Streamlit initial rendering, invalid form submission,
  mocked successful results, retrieval retries/cache behavior, metadata parsing, empty
  candidates, duplicate versions, score edge cases, source membership, and metric arithmetic.
  Retrieval diagnostics also distinguish connection failures, timeouts, and HTTP errors.
- Ruff lint and formatting checks passed after corrections.
- Live arXiv request retrieved 10 valid candidate papers with no rejected records in the
  smoke run. Both ranking modes returned three results. IDs and timing provenance are in
  evaluation/smoke.json. Subsequent smoke runs use the same cached real response.
- Streamlit started on localhost:8501 and its health endpoint returned HTTP 200 with `ok`.
- Follow-up diagnosis found outbound network access blocked when the server ran inside the
  coding sandbox. Restarting outside that sandbox restored live retrieval. The user's exact
  query `i am looking for information about transformer architecture` returned 50 validated
  candidates in an uncached live request. Source file watching is disabled to avoid unrelated
  optional-dependency scans through Transformers lazy imports.
- Structural audit of the 10 validated cached records found required-field completeness,
  selected-excerpt source membership, and source-link/identifier agreement all equal to 1.0.
  This sample excludes rejected records and these are structural checks, not quality ratings.
- CPU smoke process peak working set was 654,749,696 bytes (about 624 MiB), measured through
  Windows GetProcessMemoryInfo. This covers one process and ten candidates, excludes the UI,
  other applications and VRAM, and does not establish maximum-workload memory compliance.

## Executed judged retrieval comparison

BEIR SciFact test split: 5,183 documents and 300 queries, with supplied independent relevance
judgments. Both methods use the same complete fixed corpus and query split. Unjudged documents
are treated as zero relevance. Dataset SHA-256, per-query metrics, method configuration, and
the semantic model revision are saved in evaluation/scifact-lexical.json and
evaluation/scifact-semantic.json. The dataset preparation script preserves archive provenance.

| Method | nDCG@10 | Precision@5 | MRR | Run time (seconds) |
| --- | ---: | ---: | ---: | ---: |
| TF-IDF | 0.5664 | 0.1393 | 0.5345 | 2.34 |
| all-MiniLM-L6-v2, CPU | 0.6451 | 0.1640 | 0.6113 | 82.48 |

Run time includes encoder initialization and scoring, but excludes dataset download. These
are single runs, not rigorous performance benchmarks or UI latency measurements. MiniLM
improved aggregate ranking under this protocol, but performed worse than TF-IDF on nDCG@10
for 58 of the 300 queries. There were 62 queries with zero semantic nDCG@10 versus 81 for
TF-IDF. The ideal average Precision@5 under these sparse judgments is only 0.2260; absolute
Precision@5 must therefore be interpreted alongside judgment sparsity.

Examples with zero semantic nDCG@10 include query 13 (`5% of perinatal mortality is due to
low birth weight.`), query 48 (`A total of 1,000 people in the UK are asymptomatic carriers
of vCJD infection.`), and query 70 (`Activation of PPM1D suppresses p53 function.`). These
show failures to place judged evidence in the top ten. Whether individual failures arise
from numeric distinctions, terminology, truncation, or incomplete judgments requires
source-level review; no causal error diagnosis has been established yet.

SciFact concerns biomedical claims rather than open-ended arXiv topic search. These scores
do not evaluate arXiv candidate recall, concept extraction, or end-user usefulness. No
significance test was performed, and model pretraining overlap with the benchmark was not
audited. MiniLM truncates long inputs, which can discard relevant abstract material.

## Remaining evaluation and submission work

The runnable prototype is available; the academic evaluation is not yet complete.
Use docs/evaluation.md to collect independent human judgments for intent preservation,
live retrieval coverage, explanation support/relevance, overview coverage, and task usefulness.
Measure the actual Streamlit process at 10,000 candidates; CUDA inference has been verified
separately as described below.
The overview currently selects evidence excerpts rather than generating cross-paper synthesis.
Archive representative real demo results, complete the human review, and rehearse/record the
under-ten-minute video using docs/setup.md. No human ratings or full-system quality claims
are reported before those steps are executed.

## CUDA and reported retrieval regression

The original environment contained CPU-only PyTorch. The Windows cuda extra now selects
PyTorch 2.9.1+cu128 from the official CUDA 12.8 index; run with `uv run --extra cuda` to
preserve that dependency selection. Device availability is checked before encoder loading.
Actual inference on the RTX 4060 succeeded. The maximum absolute CPU/CUDA cosine difference
across the two checked candidate sets was below 0.000001. Diagnostic peak CUDA allocated
memory was 177,196,544 bytes, reserved memory 234,881,024 bytes, and process working set
1,548,726,272 bytes. These cover a diagnostic process with two encoders, not Streamlit's
full workload or other processes. See evaluation/cuda-verification.json for exact provenance.

The user's conversational transformer query did not retrieve arXiv:1706.03762 among its
original 50 candidates. Focused OR, focused AND, quoted topic phrase, and earliest-first
variants also failed to retrieve it in their first 50. These failures are preserved in
evaluation/retrieval-diagnosis.json. Prefix cleanup and AND retrieval reduce some obvious
irrelevant matches but do not resolve this known-item coverage failure for broad topics.

Searching the complete title in quotes retrieved the paper at API position 2. Pure semantic
ranking then placed it at position 23 among 47 usable candidates, behind some title variants.
For explicit quoted queries the app now places exact returned title matches first while
preserving all cosine scores. The final regression puts the original paper at rank 1 on
both CPU and CUDA. This rule is general to quoted titles; it contains no paper-specific
identifier override. The case was used during development and is not held-out validation.
The earlier SciFact comparison predates these changes and does not evaluate this title rule
or the changed query interpretation.

## Candidate limit increased to 10,000

The UI now defaults to 1,000 candidates and permits up to 10,000. Retrieval uses pages of
1,000 with offset/total-result tracking, cross-page deduplication, page caching, progress,
and explicit partial-results warnings. Semantic scoring retains only 256 documents' vectors
at a time, while the lexical path uses sparse TF-IDF. The tests include synthetic 10,000-entry
pagination, raw offsets despite invalid/duplicate records, exhaustion, partial-page failures
and retries, sparse scoring, progress, and the actual 10,000 setting in Streamlit.

The live transformer-topic diagnostic requested 10,000 entries from a reported 24,494
matches. It retrieved all ten pages successfully in 110.50 seconds, with no invalid records
and 9,994 usable papers after deduplication. CUDA ranking on prefixes of that same snapshot:

| Usable candidates | Ranking and result construction (seconds) |
| ---: | ---: |
| 50 | 0.37 |
| 100 | 0.20 |
| 1,000 | 1.39 |
| 9,994 | 13.68 |

These are single runs with a loaded encoder, not repeated latency benchmarks; the first
small run includes GPU warmup effects. Snapshot-prefix comparisons avoid changing candidate
sets between measurements. The UI reports live/cached retrieval separately from ranking,
and exported ui_total_seconds additionally includes model loading.

Peak working set for the diagnostic process was 1,488,252,928 bytes (about 1.39 GiB).
PyTorch CUDA peak allocated memory was 177,196,544 bytes (169 MiB), reserved memory
234,881,024 bytes (224 MiB). These exclude the Streamlit process and other applications;
reserved memory is not total GPU device usage. Chunked and previous joint-encoding scores
on the first 100 real candidates agreed within 0.000001 (actual maximum about 1.08e-7).
The original Transformer paper appears in the large candidate set but not the top 100
semantic matches for this broad query. It was absent from the smaller prefixes. This
demonstrates better coverage of this one known paper, not generally established relevance
quality. Higher best cosine scores with larger pools alone do not validate ranking quality.
Full provenance and per-prefix top results are in evaluation/candidate-scaling.json;
reproduce with `uv run --extra cuda python scripts/benchmark_candidates.py`.

## Agent development controls

agents.md sets project scope, CRISP-DM, verification and citation rules, hardware constraints,
and the README consultation requirement. The two reusable skills under .agents/skills/
provide project-specific evaluation and architecture-documentation workflows. Dependency
versions are pinned in uv.lock; the semantic model revision is fixed in src/pda/ranking.py.
Software fixtures are explicitly synthetic and are not presented as academic quality data.
