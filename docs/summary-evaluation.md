# Query-conditioned full-paper summary evaluation

This standalone command evaluates the existing summarizer using DeepEval 3.9.9 and its
native `GeminiModel` integration. It does not search, rerank, train a model, change a
summarization prompt, or change application inference. Summaries still come from local
`qwen3.5:4b` through the existing `Summarizer.summarize` workflow.

The user explicitly authorized **Google Gemini cloud inference for this offline evaluator**.
Each evaluation sends the research query, generated summary and extracted paper text (or
documented excerpts) to Google's API. The Streamlit application remains local. DeepEval
telemetry, error reporting, update checks and internal Confident tracing are disabled;
the command does not upload evaluation runs to Confident AI.

## Setup and execution

From the project root:

```powershell
uv sync --extra cuda --group evaluation
```

Create `.env` using [.env.example](../.env.example) as a template. Set `GOOGLE_API_KEY` to
your Google API key. `.env` is ignored by Git; never copy its contents into reports or commits.
An existing environment variable takes precedence over `.env`. Missing credentials produce
a configuration error before expensive summary generation, with no judge fallback.

Start Ollama with the application's existing `qwen3.5:4b` installation. No additional Ollama
model is needed or downloaded. The judge defaults to `gemini-2.5-flash`.

```powershell
# Validate the real pipeline on human query 1 / paper 1 first.
uv run --extra cuda --group evaluation python scripts/evaluate_summaries.py --smoke

# All 36 instances: generate missing summaries, judge missing/outdated scores, rebuild reports.
uv run --extra cuda --group evaluation python scripts/evaluate_summaries.py

# Generate only missing summaries locally, with no Gemini calls or API key requirement.
# Existing summaries/scores are retained; new summaries have null metric/overall scores.
uv run --extra cuda python scripts/evaluate_summaries.py --summarization-only

# Never generate summaries. Missing cached summaries remain explicitly incomplete.
uv run --extra cuda --group evaluation python scripts/evaluate_summaries.py --evaluation-only

# Retry all five judge metrics, keeping generated summaries.
uv run --extra cuda --group evaluation python scripts/evaluate_summaries.py --evaluation-only --force-evaluate

# No Qwen, Gemini or arXiv calls; rebuild scores, CSV/JSON and terminal tables from records.
uv run --extra cuda --group evaluation python scripts/evaluate_summaries.py --aggregate-only

# Explicitly regenerate summaries and invalidate their scores; expensive, never the default.
uv run --extra cuda --group evaluation python scripts/evaluate_summaries.py --force-summarize
```

After setup, `.venv\Scripts\python.exe scripts/evaluate_summaries.py ...` also works and
does not resynchronize dependencies. `--judge-model` overrides `PDA_SUMMARY_JUDGE_MODEL`.
Other `JudgeSettings` fields use `PDA_SUMMARY_JUDGE_<UPPERCASE_FIELD>`: `temperature` (0),
`seed` (42), `max_output_tokens` (4096), `max_input_bytes` (800000), `timeout_seconds` (180),
and `max_evidence_chunks` (96). Judge thinking is disabled with `thinking_budget=0`.
Generation and judge settings are independent; existing `PDA_SUMMARY_*` settings retain
their application meaning. Gemini models selected as overrides must support these settings.

The command exits 0 when all requested entries complete, 1 for failed/incomplete requested
entries, 2 for configuration errors, or 130 after Ctrl+C. Aggregate-only returns 0 after
successfully rebuilding reports even when stored records are incomplete. `--smoke` processes
one pair but initializes/reports the same 36 stable slots, so subsequent full runs reuse it.
`--summarization-only` returns 0 when all requested summaries are available, even though
their evaluations remain incomplete. It neither loads Gemini credentials nor initializes
the judge, and does not require the `evaluation` dependency group. It resumes at the first
missing summary (currently record 08 after existing records 01–07). Use the saved query,
summary, source metadata/PDF for separate web-chat judging. The evaluation object retains
the normal schema with null scores until judgments are supplied; existing judgments are
preserved. `--force-summarize` is also supported here, but is never needed just to resume.

## Fixed selection and provenance

`load_annotations(..., deduplicate=False)` loads files in sorted filename order. The two
human query batches come first, followed by ten synthetic batches. Exactly the first three
judgments in each query retain their original order. Expected labels are all 3; this is a
selection property, not a summary quality score. This population intentionally covers highly
relevant papers, so it does not measure summarization of poorly relevant retrieval results.

The repository uses `evaluation/syntetic_eval/`. The loader also accepts the correctly spelled
`evaluation/synthetic_eval/`, reporting an error if both exist. Saved `dataset_source` is
consistently `human_eval` or `synthetic_eval`. Every record preserves the original query,
query ID, selected arXiv ID, title, label, source filename and file SHA-256. IDs without a
version remain unchanged in `paper`; the resolved version is saved in summary metadata.

Missing papers/IDs or unexpected selected labels are recorded in their original slots and
never replaced by lower-ranked papers. An unexpected total query count is a configuration
error. Metadata is reused from `evaluation/annotations/prepared.json`; missing metadata uses
the existing `ArxivClient.fetch_ids`, verifying ID and title before generation. This is an
ID lookup, not a search. No ranking scores or pipelines execute.

## Durable records and resumption

`evaluation/summary_eval/summary_evaluation_01.json` through `36.json` are the permanent
query-summary caches and evaluation records. Human entries occupy 01–06, synthetic 07–36.
The original query and arXiv ID, source and query ID must match on reuse. Corrupt or unrelated
files are preserved and reported as failed cache conflicts. Resolve those files manually or
use a fresh `--output-dir`; they are never silently overwritten with another pair's output.

A successful summary is saved atomically **before** PDF evidence preparation or Gemini calls.
It stores duration, timestamp and the production `SummaryResult` provenance: resolved ID,
PDF SHA-256, model tag/digest, parser/pipeline versions, page/chunk/call counts and cache hit.
Changing the judge, rubric or evaluation configuration never regenerates it. A valid permanent
cache is also reused if application summarizer settings change; use explicit regeneration to
evaluate a changed summarizer. Forced regeneration archives the old record under `history/`.

Each metric is saved immediately after its own attempt. An interrupted/incomplete evaluation
reuses valid metrics for the same configuration and retries only missing/failed dimensions.
Changed judge settings, rubric, selected input metadata, source provenance or summary text
invalidate judge scores. `--force-evaluate` invalidates all scores without regenerating a
summary. `--aggregate-only` recalculates weighted values without inferring quality or calling
models. Existing individual scores are never rounded before saving or aggregation.

PDFs remain in the application's separate `.cache/full-paper/pdf/` store. Verified extracted
`TextUnit` records are preserved under `summary_eval/sources/`, keyed by resolved ID, PDF hash
and source parser version, with integrity checks. This preserves evaluation evidence if the
application cache is cleared. Evidence-selector progress is saved under `summary_eval/evidence/`.
Summary and judge failures have distinct structured error stages. Reports also rebuild on
Ctrl+C from the available in-memory records; the per-instance files remain the durable source.
Run one writer per output directory; concurrent evaluations of the same files are unsupported.

## Evidence and the five metrics

The evaluator reuses `PdfStore`, `extract_pdf` and `chunk_document`; it has no independent PDF
parser or scientific summarizer. The resolved paper version and PDF hash must match the
summary's provenance. Failed extraction never falls back to an abstract. The production
parser retains section/page markers, scientific equations/tables and substantive sections,
and removes bibliography where detectable. These same extracted units supply the judge.

Most papers fit Gemini's configured budget and provide **all extracted text** as context to
each metric. The 800000 UTF-8-byte budget is conservative relative to Gemini 2.5 Flash's
context window and reserves additional room for system instructions, rubrics and schema.
The complete generated G-Eval prompt is checked before dispatch. Inputs never silently truncate.

For larger inputs, the existing section-aware chunker covers every extracted unit. Gemini
inspects each chunk and selects exact source excerpts covering methods, experiments, results,
contributions, limitations, and support/contradictions for summary claims. Every quote must be
a substring of that chunk and fit its allocated budget. At least one excerpt per chunk is
required, and all chunk IDs are recorded. Selection calls are cached separately. If coverage
cannot fit, the evaluation fails explicitly. Selection is lossy, not equivalent to full text;
the judge receives this limitation and must distinguish unverified claims from contradictions.
Selection errors cannot change the permanently stored user summary.

All five dimensions use separate [G-Eval metrics](https://deepeval.com/docs/metrics-llm-evals),
because custom criteria capture query dependence and incomplete-evidence handling more
precisely than a generic answer/hallucination metric. Fixed evaluation steps avoid regenerating
rubrics each run. Native Gemini structured output supplies score/reason, and DeepEval normalizes
its 0–10 judgments to 0–1. There are five scoring calls per paper; long-paper excerpt selection
can add calls. They are not word-count, lexical-overlap or embedding metrics.

| Dimension | Weight | Criterion |
| --- | ---: | --- |
| Factual consistency | 0.35 | Supported scientific claims, exact numerical results, datasets, baselines, architecture, conditions and qualified conclusions; contradictions distinguished from missing evidence |
| Query relevance | 0.30 | Specific relationship to the original research query, with appropriately acknowledged indirect relevance |
| Key information coverage | 0.20 | Important query-relevant contribution, method, findings and documented limitations within a short summary |
| Clarity | 0.10 | Logical structure, readability, precise terminology and coherent presentation |
| Conciseness | 0.05 | Efficient communication without repetition or irrelevant detail; useful length is not penalized mechanically |

Python computes `S = .35F + .30R + .20C + .10H + .05B` and `S10 = 10*S`. All five valid
scores and nonempty reasons are required. No overall score is assigned to a failed or missing
metric, and provider errors never become invented scores. Rubric version, rubric text, full
configuration/fingerprint, model identifier, framework version and evidence hashes are stored.
Gemini is independent of the Qwen generator, but automated judgments remain fallible. Seed and
temperature reduce variation without guaranteeing identical provider responses. The model tag
can identify a provider alias whose implementation changes; force reevaluation when needed.

## Reports, statistics and software verification

`summary_evaluation_results.csv` has one row for every expected pair, including failures and
incomplete entries, with the five scores/reasons, both weighted scales, durations and statuses.
Unavailable numeric values are empty. JSON includes complete records and overall, dataset and
query-level statistics. Primary metric means/std use completed evaluations only. Separate
`available_metric_diagnostics` report partial metric counts explicitly. Sample standard
deviation uses `ddof=1` (null for fewer than two observations). Minimum/maximum weighted
scores, time means and status counts are included. Console titles alone are truncated.

```powershell
uv run --extra cuda --group evaluation pytest tests/test_summary_evaluation.py -q
uv run --extra cuda --group evaluation pytest -q
```

These are software correctness tests with mocked Qwen/Gemini responses, not quality results.
They cover actual dataset selection, stable numbering, missing slots, summary durability,
score calculation, partial failures/interruption, mode isolation, forced invalidation,
complete aggregates, native G-Eval execution, evidence coverage and PDF identity checks.
Only real-model results recorded in `evaluation/summary_eval/` support quality conclusions.

### Executed verification, 2026-11-10

- All 279 software tests passed, including 31 new summary-evaluation tests. Ruff passed
  for the new evaluation modules, CLI and tests.
- The real smoke instance was human query `transformer-architecture`, selected paper
  `1706.03762`, resolved to `1706.03762v7` ("Attention Is All You Need"). Local Qwen
  generated its summary in 110.38 seconds. Gemini evaluated all 15 extracted PDF pages,
  with bibliography removed, without evidence condensation.
- Actual metric scores were factual consistency 0.80, query relevance 1.00, coverage 0.90,
  clarity 1.00 and conciseness 1.00; the deterministic aggregate was 0.91 / 9.10 out of 10.
  Individual explanations are stored in `summary_evaluation_01.json`. This one observation
  is a smoke result, not a population-level claim about summary quality.
- Initial Gemini requests were blocked by the sandbox. The approved network retry reused
  the saved summary and completed the five metrics; cumulative evaluation duration was
  18.15 seconds. Historical structured errors remain available in the record.
- A subsequent normal smoke rerun with both generation methods explicitly blocked reused
  the real summary and metrics unchanged: zero summarizer calls and zero Gemini calls.
- The full 36-instance execution request was declined. Aggregate files contain all 36
  expected rows: one completed and 35 explicitly incomplete. The remaining summaries
  and scores have not been generated. The documented default command resumes from here.
- `.env` was confirmed ignored/untracked, with no supplied credential in Git-visible
  files. The pre-existing `readme.md` content was preserved byte for byte.

The CRISP-DM evaluation population and criteria are defined before inspecting scores:
12 existing research queries, three highly relevant papers per query, independent judge,
frozen source hashes and five fixed rubrics. Review low-scoring and high-scoring instances
manually against original PDFs before treating aggregate scores as evidence of usefulness.
