# Local setup and demonstration

Requirements: Python 3.12 or later and uv. Run from the repository root:

```powershell
uv sync
uv run streamlit run app.py
```

Open the localhost URL printed by Streamlit. The initial semantic search downloads the
encoder from Hugging Face. Subsequent inference runs locally. Internet is required for
uncached arXiv searches. Select TF-IDF baseline to run without any model download.

If retrieval fails, inspect the specific connection, timeout, or HTTP error shown in the
interface and the terminal log. Run the command above in your own terminal; a server
started inside a coding assistant's sandbox may render normally while outbound arXiv
connections are blocked (Windows socket error 10013). Cached searches can still succeed
in that situation, so they do not prove that live retrieval is available.

Source file watching is disabled in .streamlit/config.toml to avoid Streamlit inspecting
lazy Transformers imports for unrelated optional image/audio dependencies. Restart the
server after editing source files.

CPU is the default to avoid assuming a CUDA-enabled PyTorch installation. Select CUDA
only after checking `uv run python -c "import torch; print(torch.cuda.is_available())"`.
A CUDA-compatible PyTorch build must match your driver; follow the official PyTorch
installation selector at https://pytorch.org/get-started/locally/ if replacement is needed.
No claim of measured hardware compliance is made until a representative run is profiled.

For this Windows RTX 4060 laptop, the optional CUDA environment is pinned to PyTorch 2.9.1
from the official CUDA 12.8 wheel index. The driver must support that runtime. Install and
launch with the extra enabled (the initial GPU download is approximately 2.7 GiB):

```powershell
uv sync --extra cuda
uv run --extra cuda python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
uv run --extra cuda streamlit run app.py
```

Keep `--extra cuda` on `uv run` commands when using the GPU environment; otherwise uv may
restore the default CPU dependency selection. The CUDA build also supports CPU inference.
Selecting cuda in the interface cannot add CUDA support to a CPU-only PyTorch build.
The app checks availability before model loading and reports an actionable error.

Topic interpretation removes common leading request phrases and joins remaining concepts
with AND. This reduces broad filler matches but can lose relevant papers with missing terms.
For a known title, enter it in double quotes, for example `"Attention Is All You Need"`.
Quoted phrases retain stopwords during arXiv search. No paper IDs are inserted into results.
For quoted queries, exact title matches are shown first, with unchanged cosine scores.
Run `uv run --extra cuda python scripts/verify_cuda.py` to reproduce CPU/GPU agreement and
the quoted-title regression. Its source candidates must be cached, or internet access is needed.

The candidate control defaults to 1,000 and accepts up to 10,000. This is an upper bound on
raw arXiv entries, not a promise of that many distinct valid papers. Retrieval pages contain
at most 1,000 entries, with rate limiting and retries. Larger uncached searches can take
minutes depending on the API; cached searches still rerank the candidates locally. Watch
the retrieval/encoding progress and the actual usable candidate count. Partial retrieval
after a later page failure is labeled explicitly. Retry to reuse successful cached pages.
The interface displays retrieval time, ranking time, and whether all retrieved pages were
cached. The JSON export also includes total time with model loading and per-page provenance.

```powershell
uv run --extra cuda python scripts/benchmark_candidates.py
```

This retrieves up to 10,000 candidates for the reported transformer topic and compares
50/100/1,000/10,000 prefixes of one snapshot. It records timings, memory and the known paper's
presence in evaluation/candidate-scaling.json. This diagnostic does not supply general
relevance judgments or establish that a higher cosine score means a better paper.

```powershell
uv run pytest
uv run ruff check src tests scripts app.py
uv run python scripts/evaluate.py --help
uv run python scripts/prepare_scifact.py
uv run python scripts/evaluate.py --data evaluation/data/scifact-test.json --output evaluation/scifact-lexical.json --method lexical
uv run python scripts/evaluate.py --data evaluation/data/scifact-test.json --output evaluation/scifact-semantic.json --method semantic
```

Dataset preparation downloads BEIR SciFact data; review its original terms before redistribution.
Downloaded data is ignored by Git; evaluation reports are retained. Run
`uv run python scripts/smoke.py` for live retrieval and local-model verification, and
`uv run python scripts/audit_cached.py` for structural checks on retrieved records.

Retrieval pages are cached under .cache/arxiv for 24 hours, keyed by query, offset, page size,
and sort settings. The paginated cache uses a new namespace; older single-request caches
are left untouched but not reused. Delete individual cache
files to force a new search. Invalid cache records are ignored. No automatic stale-cache
fallback or offline replay interface is implemented yet. JSON export includes source IDs,
scores, excerpts, configuration, retrieval timestamps, and stage timings.

Suggested demo: a specific topic, a broader topic with weak candidates, an exclusion query
that triggers the baseline limitation warning, and an invalid empty query. Explain that
overview sentences are extracted from abstracts and scores are not calibrated probabilities.

For a video under 10 minutes: requirements (1 minute), architecture and one query flow
(2), live cases (3), evaluations and limitations (2), agent instructions and skills (1).
