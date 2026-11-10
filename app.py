"""Run with: uv run streamlit run app.py."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import streamlit as st

from pda.pipeline import search
from pda.query import interpret
from pda.ranking import FINETUNED_OPTIONS, LexicalEncoder, SemanticEncoder
from pda.retrieval import ArxivClient, RetrievalError
from pda.summary_config import SummaryError, SummarySettings, word_count
from pda.summary_jobs import SummaryJobs

st.set_page_config(page_title="Paper Discovery Assistant", page_icon="📚", layout="wide")
st.title("Paper Discovery Assistant")
st.write(
    "Find arXiv papers for a research question, with local ranking and on-demand full-paper summaries."
)


@st.cache_resource
def client():
    return ArxivClient()


@st.cache_resource
def encoder(method, device):
    if method in FINETUNED_OPTIONS:
        return SemanticEncoder(device, model_path=FINETUNED_OPTIONS[method])
    return SemanticEncoder(device) if method == "Semantic" else LexicalEncoder()


@st.cache_resource
def summary_jobs():
    return SummaryJobs(SummarySettings.from_env())


@st.fragment(run_every="1s")
def summary_panel(paper, research_query):
    try:
        jobs = summary_jobs()
    except ValueError as exc:
        st.error(f"Summarization configuration is invalid: {exc}")
        return
    key = jobs.key(paper, research_query)
    requested = st.session_state.setdefault("requested_summaries", set())
    request_errors = st.session_state.setdefault("summary_request_errors", {})
    job = jobs.snapshot(key) if key in requested else None
    busy = job is not None and job.status in {"queued", "running"}
    if job and job.result:
        st.write(job.result.text)
        st.caption(
            f"{word_count(job.result.text)} words · full-paper PDF text · {'cached summary' if job.result.cache_hit else 'summary ready'}"
        )
    if busy:
        st.status(job.progress, state="running", expanded=False)
    elif key in request_errors:
        st.error(request_errors[key])
    elif job and job.error:
        st.error(job.error)
    label = (
        "Summarizing…"
        if busy
        else "Regenerate summary"
        if job and job.result
        else "Retry summary"
        if (job and job.error) or key in request_errors
        else "Summarize"
    )

    def submit_summary():
        request_errors.pop(key, None)
        try:
            jobs.submit(paper, research_query, regenerate=bool(job and job.result))
            requested.add(key)
        except SummaryError as exc:
            request_errors[key] = str(exc)

    # The callback runs before this fragment redraws, so its next snapshot already
    # shows the queued job. Do not rerun the app and recreate the parent expander.
    st.button(label, key=f"summarize-{key}", disabled=busy, on_click=submit_summary)


with st.sidebar:
    st.header("Search settings")
    method = st.selectbox("Ranking", ["Semantic", "TF-IDF baseline", *FINETUNED_OPTIONS])
    device = st.selectbox(
        "Inference device",
        ["cpu", "cuda"],
        help="CUDA requires a compatible CUDA-enabled PyTorch installation.",
    )
    candidates = st.number_input(
        "Candidate papers", min_value=10, max_value=10_000, value=1000, step=10
    )
    st.caption("Retrieve up to 10,000 candidates. Larger uncached searches take longer.")
    top_k = st.slider("Displayed papers", min_value=5, max_value=50, value=10, step=5)
    st.caption(
        "MiniLM is saved locally on first use and reused on later launches. Inference stays local."
    )

with st.form("search"):
    query = st.text_area(
        "Research question or topic",
        max_chars=1000,
        placeholder="Efficient transformers for long document summarization",
        help="Enter a research topic or a complete paper title. Quotation marks are optional for titles.",
    )
    submitted = st.form_submit_button("Find papers", type="primary")

if submitted:
    status = None
    try:
        interpret(query)
        started = time.perf_counter()
        with st.spinner("Retrieving and ranking papers…"):
            status = st.progress(0, text="Loading ranking model…")
            st.session_state.result = search(
                query,
                client(),
                encoder(method, device),
                candidates,
                top_k,
                progress=lambda fraction, text: status.progress(fraction, text=text),
            )
            st.session_state.result.metadata["ui_total_seconds"] = time.perf_counter() - started
    except (ValueError, RetrievalError, OSError, RuntimeError) as exc:
        st.session_state.pop("result", None)
        st.error(str(exc))
    finally:
        if status is not None:
            status.empty()

if "result" in st.session_state:
    result = st.session_state.result
    st.subheader(result.interpretation.original)
    st.caption(
        f"Ranked {result.metadata['candidate_count']:,} usable candidates · "
        f"retrieval {result.metadata['retrieval_seconds']:.1f}s · "
        f"ranking {result.metadata['ranking_seconds']:.1f}s · "
        f"{'cached retrieval' if result.metadata.get('cache_hit') else 'live retrieval'}"
    )
    if result.metadata.get("partial_failure"):
        st.warning(
            "Only part of the requested search completed. Results use the retrieved "
            f"candidates. {result.metadata['partial_failure']}"
        )
    if (result.metadata.get("title_lookup") or {}).get("error"):
        st.warning(
            "The exact-title lookup failed; these results use topic retrieval only. "
            + result.metadata["title_lookup"]["error"]
        )
    for warning in result.interpretation.warnings:
        st.warning(warning)
    with st.expander("Search interpretation and execution details"):
        st.write("Concepts:", result.interpretation.concepts)
        st.code(result.interpretation.search_query, language=None)
        st.json(result.metadata)
    st.caption(
        "Scores are 0–100 similarity indices, not probabilities of relevance. "
        "Ranking covers only the retrieved candidates. Inspect evidence for weak matches."
    )
    if result.metadata.get("exact_title_priority"):
        st.caption("Exact title matches appear first; similarity scores are unchanged.")
    if not result.matches:
        st.info("No usable papers found. Try a more specific research topic.")
    else:
        st.subheader("Ranked papers")
        for match in result.matches:
            with st.expander(f"{match.display_score:.1f}/100 · {match.paper.title}"):
                st.write(", ".join(match.paper.authors))
                st.caption(
                    f"{match.paper.published.date()} · "
                    f"{', '.join(match.paper.categories)} · {match.paper.arxiv_id}"
                )
                st.write("Matching terms:", ", ".join(match.matched_terms) or "No exact overlap")
                st.write("Abstract:", match.paper.abstract)
                st.caption(f"Raw cosine: {match.raw_similarity:.6f}")
                st.link_button("Open arXiv abstract / paper", str(match.paper.url))
                summary_panel(match.paper, result.interpretation.original)
        st.download_button(
            "Download results and provenance",
            result.model_dump_json(indent=2),
            file_name="paper-discovery-results.json",
            mime="application/json",
        )
