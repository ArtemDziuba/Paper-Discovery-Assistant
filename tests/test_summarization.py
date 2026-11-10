"""Pipeline correctness with mocked Qwen responses, not scientific-quality judgments."""

import json
import re
import threading
import time

import httpx
import pytest
from test_fulltext import scientific_pdf
from test_pipeline import paper

from pda.fulltext import ExtractedPaper, PdfStore, TextUnit
from pda.ollama import ModelIdentity, OllamaClient
from pda.summarization import Summarizer
from pda.summary_config import SummaryError, SummaryResult, SummarySettings
from pda.summary_jobs import SummaryJobs

VALID_SUMMARY = " ".join(f"supported{i}" for i in range(120))


class FakePdf:
    def __init__(self, path):
        self.path = path
        self.requests = []

    def get(self, identifier):
        self.requests.append(identifier)
        return self.path, "pdf-digest"


class FakeOllama:
    def __init__(self, settings, finals=None):
        self.settings = settings
        self.calls = []
        self.model_digest = "installed-weights-1"
        self.finals = list(finals or [VALID_SUMMARY])

    def identity(self):
        return ModelIdentity(self.settings.model, self.model_digest, "test-server")

    def fits(self, messages):
        return OllamaClient.fits(self, messages)

    def chat(self, messages):
        assert self.fits(messages), "No oversized input may reach the local model"
        self.calls.append(messages)
        content = messages[-1]["content"]
        if content.startswith("Extract concise"):
            return "Reported chunk evidence and documented limitations. " * 16
        if content.startswith("Consolidate ALL"):
            return "Consolidated findings preserve all supplied groups, methodologies, results and limitations."
        return self.finals.pop(0) if len(self.finals) > 1 else self.finals[0]


@pytest.fixture
def setup(tmp_path, monkeypatch):
    settings = SummarySettings(cache_dir=tmp_path)
    model = FakeOllama(settings)
    source = FakePdf(tmp_path / "source.pdf")
    document = ExtractedPaper(
        [TextUnit("Introduction", 1, "Scientific software fixture evidence. " * 30)], 1, False, 0
    )
    monkeypatch.setattr("pda.summarization.extract_pdf", lambda *args: document)
    return settings, model, source, document


def test_short_paper_uses_original_query_and_direct_pass_without_abstract_fallback(setup):
    settings, model, source, _ = setup
    result = Summarizer(settings, ollama=model, pdf_store=source).summarize(
        paper(), "Original research query about graphs"
    )
    assert result.text == VALID_SUMMARY and result.chunks == 1 and len(model.calls) == 1
    user = model.calls[0][-1]["content"]
    data = json.loads(user.split("UNTRUSTED_DATA_JSON:\n")[1])
    assert data["research_query"] == "Original research query about graphs"
    assert "Scientific software fixture" in data["paper_evidence"]
    assert paper().abstract not in data["paper_evidence"]
    assert "untrusted DATA" in model.calls[0][0]["content"]


def test_real_pdf_extraction_through_mocked_ollama_http_and_separate_pdf_cache(
    tmp_path, monkeypatch
):
    settings = SummarySettings(cache_dir=tmp_path)
    pdf_requests = []
    model_requests = []

    def download(request):
        pdf_requests.append(str(request.url))
        return httpx.Response(200, content=scientific_pdf())

    def infer(request):
        if request.url.path == "/api/tags":
            return httpx.Response(
                200, json={"models": [{"name": settings.model, "digest": "test-weights"}]}
            )
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "test-server"})
        if request.url.path == "/api/show":
            return httpx.Response(200, json={"model_info": {"qwen.context_length": 262144}})
        payload = json.loads(request.content)
        model_requests.append(payload)
        instruction = payload["messages"][-1]["content"]
        content = (
            "Reported methodologies, results and limitations."
            if instruction.startswith(("Extract", "Consolidate"))
            else VALID_SUMMARY
        )
        return httpx.Response(
            200,
            json={
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 200,
                "message": {"content": content},
            },
        )

    monkeypatch.setattr("pda.fulltext.time.sleep", lambda _: None)
    service = Summarizer(
        settings,
        ollama=OllamaClient(settings, httpx.MockTransport(infer)),
        pdf_store=PdfStore(settings, httpx.MockTransport(download)),
    )
    first = service.summarize(paper(), "query one")
    assert first.pages == 5 and first.chunks > 1 and first.bibliography_removed
    assert service.summarize(paper(), "QUERY ONE").cache_hit
    assert not service.summarize(paper(), "query two").cache_hit
    assert len(pdf_requests) == 1
    assert not service.summarize(paper("1234.5678v2"), "query two").cache_hit
    assert len(pdf_requests) == 2 and pdf_requests[-1].endswith("v2")
    supplied = " ".join(p["messages"][-1]["content"] for p in model_requests)
    assert "SECTION_0_FINDING" in supplied and "SECTION_4_FINDING" in supplied
    assert "BIBLIOGRAPHY_ONLY_TOKEN" not in supplied
    assert "query one" in supplied and "query two" in supplied


def test_model_call_limit_fails_instead_of_omitting_remaining_sections(setup, monkeypatch):
    settings, model, source, _ = setup
    settings = settings.model_copy(update={"max_model_calls": 1})
    document = ExtractedPaper([TextUnit("Methods", 1, "Scientific evidence. " * 1000)], 1, False, 0)
    monkeypatch.setattr("pda.summarization.extract_pdf", lambda *args: document)
    service = Summarizer(settings, ollama=model, pdf_store=source)
    with pytest.raises(SummaryError, match="model-call processing limit"):
        service.summarize(paper(), "query")
    assert len(model.calls) == 1
    assert not (settings.cache_dir / "summaries").exists()


def test_long_paper_reads_every_section_and_reduces_all_evidence(tmp_path, monkeypatch):
    settings = SummarySettings(cache_dir=tmp_path)
    model = FakeOllama(settings)
    units = [
        TextUnit(f"Section {i}", i + 1, f"IMPORTANT_FINDING_{i} " + "scientific evidence " * 40)
        for i in range(40)
    ]
    document = ExtractedPaper(units, 40, False, 0)
    monkeypatch.setattr("pda.summarization.extract_pdf", lambda *args: document)
    result = Summarizer(
        settings, ollama=model, pdf_store=FakePdf(tmp_path / "source.pdf")
    ).summarize(paper(), "graph query")
    chunk_calls = [
        request for request in model.calls if request[-1]["content"].startswith("Extract concise")
    ]
    assert result.chunks == len(chunk_calls) > 1
    observed = " ".join(call[-1]["content"] for call in chunk_calls)
    assert all(f"IMPORTANT_FINDING_{i} " in observed for i in range(40))
    assert any(call[-1]["content"].startswith("Consolidate ALL") for call in model.calls)
    assert result.model_calls == len(model.calls)


def test_cache_normalizes_query_but_invalidates_for_new_query_model_digest_and_settings(setup):
    settings, model, source, _ = setup
    service = Summarizer(settings, ollama=model, pdf_store=source)
    service.summarize(paper(), "Graph learning")
    assert service.summarize(paper(), "  GRAPH  learning ").cache_hit
    assert len(model.calls) == 1
    assert not service.summarize(paper(), "CNN recognition").cache_hit
    model.model_digest = "installed-weights-2"
    assert not service.summarize(paper(), "Graph learning").cache_hit
    assert len(model.calls) == 3
    changed = settings.model_copy(update={"temperature": 0.3})
    assert (
        not Summarizer(changed, ollama=model, pdf_store=source)
        .summarize(paper(), "Graph learning")
        .cache_hit
    )
    assert len(model.calls) == 4


def test_regeneration_bypasses_summary_cache_and_pipeline_version_invalidates_it(
    setup, monkeypatch
):
    settings, model, source, _ = setup
    service = Summarizer(settings, ollama=model, pdf_store=source)
    service.summarize(paper(), "graph query")
    service.summarize(paper(), "graph query", regenerate=True)
    assert len(model.calls) == 2
    monkeypatch.setattr("pda.summary_config.PIPELINE_VERSION", "new-prompt-version")
    result = service.summarize(paper(), "graph query")
    assert not result.cache_hit and result.pipeline_version == "new-prompt-version"
    assert len(model.calls) == 3


def test_parser_version_invalidates_summary_cache(setup, monkeypatch):
    settings, model, source, _ = setup
    service = Summarizer(settings, ollama=model, pdf_store=source)
    service.summarize(paper(), "graph query")
    monkeypatch.setattr("pda.summary_config.PDF_PARSER_VERSION", "future-parser-version")
    result = service.summarize(paper(), "graph query")
    assert not result.cache_hit and result.parser_version == "future-parser-version"
    assert len(model.calls) == 2


def test_invalid_length_is_repaired_once_and_invalid_summaries_are_not_cached(setup):
    settings, _, source, _ = setup
    model = FakeOllama(settings, ["too short", VALID_SUMMARY])
    result = Summarizer(settings, ollama=model, pdf_store=source).summarize(paper(), "graph query")
    assert result.text == VALID_SUMMARY and len(model.calls) == 2
    invalid = FakeOllama(settings, ["too short"])
    with pytest.raises(SummaryError, match="words"):
        Summarizer(settings, ollama=invalid, pdf_store=source).summarize(paper(), "different query")
    assert len(invalid.calls) == 2
    assert len(list((settings.cache_dir / "summaries").glob("*.json"))) == 1


@pytest.mark.parametrize("count", [50, 75, 99, 153])
def test_summary_within_shadow_bounds_is_accepted_and_cached_without_rewriting(setup, count):
    settings, _, source, _ = setup
    text = " ".join(["evidence"] * count)
    model = FakeOllama(settings, [text])
    service = Summarizer(settings, ollama=model, pdf_store=source)
    assert service.summarize(paper(), "graph query").text == text
    assert service.summarize(paper(), "graph query").cache_hit
    assert len(model.calls) == 1


def test_summary_below_50_words_is_rewritten_with_query_paper_and_draft(setup):
    settings, _, source, document = setup
    too_short = " ".join(["evidence"] * 49)
    model = FakeOllama(settings, [too_short, VALID_SUMMARY])
    result = Summarizer(settings, ollama=model, pdf_store=source).summarize(paper(), "graph query")
    assert result.text == VALID_SUMMARY and result.model_calls == 2
    instruction, data = model.calls[-1][-1]["content"].split("UNTRUSTED_DATA_JSON:\n")
    data = json.loads(data)
    assert data["previous_summary"] == too_short
    assert data["paper_evidence"] == document.text
    assert data["research_query"] == "graph query"
    assert "100–150" in instruction
    assert not re.search(r"\b(?:50|180)\b", instruction)


def test_over_shadow_limit_rewrites_with_paper_query_and_untrusted_previous_draft(setup):
    settings, _, source, document = setup
    too_long = " ".join(["overlong"] * 181)
    model = FakeOllama(settings, [too_long, VALID_SUMMARY])
    result = Summarizer(settings, ollama=model, pdf_store=source).summarize(paper(), "graph query")
    assert result.text == VALID_SUMMARY and result.model_calls == 2
    first, second = (request[-1]["content"] for request in model.calls)
    first_data = json.loads(first.split("UNTRUSTED_DATA_JSON:\n")[1])
    repair_instruction, repair_json = second.split("UNTRUSTED_DATA_JSON:\n")
    repair_data = json.loads(repair_json)
    assert "previous_summary" not in first_data
    assert repair_data["previous_summary"] == too_long
    assert repair_data["paper_evidence"] == document.text
    assert repair_data["research_query"] == "graph query"
    assert "100–150" in repair_instruction and "180" not in repair_instruction
    assert "not evidence" in repair_instruction
    assert all("180" not in request[0]["content"] for request in model.calls)


def test_repeated_overlong_outputs_fail_after_bounded_repair_without_caching(setup):
    settings, _, source, _ = setup
    too_long = " ".join(["overlong"] * 181)
    model = FakeOllama(settings, [too_long])
    with pytest.raises(SummaryError, match="181 words"):
        Summarizer(settings, ollama=model, pdf_store=source).summarize(paper(), "graph query")
    assert len(model.calls) == 2
    assert not (settings.cache_dir / "summaries").exists()


def test_repair_keeps_draft_and_all_paper_sections_when_context_requires_chunking(
    setup, monkeypatch
):
    from pda.summarization import FINAL, REPAIR, messages

    settings, _, source, _ = setup
    too_long = " ".join(["overlong"] * 450)
    model = FakeOllama(settings, [too_long, VALID_SUMMARY])
    document = ExtractedPaper(
        [TextUnit("Results", 1, "PAPER_FINDING " + "paper evidence " * 240)], 1, False, 0
    )
    monkeypatch.setattr("pda.summarization.extract_pdf", lambda *args: document)
    assert model.fits(messages("graph query", paper().title, document.text, FINAL))
    assert not model.fits(
        messages("graph query", paper().title, document.text, REPAIR, previous_summary=too_long)
    )
    result = Summarizer(settings, ollama=model, pdf_store=source).summarize(paper(), "graph query")
    assert result.text == VALID_SUMMARY and result.chunks > 1
    read_calls = [c[-1]["content"] for c in model.calls if c[-1]["content"].startswith("Extract")]
    evidence = " ".join(
        json.loads(c.split("UNTRUSTED_DATA_JSON:\n")[1])["paper_evidence"] for c in read_calls
    )
    assert "PAPER_FINDING" in evidence
    recovered = re.sub(r"\[Section: [^\n]*; page \d+\]\n", "", evidence)
    assert recovered.split() == document.units[0].text.split()
    repair_data = json.loads(model.calls[-1][-1]["content"].split("UNTRUSTED_DATA_JSON:\n")[1])
    assert repair_data["previous_summary"] == too_long
    assert "Evidence from chunks" in repair_data["paper_evidence"]


def test_model_and_extraction_failures_do_not_produce_abstract_summaries(setup, monkeypatch):
    settings, model, source, _ = setup
    monkeypatch.setattr(
        "pda.summarization.extract_pdf",
        lambda *args: (_ for _ in ()).throw(SummaryError("insufficient PDF text")),
    )
    with pytest.raises(SummaryError, match="insufficient PDF"):
        Summarizer(settings, ollama=model, pdf_store=source).summarize(paper(), "graph")
    assert not model.calls
    assert not (settings.cache_dir / "summaries").exists()


def test_missing_model_fails_before_pdf_download(setup):
    settings, model, source, _ = setup

    def missing():
        raise SummaryError("Qwen is not installed")

    model.identity = missing
    with pytest.raises(SummaryError, match="not installed"):
        Summarizer(settings, ollama=model, pdf_store=source).summarize(paper(), "graph")
    assert not source.requests


def test_unversioned_metadata_resolves_to_actual_version(setup):
    settings, model, source, _ = setup
    requested = []

    class Metadata:
        def fetch_ids(self, identifiers):
            requested.extend(identifiers)
            return [paper("1234.5678v9")], {}

    unversioned = paper("1234.5678")
    result = Summarizer(
        settings, ollama=model, pdf_store=source, metadata_client=Metadata()
    ).summarize(unversioned, "graph")
    assert requested == ["1234.5678"] and source.requests == ["1234.5678v9"]
    assert result.arxiv_id == "1234.5678v9"


def test_processing_limit_rejects_long_paper_without_discarding_sections(setup, monkeypatch):
    settings, model, source, _ = setup
    limited = settings.model_copy(update={"max_chunks": 1})
    monkeypatch.setattr(
        "pda.summarization.extract_pdf",
        lambda *args: ExtractedPaper([TextUnit("Results", 1, "evidence " * 3000)], 1, False, 0),
    )
    with pytest.raises(SummaryError, match="no sections were omitted"):
        Summarizer(limited, ollama=model, pdf_store=source).summarize(paper(), "graph")
    assert not model.calls


def result_for(query):
    return SummaryResult(
        text=VALID_SUMMARY,
        arxiv_id="1234.5678v1",
        query=query,
        model="qwen3.5:4b",
        model_digest="test",
        pdf_sha256="test",
        pages=2,
        chunks=1,
        model_calls=1,
    )


def wait_for(jobs, key, status):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        job = jobs.snapshot(key)
        if job and job.status == status:
            return job
        time.sleep(0.01)
    pytest.fail(f"Job did not reach {status}")


def test_background_job_deduplicates_running_requests_and_supports_retry(tmp_path):
    entered, release = threading.Event(), threading.Event()
    calls = []

    class Service:
        def summarize(self, source, query, **kwargs):
            calls.append(query)
            entered.set()
            release.wait(3)
            if len(calls) == 1:
                raise SummaryError("Ollama unavailable")
            return result_for(query)

    jobs = SummaryJobs(SummarySettings(cache_dir=tmp_path), Service())
    try:
        key = jobs.submit(paper(), "graph")
        assert entered.wait(2)
        assert jobs.submit(paper(), "GRAPH") == key
        assert len(calls) == 1 and jobs.snapshot(key).status == "running"
        release.set()
        assert wait_for(jobs, key, "failed").error == "Ollama unavailable"
        jobs.submit(paper(), "graph")
        assert wait_for(jobs, key, "succeeded").result.text == VALID_SUMMARY
        assert len(calls) == 2
    finally:
        release.set()
        jobs.close()
