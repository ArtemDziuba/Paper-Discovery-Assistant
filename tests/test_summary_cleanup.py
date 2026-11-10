"""Regression checks for evidence reuse and semantic summary-cache identity."""

import json
import shutil

import pytest
from test_pipeline import paper
from test_summarization import VALID_SUMMARY, FakeOllama, FakePdf

from pda.fulltext import ExtractedPaper, TextUnit
from pda.summarization import Summarizer
from pda.summary_config import SummaryError, SummarySettings
from pda.summary_jobs import SummaryJobs


def request_data(request):
    return json.loads(request[-1]["content"].split("UNTRUSTED_DATA_JSON:\n")[1])


def request_kind(request):
    content = request[-1]["content"]
    if content.startswith("Extract concise"):
        return "extract"
    if content.startswith("Consolidate ALL"):
        return "reduce"
    return "final"


@pytest.fixture
def short_source(tmp_path, monkeypatch):
    settings = SummarySettings(cache_dir=tmp_path / "initial-cache")
    document = ExtractedPaper(
        [TextUnit("Methods", 1, "Supported paper evidence. " * 30)], 1, False, 0
    )
    monkeypatch.setattr("pda.summarization.extract_pdf", lambda *args: document)
    return settings, FakePdf(tmp_path / "paper.pdf")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("endpoint", "http://127.0.0.1:11435"),
        ("timeout_seconds", 45),
        ("keep_alive", "30m"),
        ("max_runtime_seconds", 90),
        ("max_pdf_bytes", 10 * 1024 * 1024),
        ("max_pages", 50),
        ("max_text_chars", 200_000),
        ("max_chunks", 48),
        ("max_model_calls", 40),
        ("max_reduce_rounds", 3),
    ],
)
def test_operational_changes_reuse_summary_but_keep_jobs_separate(short_source, field, value):
    settings, source = short_source
    initial_model = FakeOllama(settings)
    Summarizer(settings, ollama=initial_model, pdf_store=source).summarize(paper(), "graph query")
    changed = settings.model_copy(update={field: value})
    changed_model = FakeOllama(changed)
    checks = []
    identity = changed_model.identity

    def verified_identity():
        checks.append(True)
        return identity()

    changed_model.identity = verified_identity
    result = Summarizer(changed, ollama=changed_model, pdf_store=source).summarize(
        paper(), "graph query"
    )
    assert result.cache_hit and checks == [True]
    assert not changed_model.calls
    assert settings.content_fingerprint() == changed.content_fingerprint()
    first_jobs = SummaryJobs(settings, summarizer=object())
    changed_jobs = SummaryJobs(changed, summarizer=object())
    try:
        assert first_jobs.key(paper(), "graph query") != changed_jobs.key(paper(), "graph query")
    finally:
        first_jobs.close()
        changed_jobs.close()


def test_moving_cached_summaries_does_not_invalidate_content(short_source, tmp_path):
    settings, source = short_source
    model = FakeOllama(settings)
    service = Summarizer(settings, ollama=model, pdf_store=source)
    service.summarize(paper(), "graph query")
    relocated = settings.model_copy(update={"cache_dir": tmp_path / "relocated-cache"})
    shutil.copytree(settings.cache_dir / "summaries", relocated.cache_dir / "summaries")
    result = Summarizer(relocated, ollama=model, pdf_store=source).summarize(paper(), "graph query")
    assert result.cache_hit and len(model.calls) == 1
    assert settings.fingerprint() != relocated.fingerprint()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model", "qwen3.5:4b-custom"),
        ("num_ctx", 16384),
        ("temperature", 0.3),
        ("num_predict", 1024),
        ("seed", 7),
        ("think", True),
        ("template_reserve", 768),
    ],
)
def test_generation_and_context_changes_invalidate_summary_cache(short_source, field, value):
    settings, source = short_source
    Summarizer(settings, ollama=FakeOllama(settings), pdf_store=source).summarize(
        paper(), "graph query"
    )
    changed = settings.model_copy(update={field: value})
    model = FakeOllama(changed)
    result = Summarizer(changed, ollama=model, pdf_store=source).summarize(paper(), "graph query")
    assert not result.cache_hit and len(model.calls) == 1


class RepairModel(FakeOllama):
    def __init__(self, settings, *, initial_reduction=False):
        super().__init__(settings, [" ".join(["overlong"] * 450), VALID_SUMMARY])
        self.initial_reduction = initial_reduction

    def chat(self, messages):
        reply = super().chat(messages)
        if self.initial_reduction and request_kind(messages) == "reduce":
            has_final = any(request_kind(request) == "final" for request in self.calls)
            if not has_final:
                return "INITIAL_CONSOLIDATED_FINDINGS " + "paper evidence " * 75
        return reply


@pytest.mark.parametrize("initial_reduction", [False, True])
def test_repair_reuses_all_chunk_findings_without_reextracting(
    tmp_path, monkeypatch, initial_reduction
):
    settings = SummarySettings(cache_dir=tmp_path)
    units = [
        TextUnit(f"Section {i}", i + 1, f"IMPORTANT_FINDING_{i} " + "scientific evidence " * 40)
        for i in range(40 if initial_reduction else 8)
    ]
    document = ExtractedPaper(units, len(units), False, 0)
    monkeypatch.setattr("pda.summarization.extract_pdf", lambda *args: document)
    model = RepairModel(settings, initial_reduction=initial_reduction)
    result = Summarizer(
        settings, ollama=model, pdf_store=FakePdf(tmp_path / "paper.pdf")
    ).summarize(paper(), "graph query")
    extracts = [r for r in model.calls if request_kind(r) == "extract"]
    finals = [r for r in model.calls if request_kind(r) == "final"]
    first_final = next(i for i, r in enumerate(model.calls) if request_kind(r) == "final")
    repair_reductions = [r for r in model.calls[first_final + 1 :] if request_kind(r) == "reduce"]
    assert result.text == VALID_SUMMARY and result.chunks == len(extracts) > 1
    assert any(request_kind(r) == "extract" for r in model.calls[:first_final])
    assert len(finals) == 2 and repair_reductions
    observed = " ".join(request_data(r)["paper_evidence"] for r in extracts)
    assert all(f"IMPORTANT_FINDING_{i} " in observed for i in range(len(units)))
    assert all(request_kind(r) != "extract" for r in model.calls[first_final + 1 :])
    if initial_reduction:
        assert (
            "INITIAL_CONSOLIDATED_FINDINGS" in request_data(repair_reductions[0])["paper_evidence"]
        )
    repair = request_data(finals[-1])
    assert repair["research_query"] == "graph query"
    assert repair["previous_summary"] == " ".join(["overlong"] * 450)
    assert (
        f"chunks {','.join(str(i + 1) for i in range(result.chunks))}" in repair["paper_evidence"]
    )
    assert result.model_calls == len(model.calls)


def test_repair_does_not_reset_model_call_limit(short_source):
    settings, source = short_source
    limited = settings.model_copy(update={"max_model_calls": 1})
    model = FakeOllama(limited, ["too short", VALID_SUMMARY])
    with pytest.raises(SummaryError, match="model-call processing limit"):
        Summarizer(limited, ollama=model, pdf_store=source).summarize(paper(), "graph query")
    assert len(model.calls) == 1
    assert not (limited.cache_dir / "summaries").exists()


def test_repair_shares_the_configured_consolidation_depth_limit(tmp_path, monkeypatch):
    settings = SummarySettings(cache_dir=tmp_path, max_reduce_rounds=1)
    units = [TextUnit(f"Section {i}", i + 1, "scientific evidence " * 40) for i in range(40)]
    monkeypatch.setattr(
        "pda.summarization.extract_pdf", lambda *args: ExtractedPaper(units, 40, False, 0)
    )
    model = RepairModel(settings, initial_reduction=True)
    with pytest.raises(SummaryError, match="hierarchical processing limits"):
        Summarizer(settings, ollama=model, pdf_store=FakePdf(tmp_path / "paper.pdf")).summarize(
            paper(), "graph query"
        )
    first_final = next(i for i, r in enumerate(model.calls) if request_kind(r) == "final")
    assert first_final > 0 and first_final == len(model.calls) - 1
    assert not (settings.cache_dir / "summaries").exists()
