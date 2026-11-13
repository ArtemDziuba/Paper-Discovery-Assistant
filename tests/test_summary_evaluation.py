"""Evaluation plumbing with mocked generators/judges; not scientific quality evidence."""

import csv
import json
import runpy
from copy import deepcopy
from types import SimpleNamespace

import pytest

from pda.fulltext import ExtractedPaper, TextUnit
from pda.summary_config import PROJECT_ROOT, SummaryResult, SummarySettings
from pda.summary_eval_data import (
    WEIGHTS,
    load_instances,
    new_record,
    read_record,
    save_record,
    valid_metric,
    weighted_score,
)
from pda.summary_eval_judge import (
    DeepEvalJudge,
    JudgeConfigurationError,
    JudgeSettings,
    load_credentials,
    private_deepeval_imports,
)
from pda.summary_eval_reports import print_report
from pda.summary_evaluation import SummaryEvaluation


@pytest.fixture
def instances():
    return load_instances(PROJECT_ROOT / "evaluation")


@pytest.fixture
def document():
    return ExtractedPaper(
        [
            TextUnit("Methods", 1, "The scientific method uses attention."),
            TextUnit("Results", 2, "Results support representation learning with limitations."),
        ],
        2,
        False,
        0,
    )


class FakeJudge:
    def __init__(self, failures=(), version="one", before_measure=None):
        self.fingerprint = version
        self.configuration = {"rubric_version": version}
        self.settings = SimpleNamespace(model="mock-independent-judge")
        self.failures = set(failures)
        self.calls = []
        self.before_measure = before_measure

    def evidence(self, document, query, summary, cache):
        return {"text": document.text, "source_sha256": "source", "condensed": False}

    def measure(self, name, query, summary, evidence):
        self.calls.append((name, query, summary))
        if self.before_measure:
            self.before_measure(name)
        if name in self.failures:
            raise RuntimeError("Mock judge unavailable")
        return {"status": "completed", "score": 0.8, "reason": "Mocked plumbing response."}


def fake_summary():
    return SummaryResult(
        text="Software test summary. " * 40,
        arxiv_id="1706.03762v7",
        query="query",
        model="mock-generator",
        model_digest="fixture",
        pdf_sha256="fixture-pdf",
        pages=2,
        chunks=1,
        model_calls=1,
    )


@pytest.fixture
def setup_run(tmp_path, instances, document):
    judge = FakeJudge()
    generated = []

    def summarize(paper, query, **kwargs):
        generated.append((query, kwargs))
        return fake_summary()

    def runner(rows=None, selected_judge=None):
        result = SummaryEvaluation(
            rows or instances[:1],
            tmp_path,
            summary_settings=SummarySettings(cache_dir=tmp_path),
            summarizer_factory=lambda: SimpleNamespace(summarize=summarize),
            judge_factory=lambda: selected_judge or judge,
            document_loader=lambda r: document,
            log=lambda s: None,
        )
        result._paper = lambda r: object()
        return result

    return tmp_path, runner, judge, generated


def test_actual_dataset_has_36_stable_instances_with_first_three_in_original_order(instances):
    assert len(instances) == 36
    assert [r["evaluation_id"] for r in instances] == [
        f"summary_evaluation_{i:02}" for i in range(1, 37)
    ]
    assert [r["dataset_source"] for r in instances] == ["human_eval"] * 6 + ["synthetic_eval"] * 30
    for offset in range(0, 36, 3):
        record = instances[offset]
        folder = "human_eval" if offset < 6 else "syntetic_eval"
        original = json.loads(
            (PROJECT_ROOT / "evaluation" / folder / record["source_file"]).read_text()
        )
        assert [r["paper"]["arxiv_id"] for r in instances[offset : offset + 3]] == [
            j["arxiv_id"] for j in original["judgments"][:3]
        ]
    assert not any(r["dataset_errors"] for r in instances)


def test_loader_reports_missing_papers_and_ids_without_replacement(tmp_path, instances):
    for source, count in (("human_eval", 2), ("synthetic_eval", 10)):
        folder = tmp_path / source
        folder.mkdir()
        for i in range(count):
            save_record(
                folder / f"batch_{i:02}.json",
                {
                    "query_id": f"q{i}",
                    "query": "test query",
                    "judgments": [{"title": "Missing ID", "relevance": 3}],
                },
            )
    rows = load_instances(tmp_path)
    assert len(rows) == 36
    assert rows[0]["paper"]["title"] == "Missing ID"
    assert rows[1]["paper"]["arxiv_id"] is None
    assert all(row["dataset_errors"] for row in rows)


def test_loader_rejects_unexpected_query_count(tmp_path):
    (tmp_path / "human_eval").mkdir()
    save_record(
        tmp_path / "human_eval" / "one.json",
        {
            "query_id": "q",
            "query": "query",
            "judgments": [],
        },
    )
    with pytest.raises(ValueError, match="Expected 2 queries"):
        load_instances(tmp_path)


def test_summary_is_saved_before_judging_and_reused_on_repeat(setup_run):
    directory, runner, judge, generated = setup_run
    path = directory / "summary_evaluation_01.json"
    judge.before_measure = lambda name: read_record(path, runner().instances[0])["summarization"][
        "summary"
    ]
    first = runner().run()
    assert first["statistics"]["completed"] == 1
    assert len(generated) == 1 and len(judge.calls) == 5
    runner().run()
    assert len(generated) == 1 and len(judge.calls) == 5


def test_weighted_score_exact_weights_and_missing_or_failed_metrics():
    values = [0.9, 0.85, 0.95, 0.8, 0.75]
    metrics = {
        name: {"score": score, "reason": "fixture"}
        for name, score in zip(WEIGHTS, values, strict=True)
    }
    assert weighted_score(metrics) == sum(
        w * v for w, v in zip(WEIGHTS.values(), values, strict=True)
    )
    del metrics["clarity"]
    assert weighted_score(metrics) is None
    metrics["clarity"] = {"status": "failed", "score": 0.9, "reason": "invalid"}
    assert weighted_score(metrics) is None


@pytest.mark.parametrize("score", [-0.1, 1.1, float("nan"), float("inf"), "0.5", True, None])
def test_invalid_scores_are_not_accepted(score):
    assert not valid_metric({"score": score, "reason": "fixture"})


def test_partial_metrics_retry_only_failed_dimension(setup_run):
    _directory, runner, judge, generated = setup_run
    judge.failures.add("clarity")
    first = runner().run()
    assert first["statistics"]["incomplete"] == 1
    assert first["records"][0]["evaluation"]["weighted_score"] is None
    assert first["statistics"]["scores"]["factual_consistency"]["count"] == 0
    assert first["statistics"]["available_metric_diagnostics"]["factual_consistency"]["count"] == 1
    judge.failures.clear()
    result = runner().run(mode="evaluation-only")
    assert len(generated) == 1
    assert len(judge.calls) == 6 and judge.calls[-1][0] == "clarity"
    assert result["statistics"]["completed"] == 1


def test_interrupted_judge_preserves_summary_and_completed_metric(setup_run):
    directory, runner, judge, generated = setup_run

    def interrupt(name):
        if name == "query_relevance":
            raise KeyboardInterrupt

    judge.before_measure = interrupt
    with pytest.raises(KeyboardInterrupt):
        runner().run()
    stored = json.loads((directory / "summary_evaluation_01.json").read_text())
    assert stored["summarization"]["status"] == "completed"
    assert stored["evaluation"]["metrics"]["factual_consistency"]["score"] == 0.8
    judge.before_measure = None
    result = runner().run()
    assert result["statistics"]["completed"] == 1
    assert len(generated) == 1
    assert [call[0] for call in judge.calls].count("factual_consistency") == 1


def test_force_evaluation_and_changed_judge_preserve_summary(setup_run):
    _directory, runner, judge, generated = setup_run
    runner().run()
    runner().run(force_evaluate=True)
    assert len(judge.calls) == 10 and len(generated) == 1
    replacement = FakeJudge(version="two")
    runner(selected_judge=replacement).run()
    assert len(replacement.calls) == 5 and len(generated) == 1


def test_forced_generation_invalidates_scores_and_archives_previous_summary(setup_run):
    directory, runner, judge, generated = setup_run
    runner().run()
    runner().run(force_summarize=True)
    assert len(generated) == 2 and generated[-1][1]["regenerate"] is True
    assert len(judge.calls) == 10
    assert len(list((directory / "history").glob("*.json"))) == 1


def test_evaluation_only_never_generates_missing_summaries(setup_run):
    _directory, runner, judge, generated = setup_run
    result = runner().run(mode="evaluation-only")
    assert not generated and not judge.calls
    assert result["statistics"]["incomplete"] == 1
    assert "generation disabled" in result["records"][0]["evaluation"]["errors"][0]["message"]


def test_aggregate_only_recomputes_scores_and_calls_no_models(setup_run):
    directory, runner, judge, generated = setup_run
    runner().run()
    path = directory / "summary_evaluation_01.json"
    record = json.loads(path.read_text())
    record["evaluation"]["metrics"]["clarity"]["score"] = 0.2
    save_record(path, record)
    result = runner().run(mode="aggregate-only")
    assert len(generated) == 1 and len(judge.calls) == 5
    assert result["records"][0]["evaluation"]["weighted_score"] == pytest.approx(0.74)


def test_aggregate_includes_all_36_instances_with_failures_and_full_reasons(
    setup_run, instances, capsys
):
    directory, runner, _judge, _generated = setup_run
    runner().run()
    result = runner(instances).run(mode="aggregate-only")
    rows = list(
        csv.DictReader((directory / "summary_evaluation_results.csv").open(encoding="utf-8"))
    )
    aggregate = json.loads((directory / "summary_evaluation_results.json").read_text())
    assert len(rows) == len(aggregate["records"]) == 36
    assert rows[0]["clarity_reason"] == "Mocked plumbing response."
    assert rows[1]["weighted_score"] == ""
    assert aggregate["statistics"]["completed"] == 1
    assert aggregate["statistics"]["incomplete"] == 35
    assert len(aggregate["by_query"]) == 12
    assert aggregate["by_source"]["human_eval"]["expected"] == 6
    assert aggregate["by_source"]["synthetic_eval"]["expected"] == 30
    print_report(result)
    assert "36 synthetic" in capsys.readouterr().out


def test_smoke_processes_one_pair_but_creates_stable_36_records(setup_run, instances):
    directory, runner, judge, generated = setup_run
    result = runner(instances).run(smoke=True)
    assert len(generated) == 1 and len(judge.calls) == 5
    assert len(result["records"]) == 36
    assert len(list(directory.glob("summary_evaluation_[0-9][0-9].json"))) == 36


def test_changed_query_or_paper_cannot_reuse_or_overwrite_cache(setup_run):
    directory, runner, _judge, generated = setup_run
    runner().run()
    path = directory / "summary_evaluation_01.json"
    original = path.read_bytes()
    rows = deepcopy(runner().instances)
    rows[0]["query"] = "different query"
    result = runner(rows).run(mode="aggregate-only")
    assert path.read_bytes() == original
    assert result["statistics"]["failed"] == 1
    assert len(generated) == 1


def test_generation_failures_continue_to_next_entry(setup_run, instances):
    _directory, runner, _judge, _generated = setup_run
    active = runner(instances[:2])
    generate = active._generate

    def fail_first(record, force):
        if record["evaluation_id"].endswith("01"):
            raise RuntimeError("PDF unavailable")
        return generate(record, force)

    active._generate = fail_first
    result = active.run()
    assert result["statistics"]["failed"] == result["statistics"]["completed"] == 1


def test_missing_google_key_is_clear_and_has_no_fallback(tmp_path, monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(JudgeConfigurationError, match="Missing GOOGLE_API_KEY"):
        load_credentials(tmp_path / "absent.env")


def test_full_paper_evidence_and_native_geval_with_mocked_gemini(document, tmp_path, monkeypatch):
    pytest.importorskip("deepeval")
    GEval, GeminiModel, Case, Params = private_deepeval_imports()
    from deepeval.metrics.g_eval.schema import ReasonScore

    judge = object.__new__(DeepEvalJudge)
    judge.settings = JudgeSettings()
    judge.GEval, judge.TestCase, judge.Params = GEval, Case, Params
    model = object.__new__(GeminiModel)
    model.name = "gemini-2.5-flash"
    model.model_data = SimpleNamespace(supports_log_probs=False)
    calls = []

    def generate(prompt, schema=None):
        calls.append(prompt)
        assert "original research question" in prompt and document.units[-1].text in prompt
        return ReasonScore(score=8.5, reason="Mock native Gemini response."), 0

    monkeypatch.setattr(model, "generate", generate)
    judge.model = model
    evidence = judge.evidence(document, "query", "summary", tmp_path / "evidence.json")
    assert evidence["text"] == document.text and evidence["condensed"] is False
    for name in WEIGHTS:
        measured = judge.measure(name, "original research question", "fixture summary", evidence)
        assert measured["score"] == 0.85 and measured["reason"] == "Mock native Gemini response."
    assert len(calls) == 5


def test_long_evidence_covers_all_chunks_and_reuses_verified_excerpts(tmp_path):
    judge = object.__new__(DeepEvalJudge)
    judge.settings = JudgeSettings(max_input_bytes=32768)
    judge.fingerprint = "fixture"
    document = ExtractedPaper(
        [
            TextUnit(
                section, i + 1, f"Important {section} finding. " + ("Scientific context. " * 150)
            )
            for i, section in enumerate(["Methods", "Experiments", "Results", "Limitations"])
        ],
        4,
        False,
        0,
    )
    from pda.summary_eval_judge import SelectedEvidence

    calls = []

    def generate(prompt, schema):
        data = json.loads(prompt[prompt.index('{"query"') :])
        chunk = data["paper_chunk"]
        quote = next(line for line in chunk.splitlines() if "Important" in line).split(".")[0] + "."
        calls.append(quote)
        return SelectedEvidence(passages=[{"category": "findings", "quote": quote}]), 0

    judge.model = SimpleNamespace(generate=generate)
    cache = tmp_path / "evidence.json"
    evidence = judge.evidence(document, "research", "summary", cache)
    assert evidence["condensed"] is True
    assert evidence["covered_chunks"] == list(range(1, evidence["source_chunks"] + 1))
    assert len(calls) == evidence["source_chunks"]
    judge.evidence(document, "research", "summary", cache)
    assert len(calls) == evidence["source_chunks"]


def test_source_pdf_hash_mismatch_is_rejected(setup_run, monkeypatch):
    directory, runner, _judge, _generated = setup_run
    active = runner()
    record = new_record(active.instances[0])
    record["summarization"]["metadata"] = fake_summary().model_dump()
    monkeypatch.setattr(
        "pda.summary_evaluation.PdfStore",
        lambda settings: SimpleNamespace(
            get=lambda identifier: (directory / "wrong.pdf", "different-pdf"),
        ),
    )
    with pytest.raises(ValueError, match="PDF changed"):
        active._document(record)


def test_aggregate_recovers_interruption_after_last_metric_before_completion(setup_run):
    directory, runner, _judge, _generated = setup_run
    runner().run()
    path = directory / "summary_evaluation_01.json"
    record = json.loads(path.read_text())
    record["evaluation"].update(status="incomplete", evaluated_at=None, weighted_score=None)
    save_record(path, record)
    result = runner().run(mode="aggregate-only")
    assert result["statistics"]["completed"] == 1
    assert result["records"][0]["evaluation"]["evaluated_at"]


def test_failure_during_explicit_regeneration_keeps_archived_original(setup_run):
    directory, runner, _judge, _generated = setup_run
    runner().run()
    active = runner()
    active._generate = lambda *args: (_ for _ in ()).throw(RuntimeError("Mock generation failed"))
    report = active.run(force_summarize=True)
    previous = json.loads(next((directory / "history").glob("*.json")).read_text())
    assert previous["summarization"]["summary"] == fake_summary().text
    assert report["records"][0]["evaluation"]["weighted_score"] is None


def test_malformed_cached_metric_is_reported_without_console_crash(setup_run, capsys):
    directory, runner, _judge, _generated = setup_run
    runner().run()
    path = directory / "summary_evaluation_01.json"
    record = json.loads(path.read_text())
    record["evaluation"]["metrics"]["clarity"]["score"] = "malformed"
    save_record(path, record)
    report = runner().run(mode="aggregate-only")
    print_report(report)
    assert report["statistics"]["incomplete"] == 1
    assert "incomplete" in capsys.readouterr().out


def test_exception_messages_redact_the_configured_key(monkeypatch):
    from pda.summary_eval_data import error

    monkeypatch.setenv("GOOGLE_API_KEY", "fake-secret-for-software-test")
    result = error("judge", RuntimeError("Rejected fake-secret-for-software-test"))
    assert result["message"] == "Rejected [REDACTED]"


def test_all_judge_metrics_failing_preserves_the_generated_summary(setup_run):
    _directory, runner, judge, generated = setup_run
    judge.failures.update(WEIGHTS)
    result = runner().run()
    assert result["statistics"]["failed"] == 1
    assert result["records"][0]["summarization"]["summary"] == fake_summary().text
    assert len(generated) == 1
    judge.failures.clear()
    runner().run(mode="evaluation-only")
    assert len(generated) == 1 and len(judge.calls) == 10


def test_summarization_only_generates_missing_summaries_with_null_scores(setup_run, instances):
    directory, runner, judge, generated = setup_run
    active = runner(instances[:3])

    def forbidden():
        pytest.fail("Summarization-only must not initialize the judge")

    active.judge_factory = forbidden
    report = active.run(mode="summarization-only")
    assert len(generated) == 3 and not judge.calls
    for record in report["records"]:
        assert record["summarization"]["status"] == "completed"
        assert record["evaluation"]["judge_model"] is None
        assert record["evaluation"]["weighted_score"] is None
        assert all(m["score"] is None for m in record["evaluation"]["metrics"].values())
    stored = json.loads((directory / "summary_evaluation_03.json").read_text())
    assert stored["summarization"]["summary"] == fake_summary().text


def test_summarization_only_reuses_summaries_and_preserves_existing_scores(setup_run, instances):
    directory, runner, judge, generated = setup_run
    runner().run()
    previous = json.loads((directory / "summary_evaluation_01.json").read_text())
    runner(instances[:2]).run(mode="summarization-only")
    runner(instances[:2]).run(mode="summarization-only")
    current = json.loads((directory / "summary_evaluation_01.json").read_text())
    assert len(generated) == 2 and len(judge.calls) == 5
    assert current["summarization"] == previous["summarization"]
    assert current["evaluation"] == previous["evaluation"]


def test_summarization_only_force_regeneration_invalidates_scores_without_judging(setup_run):
    _directory, runner, judge, generated = setup_run
    runner().run()
    report = runner().run(mode="summarization-only", force_summarize=True)
    assert len(generated) == 2 and len(judge.calls) == 5
    assert generated[-1][1]["regenerate"] is True
    assert report["records"][0]["evaluation"]["weighted_score"] is None


def test_summarization_only_rejects_force_evaluate(setup_run):
    _directory, runner, _judge, _generated = setup_run
    with pytest.raises(ValueError, match="cannot be used in summarization-only"):
        runner().run(mode="summarization-only", force_evaluate=True)


def test_summarization_only_cli_needs_no_key_or_judge_settings(setup_run, monkeypatch):
    directory, runner, judge, generated = setup_run
    namespace = runpy.run_path(str(PROJECT_ROOT / "scripts" / "evaluate_summaries.py"))
    main = namespace["main"]
    active = runner()

    def forbidden(*args, **kwargs):
        pytest.fail("Judge configuration/credentials must not be used in summarization-only")

    for name in ("load_credentials", "DeepEvalJudge"):
        monkeypatch.setitem(main.__globals__, name, forbidden)
    monkeypatch.setitem(main.__globals__, "JudgeSettings", SimpleNamespace(from_env=forbidden))
    monkeypatch.setitem(main.__globals__, "load_instances", lambda folder: active.instances)
    monkeypatch.setitem(main.__globals__, "SummaryEvaluation", lambda *args, **kwargs: active)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    assert main(["--summarization-only", "--output-dir", str(directory)]) == 0
    assert len(generated) == 1 and not judge.calls
