"""Exercise on-demand summary UI states without networking or a language model."""

from dataclasses import replace
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest
from test_pipeline import paper
from test_summarization import result_for

from pda.retrieval import ArxivClient
from pda.summary_config import SummaryError, SummarySettings, digest, normalized_query
from pda.summary_jobs import SummaryJob

APP = Path(__file__).resolve().parents[1] / "app.py"


class FakeJobs:
    def __init__(self):
        self.jobs = {}
        self.calls = []

    def key(self, source, query):
        return digest([source.arxiv_id, normalized_query(query)])

    def snapshot(self, key):
        return replace(self.jobs[key]) if key in self.jobs else None

    def submit(self, source, query, *, regenerate=False):
        self.calls.append((source.arxiv_id, query, regenerate))
        key = self.key(source, query)
        self.jobs[key] = SummaryJob(status="running", progress="Reading full-paper chunk 1 of 3")
        return key


@pytest.fixture
def ui(monkeypatch):
    from pda import summary_jobs

    manager = FakeJobs()
    st.cache_resource.clear()
    monkeypatch.setattr(summary_jobs, "SummaryJobs", lambda settings: manager)
    monkeypatch.setattr(ArxivClient, "search", lambda *args, **kwargs: ([paper()], {}))
    app = AppTest.from_file(APP).run(timeout=30)
    app.selectbox[0].select("TF-IDF baseline")
    app.text_area[0].input("graph learning")
    app.button[0].click().run(timeout=30)
    assert not app.exception
    yield app, manager
    st.cache_resource.clear()


def button(app, label):
    return next(b for b in app.button if b.label == label)


def test_search_does_not_generate_and_removes_selected_evidence(ui):
    app, jobs = ui
    assert not jobs.calls
    assert button(app, "Summarize").disabled is False
    rendered = " ".join(item.value for item in app.markdown)
    assert "Selected abstract evidence:" not in rendered
    assert "Abstract:" in rendered and "Matching terms:" in rendered
    assert "Test Author" in rendered


def test_loading_success_regenerate_and_original_query(ui):
    app, jobs = ui
    # Editing the form without submitting must not change the displayed results' query.
    app.text_area[0].input("a different unsubmitted question")
    button(app, "Summarize").click().run(timeout=30)
    assert not app.exception
    assert jobs.calls == [(paper().arxiv_id, "graph learning", False)]
    assert button(app, "Summarizing…").disabled
    assert app.status[0].label == "Reading full-paper chunk 1 of 3"
    key = jobs.key(paper(), "graph learning")
    jobs.jobs[key] = SummaryJob(status="succeeded", result=result_for("graph learning"))
    app.run(timeout=30)
    assert not app.exception
    assert any(result_for("graph learning").text == item.value for item in app.markdown)
    button(app, "Regenerate summary").click().run(timeout=30)
    assert jobs.calls[-1] == (paper().arxiv_id, "graph learning", True)
    assert button(app, "Summarizing…").disabled


def test_failure_retry_and_query_change(ui):
    app, jobs = ui
    button(app, "Summarize").click().run(timeout=30)
    key = jobs.key(paper(), "graph learning")
    jobs.jobs[key] = SummaryJob(
        status="failed", error="Cannot connect to Ollama. Start Ollama locally."
    )
    app.run(timeout=30)
    assert "Cannot connect to Ollama" in app.error[0].value
    button(app, "Retry summary").click().run(timeout=30)
    assert len(jobs.calls) == 2
    assert button(app, "Summarizing…").disabled
    jobs.jobs[key] = SummaryJob(status="succeeded", result=result_for("graph learning"))
    app.text_area[0].input("neural networks")
    button(app, "Find papers").click().run(timeout=30)
    assert not app.exception
    assert button(app, "Summarize").disabled is False
    assert not any(item.value == result_for("graph learning").text for item in app.markdown)
    assert len(jobs.calls) == 2


def test_summary_actions_do_not_request_app_reruns_that_recreate_paper_expanders(ui, monkeypatch):
    app, jobs = ui
    reruns = []
    monkeypatch.setattr(st, "rerun", lambda **kwargs: reruns.append(kwargs))
    key = jobs.key(paper(), "graph learning")
    button(app, "Summarize").click().run(timeout=30)
    assert not app.exception and button(app, "Summarizing…").disabled
    jobs.jobs[key] = SummaryJob(status="succeeded", result=result_for("graph learning"))
    app.run(timeout=30)
    button(app, "Regenerate summary").click().run(timeout=30)
    assert not app.exception and button(app, "Summarizing…").disabled
    jobs.jobs[key] = SummaryJob(status="failed", error="Ollama unavailable")
    app.run(timeout=30)
    button(app, "Retry summary").click().run(timeout=30)
    assert not app.exception and button(app, "Summarizing…").disabled
    assert len(jobs.calls) == 3 and reruns == []


def test_submission_error_is_displayed_and_can_be_retried(ui, monkeypatch):
    app, jobs = ui
    submit = jobs.submit

    def full_queue(*args, **kwargs):
        raise SummaryError("The summary queue is full. Wait and retry.")

    monkeypatch.setattr(jobs, "submit", full_queue)
    button(app, "Summarize").click().run(timeout=30)
    assert not app.exception
    assert "queue is full" in app.error[0].value
    assert not button(app, "Retry summary").disabled
    monkeypatch.setattr(jobs, "submit", submit)
    button(app, "Retry summary").click().run(timeout=30)
    assert not app.exception and not app.error
    assert button(app, "Summarizing…").disabled


def test_configuration_error_is_readable(monkeypatch):
    st.cache_resource.clear()
    monkeypatch.setenv("PDA_SUMMARY_ENDPOINT", "https://remote.example")
    monkeypatch.setattr(ArxivClient, "search", lambda *args, **kwargs: ([paper()], {}))
    app = AppTest.from_file(APP).run(timeout=30)
    app.selectbox[0].select("TF-IDF baseline")
    app.text_area[0].input("graph learning")
    button(app, "Find papers").click().run(timeout=30)
    assert not app.exception
    assert "configuration is invalid" in app.error[0].value
    st.cache_resource.clear()


def test_environment_settings(monkeypatch):
    monkeypatch.setenv("PDA_SUMMARY_TEMPERATURE", "0.15")
    monkeypatch.setenv("PDA_SUMMARY_THINK", "false")
    monkeypatch.setenv("PDA_SUMMARY_NUM_CTX", "16384")
    settings = SummarySettings.from_env()
    assert settings.temperature == 0.15 and settings.think is False
    assert settings.num_ctx == 16384
