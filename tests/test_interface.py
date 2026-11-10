"""Exercise the actual Streamlit form without external requests."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app.py"


def test_initial_page_and_empty_input():
    app = AppTest.from_file(APP).run(timeout=30)
    assert not app.exception
    assert app.title[0].value == "Paper Discovery Assistant"
    assert "syntetic finetune v1" in app.selectbox[0].options
    app.selectbox[0].select("TF-IDF baseline")
    app.button[0].click().run(timeout=30)
    assert not app.exception
    assert "research topic" in app.error[0].value


def test_ranked_results_without_overview(monkeypatch):
    from test_pipeline import paper

    from pda.retrieval import ArxivClient

    monkeypatch.setattr(
        ArxivClient, "search", lambda *args, **kwargs: ([paper()], {"cache_hit": False})
    )
    app = AppTest.from_file(APP).run(timeout=30)
    app.selectbox[0].select("TF-IDF baseline")
    app.text_area[0].input("graph learning")
    app.button[0].click().run(timeout=30)
    assert not app.exception
    assert not app.error
    assert any(s.value == "Ranked papers" for s in app.subheader)
    assert all(s.value != "Abstract-based overview" for s in app.subheader)


@pytest.mark.parametrize("displayed", [5, 50])
def test_displayed_paper_limits(monkeypatch, displayed):
    from test_pipeline import paper

    from pda.retrieval import ArxivClient

    papers = [paper(f"1234.{index:04}v1", f"Graph paper {index}") for index in range(80)]
    monkeypatch.setattr(ArxivClient, "search", lambda *args, **kwargs: (papers, {}))
    app = AppTest.from_file(APP).run(timeout=30)
    app.selectbox[0].select("TF-IDF baseline")
    slider = app.slider[0]
    assert (slider.min, slider.max, slider.step) == (5, 50, 5)
    slider.set_value(displayed)
    app.text_area[0].input("graph learning")
    app.button[0].click().run(timeout=30)
    assert not app.exception and not app.error
    assert len([e for e in app.expander if "/100" in e.label]) == displayed


def test_10000_limit_and_partial_warning(monkeypatch):
    from test_pipeline import paper

    from pda.retrieval import ArxivClient

    limits = []

    def partial(client, query, limit, **kwargs):
        limits.append(limit)
        return [paper()], {"cache_hit": False, "partial_failure": "arXiv returned HTTP 503."}

    monkeypatch.setattr(ArxivClient, "search", partial)
    app = AppTest.from_file(APP).run(timeout=30)
    app.selectbox[0].select("TF-IDF baseline")
    app.number_input[0].set_value(10000)
    app.text_area[0].input("graph learning")
    app.button[0].click().run(timeout=30)
    assert not app.exception
    assert limits == [50, 10000]  # Auxiliary title lookup, then requested topic retrieval.
    assert any("Only part" in warning.value for warning in app.warning)


@pytest.mark.parametrize("label", ["syntetic finetune v1", "syntetic finetune v2"])
def test_finetuned_selection_loads_local_artifact(monkeypatch, label):
    import numpy as np
    from test_pipeline import paper

    from pda import ranking
    from pda.retrieval import ArxivClient

    calls = []

    class Encoder:
        model = object()
        model_name = (
            ranking.FINETUNED_V2_MODEL
            if label == ranking.FINETUNED_V2_LABEL
            else ranking.FINETUNED_MODEL
        )
        model_revision = "test-run"

        def score_documents(self, query, documents, progress=None):
            return np.ones(len(documents))

    def factory(device, *, model_path=None):
        calls.append((device, model_path))
        return Encoder()

    monkeypatch.setattr(ranking, "SemanticEncoder", factory)
    monkeypatch.setattr(ArxivClient, "search", lambda *args, **kwargs: ([paper()], {}))
    app = AppTest.from_file(APP).run(timeout=30)
    app.selectbox[0].select(label)
    app.text_area[0].input("graph learning")
    app.button[0].click().run(timeout=30)
    assert not app.exception and not app.error
    assert calls == [("cpu", ranking.FINETUNED_OPTIONS[label])]
    assert app.session_state.result.metadata["encoder"] == Encoder.model_name
