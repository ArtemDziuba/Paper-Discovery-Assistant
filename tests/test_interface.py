"""Exercise the actual Streamlit form without external requests."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app.py"


def test_initial_page_and_empty_input():
    app = AppTest.from_file(APP).run(timeout=30)
    assert not app.exception
    assert app.title[0].value == "Paper Discovery Assistant"
    app.selectbox[0].select("TF-IDF baseline")
    app.button[0].click().run(timeout=30)
    assert not app.exception
    assert "research topic" in app.error[0].value


def test_results_and_overview(monkeypatch):
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
    assert any(s.value == "Abstract-based overview" for s in app.subheader)


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
    assert limits == [10000]
    assert any("Only part" in warning.value for warning in app.warning)
