"""Synthetic software fixtures, not evidence of NLP effectiveness."""

from datetime import UTC, datetime

import httpx
import numpy as np
import pytest

from pda.models import Paper
from pda.pipeline import evidence, search
from pda.query import interpret
from pda.ranking import LexicalEncoder, cosine_scores
from pda.retrieval import ArxivClient, RetrievalError, parse_feed


def paper(identifier="1234.5678v1", title="Graph learning", abstract="Graph models learn links."):
    return Paper(
        arxiv_id=identifier,
        title=title,
        abstract=abstract,
        authors=["Test Author"],
        published=datetime(2024, 1, 1, tzinfo=UTC),
        categories=["cs.LG"],
        url=f"https://arxiv.org/abs/{identifier}",
    )


def feed(identifier="1234.5678v1", title="Graph learning"):
    return f"""<feed xmlns="http://www.w3.org/2005/Atom"><entry>
    <id>http://arxiv.org/abs/{identifier}</id><title>{title}</title>
    <summary>Graph models learn links.</summary><published>2024-01-01T00:00:00Z</published>
    <author><name>Test Author</name></author><category term="cs.LG"/>
    </entry></feed>"""


@pytest.mark.parametrize("query", ["", "  ", "a", "the and for", "x" * 1001])
def test_invalid_queries(query):
    with pytest.raises(ValueError):
        interpret(query)


def test_interpret_preserves_query_and_blocks_search_syntax():
    result = interpret('graph AND cat:physics "networks"')
    assert result.original == 'graph AND cat:physics "networks"'
    assert "cat:physics" not in result.search_query
    assert result.concepts == ["graph", "cat", "physics", "networks"]
    assert interpret("graphs without supervision").warnings


def test_conversational_query_focus_and_exact_title():
    result = interpret("i am looking for information about transformer architecture")
    assert result.concepts == ["transformer", "architecture"]
    assert result.search_query == 'all:"transformer" AND all:"architecture"'
    assert result.original.startswith("i am looking")
    assert (
        interpret('"Attention Is All You Need"').search_query == 'all:"Attention Is All You Need"'
    )
    assert "information" in interpret("information retrieval methods").concepts


def test_unavailable_cuda_gives_actionable_error(monkeypatch):
    import torch

    from pda.ranking import SemanticEncoder

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="uv sync --extra cuda"):
        SemanticEncoder("cuda")


def test_quoted_exact_title_precedes_semantic_paraphrases():
    class Client:
        def search(self, query, limit):
            return [paper(title="Graph Learning"), paper("9999.9999v1", "Other graph paper")], {}

    class Encoder:
        def encode_pair(self, query, documents):
            return np.array([1, 0]), np.array([[0, 1], [1, 0]])

    result = search('"Graph Learning"', Client(), Encoder())
    assert result.matches[0].paper.title == "Graph Learning"
    assert result.matches[0].raw_similarity == 0
    assert result.matches[1].raw_similarity == 1
    assert result.metadata["exact_title_priority"]


def test_parse_metadata_and_missing_abstract():
    papers, skipped = parse_feed(feed())
    assert skipped == 0
    assert papers[0].arxiv_id == "1234.5678v1"
    assert papers[0].authors == ["Test Author"]
    assert parse_feed(feed().replace("Graph models learn links.", "")) == ([], 1)


def test_deduplicate_versions():
    # Build two entries explicitly to preserve valid Atom structure.
    xml = feed().replace("</feed>", "<entry>" + feed("1234.5678v2").split("<entry>")[1])
    assert len(parse_feed(xml)[0]) == 1


@pytest.mark.parametrize(
    "xml",
    [
        "broken",
        "<html/>",
        feed().replace("http://arxiv.org/abs/1234.5678v1", "http://arxiv.org/api/errors#bad"),
    ],
)
def test_bad_feed(xml):
    with pytest.raises(RetrievalError):
        parse_feed(xml)


def test_cosine_edge_cases():
    np.testing.assert_allclose(
        cosine_scores(np.array([1, 0]), np.array([[1, 0], [0, 1], [-1, 0], [0, 0]])), [1, 0, -1, 0]
    )
    with pytest.raises(ValueError):
        cosine_scores(np.array([np.nan]), np.array([[1]]))
    with pytest.raises(ValueError):
        cosine_scores(np.array([1, 2]), np.array([[1]]))


def test_lexical_and_grounded_pipeline():
    class Client:
        def search(self, query, limit):
            return [paper(), paper("9999.9999v1", "Stellar physics", "Stars emit light.")], {}

    result = search("graph learning", Client(), LexicalEncoder())
    assert result.matches[0].paper.title == "Graph learning"
    assert result.matches[0].display_score > result.matches[1].display_score
    assert all(m.evidence in m.paper.abstract for m in result.matches)
    assert {o["arxiv_id"] for o in result.overview} <= {m.paper.arxiv_id for m in result.matches}
    assert "not a probability" in result.metadata["score_transform"]


def test_empty_results_do_not_encode():
    class Client:
        def search(self, query, limit):
            return [], {}

    assert not search("graph learning", Client(), object()).matches


def test_evidence_selects_matching_source_sentence():
    abstract = "Stars emit light. Graph models learn links."
    assert evidence(abstract, ["graph"]) == "Graph models learn links."


def test_batched_lexical_scores_match_pairwise():
    encoder = LexicalEncoder()
    documents = ["graph graph networks", "stellar physics", "networks in biology"]
    queries = ["graph networks", "stellar", "unseenword"]
    batch = encoder.score_queries(queries, documents)
    for index, query in enumerate(queries):
        q, docs = encoder.encode_pair(query, documents)
        np.testing.assert_allclose(batch[index], cosine_scores(q, docs), atol=1e-12)


def test_lexical_empty_vocabulary():
    q, docs = LexicalEncoder().encode_pair("graphs", ["the and", "a for"])
    np.testing.assert_array_equal(cosine_scores(q, docs), [0, 0])


def test_retrieval_cache_and_retry(tmp_path, monkeypatch):
    calls = []

    def get(url, **kwargs):
        calls.append(kwargs)
        return httpx.Response(
            503 if len(calls) == 1 else 200, text=feed(), request=httpx.Request("GET", url)
        )

    monkeypatch.setattr(httpx, "get", get)
    monkeypatch.setattr("pda.retrieval.time.sleep", lambda _: None)
    client = ArxivClient(tmp_path)
    papers, metadata = client.search("all:graph")
    assert len(calls) == 2 and papers
    assert not metadata["cache_hit"]
    assert client.search("all:graph")[1]["cache_hit"]
    assert len(calls) == 2


def test_retrieval_failure(tmp_path, monkeypatch):
    def get(url, **kwargs):
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(httpx, "get", get)
    monkeypatch.setattr("pda.retrieval.time.sleep", lambda _: None)
    with pytest.raises(RetrievalError, match="Could not connect"):
        ArxivClient(tmp_path).search("all:graph")


@pytest.mark.parametrize(
    ("failure", "expected"),
    [("timeout", "timed out"), (403, "HTTP 403"), (503, "HTTP 503")],
)
def test_retrieval_error_diagnostics(tmp_path, monkeypatch, failure, expected):
    def get(url, **kwargs):
        if failure == "timeout":
            raise httpx.ReadTimeout("timeout")
        return httpx.Response(failure, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", get)
    monkeypatch.setattr("pda.retrieval.time.sleep", lambda _: None)
    with pytest.raises(RetrievalError, match=expected):
        ArxivClient(tmp_path).search("all:graph")
