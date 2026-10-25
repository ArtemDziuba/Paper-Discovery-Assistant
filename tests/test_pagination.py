"""Synthetic paginated feeds validate software behavior, not retrieval quality."""

import httpx
import pytest
from test_pipeline import feed

from pda.retrieval import ArxivClient


def page(identifiers, total, invalid=False):
    entries = [feed(i).split("<entry>")[1].split("</entry>")[0] for i in identifiers]
    if invalid:
        entries[-1] = entries[-1].replace("Graph models learn links.", "")
    return (
        '<feed xmlns="http://www.w3.org/2005/Atom" '
        'xmlns:o="http://a9.com/-/spec/opensearch/1.1/">'
        f"<o:totalResults>{total}</o:totalResults>"
        + "".join(f"<entry>{entry}</entry>" for entry in entries)
        + "</feed>"
    )


def test_page_offsets_follow_raw_entries_and_cache(tmp_path, monkeypatch):
    monkeypatch.setattr("pda.retrieval.PAGE_SIZE", 2)
    monkeypatch.setattr("pda.retrieval.time.sleep", lambda _: None)
    calls = []
    responses = {
        0: page(["1234.0001v1", "1234.0002v1"], 5),
        2: page(["1234.0001v2", "1234.0004v1"], 5, invalid=True),
        4: page(["1234.0003v1"], 5),
    }

    def get(url, **kwargs):
        params = kwargs["params"]
        calls.append((params["start"], params["max_results"]))
        return httpx.Response(
            200, text=responses[params["start"]], request=httpx.Request("GET", url)
        )

    monkeypatch.setattr(httpx, "get", get)
    client = ArxivClient(tmp_path)
    progress = []
    papers, metadata = client.search(
        "graphs", 5, progress=lambda fraction, _: progress.append(fraction)
    )
    assert calls == [(0, 2), (2, 2), (4, 1)]
    assert len(papers) == 3
    assert papers[0].arxiv_id == "1234.0001v1"
    assert metadata["skipped_records"] == 1
    assert metadata["scanned_entries"] == 5
    assert progress == sorted(progress) and progress[-1] == 1
    assert client.search("graphs", 5)[1]["cache_hit"]
    assert len(calls) == 3


def test_10000_candidates_and_exhaustion(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr("pda.retrieval.time.sleep", lambda _: None)

    def get(url, **kwargs):
        params = kwargs["params"]
        calls.append(params)
        start = params["start"]
        identifiers = [f"1234.{i:05}v1" for i in range(start, start + params["max_results"])]
        return httpx.Response(200, text=page(identifiers, 12000), request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", get)
    papers, metadata = ArxivClient(tmp_path).search("graphs", 10000)
    assert len(papers) == 10000
    assert len(calls) == 10
    assert [p["start"] for p in calls] == list(range(0, 10000, 1000))
    assert all(p["max_results"] <= 1000 for p in calls)
    assert metadata["requested_candidates"] == 10000
    assert metadata["partial_failure"] is None


def test_stop_at_total_results(tmp_path, monkeypatch):
    monkeypatch.setattr("pda.retrieval.time.sleep", lambda _: None)
    calls = []

    def get(url, **kwargs):
        calls.append(kwargs)
        return httpx.Response(200, text=page(["1234.0001v1"], 1), request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", get)
    papers, metadata = ArxivClient(tmp_path).search("graphs", 10000)
    assert len(papers) == 1 and len(calls) == 1
    assert metadata["total_results"] == 1


def test_partial_failure_preserves_pages_for_retry(tmp_path, monkeypatch):
    monkeypatch.setattr("pda.retrieval.PAGE_SIZE", 1)
    monkeypatch.setattr("pda.retrieval.time.sleep", lambda _: None)
    failed = True
    starts = []

    def get(url, **kwargs):
        start = kwargs["params"]["start"]
        starts.append(start)
        return httpx.Response(
            503 if start == 1 and failed else 200,
            text=page([f"1234.{start:04}v1"], 2),
            request=httpx.Request("GET", url),
        )

    monkeypatch.setattr(httpx, "get", get)
    client = ArxivClient(tmp_path)
    papers, metadata = client.search("graphs", 2)
    assert len(papers) == 1 and "HTTP 503" in metadata["partial_failure"]
    failed = False
    papers, metadata = client.search("graphs", 2)
    assert len(papers) == 2 and metadata["partial_failure"] is None
    assert starts.count(0) == 1


@pytest.mark.parametrize("limit", [0, 10001])
def test_candidate_limit_validation(tmp_path, limit):
    with pytest.raises(ValueError, match="10,000"):
        ArxivClient(tmp_path).search("graphs", limit)
