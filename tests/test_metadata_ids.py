"""Metadata hydration uses explicit IDs and preserves returned versions."""

import httpx
import pytest
from test_pipeline import feed

from pda.retrieval import ArxivClient


def test_fetch_ids_uses_id_list_and_cache(tmp_path, monkeypatch):
    calls = []

    def get(url, **kwargs):
        calls.append(kwargs["params"])
        return httpx.Response(200, text=feed(), request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", get)
    client = ArxivClient(tmp_path)
    papers, _ = client.fetch_ids(["1234.5678"])
    assert papers[0].arxiv_id == "1234.5678v1"
    assert calls[0]["id_list"] == "1234.5678" and "search_query" not in calls[0]
    assert client.fetch_ids(["1234.5678"])[1]["cache_hit"]
    assert len(calls) == 1


def test_fetch_ids_rejects_query_injection(tmp_path):
    with pytest.raises(ValueError, match="Invalid arXiv"):
        ArxivClient(tmp_path).fetch_ids(['1234.5678 OR all:"graphs"'])
