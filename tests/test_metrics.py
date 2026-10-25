"""Hand-calculated metric expectations; not an NLP quality experiment."""

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("evaluate", Path("scripts/evaluate.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_perfect_and_missed_ranking():
    assert module.metrics(["a", "b"], {"a": 1}) == {
        "ndcg@10": 1,
        "precision@5": 0.2,
        "mrr": 1,
    }
    assert module.metrics(["b", "a"], {"a": 1})["mrr"] == 0.5
    assert module.metrics(["b", "a"], {"a": 1})["ndcg@10"] == pytest.approx(0.6309297536)
    assert module.metrics(["b"], {"a": 1})["ndcg@10"] == 0
