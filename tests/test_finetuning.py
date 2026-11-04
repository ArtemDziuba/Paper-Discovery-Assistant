"""Training-data leakage, judgment validation, and local model selection contracts."""

import json

import pytest

from pda.evaluation import metrics
from pda.finetuning_data import hydrate, load_annotations, make_preferences, split_queries
from pda.ranking import SemanticEncoder


def query(identifier, papers):
    return {
        "query_id": identifier,
        "query": identifier,
        "judgments": [
            {
                "arxiv_id": paper,
                "title": f"Title {paper}",
                "relevance": grade,
                "paper": {"title": f"Title {paper}", "abstract": "Evidence."},
            }
            for paper, grade in papers
        ],
    }


def test_split_is_deterministic_and_paper_disjoint():
    synthetic = [query(f"q{i}", [("1234.0001", 3), (f"1234.{i + 10:04}", 0)]) for i in range(10)]
    human = [query("human", [("1234.0001v2", 3)])]
    split = split_queries(synthetic, human)
    assert split == split_queries(synthetic, human)
    training_ids = {j["arxiv_id"] for q in split["train"] for j in q["judgments"]}
    blocked_ids = {j["arxiv_id"] for q in split["synthetic_dev"] for j in q["judgments"]}
    assert "1234.0001" not in training_ids
    assert not training_ids & blocked_ids
    assert not {q["query_id"] for q in split["train"]} & {
        q["query_id"] for q in split["synthetic_dev"]
    }


def test_metadata_mismatch_and_missing_are_reported_not_replaced():
    annotations = [query("q", [("1234.0001", 3), ("1234.0002", 0)])]
    records = {
        "1234.0001": {"title": "Entirely unrelated astronomy investigation", "abstract": "x"}
    }
    prepared, excluded = hydrate(annotations, records)
    assert not prepared[0]["judgments"]
    assert {e["reason"] for e in excluded} == {"title_mismatch", "metadata_not_returned"}
    assert len(annotations[0]["judgments"]) == 2


def test_preference_sampling_uses_only_ordered_human_or_synthetic_grades():
    samples = make_preferences([query("q", [("1234.0001", 3), ("1234.0002", 0)])], 42)
    assert len(samples) == 1
    assert "0001" in samples[0][1] and "0002" in samples[0][2]
    assert samples[0][3] == pytest.approx(0.3)


def test_annotation_shape_and_invalid_grade(tmp_path):
    record = query("q", [("1234.0001", 3)])
    (tmp_path / "batch.json").write_text(json.dumps([record]), encoding="utf-8")
    assert len(load_annotations(tmp_path, "synthetic")[0]) == 1
    record["judgments"][0]["relevance"] = True
    (tmp_path / "batch.json").write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="integer relevance"):
        load_annotations(tmp_path, "human")


def test_binary_threshold_does_not_change_graded_ndcg():
    qrels = {"a": 1, "b": 3}
    permissive = metrics(["a", "b"], qrels)
    strict = metrics(["a", "b"], qrels, relevant_min=2)
    assert permissive["ndcg@10"] == strict["ndcg@10"]
    assert permissive["mrr"] == 1 and strict["mrr"] == 0.5
    assert permissive["precision@5"] == 0.4 and strict["precision@5"] == 0.2
    assert metrics(["a"], {"a": 0.5})["mrr"] == 1


def test_missing_finetuned_model_is_not_silently_base_model(tmp_path):
    with pytest.raises(RuntimeError, match="fine-tuned model is not available"):
        SemanticEncoder("cpu", model_path=tmp_path)
