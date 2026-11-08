"""Occurrence preservation, combined development weighting, and the shared training loop."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import train_ranker_v2
from prepare_finetuning_v2 import combine_development, occurrence_audit
from train_ranker import evaluate, train_epoch
from train_ranker_v2 import plot_history

from pda.finetuning_data import load_annotations, make_preferences
from pda.ranking import FINETUNED_V2_MODEL, SemanticEncoder


def group(identifier, grades=(3, 0)):
    return {
        "query_id": identifier,
        "query": identifier,
        "judgments": [
            {
                "arxiv_id": f"1234.000{i + 1}",
                "title": f"Title {i}",
                "relevance": grade,
                "paper": {"title": f"Title {i}", "abstract": "Synthetic software fixture."},
            }
            for i, grade in enumerate(grades)
        ],
    }


def test_training_preserves_repeated_rows_and_query_dependent_paper_grades(tmp_path):
    first, second = group("first"), group("second", (0, 3))
    (tmp_path / "batch1.json").write_text(json.dumps([first, first]), encoding="utf-8")
    (tmp_path / "batch2.json").write_text(json.dumps(second), encoding="utf-8")
    rows, hashes = load_annotations(tmp_path, "training", deduplicate=False)
    assert len(rows) == 3 and len(hashes) == 2
    assert sum(len(q["judgments"]) for q in rows) == 6
    assert [q["judgments"][0]["relevance"] for q in rows] == [3, 3, 0]
    samples = make_preferences(rows, 42, 64)
    assert len(samples) == 3
    assert "Title 0" in samples[0][1] and "Title 1" in samples[-1][1]
    assert occurrence_audit(rows)["papers_with_different_grades"] == 2


def test_combined_development_uses_one_query_macro_mean_without_merging_candidates():
    human = group("shared")
    human["query"] = "human"
    combined = combine_development(
        {"human": [human], "synthetic": [group("shared"), group("other")]}
    )
    assert len({q["query_id"] for q in combined}) == 3
    assert all(len(q["judgments"]) == 2 for q in combined)

    class Encoder:
        def score_documents(self, query, documents):
            return np.array([1.0, 0.0] if query == "human" else [0.0, 1.0])

    report = evaluate(Encoder(), combined, mode="full")
    assert report["query_count"] == 3
    query_values = [row["metrics"]["ndcg@10"] for row in report["per_query"].values()]
    assert report["aggregate"]["ndcg@10"] == pytest.approx(sum(query_values) / 3)
    assert report["aggregate"]["ndcg@10"] != pytest.approx((query_values[0] + query_values[1]) / 2)


def test_shared_epoch_updates_weights_and_scheduler_and_returns_observed_loss():
    class TinyEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = torch.nn.Linear(2, 2, bias=False)
            self.projection.weight.data.copy_(torch.eye(2))

        def preprocess(self, texts):
            values = {"query": [1.0, 0.0], "better": [0.0, 1.0], "worse": [1.0, 0.0]}
            return {"inputs": torch.tensor([values[t] for t in texts])}

        def forward(self, features):
            return {"sentence_embedding": self.projection(features["inputs"])}

    model = TinyEncoder()
    before = model.projection.weight.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    samples = [("query", "better", "worse", 0.3)] * 9
    losses = train_epoch(
        model, samples, optimizer, scheduler, {"seed": 42, "batch_size": 8, "device": "cpu"}, 1
    )
    assert len(losses) == 2 and all(np.isfinite(loss) and loss > 0 for loss in losses)
    assert not torch.equal(before, model.projection.weight.detach())
    assert scheduler.last_epoch == 2


def test_epoch_plots_are_exported_for_loss_and_all_four_development_cutoffs(tmp_path):
    history = [
        {
            "epoch": epoch,
            "training_loss": 0.3 / epoch,
            **{f"development_ndcg@{k}": 0.5 + 0.01 * epoch for k in (5, 10, 20, 50)},
        }
        for epoch in (1, 2, 3)
    ]
    baseline = {"aggregate": {f"ndcg@{k}": 0.5 for k in (5, 10, 20, 50)}}
    plot_history(history, baseline, 2, tmp_path)
    assert len(list(tmp_path.glob("*.png"))) == 5
    assert len(list(tmp_path.glob("*.svg"))) == 5
    assert all(path.stat().st_size > 1000 for path in tmp_path.iterdir())


def test_v2_local_model_exports_its_own_identity(tmp_path, monkeypatch):
    (tmp_path / "training_manifest.json").write_text(
        json.dumps({"run_id": "v2-test", "model_name": FINETUNED_V2_MODEL}), encoding="utf-8"
    )
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=lambda *args, **kwargs: object()),
    )
    encoder = SemanticEncoder("cpu", model_path=tmp_path)
    assert encoder.model_name == FINETUNED_V2_MODEL
    assert encoder.model_revision == "v2-test"


def test_retraining_refuses_to_overwrite_existing_epoch_results(tmp_path, monkeypatch):
    (tmp_path / "training-results.json").write_text("existing experiment", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_ranker_v2.py",
            "--data",
            str(tmp_path / "prepared.json"),
            "--output",
            str(tmp_path / "new-model"),
        ],
    )
    with pytest.raises(ValueError, match="new --results folder"):
        train_ranker_v2.main()
    assert (tmp_path / "training-results.json").read_text(encoding="utf-8") == "existing experiment"
