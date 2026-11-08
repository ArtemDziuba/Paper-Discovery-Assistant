"""Fixed evaluation inputs, metadata exclusions, model coverage, and exposure auditing."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from test_pipeline import paper

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import evaluate_annotations
import prepare_evaluation


def annotation(identifier, paper_id="1234.0001"):
    return {
        "query_id": identifier,
        "query": identifier,
        "judgments": [{"arxiv_id": paper_id, "title": "Graph learning", "relevance": 3}],
    }


@pytest.fixture
def annotation_folders(tmp_path, monkeypatch):
    folders = {name: tmp_path / name for name in ("human", "synthetic")}
    for name, folder in folders.items():
        folder.mkdir()
        (folder / "batch.json").write_text(json.dumps(annotation(name)), encoding="utf-8")
    monkeypatch.setattr(prepare_evaluation, "ANNOTATION_FOLDERS", folders)
    return folders


@pytest.mark.parametrize("change", ["edit", "add", "remove"])
def test_snapshot_rejects_changed_annotation_file_set(annotation_folders, change):
    _, hashes = prepare_evaluation.load_datasets()
    prepared = {"source_hashes": hashes}
    prepare_evaluation.validate_snapshot(prepared)
    folder = annotation_folders["human"]
    original = folder / "batch.json"
    if change == "edit":
        original.write_text(json.dumps(annotation("different")), encoding="utf-8")
    elif change == "add":
        (folder / "new.json").write_text(json.dumps(annotation("new")), encoding="utf-8")
    else:
        original.rename(folder / "renamed.json")
    with pytest.raises(ValueError, match="annotations changed"):
        prepare_evaluation.validate_snapshot(prepared)


def test_preparation_freezes_real_metadata_and_reports_title_mismatch(
    annotation_folders, tmp_path, monkeypatch
):
    synthetic = annotation("synthetic", "1234.0002")
    synthetic["judgments"].append(
        {"arxiv_id": "1234.0003", "title": "Claimed title", "relevance": 2}
    )
    (annotation_folders["synthetic"] / "batch.json").write_text(
        json.dumps(synthetic), encoding="utf-8"
    )
    requests = []

    class Client:
        def fetch_ids(self, identifiers):
            requests.extend(identifiers)
            return [paper(i) for i in identifiers], {"retrieved_at": 123, "cache_hit": False}

    monkeypatch.setattr(prepare_evaluation, "ArxivClient", Client)
    target = tmp_path / "prepared.json"
    monkeypatch.setattr(sys, "argv", ["prepare_evaluation.py", "--output", str(target)])
    prepare_evaluation.main()
    snapshot = json.loads(target.read_bytes())
    assert requests == ["1234.0001", "1234.0002", "1234.0003"]
    assert snapshot["source_judgment_counts"] == {"human": 1, "synthetic": 2}
    assert len(snapshot["datasets"]["synthetic"][0]["judgments"]) == 1
    assert snapshot["exclusions"]["synthetic"][0]["reason"] == "title_mismatch"
    assert snapshot["datasets"]["human"][0]["judgments"][0]["paper"]["abstract"]
    assert "splits" not in snapshot
    with pytest.raises(ValueError, match="already exists"):
        prepare_evaluation.main()


def test_exposure_audit_preserves_test_candidates_and_collapses_versions():
    datasets = {"synthetic": [annotation("new-topic", "1234.0001v2")]}
    original = {"train": [annotation("old-topic", "1234.0001v1")], "synthetic_dev": []}
    exposure = evaluate_annotations.training_overlap(datasets, original)
    seen = exposure["synthetic"]["new-topic"]["train"]
    assert not seen["query_seen"]
    assert seen["paper_ids_seen"] == ["1234.0001"]
    assert datasets["synthetic"][0]["judgments"][0]["arxiv_id"] == "1234.0001v2"


@pytest.mark.parametrize("mode", ["truncated", "full"])
@pytest.mark.parametrize("include_v2", [False, True])
def test_runner_evaluates_every_model_on_both_sources(
    annotation_folders, tmp_path, monkeypatch, mode, include_v2
):
    datasets, hashes = prepare_evaluation.load_datasets()
    for rows in datasets.values():
        rows[0]["judgments"][0]["paper"] = paper().model_dump(mode="json")
    prepared = {
        "source_hashes": hashes,
        "source_judgment_counts": {"human": 1, "synthetic": 1},
        "datasets": datasets,
        "exclusions": {"human": [], "synthetic": []},
    }
    prepared_path = tmp_path / "prepared.json"
    prepared_path.write_text(json.dumps(prepared), encoding="utf-8")
    provenance = tmp_path / "training.json"
    provenance.write_text(json.dumps({"splits": {"train": [], "synthetic_dev": []}}))
    model = tmp_path / "model"
    model.mkdir()
    (model / "training_manifest.json").write_text(
        json.dumps(
            {
                "run_id": "original-run",
                "dataset_sha256": hashlib.sha256(provenance.read_bytes()).hexdigest(),
            }
        )
    )
    (model / "model.safetensors").write_bytes(b"explicit synthetic software fixture")
    v2_model = tmp_path / "v2"
    if include_v2:
        v2_model.mkdir()
        v2_prepared_path = tmp_path / "v2-prepared.json"
        v2_prepared_path.write_text(
            json.dumps(
                {
                    "training": datasets["human"],
                    "combined_development": [q for rows in datasets.values() for q in rows],
                }
            ),
            encoding="utf-8",
        )
        (v2_model / "training_manifest.json").write_text(
            json.dumps(
                {
                    "run_id": "v2-run",
                    "prepared_path": str(v2_prepared_path),
                    "dataset_sha256": hashlib.sha256(v2_prepared_path.read_bytes()).hexdigest(),
                    "development_snapshot_sha256": hashlib.sha256(
                        prepared_path.read_bytes()
                    ).hexdigest(),
                }
            ),
            encoding="utf-8",
        )
        (v2_model / "model.safetensors").write_bytes(b"explicit synthetic v2 software fixture")
    calls = []

    class Encoder:
        def __init__(self, *args, **kwargs):
            calls.append(kwargs)

        def score_documents(self, query, documents):
            return np.ones(len(documents))

    monkeypatch.setattr(evaluate_annotations, "FINETUNED_PATH", model)
    monkeypatch.setattr(evaluate_annotations, "FINETUNED_V2_PATH", v2_model)
    monkeypatch.setattr(evaluate_annotations, "LexicalEncoder", Encoder)
    monkeypatch.setattr(evaluate_annotations, "SemanticEncoder", Encoder)
    target = tmp_path / "results.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_annotations.py",
            "--device",
            "cpu",
            "--prepared",
            str(prepared_path),
            "--training-provenance",
            str(provenance),
            "--output",
            str(target),
            "--mode",
            mode,
        ],
    )
    evaluate_annotations.main()
    report = json.loads(target.read_bytes())
    assert len(calls) == (4 if include_v2 else 3)
    assert calls[-1] == {"model_path": v2_model if include_v2 else model}
    for methods in report["results"].values():
        expected_models = {"tfidf", "semantic", "finetuned"}
        if include_v2:
            expected_models.add("finetuned_v2")
        assert set(methods) == expected_models
        assert all(result["query_count"] == 1 for result in methods.values())
        assert all(result["aggregate"]["ndcg@10"] == 1 for result in methods.values())
        assert all(result["aggregate"]["map@10"] == 1 for result in methods.values())
        pairwise_name = "pairwise_accuracy" if mode == "truncated" else "pairwise_accuracy@50"
        assert all(result["aggregate"][pairwise_name] is None for result in methods.values())
        assert all(result["pairwise_valid_query_count"] == 0 for result in methods.values())
        assert all(result["evaluation_mode"] == mode for result in methods.values())
        assert all(
            len(result["aggregate"]) == (4 if mode == "truncated" else 12)
            for result in methods.values()
        )
    assert report["evaluation_mode"] == mode
    assert report["models"]["finetuned"]["label"] == "syntetic finetune v1"
    assert report["finetuned_artifact"]["training_run_id"] == "original-run"
    if include_v2:
        assert report["finetuned_v2_artifact"]["evaluation_used_for_checkpoint_selection"]
        assert report["v2_training_and_development_overlap"]["human"]["human"]["synthetic_dev"][
            "query_seen"
        ]
    else:
        assert report["finetuned_v2_artifact"] is None


@pytest.mark.parametrize("mode", ["truncated", "full"])
def test_judged_runner_uses_raw_scores_for_pairwise_ties(mode):
    class Encoder:
        def score_documents(self, query, documents):
            return np.array([0.5, 0.5])

    query = annotation("graph")
    query["judgments"][0].update(relevance=0, paper=paper().model_dump(mode="json"))
    query["judgments"].append(
        {"arxiv_id": "1234.0002", "relevance": 3, "paper": paper().model_dump(mode="json")}
    )
    result = evaluate_annotations.evaluate(Encoder(), [query], mode=mode)
    pairwise_name = "pairwise_accuracy" if mode == "truncated" else "pairwise_accuracy@50"
    assert result["aggregate"][pairwise_name] == 0.5
    assert result["pairwise_valid_query_count"] == 1
    assert result["aggregate"]["map@10"] == 0.5
