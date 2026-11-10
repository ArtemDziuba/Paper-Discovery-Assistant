"""Dataset relocation, metadata fetching and historical workflow guards use mocks only."""

import hashlib
import json
import sys
from pathlib import Path

import httpx
import numpy as np
import pytest
from test_finetuning import query
from test_pipeline import paper

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import evaluate_selected_batches
import prepare_finetuning

from pda.finetuning_data import (
    fetch_metadata,
    load_annotations,
    make_preferences,
    match_frozen_source,
    split_queries,
    v1_batch_paths,
)
from pda.retrieval import RetrievalError


def test_occurrence_loading_keeps_conflicting_repeated_rows_without_deduplication(tmp_path):
    row = query("same-query", [("1234.0001v1", 3), ("1234.0001v2", 0)])
    path = tmp_path / "batch.json"
    path.write_text(json.dumps([row, row]), encoding="utf-8")
    result, hashes = load_annotations(tmp_path, "training", deduplicate=False)
    assert len(result) == 2 and len(hashes) == 1
    assert [[j["relevance"] for j in q["judgments"]] for q in result] == [[3, 0], [3, 0]]
    assert result[0]["judgments"][0]["arxiv_id"] == "1234.0001v1"
    with pytest.raises(ValueError, match="Conflicting paper judgment"):
        load_annotations(tmp_path, "training")


def test_first_fifteen_file_selection_does_not_read_later_batches(tmp_path):
    expected = []
    for number in range(1, 16):
        path = tmp_path / f"synthetic_finetuning_batch_{number:02}.json"
        path.write_text(json.dumps(query(f"q{number}", [(f"1234.{number:04}", 3)])))
        expected.append(path)
    (tmp_path / "synthetic_finetuning_batch_16.json").write_text("unrelated invalid JSON")
    selected = v1_batch_paths(tmp_path)
    rows, hashes = load_annotations(tmp_path, "synthetic", files=selected)
    assert selected == expected and len(rows) == 15
    assert set(hashes) == {str(path) for path in expected}


@pytest.mark.parametrize("numbers", [[], [0], [16], [1, 1]])
def test_v1_file_selection_rejects_nonhistorical_or_duplicate_batches(tmp_path, numbers):
    with pytest.raises(ValueError):
        v1_batch_paths(tmp_path, numbers)


def test_v1_file_selection_requires_every_selected_file(tmp_path):
    with pytest.raises(ValueError, match="Missing original v1"):
        v1_batch_paths(tmp_path)


def test_relocated_source_requires_unique_filename_and_identical_content():
    source = Path("finetune_data/encoder/synthetic_finetuning_batch_01.json")
    old = "evaluation\\syntetic_eval\\synthetic_finetuning_batch_01.json"
    assert match_frozen_source(source, "same-digest", {old: "same-digest"}) == old
    with pytest.raises(ValueError, match="Source changed"):
        match_frozen_source(source, "changed-digest", {old: "same-digest"})
    with pytest.raises(ValueError, match="unambiguous"):
        match_frozen_source(source, "same-digest", {})
    with pytest.raises(ValueError, match="unambiguous"):
        match_frozen_source(
            source, "same-digest", {old: "same-digest", f"archive/{source.name}": "same-digest"}
        )


class MetadataClient:
    def __init__(self, status=400):
        self.calls = []
        self.status = status

    def fetch_ids(self, identifiers):
        self.calls.append(list(identifiers))
        if "1234.0002" in identifiers:
            response = httpx.Response(
                self.status, request=httpx.Request("GET", "https://example.test")
            )
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise RetrievalError("Rejected metadata batch") from exc
        return [paper(f"{identifier}v1") for identifier in identifiers], {"cache_hit": False}


def test_shared_metadata_fetch_splits_400_and_records_singletons_for_current_workflows():
    client = MetadataClient()
    progress = []
    records, batches = fetch_metadata(
        client,
        ["1234.0001", "1234.0002", "1234.0003"],
        progress=lambda done, total: progress.append((done, total)),
    )
    assert set(records) == {"1234.0001", "1234.0003"}
    assert records["1234.0001"]["arxiv_id"] == "1234.0001v1"
    assert [b["requested_ids"] for b in batches] == [["1234.0001"], ["1234.0002"], ["1234.0003"]]
    assert "error" in batches[1] and progress == [(3, 3)]


def test_v1_metadata_fetch_preserves_strict_singleton_400_behavior():
    with pytest.raises(RetrievalError):
        fetch_metadata(MetadataClient(), ["1234.0002"], strict_singletons=True)


def test_metadata_fetch_does_not_split_other_http_failures():
    client = MetadataClient(status=503)
    with pytest.raises(RetrievalError):
        fetch_metadata(client, ["1234.0001", "1234.0002"])
    assert client.calls == [["1234.0001", "1234.0002"]]


def test_metadata_fetch_preserves_outer_batch_progress_and_order():
    progress = []
    client = MetadataClient()
    records, batches = fetch_metadata(
        client,
        ["1234.0001", "1234.0003", "1234.0004"],
        batch_size=2,
        progress=lambda done, total: progress.append((done, total)),
    )
    assert list(records) == ["1234.0001", "1234.0003", "1234.0004"]
    assert len(batches) == 2 and progress == [(2, 3), (3, 3)]


def test_v1_preparation_uses_encoder_fifteen_and_preserves_splitting(tmp_path, monkeypatch):
    training, human = tmp_path / "encoder", tmp_path / "human"
    training.mkdir()
    human.mkdir()
    synthetic = []
    for number in range(1, 16):
        row = query(f"q{number:02}", [(f"1234.{number:04}", 3)])
        synthetic.append(row)
        (training / f"synthetic_finetuning_batch_{number:02}.json").write_text(json.dumps(row))
    (training / "synthetic_finetuning_batch_16.json").write_text("must not read this batch")
    human_row = query("human", [("1234.0099", 3)])
    (human / "human_evaluation_batch_01.json").write_text(json.dumps(human_row))
    requested = []

    class Client:
        def fetch_ids(self, identifiers):
            requested.extend(identifiers)
            return [
                paper(identifier, f"Title {identifier}", "Evidence.") for identifier in identifiers
            ], {}

    monkeypatch.setattr(prepare_finetuning, "ArxivClient", Client)
    output = tmp_path / "prepared.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_finetuning.py",
            "--training",
            str(training),
            "--human",
            str(human),
            "--output",
            str(output),
        ],
    )
    prepare_finetuning.main()
    result = json.loads(output.read_bytes())
    assert len(result["source_hashes"]) == 16
    assert set(requested) == {f"1234.{n:04}" for n in [*range(1, 16), 99]}
    expected = split_queries(synthetic, [human_row])
    assert {
        key: [q["query_id"] for q in result["splits"][key]]
        for key in ("train", "synthetic_dev", "human_test")
    } == {
        key: [q["query_id"] for q in expected[key]]
        for key in ("train", "synthetic_dev", "human_test")
    }
    with pytest.raises(ValueError, match="Snapshot already exists"):
        prepare_finetuning.main()


def test_selected_batch_evaluator_validates_relocated_hashes_and_preserves_report(
    tmp_path, monkeypatch
):
    training = tmp_path / "encoder"
    training.mkdir()
    source = training / "synthetic_finetuning_batch_01.json"
    row = query("q01", [("1234.0001", 3), ("1234.0002", 0)])
    source.write_text(json.dumps(row), encoding="utf-8")
    old_path = "evaluation\\syntetic_eval\\" + source.name
    prepared = tmp_path / "prepared.json"
    prepared.write_text(
        json.dumps(
            {
                "source_hashes": {old_path: hashlib.sha256(source.read_bytes()).hexdigest()},
                "splits": {"train": [row], "synthetic_dev": [], "human_test": []},
            }
        )
    )
    model = tmp_path / "model"
    model.mkdir()
    (model / "training_manifest.json").write_text(
        json.dumps({"dataset_sha256": hashlib.sha256(prepared.read_bytes()).hexdigest()})
    )
    calls = []

    class Encoder:
        def __init__(self, *args, **kwargs):
            calls.append(kwargs)

        def score_documents(self, query, documents):
            return np.array([1.0, 0.0])

    monkeypatch.setattr(evaluate_selected_batches, "LexicalEncoder", Encoder)
    monkeypatch.setattr(evaluate_selected_batches, "SemanticEncoder", Encoder)
    output = tmp_path / "report.json"
    args = [
        "evaluate_selected_batches.py",
        "--batches",
        "1",
        "--training",
        str(training),
        "--prepared",
        str(prepared),
        "--model-path",
        str(model),
        "--output",
        str(output),
    ]
    monkeypatch.setattr(sys, "argv", args)
    evaluate_selected_batches.main()
    report = json.loads(output.read_bytes())
    assert set(report["results"]) == {"tfidf", "semantic", "finetuned"}
    assert report["batches"][0]["frozen_source_path"] == old_path
    assert report["batches"][0]["source_path"] == str(source)
    assert len(calls) == 3
    with pytest.raises(ValueError, match="Report already exists"):
        evaluate_selected_batches.main()
    source.write_text(json.dumps({**row, "query": "changed query"}), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [*args[:-1], str(tmp_path / "another-report.json")])
    with pytest.raises(ValueError, match="Source changed"):
        evaluate_selected_batches.main()
    assert len(calls) == 3


def test_configured_margin_changes_preference_margins_without_changing_pair_selection():
    rows = [query("q", [("1234.0001", 3), ("1234.0002", 0)])]
    original = make_preferences(rows, 42)
    changed = make_preferences(rows, 42, margin_per_grade=0.2)
    assert original[0][:3] == changed[0][:3]
    assert original[0][3] == pytest.approx(0.3) and changed[0][3] == pytest.approx(0.6)
