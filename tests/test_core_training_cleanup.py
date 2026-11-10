"""Preserve retrieval contracts and original training numerics during consolidation."""

import random
import sys
from pathlib import Path

import httpx
import numpy as np
import pytest
import torch
from test_pagination import page
from test_pipeline import feed, paper

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from train_ranker import build_optimizer_and_scheduler, evaluate, seed_training, train_epoch

from pda import pipeline, retrieval
from pda.ranking import LexicalEncoder


def training_config(**overrides):
    return {
        "seed": 42,
        "epochs": 3,
        "batch_size": 8,
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "warmup_fraction": 0.1,
        "device": "cpu",
        **overrides,
    }


class TinyEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(2, 2, bias=False)
        self.projection.weight.data.copy_(torch.eye(2))

    def preprocess(self, texts):
        values = {"query": [1.0, 0.0], "better": [0.0, 1.0], "worse": [1.0, 0.0]}
        return {"inputs": torch.tensor([values[text] for text in texts])}

    def forward(self, features):
        return {"sentence_embedding": self.projection(features["inputs"])}


def test_atom_response_is_parsed_once_with_raw_counts_and_public_api_preserved(
    tmp_path, monkeypatch
):
    xml = page(["1234.0001v1", "1234.0001v2", "1234.0002v1"], 3, invalid=True)
    original = retrieval.ET.fromstring
    parsed = []

    def parse(content):
        parsed.append(content)
        return original(content)

    monkeypatch.setattr(retrieval.ET, "fromstring", parse)
    monkeypatch.setattr(
        httpx,
        "get",
        lambda url, **kwargs: httpx.Response(200, text=xml, request=httpx.Request("GET", url)),
    )
    client = retrieval.ArxivClient(tmp_path)
    records, metadata = client.search("graphs", 3)
    assert len(parsed) == 1
    assert [p.arxiv_id for p in records] == ["1234.0001v1"]
    assert metadata["scanned_entries"] == 3 and metadata["skipped_records"] == 1
    assert metadata["total_results"] == 3
    assert client.search("graphs", 3)[1]["cache_hit"]
    assert len(parsed) == 1
    public_records, skipped = retrieval.parse_feed(xml)
    assert public_records == records and skipped == 1 and len(parsed) == 2


@pytest.mark.parametrize("total, expected", [(None, None), ("unknown", None), (-5, 0), (7, 7)])
def test_single_parse_retains_optional_invalid_and_clamped_total_results(
    tmp_path, monkeypatch, total, expected
):
    xml = feed()
    if total is not None:
        xml = xml.replace(
            "<entry>",
            f'<totalResults xmlns="http://a9.com/-/spec/opensearch/1.1/">{total}</totalResults><entry>',
            1,
        )
    monkeypatch.setattr(
        httpx,
        "get",
        lambda url, **kwargs: httpx.Response(200, text=xml, request=httpx.Request("GET", url)),
    )
    records, metadata = retrieval.ArxivClient(tmp_path).search("graphs", 1)
    assert len(records) == 1 and metadata["total_results"] == expected


def test_pipeline_reuses_normalized_titles_for_ordering_and_metadata(monkeypatch):
    originals = [paper(f"1234.{i:04}v1", f"Paper {i}") for i in range(8)]
    calls = []
    normalize = pipeline.normalize_title

    class Client:
        def search(self, query, limit):
            return originals, {}

    def observed(text):
        calls.append(text)
        return normalize(text)

    monkeypatch.setattr(pipeline, "normalize_title", observed)
    result = pipeline.search("UNMATCHED_QUERY", Client(), LexicalEncoder())
    assert not result.metadata["exact_title_priority"]
    assert calls.count("UNMATCHED_QUERY") == 1
    assert all(calls.count(p.title) == 1 for p in originals)
    assert [m.paper.arxiv_id for m in result.matches] == [p.arxiv_id for p in originals]


def test_title_lookup_normalizes_input_once(monkeypatch, tmp_path):
    records = [paper(f"1234.{i:04}v1", f"Paper {i}") for i in range(8)]
    monkeypatch.setattr(retrieval.ArxivClient, "search", lambda *args: (records, {}))
    calls = []
    normalize = retrieval.normalize_title

    def observed(text):
        calls.append(text)
        return normalize(text)

    monkeypatch.setattr(retrieval, "normalize_title", observed)
    result, metadata = retrieval.ArxivClient(tmp_path).find_exact_titles("GRAPH PAPER")
    assert not result and metadata["returned_candidates"] == 8
    assert calls.count("GRAPH PAPER") == 1


def test_shared_optimizer_preserves_original_schedule_and_default_hyperparameters():
    model = TinyEncoder()
    optimizer, scheduler, steps, warmup = build_optimizer_and_scheduler(model, 9, training_config())
    assert steps == 2 and warmup == 1
    assert optimizer.defaults["lr"] == 2e-5 and optimizer.defaults["weight_decay"] == 0.01
    trace = [scheduler.get_last_lr()[0] / 2e-5]
    for _ in range(6):
        optimizer.step()
        scheduler.step()
        trace.append(scheduler.get_last_lr()[0] / 2e-5)
    assert trace == pytest.approx([1, 1, 0.8, 0.6, 0.4, 0.2, 0])


def test_shared_schedule_keeps_warmup_boundaries_and_uses_configured_batch_size():
    model = TinyEncoder()
    optimizer, scheduler, steps, warmup = build_optimizer_and_scheduler(
        model, 80, training_config()
    )
    assert (steps, warmup) == (10, 3)
    trace = [scheduler.get_last_lr()[0] / 2e-5]
    for _ in range(30):
        optimizer.step()
        scheduler.step()
        trace.append(scheduler.get_last_lr()[0] / 2e-5)
    assert trace[:5] == pytest.approx([1 / 3, 2 / 3, 1, 1, 26 / 27])
    assert trace[-2:] == pytest.approx([1 / 27, 0])
    _, _, configured_steps, _ = build_optimizer_and_scheduler(
        model, 9, training_config(batch_size=4)
    )
    assert configured_steps == 3


def test_default_training_step_keeps_original_hinge_loss_and_adamw_update():
    model = TinyEncoder()
    config = training_config()
    optimizer, scheduler, _, _ = build_optimizer_and_scheduler(model, 1, config)
    losses = train_epoch(
        model, [("query", "better", "worse", 0.3)], optimizer, scheduler, config, 1
    )
    assert losses == pytest.approx([1.3])
    expected = torch.tensor([[1 - 2e-7, 2e-5], [2e-5, 1 - 2e-7]])
    torch.testing.assert_close(model.projection.weight, expected, rtol=1e-7, atol=1e-10)
    assert scheduler.last_epoch == 1


@pytest.mark.parametrize("clip_norm", [None, 0.25])
def test_epoch_applies_default_or_configured_gradient_clip_norm(clip_norm):
    model = TinyEncoder()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)
    config = training_config(**({"gradient_clip_norm": clip_norm} if clip_norm is not None else {}))
    train_epoch(model, [("query", "better", "worse", 0.3)], optimizer, scheduler, config, 1)
    norm = torch.linalg.vector_norm(torch.stack([p.grad.norm() for p in model.parameters()]))
    assert float(norm) == pytest.approx(1.0 if clip_norm is None else clip_norm, abs=1e-6)


def test_shared_seed_setup_reproduces_python_numpy_and_torch_sequences(monkeypatch):
    before_python, before_numpy = random.getstate(), np.random.get_state()
    before_torch, before_threads = torch.get_rng_state(), torch.get_num_threads()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    try:
        seed_training(training_config())
        first = (random.random(), np.random.random(), torch.rand(3))
        seed_training(training_config())
        second = (random.random(), np.random.random(), torch.rand(3))
        assert first[:2] == second[:2]
        assert torch.equal(first[2], second[2])
        assert torch.get_num_threads() == 4
    finally:
        random.setstate(before_python)
        np.random.set_state(before_numpy)
        torch.set_rng_state(before_torch)
        torch.set_num_threads(before_threads)


def test_shared_seed_setup_preserves_deterministic_cuda_configuration(monkeypatch):
    observed = []
    monkeypatch.setattr(random, "seed", lambda seed: observed.append(("python", seed)))
    monkeypatch.setattr(np.random, "seed", lambda seed: observed.append(("numpy", seed)))
    monkeypatch.setattr(torch, "manual_seed", lambda seed: observed.append(("torch", seed)))
    monkeypatch.setattr(torch, "set_num_threads", lambda count: observed.append(("threads", count)))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda seed: observed.append(("cuda", seed)))
    monkeypatch.setattr(torch.backends.cudnn, "deterministic", False)
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", True)
    seed_training(training_config(seed=19))
    assert observed == [("python", 19), ("numpy", 19), ("torch", 19), ("threads", 4), ("cuda", 19)]
    assert torch.backends.cudnn.deterministic and not torch.backends.cudnn.benchmark


def test_training_evaluation_uses_configured_binary_threshold_without_changing_ndcg():
    class Encoder:
        def score_documents(self, query, documents):
            return np.array([1.0, 0.5, 0.0])

    groups = [
        {
            "query_id": "q",
            "query": "graph query",
            "judgments": [
                {
                    "arxiv_id": identifier,
                    "relevance": grade,
                    "paper": {"title": identifier, "abstract": "Fixture evidence."},
                }
                for identifier, grade in [("low", 1), ("high", 3), ("irrelevant", 0)]
            ],
        }
    ]
    default = evaluate(Encoder(), groups, mode="full")
    permissive = evaluate(Encoder(), groups, mode="full", relevant_min=1)
    assert default["aggregate"]["ndcg@10"] == permissive["aggregate"]["ndcg@10"]
    assert default["aggregate"]["precision@5"] == 0.2
    assert permissive["aggregate"]["precision@5"] == 0.4
    assert default["aggregate"]["mrr@50"] == 0.5
    assert permissive["aggregate"]["mrr@50"] == 1
    assert "binary relevance >=1" in permissive["protocol"]
