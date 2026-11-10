"""Train a local preference ranker on synthetic groups; reserve human judgments for testing."""

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from torch.nn import functional as F

from pda.evaluation import (
    aggregate_metrics,
    evaluation_profile,
    metric_definitions,
    metrics,
    pairwise_metric_name,
)
from pda.finetuning_data import make_preferences
from pda.ranking import FINETUNED_PATH, MODEL, REVISION, LexicalEncoder, SemanticEncoder
from pda.resources import peak_working_set_bytes


def evaluate(
    encoder, queries: list[dict], mode: str = "truncated", *, relevant_min: float = 2
) -> dict:
    profile = evaluation_profile(mode)
    rows = {}
    score_groups = {str(grade): [] for grade in range(4)}
    for query in queries:
        items = query["judgments"]
        if not items:
            continue
        texts = [j["paper"]["title"] + "\n" + j["paper"]["abstract"] for j in items]
        scores = encoder.score_documents(query["query"], texts)
        ranked = sorted(range(len(items)), key=lambda i: -float(scores[i]))
        judgments = {j["arxiv_id"]: j["relevance"] for j in items}
        result = metrics(
            [items[i]["arxiv_id"] for i in ranked],
            judgments,
            relevant_min=relevant_min,
            scores={j["arxiv_id"]: float(score) for j, score in zip(items, scores, strict=True)},
            mode=mode,
        )
        for item, score in zip(items, scores, strict=True):
            score_groups[str(item["relevance"])].append(float(score))
        rows[query["query_id"]] = {
            "query": query["query"],
            "metrics": result,
            "judged_candidates": len(items),
            "ranking": [
                {
                    "arxiv_id": items[i]["arxiv_id"],
                    "grade": items[i]["relevance"],
                    "cosine": float(scores[i]),
                }
                for i in ranked
            ],
        }
    return {
        "query_count": len(rows),
        "evaluation_mode": mode,
        "evaluation_configuration": profile,
        "per_query": rows,
        "aggregate": aggregate_metrics([r["metrics"] for r in rows.values()]) if rows else None,
        "pairwise_valid_query_count": sum(
            row["metrics"][pairwise_metric_name(mode)] is not None for row in rows.values()
        ),
        "mean_cosine_by_grade": {
            grade: float(np.mean(scores)) if scores else None
            for grade, scores in score_groups.items()
        },
        "metric_definitions": metric_definitions(mode),
        "protocol": f"Per-query judged candidates only; no title override; binary relevance >={relevant_min:g} "
        "for Precision, reciprocal rank and AP. Cutoffs follow evaluation_configuration; "
        "pairwise accuracy uses graded judgments and raw score ties.",
    }


def train_epoch(model, samples, optimizer, scheduler, config: dict, epoch: int) -> list[float]:
    """Run v1's seeded pairwise training loop; return every observed batch loss."""
    model.train()
    batches = list(samples)
    random.Random(config["seed"] + epoch).shuffle(batches)
    losses = []
    for start in range(0, len(batches), config["batch_size"]):
        batch = batches[start : start + config["batch_size"]]
        optimizer.zero_grad(set_to_none=True)
        embeddings = []
        for column in range(3):
            features = model.preprocess([item[column] for item in batch])
            features = {
                key: value.to(config["device"]) if isinstance(value, torch.Tensor) else value
                for key, value in features.items()
            }
            embeddings.append(model(features)["sentence_embedding"])
        margins = torch.tensor([item[3] for item in batch], device=config["device"])
        positive = F.cosine_similarity(embeddings[0], embeddings[1])
        negative = F.cosine_similarity(embeddings[0], embeddings[2])
        loss = F.relu(margins - positive + negative).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.get("gradient_clip_norm", 1.0))
        optimizer.step()
        scheduler.step()
        losses.append(float(loss.detach()))
        if start % 128 == 0:
            print(
                f"epoch {epoch}: preferences {start}/{len(batches)}, loss {losses[-1]:.4f}",
                flush=True,
            )
    return losses


def seed_training(config: dict) -> None:
    """Apply the shared original RNG, thread-count and deterministic CUDA settings."""
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.set_num_threads(4)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_optimizer_and_scheduler(
    model, sample_count: int, config: dict
) -> tuple[torch.optim.AdamW, torch.optim.lr_scheduler.LambdaLR, int, int]:
    """Build the original AdamW and warmup/decay schedule from stored parameters."""
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    batch_size = config["batch_size"]
    steps_per_epoch = (sample_count + batch_size - 1) // batch_size
    total_steps = steps_per_epoch * config["epochs"]
    warmup = max(1, int(total_steps * config["warmup_fraction"]))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min(
            (step + 1) / warmup, max(0, (total_steps - step) / max(1, total_steps - warmup))
        ),
    )
    return optimizer, scheduler, steps_per_epoch, warmup


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("evaluation/finetuning/prepared.json"))
    parser.add_argument("--output", type=Path, default=FINETUNED_PATH)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    if args.epochs < 1 or (args.output / "training_manifest.json").exists():
        raise ValueError("Use positive epochs and a new output folder for a reproducible run.")
    raw = args.data.read_bytes()
    data = json.loads(raw)
    splits = data["splits"]
    config = {
        "seed": 42,
        "epochs": args.epochs,
        "batch_size": 8,
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "warmup_fraction": 0.1,
        "preferences_per_query": 64,
        "margin_per_grade": 0.1,
        "loss": "same-query pairwise cosine hinge",
        "device": args.device,
        "binary_relevant_min": 2,
        "selection": "best synthetic development nDCG@10; never human labels",
    }
    dataset_hash = hashlib.sha256(raw).hexdigest()
    run_id = hashlib.sha256(
        (dataset_hash + json.dumps(config, sort_keys=True)).encode()
    ).hexdigest()[:16]
    seed_training(config)
    samples = make_preferences(
        splits["train"],
        config["seed"],
        config["preferences_per_query"],
        margin_per_grade=config["margin_per_grade"],
    )
    encoder = SemanticEncoder(args.device)
    model = encoder.model
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    relevant_min = config["binary_relevant_min"]
    baseline_dev = evaluate(encoder, splits["synthetic_dev"], relevant_min=relevant_min)
    if not baseline_dev["query_count"]:
        raise ValueError("No verified synthetic development queries are available.")
    optimizer, scheduler, _, _ = build_optimizer_and_scheduler(model, len(samples), config)
    best_score, best_epoch = -1.0, None
    history = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        losses = train_epoch(model, samples, optimizer, scheduler, config, epoch)
        model.eval()
        dev = evaluate(encoder, splits["synthetic_dev"], relevant_min=relevant_min)
        history.append(
            {"epoch": epoch, "mean_training_loss": float(np.mean(losses)), "synthetic_dev": dev}
        )
        print("epoch", epoch, "synthetic development", dev["aggregate"], flush=True)
        if dev["aggregate"]["ndcg@10"] > best_score:
            best_score, best_epoch = dev["aggregate"]["ndcg@10"], epoch
            model.save_pretrained(str(args.output))
    training_seconds = time.perf_counter() - started
    manifest = {
        "run_id": run_id,
        "base_model": MODEL,
        "base_revision": REVISION,
        "dataset_sha256": dataset_hash,
        "source_hashes": data["source_hashes"],
        "config": config,
        "best_epoch": best_epoch,
        "training_preferences": len(samples),
        "synthetic_training_queries": [q["query_id"] for q in splits["train"]],
        "synthetic_dev_queries": [q["query_id"] for q in splits["synthetic_dev"]],
        "human_test_queries": [q["query_id"] for q in splits["human_test"]],
        "torch": torch.__version__,
        "training_seconds": training_seconds,
        "peak_process_working_set_bytes": peak_working_set_bytes(),
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated()
        if args.device == "cuda"
        else None,
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved()
        if args.device == "cuda"
        else None,
        "limitations": "Synthetic training judgments; only two human test queries. "
        "Excluded unverified ID/title pairs. Scores are uncalibrated cosine.",
    }
    (args.output / "training_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    del optimizer, model, encoder
    if args.device == "cuda":
        torch.cuda.empty_cache()
    baseline = SemanticEncoder(args.device)
    comparison = {
        "synthetic_dev_before": baseline_dev,
        "human_test": {
            "lexical": evaluate(LexicalEncoder(), splits["human_test"], relevant_min=relevant_min),
            "semantic": evaluate(baseline, splits["human_test"], relevant_min=relevant_min),
        },
    }
    del baseline
    tuned = SemanticEncoder(args.device, model_path=args.output)
    comparison["human_test"]["finetuned"] = evaluate(
        tuned, splits["human_test"], relevant_min=relevant_min
    )
    comparison["synthetic_dev_selected"] = evaluate(
        tuned, splits["synthetic_dev"], relevant_min=relevant_min
    )
    report = {
        "manifest": manifest,
        "history": history,
        "comparison": comparison,
        "metadata_exclusion_count": len(data["exclusions"]),
        "removed_training_overlap": splits["removed_training_overlap"],
    }
    target = args.data.parent / "training-results.json"
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Saved model:", args.output, flush=True)
    print(
        {name: result["aggregate"] for name, result in comparison["human_test"].items()}, flush=True
    )


if __name__ == "__main__":
    main()
