"""Train original MiniLM on all verified encoder batches with one combined development set."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from prepare_evaluation import validate_snapshot
from train_ranker import build_optimizer_and_scheduler, evaluate, seed_training, train_epoch

from pda.finetuning_data import load_annotations, make_preferences
from pda.ranking import (
    FINETUNED_PATH,
    FINETUNED_V2_LABEL,
    FINETUNED_V2_MODEL,
    FINETUNED_V2_PATH,
    MODEL,
    REVISION,
    SemanticEncoder,
)
from pda.resources import peak_working_set_bytes

CUTOFFS = (5, 10, 20, 50)


def plot_history(history: list[dict], baseline: dict, best_epoch: int, folder: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder.mkdir(parents=True, exist_ok=True)
    series = [("training_loss", "Training loss", "training-loss", None)] + [
        (
            f"development_ndcg@{k}",
            f"Combined development nDCG@{k}",
            f"development-ndcg-at-{k}",
            baseline["aggregate"][f"ndcg@{k}"],
        )
        for k in CUTOFFS
    ]
    for key, title, stem, reference in series:
        fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
        ax.plot(
            [row["epoch"] for row in history],
            [row[key] for row in history],
            marker="o",
            color="#176b94",
            linewidth=2,
        )
        if reference is not None:
            ax.axhline(reference, color="#6b7280", linestyle="--", label="Original MiniLM")
            ax.set_ylim(0, 1)
        ax.axvline(best_epoch, color="#b45309", linestyle=":", label="Selected epoch (nDCG@10)")
        ax.set(
            title=title, xlabel="Epoch", ylabel="Mean hinge loss" if reference is None else "nDCG"
        )
        ax.set_xticks([row["epoch"] for row in history])
        ax.grid(alpha=0.2)
        ax.legend(loc="lower left" if reference is not None else "upper right")
        fig.savefig(folder / f"{stem}.png", dpi=180)
        fig.savefig(folder / f"{stem}.svg")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("evaluation/finetuning-v2/prepared.json"))
    parser.add_argument("--output", type=Path, default=FINETUNED_V2_PATH)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument(
        "--results",
        type=Path,
        help="History, CSV and plots folder; defaults to the prepared-data folder.",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(
            "Choose a new model output folder; existing artifacts will not be overwritten."
        )
    args.results = args.results or args.data.parent
    if any(
        (args.results / name).exists()
        for name in ("training-results.json", "epoch-metrics.csv", "plots")
    ):
        raise ValueError("Training results already exist; choose a new --results folder.")
    # Fail before training if the plotting setup is missing.
    import matplotlib  # noqa: F401

    raw = args.data.read_bytes()
    data = json.loads(raw)
    _, source_hashes = load_annotations(
        Path(data.get("training_folder", "finetune_data/encoder")),
        "synthetic_training",
        deduplicate=False,
    )
    if source_hashes != data["source_hashes"]:
        raise ValueError("Training files changed since arXiv verification; prepare a new snapshot.")
    development_raw = Path(data["development_snapshot_path"]).read_bytes()
    if hashlib.sha256(development_raw).hexdigest() != data["development_snapshot_sha256"]:
        raise ValueError("Development snapshot changed since preparation.")
    validate_snapshot(json.loads(development_raw))
    original_manifest_raw = (FINETUNED_PATH / "training_manifest.json").read_bytes()
    original = json.loads(original_manifest_raw)
    config = {
        **original["config"],
        "device": args.device,
        "selection": "best combined 12-query development nDCG@10; first epoch wins ties",
        "development_pooling": "one unweighted macro mean across 10 synthetic and 2 human query pools",
        "dataset_deduplication": False,
        "gradient_clip_norm": original["config"].get("gradient_clip_norm", 1.0),
        # Describes the fixed float32 path; this is not a precision-selection option.
        "precision": "float32",
    }
    dataset_hash = hashlib.sha256(raw).hexdigest()
    run_id = hashlib.sha256(
        (dataset_hash + json.dumps(config, sort_keys=True)).encode()
    ).hexdigest()[:16]
    seed_training(config)
    samples = make_preferences(
        data["training"],
        config["seed"],
        config["preferences_per_query"],
        margin_per_grade=config["margin_per_grade"],
    )
    encoder = SemanticEncoder(args.device)  # Always the original base, never v1 weights.
    model = encoder.model
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    relevant_min = config["binary_relevant_min"]
    baseline = evaluate(
        encoder, data["combined_development"], mode="full", relevant_min=relevant_min
    )
    optimizer, scheduler, steps_per_epoch, warmup = build_optimizer_and_scheduler(
        model, len(samples), config
    )
    history, best_epoch, best_score = [], None, -1.0
    started = time.perf_counter()
    args.results.mkdir(parents=True, exist_ok=True)
    report_path = args.results / "training-results.json"
    args.output.mkdir(parents=True)
    for epoch in range(1, config["epochs"] + 1):
        losses = train_epoch(model, samples, optimizer, scheduler, config, epoch)
        model.eval()
        development = evaluate(
            encoder, data["combined_development"], mode="full", relevant_min=relevant_min
        )
        row = {
            "epoch": epoch,
            "training_loss": float(np.mean(losses)),
            **{f"development_ndcg@{k}": development["aggregate"][f"ndcg@{k}"] for k in CUTOFFS},
            "batch_losses": losses,
            "combined_development": development,
        }
        history.append(row)
        if row["development_ndcg@10"] > best_score:
            best_score, best_epoch = row["development_ndcg@10"], epoch
            model.save_pretrained(str(args.output), create_model_card=False)
        # Save the actual observed history after each epoch, even before plots are created.
        report_path.write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "config": config,
                    "baseline_combined_development": baseline,
                    "history": history,
                    "best_epoch": best_epoch,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(
            "epoch",
            epoch,
            {
                key: value
                for key, value in row.items()
                if key not in {"batch_losses", "combined_development"}
            },
            flush=True,
        )
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "run_id": run_id,
        "model_name": FINETUNED_V2_MODEL,
        "display_name": FINETUNED_V2_LABEL,
        "base_model": MODEL,
        "base_revision": REVISION,
        "initialized_from": "original pinned MiniLM, not v1",
        "dataset_sha256": dataset_hash,
        "prepared_path": str(args.data),
        "source_hashes": data["source_hashes"],
        "development_snapshot_sha256": data["development_snapshot_sha256"],
        "development_source_hashes": data["development_source_hashes"],
        "original_parameters_manifest_sha256": hashlib.sha256(original_manifest_raw).hexdigest(),
        "config": config,
        "best_epoch": best_epoch,
        "training_preferences": len(samples),
        "steps_per_epoch": steps_per_epoch,
        "warmup_steps": warmup,
        "training_query_count": len(data["training"]),
        "training_judgments": data["verified_occurrences"]["judgments"],
        "development_counts": data["development_counts"],
        "model_max_seq_length": model.max_seq_length,
        "training_seconds": time.perf_counter() - started,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "sentence-transformers", "transformers", "numpy", "matplotlib")
        },
        "peak_process_working_set_bytes": peak_working_set_bytes(),
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated()
        if args.device == "cuda"
        else None,
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved()
        if args.device == "cuda"
        else None,
        "limitations": "Both human and synthetic evaluation queries select the checkpoint; subsequent scores on them are development performance. Training/development paper overlap is preserved and audited.",
    }
    (args.output / "training_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    report = json.loads(report_path.read_bytes())
    report["manifest"] = manifest
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    columns = ["epoch", "training_loss", *[f"development_ndcg@{k}" for k in CUTOFFS]]
    with (args.results / "epoch-metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: row[key] for key in columns} for row in history)
    plot_history(history, baseline, best_epoch, args.results / "plots")
    print("Saved v2:", args.output, "selected epoch:", best_epoch, flush=True)


if __name__ == "__main__":
    main()
