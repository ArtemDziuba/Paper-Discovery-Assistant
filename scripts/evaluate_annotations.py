"""Compare available rankers on the fixed human and synthetic evaluation batches."""

import argparse
import hashlib
import importlib.metadata
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prepare_evaluation import PREPARED_PATH, validate_snapshot
from train_ranker import evaluate

from pda.evaluation import EVALUATION_MODES, evaluation_profile, metric_definitions
from pda.finetuning_data import base_id
from pda.query import normalize_title
from pda.ranking import (
    FINETUNED_LABEL,
    FINETUNED_PATH,
    FINETUNED_V2_LABEL,
    FINETUNED_V2_PATH,
    LexicalEncoder,
    SemanticEncoder,
)


def training_overlap(datasets: dict, original_splits: dict) -> dict:
    """Audit exposure to v1's original training/development data without filtering tests."""
    exposure = {}
    for split in ("train", "synthetic_dev"):
        queries = original_splits[split]
        exposure[split] = {
            "texts": {normalize_title(q["query"]) for q in queries},
            "ids": {q["query_id"] for q in queries},
            "papers": {base_id(j["arxiv_id"]) for q in queries for j in q["judgments"]},
        }
    return {
        name: {
            q["query_id"]: {
                split: {
                    "query_seen": q["query_id"] in known["ids"]
                    or normalize_title(q["query"]) in known["texts"],
                    "paper_ids_seen": sorted(
                        {base_id(j["arxiv_id"]) for j in q["judgments"]} & known["papers"]
                    ),
                }
                for split, known in exposure.items()
            }
            for q in rows
        }
        for name, rows in datasets.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--mode", choices=list(EVALUATION_MODES), default="truncated")
    parser.add_argument("--prepared", type=Path, default=PREPARED_PATH)
    parser.add_argument(
        "--v2-model",
        type=Path,
        default=FINETUNED_V2_PATH,
        help="Include v2 if this folder contains its training manifest.",
    )
    parser.add_argument(
        "--output", type=Path, help="Default: evaluation/results/ranking-comparison-MODE.json"
    )
    parser.add_argument(
        "--training-provenance", type=Path, default=Path("evaluation/finetuning/prepared.json")
    )
    args = parser.parse_args()
    if args.output is None:
        args.output = Path(f"evaluation/results/ranking-comparison-{args.mode}.json")
    if args.output.exists():
        raise ValueError("Report already exists; choose a new --output path to preserve the run.")
    prepared_raw = args.prepared.read_bytes()
    prepared = json.loads(prepared_raw)
    validate_snapshot(prepared)
    datasets = prepared["datasets"]
    provenance_raw = args.training_provenance.read_bytes()
    manifest_raw = (FINETUNED_PATH / "training_manifest.json").read_bytes()
    manifest = json.loads(manifest_raw)
    if hashlib.sha256(provenance_raw).hexdigest() != manifest["dataset_sha256"]:
        raise ValueError("Original training provenance does not match the v1 model manifest.")
    overlap = training_overlap(datasets, json.loads(provenance_raw)["splits"])
    v2_artifact, v2_overlap = None, None
    v2_manifest_path = args.v2_model / "training_manifest.json"
    if v2_manifest_path.is_file():
        v2_manifest_raw = v2_manifest_path.read_bytes()
        v2_manifest = json.loads(v2_manifest_raw)
        v2_prepared_path = Path(v2_manifest["prepared_path"])
        v2_prepared_raw = v2_prepared_path.read_bytes()
        if hashlib.sha256(v2_prepared_raw).hexdigest() != v2_manifest["dataset_sha256"]:
            raise ValueError("V2 preparation does not match its training manifest.")
        v2_prepared = json.loads(v2_prepared_raw)
        v2_overlap = training_overlap(
            datasets,
            {
                "train": v2_prepared["training"],
                "synthetic_dev": v2_prepared["combined_development"],
            },
        )
        v2_artifact = {
            "path": str(args.v2_model),
            "manifest_sha256": hashlib.sha256(v2_manifest_raw).hexdigest(),
            "weights_sha256": hashlib.sha256(
                (args.v2_model / "model.safetensors").read_bytes()
            ).hexdigest(),
            "training_run_id": v2_manifest["run_id"],
            "evaluation_used_for_checkpoint_selection": v2_manifest["development_snapshot_sha256"]
            == hashlib.sha256(prepared_raw).hexdigest(),
        }
    results = {name: {} for name in datasets}
    identities, timings = {}, {}
    factories = {
        "tfidf": ("TF-IDF baseline", LexicalEncoder),
        "semantic": ("Semantic", lambda: SemanticEncoder(args.device)),
        "finetuned": (
            FINETUNED_LABEL,
            lambda: SemanticEncoder(args.device, model_path=FINETUNED_PATH),
        ),
    }
    if v2_artifact is not None:
        factories["finetuned_v2"] = (
            FINETUNED_V2_LABEL,
            lambda: SemanticEncoder(args.device, model_path=args.v2_model),
        )
    for name, (label, factory) in factories.items():
        started = time.perf_counter()
        encoder = factory()
        identities[name] = {
            "label": label,
            "model": getattr(encoder, "model_name", "tf-idf"),
            "revision": getattr(encoder, "model_revision", None),
        }
        timings[name] = {"initialization_seconds": time.perf_counter() - started}
        for group, queries in datasets.items():
            started = time.perf_counter()
            results[group][name] = evaluate(encoder, queries, mode=args.mode)
            timings[name][f"{group}_seconds"] = time.perf_counter() - started
            print(group, label, results[group][name]["aggregate"], flush=True)
        del encoder
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "prepared_path": str(args.prepared),
        "prepared_sha256": hashlib.sha256(prepared_raw).hexdigest(),
        "source_hashes": prepared["source_hashes"],
        "source_judgment_counts": prepared["source_judgment_counts"],
        "verified_judgment_counts": {
            name: sum(len(q["judgments"]) for q in queries) for name, queries in datasets.items()
        },
        "exclusions": prepared["exclusions"],
        "device": args.device,
        "evaluation_mode": args.mode,
        "evaluation_configuration": evaluation_profile(args.mode),
        "versions": {
            package: importlib.metadata.version(package)
            for package in (
                "torch",
                "sentence-transformers",
                "transformers",
                "numpy",
                "scikit-learn",
            )
        },
        "models": identities,
        "finetuned_artifact": {
            "path": str(FINETUNED_PATH),
            "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
            "weights_sha256": hashlib.sha256(
                (FINETUNED_PATH / "model.safetensors").read_bytes()
            ).hexdigest(),
            "training_run_id": manifest["run_id"],
        },
        "original_training_overlap": overlap,
        "finetuned_v2_artifact": v2_artifact,
        "v2_training_and_development_overlap": v2_overlap,
        "timings": timings,
        "metric_definitions": metric_definitions(args.mode),
        "protocol": "Rank each query's explicitly judged pool by title+abstract cosine; "
        "all methods use identical verified candidates, including synthetic batches. "
        "No title-priority rule or live topic retrieval. nDCG uses grades 0–3. "
        "Precision, reciprocal rank and AP use grade >=2. MAP@k averages AP@k with denominator "
        "min(k, total relevant judgments). Metric cutoffs follow evaluation_configuration. "
        "Pairwise accuracy uses unequal-grade judged pairs within its configured scope "
        "and gives half credit for exact score ties; undefined queries are excluded "
        "from its mean and counted separately. Macro averages weight queries equally. "
        "Human and synthetic results are separate. Training overlap is reported, not filtered. "
        "Synthetic-label agreement is not human ground truth; two human queries limit inference. "
        "No training or model updates occur during this evaluation. "
        "If v2 is included, its training uses all verified encoder batches and these evaluation "
        "queries were combined for checkpoint selection; v2 scores here are development results.",
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Saved", args.output, flush=True)


if __name__ == "__main__":
    main()
