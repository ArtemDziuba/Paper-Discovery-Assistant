"""Freeze arXiv metadata for evaluation annotations, independently of training data."""

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pda.finetuning_data import base_id, fetch_metadata, hydrate, load_annotations
from pda.retrieval import ArxivClient

PREPARED_PATH = Path("evaluation/annotations/prepared.json")
ANNOTATION_FOLDERS = {
    "human": Path("evaluation/human_eval"),
    "synthetic": Path("evaluation/syntetic_eval"),
}


def load_datasets(folders: dict[str, Path] | None = None) -> tuple[dict, dict]:
    datasets, hashes = {}, {}
    for name, folder in (ANNOTATION_FOLDERS if folders is None else folders).items():
        datasets[name], source_hashes = load_annotations(folder, name)
        hashes.update(source_hashes)
    return datasets, hashes


def validate_snapshot(prepared: dict, folders: dict[str, Path] | None = None) -> None:
    """Require the complete current annotation file set and contents to be unchanged."""
    _, hashes = load_datasets(folders)
    if hashes != prepared["source_hashes"]:
        raise ValueError(
            "Evaluation annotations changed since metadata preparation. "
            "Run scripts/prepare_evaluation.py with a new --output snapshot."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PREPARED_PATH)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Snapshot already exists; use a new --output path to preserve provenance.")
    datasets, hashes = load_datasets()
    identifiers = sorted(
        {base_id(j["arxiv_id"]) for rows in datasets.values() for q in rows for j in q["judgments"]}
    )
    records, batches = fetch_metadata(
        ArxivClient(),
        identifiers,
        progress=lambda done, total: print(f"Metadata: {done}/{total} IDs", flush=True),
    )
    verified, exclusions = {}, {}
    for name, rows in datasets.items():
        verified[name], exclusions[name] = hydrate(rows, records, threshold=0.85)
        if any(not q["judgments"] for q in verified[name]):
            raise ValueError(f"A {name} query has no verified papers; review the annotations.")
    snapshot = {
        "created_at": datetime.now(UTC).isoformat(),
        "source_hashes": hashes,
        "title_agreement_threshold": 0.85,
        "metadata_batches": batches,
        "source_judgment_counts": {
            name: sum(len(q["judgments"]) for q in rows) for name, rows in datasets.items()
        },
        "datasets": verified,
        "exclusions": exclusions,
        "scope": "Evaluation annotations only; finetune_data is not read or used for training.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    print("Saved", args.output, flush=True)
    print("Snapshot SHA256:", hashlib.sha256(args.output.read_bytes()).hexdigest(), flush=True)
    print("Excluded judgments:", {name: len(rows) for name, rows in exclusions.items()}, flush=True)


if __name__ == "__main__":
    main()
