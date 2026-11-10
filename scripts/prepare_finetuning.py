"""Prepare v1 from exactly the first fifteen encoder batches and reserved human judgments."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pda.finetuning_data import (
    fetch_metadata,
    hydrate,
    load_annotations,
    split_queries,
    v1_batch_paths,
)
from pda.retrieval import ArxivClient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, default=Path("finetune_data/encoder"))
    parser.add_argument("--human", type=Path, default=Path("evaluation/human_eval"))
    parser.add_argument("--output", type=Path, default=Path("evaluation/finetuning/prepared.json"))
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Snapshot already exists; choose a new --output path to preserve the run.")
    human, human_hashes = load_annotations(args.human, "human")
    synthetic, synthetic_hashes = load_annotations(
        args.training, "synthetic", files=v1_batch_paths(args.training)
    )
    identifiers = sorted({j["arxiv_id"] for q in [*human, *synthetic] for j in q["judgments"]})
    records, batches = fetch_metadata(
        ArxivClient(),
        identifiers,
        strict_singletons=True,
        progress=lambda done, total: print(f"Verified metadata batch {done}/{total}", flush=True),
    )
    prepared_human, excluded_human = hydrate(human, records)
    prepared_synthetic, excluded_synthetic = hydrate(synthetic, records)
    splits = split_queries(prepared_synthetic, prepared_human)
    output = {
        "source_hashes": {**human_hashes, **synthetic_hashes},
        "training_folder": str(args.training),
        "training_batches": list(range(1, 16)),
        "human_folder": str(args.human),
        "title_agreement_threshold": 0.85,
        "metadata_batches": batches,
        "exclusions": excluded_human + excluded_synthetic,
        "splits": splits,
        "annotation_provenance": {
            "human": "user supplied",
            "synthetic": "user supplied synthetic labels; generator not recorded",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(
        {
            key: sum(len(q["judgments"]) for q in splits[key])
            for key in ["train", "synthetic_dev", "human_test"]
        },
        flush=True,
    )
    print("Excluded metadata/ID-title mismatches:", len(output["exclusions"]))


if __name__ == "__main__":
    main()
