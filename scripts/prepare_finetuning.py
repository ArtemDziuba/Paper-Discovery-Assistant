"""Verify annotation paper IDs and freeze abstracts without changing source judgments."""

import argparse
import json
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pda.finetuning_data import base_id, hydrate, load_annotations, split_queries
from pda.retrieval import ArxivClient, RetrievalError


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("evaluation/finetuning/prepared.json"))
    args = parser.parse_args()
    human, human_hashes = load_annotations(Path("evaluation/human_eval"), "human")
    synthetic, synthetic_hashes = load_annotations(Path("evaluation/syntetic_eval"), "synthetic")
    identifiers = sorted({j["arxiv_id"] for q in [*human, *synthetic] for j in q["judgments"]})
    client = ArxivClient()
    records, batches = {}, []

    def fetch(batch):
        try:
            papers, metadata = client.fetch_ids(batch)
        except RetrievalError as exc:
            if (
                len(batch) == 1
                or not isinstance(exc.__cause__, httpx.HTTPStatusError)
                or exc.__cause__.response.status_code != 400
            ):
                raise
            midpoint = len(batch) // 2
            fetch(batch[:midpoint])
            fetch(batch[midpoint:])
            return
        records.update({base_id(p.arxiv_id): p.model_dump(mode="json") for p in papers})
        batches.append({"requested_ids": batch, "metadata": metadata})

    for start in range(0, len(identifiers), 50):
        fetch(identifiers[start : start + 50])
        print(
            f"Verified metadata batch {min(start + 50, len(identifiers))}/{len(identifiers)}",
            flush=True,
        )
    prepared_human, excluded_human = hydrate(human, records)
    prepared_synthetic, excluded_synthetic = hydrate(synthetic, records)
    splits = split_queries(prepared_synthetic, prepared_human)
    output = {
        "source_hashes": {**human_hashes, **synthetic_hashes},
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
