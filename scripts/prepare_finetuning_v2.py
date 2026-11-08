"""Verify the 45 encoder-training batches without deduplicating their judgments."""

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prepare_evaluation import validate_snapshot

from pda.finetuning_data import base_id, hydrate, load_annotations
from pda.retrieval import ArxivClient, RetrievalError


def occurrence_audit(queries: list[dict]) -> dict:
    occurrences = defaultdict(list)
    for query in queries:
        for item in query["judgments"]:
            occurrences[base_id(item["arxiv_id"])].append(
                {"query_id": query["query_id"], "relevance": item["relevance"]}
            )
    return {
        "judgments": sum(len(q["judgments"]) for q in queries),
        "unique_paper_ids": len(occurrences),
        "repeated_papers": {key: rows for key, rows in occurrences.items() if len(rows) > 1},
        "papers_with_different_grades": sum(
            len({r["relevance"] for r in rows}) > 1 for rows in occurrences.values()
        ),
    }


def combine_development(datasets: dict[str, list[dict]]) -> list[dict]:
    """Preserve individual query pools while giving all sources one query-level mean."""
    return [
        {
            **query,
            "original_query_id": query["query_id"],
            "query_id": f"{source}:{query['query_id']}",
            "evaluation_source": source,
        }
        for source, queries in datasets.items()
        for query in queries
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, default=Path("finetune_data/encoder"))
    parser.add_argument(
        "--development", type=Path, default=Path("evaluation/annotations/prepared.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("evaluation/finetuning-v2/prepared.json")
    )
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Prepared data already exists; choose a new --output path.")
    training, hashes = load_annotations(args.training, "synthetic_training", deduplicate=False)
    development_raw = args.development.read_bytes()
    development = json.loads(development_raw)
    validate_snapshot(development)
    identifiers = sorted({base_id(j["arxiv_id"]) for q in training for j in q["judgments"]})
    client = ArxivClient()
    records, batches = {}, []

    def fetch(batch: list[str]) -> None:
        try:
            papers, metadata = client.fetch_ids(batch)
        except RetrievalError as exc:
            if (
                not isinstance(exc.__cause__, httpx.HTTPStatusError)
                or exc.__cause__.response.status_code != 400
            ):
                raise
            if len(batch) == 1:
                batches.append({"requested_ids": batch, "error": str(exc)})
                return
            midpoint = len(batch) // 2
            fetch(batch[:midpoint])
            fetch(batch[midpoint:])
            return
        records.update({base_id(p.arxiv_id): p.model_dump(mode="json") for p in papers})
        batches.append({"requested_ids": batch, "metadata": metadata})

    for start in range(0, len(identifiers), 50):
        fetch(identifiers[start : start + 50])
        print(
            f"arXiv verification: {min(start + 50, len(identifiers))}/{len(identifiers)} IDs",
            flush=True,
        )
    verified, exclusions = hydrate(training, records, threshold=0.85)
    if any(not q["judgments"] for q in verified):
        raise ValueError("A training batch has no verified papers; review its IDs and titles.")
    # Keep query pools separate for scoring; concatenate query groups for one macro mean.
    combined = combine_development(development["datasets"])
    training_ids = {base_id(j["arxiv_id"]) for q in verified for j in q["judgments"]}
    overlap = {
        q["query_id"]: sorted(training_ids & {base_id(j["arxiv_id"]) for j in q["judgments"]})
        for q in combined
    }
    output = {
        "created_at": datetime.now(UTC).isoformat(),
        "source_hashes": hashes,
        "training_folder": str(args.training),
        "development_snapshot_path": str(args.development),
        "development_snapshot_sha256": hashlib.sha256(development_raw).hexdigest(),
        "development_source_hashes": development["source_hashes"],
        "training": verified,
        "combined_development": combined,
        "development_counts": dict(Counter(q["evaluation_source"] for q in combined)),
        "title_agreement_threshold": 0.85,
        "metadata_batches": batches,
        "exclusions": exclusions,
        "source_occurrences": occurrence_audit(training),
        "verified_occurrences": occurrence_audit(verified),
        "development_training_paper_overlap": overlap,
        "policy": "All training query/judgment occurrences are preserved; no deduplication or overlap removal. "
        "Unique IDs are fetched once solely to reuse metadata. Unverified ID/title pairs are excluded and reported. "
        "The 12 evaluation queries form one equally weighted development set, used for checkpoint selection.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print("Saved", args.output, flush=True)
    print(
        "Training judgments:",
        output["verified_occurrences"]["judgments"],
        "excluded:",
        len(exclusions),
        flush=True,
    )


if __name__ == "__main__":
    main()
