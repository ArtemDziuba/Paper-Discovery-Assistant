"""Compare rankers on full selected synthetic batches using frozen verified metadata."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from train_ranker import evaluate

from pda.finetuning_data import hydrate, load_annotations
from pda.ranking import FINETUNED_PATH, LexicalEncoder, SemanticEncoder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 5, 7, 8, 13, 15])
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument(
        "--output", type=Path, default=Path("evaluation/finetuning/selected-batches.json")
    )
    args = parser.parse_args()
    prepared_path = Path("evaluation/finetuning/prepared.json")
    prepared_raw = prepared_path.read_bytes()
    prepared = json.loads(prepared_raw)
    synthetic, hashes = load_annotations(Path("evaluation/syntetic_eval"), "synthetic")
    by_id = {q["query_id"]: q for q in synthetic}
    selected, descriptions = [], []
    records = {}
    for split in ["train", "synthetic_dev", "human_test"]:
        for query in prepared["splits"][split]:
            records.update({j["arxiv_id"]: j["paper"] for j in query["judgments"]})
    for number in args.batches:
        path = Path(f"evaluation/syntetic_eval/synthetic_finetuning_batch_{number:02}.json")
        digest = hashes[str(path)]
        if digest != prepared["source_hashes"].get(str(path)):
            raise ValueError(f"Source changed after preparation: {path}")
        contents = json.loads(path.read_text(encoding="utf-8-sig"))
        rows = contents if isinstance(contents, list) else [contents]
        for row in rows:
            query = by_id[row["query_id"]]
            hydrated, excluded = hydrate([query], records)
            if excluded:
                raise ValueError(f"Missing verified metadata in full batch {number}: {excluded}")
            selected.extend(hydrated)
            role = next(
                (
                    split
                    for split in ["train", "synthetic_dev"]
                    if any(q["query_id"] == query["query_id"] for q in prepared["splits"][split])
                ),
                "excluded",
            )
            descriptions.append(
                {
                    "batch": number,
                    "query_id": query["query_id"],
                    "query": query["query"],
                    "candidates": len(query["judgments"]),
                    "role_during_finetuning": role,
                    "source_sha256": digest,
                }
            )
    results = {}
    factories = {
        "tfidf": LexicalEncoder,
        "semantic": lambda: SemanticEncoder(args.device),
        "finetuned": lambda: SemanticEncoder(args.device, model_path=FINETUNED_PATH),
    }
    for name, factory in factories.items():
        encoder = factory()
        results[name] = evaluate(encoder, selected)
        print(name, results[name]["aggregate"], flush=True)
        del encoder
    report = {
        "prepared_sha256": hashlib.sha256(prepared_raw).hexdigest(),
        "batches": descriptions,
        "device": args.device,
        "finetuned_manifest": json.loads((FINETUNED_PATH / "training_manifest.json").read_text()),
        "protocol": "All 50 judged papers per batch, including overlap-filtered papers; "
        "macro average across queries; no title priority or live retrieval. "
        "These are training/development queries, not held-out test queries.",
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for row in descriptions:
        key = row["query_id"]
        print(
            row["batch"],
            row["query"],
            {
                name: value["per_query"][key]["metrics"]["ndcg@10"]
                for name, value in results.items()
            },
        )


if __name__ == "__main__":
    main()
