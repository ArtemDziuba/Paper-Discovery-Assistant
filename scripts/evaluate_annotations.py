"""Evaluate human labels with all three methods and synthetic labels with the two baselines."""

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
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument(
        "--prepared", type=Path, default=Path("evaluation/finetuning/prepared.json")
    )
    parser.add_argument("--output", type=Path, default=Path("evaluation/ranking-comparison.json"))
    args = parser.parse_args()
    prepared_raw = args.prepared.read_bytes()
    prepared = json.loads(prepared_raw)
    records = {}
    for name in ["train", "synthetic_dev", "human_test"]:
        for query in prepared["splits"][name]:
            records.update({j["arxiv_id"]: j["paper"] for j in query["judgments"]})
    datasets, exclusions, source_hashes = {}, {}, {}
    known_exclusions = {(e["query_id"], e["arxiv_id"]): e for e in prepared["exclusions"]}
    source_counts = {}
    for name, folder in [
        ("human", "evaluation/human_eval"),
        ("synthetic", "evaluation/syntetic_eval"),
    ]:
        annotations, hashes = load_annotations(Path(folder), name)
        for path, digest in hashes.items():
            if prepared["source_hashes"].get(path) != digest:
                raise ValueError(
                    f"Annotations changed: {path}. Refresh metadata preparation first."
                )
        source_hashes.update(hashes)
        source_counts[name] = sum(len(q["judgments"]) for q in annotations)
        datasets[name], rejected = hydrate(annotations, records)
        exclusions[name] = []
        for item in rejected:
            key = item["query_id"], item["arxiv_id"]
            if key not in known_exclusions:
                raise ValueError(f"Missing frozen metadata for {key}; prepare metadata first.")
            exclusions[name].append(known_exclusions[key])
        if any(not q["judgments"] for q in datasets[name]):
            raise ValueError(f"A {name} query has no verified candidates.")
    results = {"human": {}, "synthetic": {}}
    identities = {}
    factories = {
        "tfidf": LexicalEncoder,
        "semantic": lambda: SemanticEncoder(args.device),
        "finetuned": lambda: SemanticEncoder(args.device, model_path=FINETUNED_PATH),
    }
    for name, factory in factories.items():
        encoder = factory()
        identities[name] = {
            "model": getattr(encoder, "model_name", "tf-idf"),
            "revision": getattr(encoder, "model_revision", None),
        }
        results["human"][name] = evaluate(encoder, datasets["human"])
        print("human", name, results["human"][name]["aggregate"], flush=True)
        if name != "finetuned":
            results["synthetic"][name] = evaluate(encoder, datasets["synthetic"])
            print("synthetic", name, results["synthetic"][name]["aggregate"], flush=True)
        del encoder
    report = {
        "prepared_sha256": hashlib.sha256(prepared_raw).hexdigest(),
        "source_hashes": source_hashes,
        "source_judgment_counts": source_counts,
        "verified_judgment_counts": {
            name: sum(len(q["judgments"]) for q in queries) for name, queries in datasets.items()
        },
        "exclusions": exclusions,
        "device": args.device,
        "models": identities,
        "protocol": "Rank each query's explicitly judged pool by title+abstract cosine; "
        "no title-priority rule or live candidate retrieval. nDCG uses grades 0–3. "
        "Precision@5 and MRR use grade >=2. Macro averages weight queries equally. "
        "Full synthetic batches include overlap-filtered training papers. "
        "Synthetic results are synthetic-label agreement, not human ground truth.",
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Saved", args.output, flush=True)


if __name__ == "__main__":
    main()
