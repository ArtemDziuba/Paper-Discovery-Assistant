"""Evaluate a fixed judged corpus, independently of live arXiv candidate retrieval."""

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from pda.ranking import MODEL, REVISION, LexicalEncoder, SemanticEncoder


def metrics(ranked: list[str], judgments: dict[str, float]) -> dict[str, float]:
    def dcg(values):
        return sum((2**value - 1) / math.log2(i + 2) for i, value in enumerate(values))

    ideal = dcg(sorted(judgments.values(), reverse=True)[:10])
    return {
        "ndcg@10": dcg([judgments.get(i, 0) for i in ranked[:10]]) / ideal if ideal else 0,
        "precision@5": sum(judgments.get(i, 0) > 0 for i in ranked[:5]) / 5,
        "mrr": next((1 / (i + 1) for i, doc in enumerate(ranked) if judgments.get(doc, 0) > 0), 0),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=["lexical", "semantic"], default="lexical")
    args = parser.parse_args()
    raw = args.data.read_bytes()
    data = json.loads(raw)
    documents, queries, qrels = data["documents"], data["queries"], data["qrels"]
    ids = [d["id"] for d in documents]
    query_ids = [q["id"] for q in queries]
    if not ids or len(set(ids)) != len(ids) or not queries or len(set(query_ids)) != len(queries):
        raise ValueError("Require nonempty documents/queries with unique IDs.")
    for query in queries:
        judgments = qrels[query["id"]]
        if not judgments or not set(judgments) <= set(ids):
            raise ValueError("Judgments must reference corpus document IDs.")
        if not all(
            isinstance(v, (int, float)) and math.isfinite(v) and 0 <= v <= 10
            for v in judgments.values()
        ):
            raise ValueError("Relevance grades must be finite numbers between 0 and 10.")
        if not any(v > 0 for v in judgments.values()):
            raise ValueError("Each query must have a known relevant document.")
    started = time.perf_counter()
    encoder = SemanticEncoder() if args.method == "semantic" else LexicalEncoder()
    texts = [f"{d['title']}\n{d['abstract']}" for d in documents]
    results = {}
    scores = encoder.score_queries([q["text"] for q in queries], texts)
    for query, query_scores in zip(queries, scores, strict=True):
        ranked = [ids[i] for i in np.argsort(-query_scores, kind="stable")]
        results[query["id"]] = metrics(ranked, qrels[query["id"]])
    report = {
        "dataset_sha256": hashlib.sha256(raw).hexdigest(),
        "method": args.method,
        "model": MODEL if args.method == "semantic" else None,
        "revision": REVISION if args.method == "semantic" else None,
        "corpus_size": len(ids),
        "query_count": len(queries),
        "elapsed_seconds": time.perf_counter() - started,
        "per_query": results,
        "unjudged_convention": "zero relevance; incomplete judgments limit interpretation",
        "aggregate": {
            m: float(np.mean([r[m] for r in results.values()]))
            for m in next(iter(results.values()))
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["aggregate"], indent=2))


if __name__ == "__main__":
    main()
