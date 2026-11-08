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

from pda.evaluation import (
    EVALUATION_MODES,
    aggregate_metrics,
    evaluation_profile,
    metric_definitions,
    metrics,
    pairwise_metric_name,
)
from pda.ranking import FINETUNED_PATH, LexicalEncoder, SemanticEncoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=["lexical", "semantic", "finetuned"], default="lexical")
    parser.add_argument("--model-path", type=Path, default=FINETUNED_PATH)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--mode", choices=list(EVALUATION_MODES), default="truncated")
    parser.add_argument(
        "--relevant-min", type=float, help="Binary metric threshold; default preserves grade > 0."
    )
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
    if args.method == "finetuned":
        encoder = SemanticEncoder(args.device, model_path=args.model_path)
    else:
        encoder = SemanticEncoder(args.device) if args.method == "semantic" else LexicalEncoder()
    texts = [f"{d['title']}\n{d['abstract']}" for d in documents]
    results = {}
    scores = encoder.score_queries([q["text"] for q in queries], texts)
    for query, query_scores in zip(queries, scores, strict=True):
        ranked = [ids[i] for i in np.argsort(-query_scores, kind="stable")]
        results[query["id"]] = metrics(
            ranked,
            qrels[query["id"]],
            relevant_min=args.relevant_min,
            scores=dict(zip(ids, map(float, query_scores), strict=True)),
            mode=args.mode,
        )
    report = {
        "dataset_sha256": hashlib.sha256(raw).hexdigest(),
        "method": args.method,
        "model": getattr(encoder, "model_name", None),
        "revision": getattr(encoder, "model_revision", None),
        "binary_relevance_rule": "grade > 0"
        if args.relevant_min is None
        else f"grade >= {args.relevant_min}",
        "corpus_size": len(ids),
        "query_count": len(queries),
        "elapsed_seconds": time.perf_counter() - started,
        "per_query": results,
        "unjudged_convention": "zero relevance; incomplete judgments limit interpretation",
        "evaluation_mode": args.mode,
        "evaluation_configuration": evaluation_profile(args.mode),
        "metric_definitions": metric_definitions(args.mode),
        "pairwise_valid_query_count": sum(
            row[pairwise_metric_name(args.mode)] is not None for row in results.values()
        ),
        "aggregate": aggregate_metrics(list(results.values())),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["aggregate"], indent=2))


if __name__ == "__main__":
    main()
