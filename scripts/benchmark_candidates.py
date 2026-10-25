"""Live candidate-size diagnostic for the user-reported topic, not a general quality eval."""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from pda.pipeline import search
from pda.query import interpret
from pda.ranking import MODEL, REVISION, SemanticEncoder, cosine_scores
from pda.resources import peak_working_set_bytes
from pda.retrieval import ArxivClient

QUERY = "i am looking for information about transformer architecture"


def main() -> None:
    started = time.perf_counter()
    papers, metadata = ArxivClient().search(
        interpret(QUERY).search_query,
        10000,
        progress=lambda _, text: print(text, flush=True),
    )
    retrieval_seconds = time.perf_counter() - started
    encoder = SemanticEncoder("cuda")
    torch.cuda.reset_peak_memory_stats()

    class SnapshotClient:
        def search(self, query, limit):
            return papers[:limit], {"source": "prefix of the same retrieved snapshot"}

    runs = {}
    for limit in sorted({50, 100, 1000, 10000}):
        result = search(QUERY, SnapshotClient(), encoder, limit, 100)
        runs[str(limit)] = {
            "usable_candidates": result.metadata["candidate_count"],
            "ranking_seconds": result.metadata["ranking_seconds"],
            "target_in_candidates": any(
                p.arxiv_id.startswith("1706.03762") for p in papers[:limit]
            ),
            "target_rank_in_top100": next(
                (
                    i + 1
                    for i, m in enumerate(result.matches)
                    if m.paper.arxiv_id.startswith("1706.03762")
                ),
                None,
            ),
            "top10": [
                {"id": m.paper.arxiv_id, "title": m.paper.title, "raw_similarity": m.raw_similarity}
                for m in result.matches[:10]
            ],
        }
        print(
            limit,
            "candidates:",
            runs[str(limit)]["usable_candidates"],
            "ranking seconds:",
            runs[str(limit)]["ranking_seconds"],
            flush=True,
        )
    sample = [p.text for p in papers[:100]]
    agreement = None
    if sample:
        import numpy as np

        q, vectors = encoder.encode_pair(QUERY, sample)
        agreement = float(
            np.max(np.abs(encoder.score_documents(QUERY, sample) - cosine_scores(q, vectors)))
        )
        if agreement > 0.0001:
            raise AssertionError(f"Chunking changed scores beyond tolerance: {agreement}")
    report = {
        "query": QUERY,
        "search_query": interpret(QUERY).search_query,
        "requested_candidates": 10000,
        "max_chunked_legacy_score_difference_first100": agreement,
        "retrieval_seconds": retrieval_seconds,
        "retrieval_metadata": metadata,
        "model": MODEL,
        "revision": REVISION,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "peak_process_working_set_bytes": peak_working_set_bytes(),
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
        "scope": "Single diagnostic process, excludes Streamlit and other processes. "
        "Prefix comparisons use one snapshot; no independent relevance judgments.",
        "runs": runs,
    }
    Path("evaluation/candidate-scaling.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
