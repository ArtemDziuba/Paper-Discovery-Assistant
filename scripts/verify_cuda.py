"""Check actual GPU inference and CPU/CUDA score agreement on cached real papers."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from pda.pipeline import search
from pda.ranking import MODEL, REVISION, SemanticEncoder
from pda.resources import peak_working_set_bytes
from pda.retrieval import ArxivClient


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; install and run with the cuda extra.")
    cpu = SemanticEncoder("cpu")
    gpu = SemanticEncoder("cuda")
    torch.cuda.reset_peak_memory_stats()
    client = ArxivClient()
    cases = {}
    for query in [
        "i am looking for information about transformer architecture",
        '"Attention Is All You Need"',
    ]:
        cpu_result = search(query, client, cpu, 50, 50)
        gpu_result = search(query, client, gpu, 50, 50)
        cpu_scores = {m.paper.arxiv_id: m.raw_similarity for m in cpu_result.matches}
        difference = max(
            (abs(m.raw_similarity - cpu_scores[m.paper.arxiv_id]) for m in gpu_result.matches),
            default=0,
        )
        if difference > 0.0001:
            raise AssertionError(f"Unexpected CPU/CUDA score difference: {difference}")
        cases[query] = {
            "search_query": gpu_result.interpretation.search_query,
            "candidate_count": gpu_result.metadata["candidate_count"],
            "target_rank": next(
                (
                    i + 1
                    for i, m in enumerate(gpu_result.matches)
                    if m.paper.arxiv_id.startswith("1706.03762")
                ),
                None,
            ),
            "max_cpu_cuda_score_difference": difference,
            "top_titles": [m.paper.title for m in gpu_result.matches[:5]],
        }
    report = {
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "model": MODEL,
        "revision": REVISION,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
        "peak_process_working_set_bytes": peak_working_set_bytes(),
        "scope": "Two encoders in one diagnostic process; excludes Streamlit and other apps.",
        "cases": cases,
    }
    Path("evaluation/cuda-verification.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
