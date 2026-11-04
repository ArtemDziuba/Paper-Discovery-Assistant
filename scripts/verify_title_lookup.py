"""Known-item regression: title variants must retrieve and show the real original first."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pda.pipeline import search
from pda.ranking import MODEL, REVISION, SemanticEncoder
from pda.retrieval import ArxivClient


def main() -> None:
    client = ArxivClient()
    encoder = SemanticEncoder("cuda")
    cases = {}
    for query in [
        "attention is all you need",
        '"attention is all you need"',
        "“Attention Is All You Need”",
    ]:
        result = search(query, client, encoder, limit=100, top_k=10)
        first = result.matches[0] if result.matches else None
        if first is None or not first.paper.arxiv_id.startswith("1706.03762"):
            raise AssertionError(f"Known title did not rank first for {query!r}")
        cases[query] = {
            "first_id": first.paper.arxiv_id,
            "first_title": first.paper.title,
            "raw_similarity": first.raw_similarity,
            "metadata": result.metadata,
            "interpretation": result.interpretation.model_dump(),
        }
        print(query, "->", first.paper.arxiv_id, first.paper.title, flush=True)
    output = {
        "model": MODEL,
        "revision": REVISION,
        "device": "cuda",
        "cases": cases,
        "scope": "One development known-item regression, not a held-out quality benchmark.",
    }
    Path("evaluation/title-lookup-regression.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
