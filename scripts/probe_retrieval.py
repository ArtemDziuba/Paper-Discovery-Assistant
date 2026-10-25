"""Live diagnosis for the reported transformer query; no paper injection."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pda.retrieval import ArxivClient


def main():
    client = ArxivClient()
    queries = {
        "original": 'all:"am" OR all:"looking" OR all:"information" OR all:"transformer" OR all:"architecture"',
        "focused_or": 'all:"transformer" OR all:"architecture"',
        "focused_and": 'all:"transformer" AND all:"architecture"',
        "phrase": 'all:"transformer architecture"',
        "quoted_title": 'all:"Attention Is All You Need"',
    }
    report = {}
    for label, query in queries.items():
        papers, metadata = client.search(query, 50)
        report[label] = {
            "query": query,
            "metadata": metadata,
            "target_position": next(
                (i + 1 for i, p in enumerate(papers) if p.arxiv_id.startswith("1706.03762")), None
            ),
            "papers": [{"id": p.arxiv_id, "title": p.title} for p in papers],
        }
        print(label, "target position", report[label]["target_position"], flush=True)
    for label, query in [
        ("earliest_and", queries["focused_and"]),
        ("earliest_abstract", 'abs:"transformer" AND abs:"architecture"'),
    ]:
        papers, metadata = client.search(query, 50, sort_by="submittedDate", sort_order="ascending")
        report[label] = {
            "query": query,
            "metadata": metadata,
            "target_position": next(
                (i + 1 for i, p in enumerate(papers) if p.arxiv_id.startswith("1706.03762")), None
            ),
            "papers": [{"id": p.arxiv_id, "title": p.title} for p in papers],
        }
        print(label, "target position", report[label]["target_position"], flush=True)
    Path("evaluation/retrieval-diagnosis.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
