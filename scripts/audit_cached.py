"""Structural audit of real cached arXiv metadata; not a human quality judgment."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pda.models import Paper
from pda.pipeline import evidence
from pda.query import interpret


def main():
    rows = []
    for path in Path(".cache/arxiv").glob("*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        papers = [Paper.model_validate(p) for p in data["papers"]]
        rows.extend(papers)
    if not rows:
        raise ValueError("No cached real papers to audit; run scripts/smoke.py first.")
    concepts = interpret("graph neural networks").concepts
    report = {
        "kind": "structural checks, not NLP quality judgments",
        "records": len(rows),
        "required_field_completeness": sum(bool(p.title and p.abstract and p.authors) for p in rows)
        / len(rows),
        "selected_excerpt_exact_source_membership": sum(
            evidence(p.abstract, concepts) in p.abstract for p in rows
        )
        / len(rows),
        "source_link_id_agreement": sum(
            str(p.url) == f"https://arxiv.org/abs/{p.arxiv_id}" for p in rows
        )
        / len(rows),
        "limitations": "Validated cache only; excludes rejected records. No independent human "
        "judgment of explanation relevance, intent, or overview coverage.",
    }
    target = Path("evaluation/structural-audit.json")
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
