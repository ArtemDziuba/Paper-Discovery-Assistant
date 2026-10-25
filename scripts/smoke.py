"""Live API and local model smoke check; not an NLP quality evaluation."""

import json
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pda.pipeline import search
from pda.ranking import LexicalEncoder, SemanticEncoder
from pda.resources import peak_working_set_bytes
from pda.retrieval import ArxivClient


def main():
    report = {}
    client = ArxivClient()
    for method, factory in [("lexical", LexicalEncoder), ("semantic", SemanticEncoder)]:
        try:
            result = search("graph neural networks", client, factory(), limit=10, top_k=3)
            report[method] = {
                "ok": True,
                "papers": len(result.matches),
                "ids": [m.paper.arxiv_id for m in result.matches],
                "metadata": result.metadata,
            }
        except (OSError, RuntimeError, ValueError) as exc:
            report[method] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps({method: report[method]}), flush=True)
    report["resources"] = {
        "peak_process_working_set_bytes": peak_working_set_bytes(),
        "scope": "Smoke process only, 10 candidates, CPU inference; excludes UI and VRAM.",
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    output = Path("evaluation/smoke.json")
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
