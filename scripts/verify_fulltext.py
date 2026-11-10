"""Live PDF/extraction check using local layout inference, with no Qwen inference."""

import argparse
import json
import re
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pda.fulltext import PdfStore, atomic_write, chunk_document, extract_pdf
from pda.summarization import EXTRACT, FINAL, REDUCE, messages
from pda.summary_config import PDF_PARSER_VERSION, PIPELINE_VERSION, PROJECT_ROOT, SummarySettings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arxiv-id", default="1706.03762v7")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "evaluation" / "summarization" / "fulltext-smoke.json",
    )
    args = parser.parse_args()
    settings = SummarySettings.from_env()
    path, sha = PdfStore(settings).get(args.arxiv_id)
    started = time.perf_counter()
    document = extract_pdf(path, settings)
    extraction_seconds = time.perf_counter() - started
    query = "Efficient transformers for long document processing"
    title = "Live PDF extraction smoke check"
    overhead = max(
        sum(len(m["content"].encode()) for m in messages(query, title, "", instruction))
        for instruction in (FINAL, EXTRACT, REDUCE)
    )
    budget = (settings.num_ctx - settings.num_predict - settings.template_reserve - overhead) // 2
    chunks = chunk_document(document, budget, settings.max_chunks)
    # Verify ordered word coverage after stripping only our added source labels.
    # Table splitting may repeat column headers, so allow extra context words.
    reconstructed = re.sub(r"\[Section: [^\n]*; page \d+\]\n", "", "\n\n".join(chunks))
    original_words = " ".join(unit.text for unit in document.units).split()
    chunk_words = reconstructed.split()
    position = 0
    for word in chunk_words:
        if position < len(original_words) and word == original_words[position]:
            position += 1
    covered = position == len(original_words)
    if not covered:
        raise RuntimeError("Chunking lost or reordered extracted words")
    report = {
        "executed_at": datetime.now(UTC).isoformat(),
        "kind": "live PDF retrieval/extraction/chunking smoke check; local layout inference, no Qwen inference or summary-quality judgments",
        "arxiv_id": args.arxiv_id,
        "pdf_sha256": sha,
        "pdf_bytes": path.stat().st_size,
        "pipeline_version": PIPELINE_VERSION,
        "parser_version": PDF_PARSER_VERSION,
        "extraction_seconds": extraction_seconds,
        "pages": document.pages,
        "text_characters": len(document.text),
        "extracted_words": len(" ".join(unit.text for unit in document.units).split()),
        "sections": list(dict.fromkeys(unit.section for unit in document.units)),
        "unit_kinds": dict(Counter(unit.kind for unit in document.units)),
        "table_pages": [unit.page for unit in document.units if unit.kind == "table"],
        "bibliography_removed": document.bibliography_removed,
        "removed_artifact_lines": document.removed_artifact_lines,
        "chunks": len(chunks),
        "chunk_budget_bytes": budget,
        "largest_chunk_bytes": max(len(chunk.encode()) for chunk in chunks),
        "all_extracted_words_preserved_in_order": covered,
        "word_coverage_rule": "Original words are an ordered subsequence; repeated table headers are allowed.",
        "additional_chunk_words": len(chunk_words) - len(original_words),
        "settings": settings.model_dump(mode="json", exclude={"cache_dir"}),
    }
    atomic_write(args.output, json.dumps(report, indent=2).encode())
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
