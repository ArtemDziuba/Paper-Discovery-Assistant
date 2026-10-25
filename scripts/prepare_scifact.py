"""Download BEIR SciFact and convert its test split to the local evaluation schema."""

import hashlib
import io
import json
import zipfile
from pathlib import Path

import httpx

URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"


def main():
    response = httpx.get(URL, timeout=120, follow_redirects=True)
    response.raise_for_status()
    raw = response.content
    archive = zipfile.ZipFile(io.BytesIO(raw))

    def rows(name):
        return [json.loads(line) for line in archive.read(name).decode().splitlines()]

    corpus = rows("scifact/corpus.jsonl")
    query_rows = rows("scifact/queries.jsonl")
    qrels = {}
    for line in archive.read("scifact/qrels/test.tsv").decode().splitlines()[1:]:
        query, document, grade = line.split("\t")
        qrels.setdefault(query, {})[document] = int(grade)
    data = {
        "provenance": {
            "source": URL,
            "split": "test",
            "archive_sha256": hashlib.sha256(raw).hexdigest(),
            "caution": "Biomedical claim retrieval; not live arXiv recall.",
        },
        "documents": [{"id": d["_id"], "title": d["title"], "abstract": d["text"]} for d in corpus],
        "queries": [{"id": q["_id"], "text": q["text"]} for q in query_rows if q["_id"] in qrels],
        "qrels": qrels,
    }
    target = Path("evaluation/data/scifact-test.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data), encoding="utf-8")
    print(f"Saved {len(corpus)} documents and {len(qrels)} queries to {target}")


if __name__ == "__main__":
    main()
