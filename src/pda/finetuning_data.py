"""Validate user judgments, verify source metadata, and keep evaluation out of training."""

import hashlib
import json
import random
import re
from difflib import SequenceMatcher
from pathlib import Path

from pda.query import normalize_title


def base_id(identifier: str) -> str:
    return re.sub(r"v\d+$", "", identifier)


def load_annotations(folder: Path, source: str) -> tuple[list[dict], dict]:
    queries = {}
    hashes = {}
    for path in sorted(folder.glob("*.json")):
        raw = path.read_bytes()
        hashes[str(path)] = hashlib.sha256(raw).hexdigest()
        contents = json.loads(raw.decode("utf-8-sig"))
        rows = contents if isinstance(contents, list) else [contents]
        for row in rows:
            identifier, text = row["query_id"], row["query"]
            if not isinstance(identifier, str) or not identifier.strip() or not text.strip():
                raise ValueError(f"Empty query identity in {path}")
            judgments = {}
            for item in row["judgments"]:
                grade = item["relevance"]
                if type(grade) is not int or grade not in range(4):
                    raise ValueError(f"Expected integer relevance 0–3 in {path}")
                paper_id = base_id(item["arxiv_id"])
                if paper_id in judgments and judgments[paper_id] != item:
                    raise ValueError(f"Conflicting paper judgment {paper_id} in {path}")
                judgments[paper_id] = dict(item)
            record = {
                "query_id": identifier,
                "query": text,
                "judgments": list(judgments.values()),
                "source": source,
            }
            if identifier in queries and queries[identifier] != record:
                raise ValueError(f"Conflicting duplicate query {identifier}")
            queries[identifier] = record
    if not queries:
        raise ValueError(f"No annotation files in {folder}")
    return list(queries.values()), hashes


def title_agreement(expected: str, actual: str) -> float:
    def cleaned(text):
        return " ".join(re.findall(r"\w+", normalize_title(text)))

    return SequenceMatcher(None, cleaned(expected), cleaned(actual)).ratio()


def hydrate(queries: list[dict], records: dict, threshold: float = 0.85) -> tuple[list[dict], list]:
    prepared, exclusions = [], []
    for query in queries:
        items = []
        for item in query["judgments"]:
            identifier = base_id(item["arxiv_id"])
            record = records.get(identifier)
            if record is None:
                exclusions.append(
                    {
                        "query_id": query["query_id"],
                        "arxiv_id": item["arxiv_id"],
                        "reason": "metadata_not_returned",
                    }
                )
                continue
            agreement = title_agreement(item["title"], record["title"])
            if agreement < threshold:
                exclusions.append(
                    {
                        "query_id": query["query_id"],
                        "arxiv_id": item["arxiv_id"],
                        "reason": "title_mismatch",
                        "provided_title": item["title"],
                        "retrieved_title": record["title"],
                        "agreement": agreement,
                    }
                )
                continue
            items.append(
                {**item, "arxiv_id": identifier, "paper": record, "title_agreement": agreement}
            )
        prepared.append({**query, "judgments": items})
    return prepared, exclusions


def split_queries(synthetic: list[dict], human: list[dict], seed: int = 42) -> dict:
    if len(synthetic) < 4:
        raise ValueError("At least four synthetic query groups are required.")
    human_text = {normalize_title(q["query"]) for q in human}
    eligible = [q for q in synthetic if normalize_title(q["query"]) not in human_text]
    ordered = sorted(eligible, key=lambda q: q["query_id"])
    random.Random(seed).shuffle(ordered)
    dev_count = max(1, round(len(ordered) * 0.2))
    dev, train_candidates = ordered[:dev_count], ordered[dev_count:]
    blocked = {base_id(j["arxiv_id"]) for q in [*human, *dev] for j in q["judgments"]}
    train = []
    removed = []
    for query in train_candidates:
        items = []
        for item in query["judgments"]:
            if base_id(item["arxiv_id"]) in blocked:
                removed.append({"query_id": query["query_id"], "arxiv_id": item["arxiv_id"]})
            else:
                items.append(item)
        train.append({**query, "judgments": items})
    return {
        "train": train,
        "synthetic_dev": dev,
        "human_test": human,
        "removed_training_overlap": removed,
        "seed": seed,
        "excluded_synthetic_query_ids": [q["query_id"] for q in synthetic if q not in eligible],
    }


def make_preferences(queries: list[dict], seed: int, per_query: int = 64) -> list[tuple]:
    rng = random.Random(seed)
    samples = []
    for query in queries:
        pairs = [
            (a, b)
            for a in query["judgments"]
            for b in query["judgments"]
            if a["relevance"] > b["relevance"]
        ]
        rng.shuffle(pairs)
        for better, worse in pairs[:per_query]:

            def text(item):
                return item["paper"]["title"] + "\n" + item["paper"]["abstract"]

            samples.append(
                (
                    query["query"],
                    text(better),
                    text(worse),
                    0.1 * (better["relevance"] - worse["relevance"]),
                )
            )
    if not samples:
        raise ValueError("No graded preferences remain after source verification and splitting.")
    return samples
