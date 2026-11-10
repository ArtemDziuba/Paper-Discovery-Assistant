"""Validate user judgments, verify source metadata, and keep evaluation out of training."""

import hashlib
import json
import math
import random
import re
from collections.abc import Callable, Iterable
from difflib import SequenceMatcher
from pathlib import Path

import httpx

from pda.query import normalize_title
from pda.retrieval import ArxivClient, RetrievalError


def base_id(identifier: str) -> str:
    return re.sub(r"v\d+$", "", identifier)


def load_annotations(
    folder: Path,
    source: str,
    *,
    deduplicate: bool = True,
    files: Iterable[Path] | None = None,
) -> tuple[list[dict], dict]:
    queries = {} if deduplicate else None
    preserved = [] if not deduplicate else None
    hashes = {}
    for path in sorted(folder.glob("*.json") if files is None else files):
        raw = path.read_bytes()
        hashes[str(path)] = hashlib.sha256(raw).hexdigest()
        contents = json.loads(raw.decode("utf-8-sig"))
        rows = contents if isinstance(contents, list) else [contents]
        for row in rows:
            identifier, text = row["query_id"], row["query"]
            if not isinstance(identifier, str) or not identifier.strip() or not text.strip():
                raise ValueError(f"Empty query identity in {path}")
            judgments = {} if deduplicate else None
            occurrences = [] if not deduplicate else None
            for item in row["judgments"]:
                grade = item["relevance"]
                if type(grade) is not int or grade not in range(4):
                    raise ValueError(f"Expected integer relevance 0–3 in {path}")
                if deduplicate:
                    paper_id = base_id(item["arxiv_id"])
                    if paper_id in judgments and judgments[paper_id] != item:
                        raise ValueError(f"Conflicting paper judgment {paper_id} in {path}")
                    judgments[paper_id] = dict(item)
                else:
                    occurrences.append(dict(item))
            record = {
                "query_id": identifier,
                "query": text,
                "judgments": list(judgments.values()) if deduplicate else occurrences,
                "source": source,
            }
            if not deduplicate:
                preserved.append({**record, "source_file": str(path)})
                continue
            if identifier in queries and queries[identifier] != record:
                raise ValueError(f"Conflicting duplicate query {identifier}")
            queries[identifier] = record
    if not queries and not preserved:
        raise ValueError(f"No annotation files in {folder}")
    return list(queries.values()) if deduplicate else preserved, hashes


def v1_batch_paths(folder: Path, batches: Iterable[int] = range(1, 16)) -> list[Path]:
    """Select only the original fifteen synthetic training batches, in caller order."""
    numbers = list(batches)
    if not numbers or any(type(number) is not int or not 1 <= number <= 15 for number in numbers):
        raise ValueError("V1 batches must be numbered 1 through 15.")
    if len(set(numbers)) != len(numbers):
        raise ValueError("Select each v1 batch only once.")
    paths = [folder / f"synthetic_finetuning_batch_{number:02}.json" for number in numbers]
    for path in paths:
        if not path.is_file():
            raise ValueError(f"Missing original v1 training batch: {path}")
    return paths


def match_frozen_source(path: Path, digest: str, source_hashes: dict[str, str]) -> str:
    """Permit relocation only for an unambiguous filename with identical frozen bytes."""
    candidates = [
        stored
        for stored in source_hashes
        if Path(stored.replace("\\", "/")).name.casefold() == path.name.casefold()
    ]
    if len(candidates) != 1:
        raise ValueError(f"No unambiguous frozen source filename for {path}")
    stored = candidates[0]
    if source_hashes[stored] != digest:
        raise ValueError(f"Source changed after preparation: {path}")
    return stored


def fetch_metadata(
    client: ArxivClient,
    identifiers: list[str],
    *,
    strict_singletons: bool = False,
    batch_size: int = 50,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[dict[str, dict], list[dict]]:
    """Fetch verified metadata, splitting HTTP-400 batches without changing source rows."""
    if batch_size < 1:
        raise ValueError("Metadata batch size must be positive.")
    records, batches = {}, []

    def fetch(batch: list[str]) -> None:
        try:
            papers, metadata = client.fetch_ids(batch)
        except RetrievalError as exc:
            if (
                not isinstance(exc.__cause__, httpx.HTTPStatusError)
                or exc.__cause__.response.status_code != 400
            ):
                raise
            if len(batch) == 1:
                if strict_singletons:
                    raise
                batches.append({"requested_ids": batch, "error": str(exc)})
                return
            midpoint = len(batch) // 2
            fetch(batch[:midpoint])
            fetch(batch[midpoint:])
            return
        records.update({base_id(p.arxiv_id): p.model_dump(mode="json") for p in papers})
        batches.append({"requested_ids": batch, "metadata": metadata})

    for start in range(0, len(identifiers), batch_size):
        fetch(identifiers[start : start + batch_size])
        if progress:
            progress(min(start + batch_size, len(identifiers)), len(identifiers))
    return records, batches


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


def make_preferences(
    queries: list[dict], seed: int, per_query: int = 64, *, margin_per_grade: float = 0.1
) -> list[tuple]:
    if not math.isfinite(margin_per_grade) or margin_per_grade < 0:
        raise ValueError("Preference margin per grade must be finite and nonnegative.")
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
                    margin_per_grade * (better["relevance"] - worse["relevance"]),
                )
            )
    if not samples:
        raise ValueError("No graded preferences remain after source verification and splitting.")
    return samples
