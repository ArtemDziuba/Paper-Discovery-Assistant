"""Frozen dataset selection and durable records for summary quality evaluation."""

import hashlib
import json
import math
import os
from datetime import UTC, datetime
from pathlib import Path

from pda.finetuning_data import load_annotations
from pda.fulltext import atomic_write, checked_id
from pda.summary_config import digest

SCHEMA_VERSION = 1
WEIGHTS = {
    "factual_consistency": 0.35,
    "query_relevance": 0.30,
    "key_information_coverage": 0.20,
    "clarity": 0.10,
    "conciseness": 0.05,
}


def timestamp() -> str:
    return datetime.now(UTC).isoformat()


def error(stage: str, exc: Exception | str) -> dict:
    message = str(exc)
    secret = os.environ.get("GOOGLE_API_KEY")
    if secret:
        message = message.replace(secret, "[REDACTED]")
    return {
        "stage": stage,
        "type": type(exc).__name__ if isinstance(exc, Exception) else "ValidationError",
        "message": message,
        "at": timestamp(),
    }


def save_record(path: Path, record: dict) -> None:
    atomic_write(
        path,
        (json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"),
    )


def load_instances(evaluation_dir: Path) -> list[dict]:
    """Select three fixed slots per query, without deduplication or substitutions."""
    canonical = evaluation_dir / "synthetic_eval"
    legacy = evaluation_dir / "syntetic_eval"
    if canonical.exists() and legacy.exists():
        raise ValueError(
            "Both synthetic_eval and syntetic_eval exist; choose one dataset directory."
        )
    synthetic = canonical if canonical.exists() else legacy
    instances = []
    for source, folder, expected in (
        ("human_eval", evaluation_dir / "human_eval", 2),
        ("synthetic_eval", synthetic, 10),
    ):
        queries, hashes = load_annotations(folder, source, deduplicate=False)
        if len(queries) != expected:
            raise ValueError(f"Expected {expected} queries in {folder}, found {len(queries)}.")
        for query_index, row in enumerate(queries, 1):
            for paper_index in range(3):
                issues = []
                item = row["judgments"][paper_index] if paper_index < len(row["judgments"]) else {}
                identifier = item.get("arxiv_id")
                try:
                    if identifier is None:
                        raise ValueError("Selected paper is missing an arXiv ID.")
                    if not isinstance(identifier, str):
                        raise TypeError("Selected arXiv ID must be text.")
                    checked_id(identifier)
                except (ValueError, TypeError, RuntimeError) as exc:
                    issues.append(error("dataset", exc))
                if not item:
                    issues.append(
                        error("dataset", f"Missing selected paper slot {paper_index + 1}.")
                    )
                if item.get("relevance") != 3:
                    issues.append(
                        error("dataset", "Selected paper does not have relevance label 3.")
                    )
                if not item.get("title"):
                    issues.append(error("dataset", "Selected paper is missing its title."))
                instance = {
                    "schema_version": SCHEMA_VERSION,
                    "evaluation_id": f"summary_evaluation_{len(instances) + 1:02}",
                    "dataset_source": source,
                    "source_file": Path(row["source_file"]).name,
                    "source_sha256": hashes[row["source_file"]],
                    "query_index": query_index,
                    "paper_index": paper_index + 1,
                    "query_id": row["query_id"],
                    "query": row["query"],
                    "paper": {
                        "arxiv_id": identifier,
                        "title": item.get("title"),
                        "relevance_label": item.get("relevance"),
                    },
                    "dataset_errors": issues,
                }
                instance["selection_fingerprint"] = digest(
                    {k: instance[k] for k in ("dataset_source", "query_id", "query", "paper")}
                )
                instances.append(instance)
    return instances


def empty_evaluation() -> dict:
    return {
        "status": "incomplete",
        "framework": "deepeval",
        "judge_model": None,
        "evaluated_at": None,
        "metrics": {name: {"status": "pending", "score": None, "reason": None} for name in WEIGHTS},
        "weighted_score": None,
        "weighted_score_10": None,
        "duration_seconds": None,
        "errors": [],
        "configuration": None,
        "cache_key": None,
        "evidence": None,
    }


def new_record(instance: dict) -> dict:
    return {
        **instance,
        "summarization": {
            "status": "pending",
            "summary": None,
            "duration_seconds": None,
            "generated_at": None,
            "metadata": None,
            "errors": [],
        },
        "evaluation": empty_evaluation(),
    }


def matching_record(record: dict, instance: dict) -> bool:
    return (
        record.get("schema_version") == SCHEMA_VERSION
        and record.get("evaluation_id") == instance["evaluation_id"]
        and record.get("dataset_source") == instance["dataset_source"]
        and record.get("query_id") == instance["query_id"]
        and record.get("query") == instance["query"]
        and record.get("paper", {}).get("arxiv_id") == instance["paper"]["arxiv_id"]
    )


def read_record(path: Path, instance: dict) -> dict:
    if not path.exists():
        return new_record(instance)
    record = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(record, dict) or not matching_record(record, instance):
        raise ValueError(f"Cache identity mismatch in {path.name}; original cache was preserved.")
    for stage in ("summarization", "evaluation"):
        if not isinstance(record.get(stage), dict):
            raise TypeError(f"Invalid {stage} cache structure in {path.name}; file was preserved.")
        if not isinstance(record[stage].get("errors", []), list):
            raise TypeError(f"Invalid {stage} errors in {path.name}; file was preserved.")
    if not isinstance(record["evaluation"].get("metrics"), dict):
        raise TypeError(f"Invalid metric cache structure in {path.name}; file was preserved.")
    defaults = empty_evaluation()
    record["evaluation"] = {
        **defaults,
        **record["evaluation"],
        "metrics": {**defaults["metrics"], **record["evaluation"]["metrics"]},
    }
    # Keep generated text and scores; refresh dataset provenance, labels and display titles.
    return {**record, **instance}


def valid_summary(record: dict) -> bool:
    item = record.get("summarization", {})
    return (
        item.get("status") == "completed"
        and isinstance(item.get("summary"), str)
        and bool(item["summary"].strip())
    )


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def valid_metric(metric: dict | None) -> bool:
    if not isinstance(metric, dict) or metric.get("status", "completed") != "completed":
        return False
    score = metric.get("score")
    return (
        type(score) in (int, float)
        and math.isfinite(score)
        and 0 <= score <= 1
        and isinstance(metric.get("reason"), str)
        and bool(metric["reason"].strip())
    )


def weighted_score(metrics: dict) -> float | None:
    if not all(valid_metric(metrics.get(name)) for name in WEIGHTS):
        return None
    return sum(weight * metrics[name]["score"] for name, weight in WEIGHTS.items())


def recalculate(record: dict) -> None:
    evaluation = record["evaluation"]
    score = weighted_score(evaluation.get("metrics", {}))
    # A failed preparation, stale configuration, or changed summary cannot acquire an overall.
    if not valid_summary(record) or evaluation.get("status") not in {"completed", "incomplete"}:
        score = None
    evaluation["weighted_score"] = score
    evaluation["weighted_score_10"] = None if score is None else score * 10
    if evaluation.get("status") == "incomplete" and score is not None:
        evaluation["status"] = "completed"
        evaluation["evaluated_at"] = evaluation.get("evaluated_at") or timestamp()
    if evaluation.get("status") == "completed" and score is None:
        evaluation["status"] = "incomplete"
