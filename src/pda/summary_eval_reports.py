"""Complete CSV/JSON reports and terminal tables; no inference or paper retrieval."""

import csv
import io
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from pda.fulltext import atomic_write
from pda.summary_eval_data import WEIGHTS, recalculate, save_record, timestamp, valid_metric

SCORE_NAMES = [*WEIGHTS, "weighted_score", "weighted_score_10"]


def describe(values: list[float]) -> dict:
    return {
        "count": len(values),
        "mean": statistics.fmean(values) if values else None,
        "std": statistics.stdev(values) if len(values) > 1 else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def statistics_for(records: list[dict]) -> dict:
    completed = [
        r
        for r in records
        if r["evaluation"]["status"] == "completed"
        and r["evaluation"]["weighted_score"] is not None
    ]
    scores = {}
    for name in SCORE_NAMES:
        scores[name] = describe(
            [
                r["evaluation"]["metrics"][name]["score"]
                if name in WEIGHTS
                else r["evaluation"][name]
                for r in completed
            ]
        )
    # Explicitly separate partial metric diagnostics from the primary complete-case averages.
    partial = {
        name: describe(
            [
                r["evaluation"]["metrics"][name]["score"]
                for r in records
                if valid_metric(r["evaluation"].get("metrics", {}).get(name))
            ]
        )
        for name in WEIGHTS
    }
    durations = {}
    for stage in ("summarization", "evaluation"):
        durations[stage] = describe(
            [
                r[stage]["duration_seconds"]
                for r in records
                if r[stage].get("status") == "completed"
                and r[stage].get("duration_seconds") is not None
            ]
        )
    statuses = Counter(r["evaluation"]["status"] for r in records)
    return {
        "expected": len(records),
        "completed": len(completed),
        "failed": statuses["failed"],
        "incomplete": len(records) - len(completed) - statuses["failed"],
        "scores": scores,
        "available_metric_diagnostics": partial,
        "durations_seconds": durations,
        "std_convention": "sample standard deviation (ddof=1)",
        "averaging_policy": "Primary scores use completed evaluations only; counts are explicit.",
    }


def report_results(records: list[dict], output_dir: Path) -> dict:
    for record in records:
        recalculate(record)
    sources, queries = defaultdict(list), defaultdict(list)
    for record in records:
        sources[record["dataset_source"]].append(record)
        queries[(record["dataset_source"], record["query_id"])].append(record)
    report = {
        "created_at": timestamp(),
        "weights": WEIGHTS,
        "statistics": statistics_for(records),
        "by_source": {source: statistics_for(rows) for source, rows in sources.items()},
        "by_query": [
            {
                "dataset_source": source,
                "query_id": query_id,
                "query": rows[0]["query"],
                **statistics_for(rows),
            }
            for (source, query_id), rows in queries.items()
        ],
        "records": records,
    }
    save_record(output_dir / "summary_evaluation_results.json", report)
    columns = [
        "evaluation_id",
        "dataset_source",
        "query_id",
        "query",
        "arxiv_id",
        "paper_title",
        "relevance_label",
        *WEIGHTS,
        "weighted_score",
        "weighted_score_10",
        *(f"{name}_reason" for name in WEIGHTS),
        "summarization_duration_seconds",
        "evaluation_duration_seconds",
        "judge_model",
        "evaluation_status",
        "summarization_status",
        "errors",
    ]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    for record in records:
        evaluation = record["evaluation"]
        metrics = evaluation.get("metrics", {})
        row = {key: record[key] for key in columns[:4]}
        row.update(
            {
                "arxiv_id": record["paper"]["arxiv_id"],
                "paper_title": record["paper"]["title"],
                "relevance_label": record["paper"]["relevance_label"],
                "weighted_score": evaluation["weighted_score"],
                "weighted_score_10": evaluation["weighted_score_10"],
                "summarization_duration_seconds": record["summarization"].get("duration_seconds"),
                "evaluation_duration_seconds": evaluation.get("duration_seconds"),
                "judge_model": evaluation.get("judge_model"),
                "evaluation_status": evaluation["status"],
                "summarization_status": record["summarization"]["status"],
                "errors": str(
                    record.get("dataset_errors", [])
                    + record["summarization"].get("errors", [])
                    + evaluation.get("errors", [])
                ),
            }
        )
        for name in WEIGHTS:
            metric = metrics.get(name, {})
            row[name] = metric.get("score") if valid_metric(metric) else None
            row[f"{name}_reason"] = metric.get("reason")
        writer.writerow(row)
    atomic_write(output_dir / "summary_evaluation_results.csv", stream.getvalue().encode("utf-8"))
    return report


def display(value: float | None) -> str:
    return "--" if value is None else f"{value:.2f}"


def print_report(report: dict) -> None:
    print(
        "\nID Source    Query                    Paper                        "
        "Fact  Rel   Cover Clear Brief Overall Status"
    )
    for record in report["records"]:
        ev = record["evaluation"]
        scores = [
            display(ev["metrics"][name]["score"])
            if valid_metric(ev.get("metrics", {}).get(name))
            else "--"
            for name in WEIGHTS
        ]
        print(
            f"{record['evaluation_id'][-2:]} "
            f"{record['dataset_source'].removesuffix('_eval')[:9]:9} "
            f"{record['query'][:24]:24} {(record['paper']['title'] or '--')[:28]:28} "
            + " ".join(f"{score:5}" for score in scores)
            + f" {display(ev['weighted_score_10']):7} {ev['status']}"
        )
    print("\nMetric                        N     Mean      Std      Min      Max")
    for name, values in report["statistics"]["scores"].items():
        print(
            f"{name:28} {values['count']:3} "
            + " ".join(f"{display(values[k]):8}" for k in ("mean", "std", "min", "max"))
        )
    print("\nDataset / query                            Complete    Overall /10")
    for source, values in report["by_source"].items():
        print(
            f"{source:43} {values['completed']:3}/{values['expected']:<3}     "
            f"{display(values['scores']['weighted_score_10']['mean'])}"
        )
    for row in report["by_query"]:
        print(
            f"{row['dataset_source'] + '/' + row['query_id']:43.43} "
            f"{row['completed']:3}/{row['expected']:<3}     "
            f"{display(row['scores']['weighted_score_10']['mean'])}"
        )
    stats = report["statistics"]
    print(
        f"\nCompleted: {stats['completed']}; failed: {stats['failed']}; "
        f"incomplete: {stats['incomplete']}."
    )
    for stage, values in stats["durations_seconds"].items():
        print(f"Mean {stage} duration: {display(values['mean'])} s (n={values['count']}).")
