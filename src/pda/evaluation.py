"""Ranking metrics with an explicit binary relevance threshold."""

import math
from itertools import combinations

EVALUATION_MODES = {
    "truncated": {
        "ndcg_cutoffs": (10,),
        "precision_cutoffs": (5,),
        "map_cutoffs": (10,),
        "pairwise_cutoff": None,
        "mrr_cutoff": None,
        "primary_metrics": ("ndcg@10", "precision@5"),
    },
    "full": {
        "ndcg_cutoffs": (5, 10, 20, 50),
        "precision_cutoffs": (5, 10, 20),
        "map_cutoffs": (10, 20, 50),
        "pairwise_cutoff": 50,
        "mrr_cutoff": 50,
        "primary_metrics": (),
    },
}


def evaluation_profile(mode: str) -> dict:
    if mode not in EVALUATION_MODES:
        raise ValueError("Evaluation mode must be truncated or full.")
    return dict(EVALUATION_MODES[mode])


def pairwise_metric_name(mode: str) -> str:
    cutoff = evaluation_profile(mode)["pairwise_cutoff"]
    return "pairwise_accuracy" if cutoff is None else f"pairwise_accuracy@{cutoff}"


def metric_definitions(mode: str = "truncated") -> dict[str, str]:
    profile = evaluation_profile(mode)
    definitions = {}
    for cutoff in profile["ndcg_cutoffs"]:
        definitions[f"ndcg@{cutoff}"] = (
            f"Graded gains 2^grade - 1 with log2(rank + 1) discounts, cutoff {cutoff}; "
            "ideal ordering uses the complete judgment pool."
        )
    for cutoff in profile["precision_cutoffs"]:
        definitions[f"precision@{cutoff}"] = (
            f"Relevant papers in the first {cutoff} positions, divided by {cutoff} "
            "even when fewer papers are returned."
        )
    for cutoff in profile["map_cutoffs"]:
        definitions[f"ap@{cutoff}"] = (
            f"Sum of precision@rank at relevant hits in ranks 1–{cutoff}, divided by "
            f"min({cutoff}, total relevant judgments); zero if no relevant judgments."
        )
        definitions[f"map@{cutoff}"] = f"Unweighted source-group mean of per-query AP@{cutoff}."
    scope = (
        "All scored, explicitly judged papers"
        if profile["pairwise_cutoff"] is None
        else f"Scored, explicitly judged papers within the first {profile['pairwise_cutoff']} ranks"
    )
    definitions[pairwise_metric_name(mode)] = (
        f"{scope}: unordered pairs with unequal grades; 1 for correct score order, "
        "0 for reversal, 0.5 for an exact score tie. Equal-grade pairs are excluded. "
        "No comparable pairs yields null; means omit null queries and report the valid count."
    )
    if profile["mrr_cutoff"] is not None:
        cutoff = profile["mrr_cutoff"]
        definitions[f"mrr@{cutoff}"] = (
            f"Reciprocal rank of the first relevant paper within ranks 1–{cutoff}; "
            "zero if no relevant paper appears within that cutoff."
        )
    return definitions


def pairwise_accuracy(judgments: dict[str, float], scores: dict[str, float]) -> float | None:
    """Measure graded ordering across the scored judged pool, independently of tie breaks."""
    scored = [(grade, float(scores[doc])) for doc, grade in judgments.items() if doc in scores]
    if any(not math.isfinite(score) for _, score in scored):
        raise ValueError("Pairwise accuracy requires finite scores.")
    correct, compared = 0.0, 0
    for (grade_a, score_a), (grade_b, score_b) in combinations(scored, 2):
        if grade_a == grade_b:
            continue
        compared += 1
        if score_a == score_b:
            correct += 0.5
        elif (grade_a > grade_b) == (score_a > score_b):
            correct += 1
    return correct / compared if compared else None


def aggregate_metrics(rows: list[dict[str, float | None]]) -> dict[str, float | None]:
    """Macro-average a consistent metric set, naming means of AP@k as MAP@k."""
    if not rows:
        return {}
    if any(row.keys() != rows[0].keys() for row in rows):
        raise ValueError("Cannot aggregate different evaluation metric sets.")
    aggregates = {}
    for name in rows[0]:
        values = [row[name] for row in rows if row[name] is not None]
        aggregates[f"m{name}" if name.startswith("ap@") else name] = (
            sum(values) / len(values) if values else None
        )
    return aggregates


def metrics(
    ranked: list[str],
    judgments: dict[str, float],
    relevant_min: float | None = None,
    *,
    scores: dict[str, float] | None = None,
    mode: str = "truncated",
) -> dict[str, float | None]:
    profile = evaluation_profile(mode)

    def dcg(values):
        return sum((2**value - 1) / math.log2(i + 2) for i, value in enumerate(values))

    def relevant(identifier):
        grade = judgments.get(identifier, 0)
        return grade > 0 if relevant_min is None else grade >= relevant_min

    result = {}
    ideal_grades = sorted(judgments.values(), reverse=True)
    for cutoff in profile["ndcg_cutoffs"]:
        ideal = dcg(ideal_grades[:cutoff])
        result[f"ndcg@{cutoff}"] = (
            dcg([judgments.get(doc, 0) for doc in ranked[:cutoff]]) / ideal if ideal else 0
        )
    for cutoff in profile["precision_cutoffs"]:
        result[f"precision@{cutoff}"] = sum(relevant(doc) for doc in ranked[:cutoff]) / cutoff
    relevant_count = sum(relevant(doc) for doc in judgments)
    for cutoff in profile["map_cutoffs"]:
        hits, precision_sum = 0, 0.0
        for rank, identifier in enumerate(ranked[:cutoff], start=1):
            if relevant(identifier):
                hits += 1
                precision_sum += hits / rank
        denominator = min(cutoff, relevant_count)
        result[f"ap@{cutoff}"] = precision_sum / denominator if denominator else 0
    # Without raw scores, treat the supplied ranking as a strict order.
    pair_scores = scores if scores is not None else {doc: -i for i, doc in enumerate(ranked)}
    pairwise_cutoff = profile["pairwise_cutoff"]
    if pairwise_cutoff is not None:
        selected = set(ranked[:pairwise_cutoff])
        pair_scores = {doc: score for doc, score in pair_scores.items() if doc in selected}
    result[pairwise_metric_name(mode)] = pairwise_accuracy(judgments, pair_scores)
    mrr_cutoff = profile["mrr_cutoff"]
    if mrr_cutoff is not None:
        result[f"mrr@{mrr_cutoff}"] = next(
            (1 / (i + 1) for i, doc in enumerate(ranked[:mrr_cutoff]) if relevant(doc)), 0
        )
    return result
