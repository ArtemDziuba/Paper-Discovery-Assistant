"""Ranking metrics with an explicit binary relevance threshold."""

import math


def metrics(
    ranked: list[str], judgments: dict[str, float], relevant_min: float | None = None
) -> dict[str, float]:
    def dcg(values):
        return sum((2**value - 1) / math.log2(i + 2) for i, value in enumerate(values))

    ideal = dcg(sorted(judgments.values(), reverse=True)[:10])

    def relevant(identifier):
        grade = judgments.get(identifier, 0)
        return grade > 0 if relevant_min is None else grade >= relevant_min

    return {
        "ndcg@10": dcg([judgments.get(i, 0) for i in ranked[:10]]) / ideal if ideal else 0,
        "precision@5": sum(relevant(i) for i in ranked[:5]) / 5,
        "mrr": next((1 / (i + 1) for i, doc in enumerate(ranked) if relevant(doc)), 0),
    }
