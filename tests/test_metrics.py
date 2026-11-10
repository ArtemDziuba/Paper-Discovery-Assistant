"""Hand-calculated metric expectations; not an NLP quality experiment."""

import math

import pytest

from pda.evaluation import aggregate_metrics, evaluation_profile, metrics, pairwise_accuracy


def test_perfect_and_missed_ranking():
    assert metrics(["a", "b"], {"a": 1}) == {
        "ndcg@10": 1,
        "precision@5": 0.2,
        "ap@10": 1,
        "pairwise_accuracy": None,
    }
    assert metrics(["b", "a"], {"a": 1}, mode="full")["mrr@50"] == 0.5
    assert metrics(["b", "a"], {"a": 1})["ndcg@10"] == pytest.approx(0.6309297536)
    assert metrics(["b"], {"a": 1})["ndcg@10"] == 0


def test_ap_at_10_uses_relevance_threshold_and_precision_at_each_hit():
    result = metrics(["a", "b", "c", "d"], {"a": 3, "b": 1, "c": 2, "d": 0}, relevant_min=2)
    assert result["ap@10"] == pytest.approx((1 + 2 / 3) / 2)
    assert result["pairwise_accuracy"] == pytest.approx(5 / 6)
    assert metrics(["a", "b", "c"], {"a": 1, "b": 1, "c": 1})["ap@10"] == 1


def test_ap_at_10_denominator_includes_relevant_papers_not_returned():
    assert metrics(["a"], {"a": 1, "b": 1, "c": 1})["ap@10"] == pytest.approx(1 / 3)
    ranked = [f"doc{i}" for i in range(12)]
    assert metrics(ranked, dict.fromkeys(ranked, 2), relevant_min=2)["ap@10"] == 1
    assert metrics(ranked, {"doc10": 3}, relevant_min=2)["ap@10"] == 0
    assert metrics([], {"a": 3}, relevant_min=2)["ap@10"] == 0
    assert metrics(["a"], {"a": 1}, relevant_min=2)["ap@10"] == 0


def test_pairwise_accuracy_perfect_reversed_and_score_ties():
    judgments = {"a": 3, "b": 2, "c": 0}
    assert pairwise_accuracy(judgments, {"a": 2, "b": 1, "c": 0}) == 1
    assert pairwise_accuracy(judgments, {"a": 0, "b": 1, "c": 2}) == 0
    assert pairwise_accuracy(judgments, {"a": 1, "b": 1, "c": 0}) == pytest.approx(5 / 6)
    assert pairwise_accuracy(judgments, dict.fromkeys(judgments, 0.5)) == 0.5
    # All three pairs are compared, even though grades 3 and 2 are both binary relevant.
    assert pairwise_accuracy({"a": 3, "b": 2}, {"a": 0, "b": 1}) == 0


def test_pairwise_ignores_equal_grades_and_unjudged_or_unscored_papers():
    assert pairwise_accuracy({"a": 3, "b": 3, "c": 0}, {"a": 1, "b": 0, "c": 0.5}) == 0.5
    assert pairwise_accuracy({"a": 3, "b": 0}, {"a": 1, "b": 0, "unjudged": 2}) == 1
    assert pairwise_accuracy({"a": 3, "b": 0, "unscored": 2}, {"a": 1, "b": 0}) == 1
    assert pairwise_accuracy({"a": 3, "b": 3}, {"a": 1, "b": 0}) is None
    assert pairwise_accuracy({}, {}) is None
    with pytest.raises(ValueError, match="finite scores"):
        pairwise_accuracy({"a": 3, "b": 0}, {"a": float("nan"), "b": 0})


def test_pairwise_covers_full_pool_beyond_top_10():
    ranked = [f"doc{i}" for i in range(12)]
    judgments = dict.fromkeys(ranked, 3)
    judgments["doc10"], judgments["doc11"] = 0, 2
    # 10 high-grade papers each beat both tail papers; the last pair is reversed.
    assert metrics(ranked, judgments, relevant_min=2)["pairwise_accuracy"] == pytest.approx(20 / 21)


def test_macro_map_and_undefined_pairwise_averages():
    first = metrics(["a"], {"a": 3}, relevant_min=2)
    second = metrics(["c", "b"], {"b": 3, "c": 0}, relevant_min=2)
    aggregate = aggregate_metrics([first, second])
    assert aggregate["map@10"] == 0.75  # Per-query AP values 1.0 and 0.5.
    assert "ap@10" not in aggregate
    assert aggregate["pairwise_accuracy"] == 0
    assert aggregate_metrics([first])["pairwise_accuracy"] is None


def test_full_metric_set_and_hand_calculated_short_pool():
    result = metrics(["low", "high", "medium"], {"low": 0, "high": 3, "medium": 2}, 2, mode="full")
    assert set(result) == {
        "ndcg@5",
        "ndcg@10",
        "ndcg@20",
        "ndcg@50",
        "precision@5",
        "precision@10",
        "precision@20",
        "ap@10",
        "ap@20",
        "ap@50",
        "pairwise_accuracy@50",
        "mrr@50",
    }
    expected_ndcg = (7 / math.log2(3) + 3 / math.log2(4)) / (7 + 3 / math.log2(3))
    for cutoff in (5, 10, 20, 50):
        assert result[f"ndcg@{cutoff}"] == pytest.approx(expected_ndcg)
    for cutoff in (5, 10, 20):
        assert result[f"precision@{cutoff}"] == pytest.approx(2 / cutoff)
    for cutoff in (10, 20, 50):
        assert result[f"ap@{cutoff}"] == pytest.approx((1 / 2 + 2 / 3) / 2)
    assert result["pairwise_accuracy@50"] == pytest.approx(1 / 3)
    assert result["mrr@50"] == 0.5
    aggregate = aggregate_metrics([result])
    assert all(f"map@{cutoff}" in aggregate for cutoff in (10, 20, 50))
    assert not any(key.startswith("ap@") for key in aggregate)


@pytest.mark.parametrize("position", [5, 10, 20, 50, 51])
def test_full_cutoffs_include_boundary_and_exclude_later_hits(position):
    ranked = [f"doc{i}" for i in range(60)]
    result = metrics(ranked, {ranked[position - 1]: 3}, 2, mode="full")
    for cutoff in (5, 10, 20, 50):
        expected = 1 / math.log2(position + 1) if position <= cutoff else 0
        assert result[f"ndcg@{cutoff}"] == pytest.approx(expected)
    for cutoff in (5, 10, 20):
        assert result[f"precision@{cutoff}"] == (1 / cutoff if position <= cutoff else 0)
    for cutoff in (10, 20, 50):
        assert result[f"ap@{cutoff}"] == (1 / position if position <= cutoff else 0)
    assert result["mrr@50"] == (1 / position if position <= 50 else 0)


def test_pairwise_at_50_excludes_pairs_with_papers_below_cutoff():
    ranked = [f"doc{i}" for i in range(51)]
    judgments = dict.fromkeys(ranked, 3)
    judgments["doc49"], judgments["doc50"] = 2, 3
    truncated = metrics(ranked, judgments, 2)
    full = metrics(ranked, judgments, 2, mode="full")
    assert truncated["pairwise_accuracy"] == pytest.approx(49 / 50)
    assert full["pairwise_accuracy@50"] == 1
    # An undefined top-50 pool remains null, even if a comparable paper exists at rank 51.
    judgments["doc49"], judgments["doc50"] = 3, 0
    assert metrics(ranked, judgments, 2, mode="full")["pairwise_accuracy@50"] is None


def test_full_ap_denominator_uses_complete_judgments_at_each_cutoff():
    ranked = [f"doc{i}" for i in range(60)]
    judgments = dict.fromkeys(ranked, 2)
    result = metrics(ranked, judgments, 2, mode="full")
    assert all(result[f"ap@{cutoff}"] == 1 for cutoff in (10, 20, 50))
    result = metrics(ranked[:1], judgments, 2, mode="full")
    for cutoff in (10, 20, 50):
        assert result[f"ap@{cutoff}"] == 1 / cutoff


def test_profiles_reject_unknown_modes_and_mixed_aggregation():
    assert evaluation_profile("truncated")["primary_metrics"] == ("ndcg@10", "precision@5")
    assert len(metrics([], {})) == 4
    with pytest.raises(ValueError, match="mode"):
        metrics([], {}, mode="unknown")
    with pytest.raises(ValueError, match="different evaluation metric sets"):
        aggregate_metrics([metrics([], {}), metrics([], {}, mode="full")])
