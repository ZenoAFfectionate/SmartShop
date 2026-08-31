"""Tests for examples.ShopSimulator.common (metrics, CI, slices)."""

from __future__ import annotations

import math

import pytest

from examples.ShopSimulator.common import (
    SUB_SCORES,
    _candidate_outcome,
    official_metrics,
    prompt_text,
    wilson_ci,
)


class TestPromptText:
    def test_string_passthrough(self):
        assert prompt_text("你好") == "你好"

    def test_list_of_user_messages(self):
        prompt = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "mid"},
            {"role": "user", "content": [{"type": "text", "text": "second"}]},
        ]
        assert prompt_text(prompt) == "first\nsecond"

    def test_no_user_text_raises(self):
        with pytest.raises(ValueError, match="no user text"):
            prompt_text([{"role": "system", "content": "sys"}])
        with pytest.raises(ValueError):
            prompt_text([])


class TestWilsonCi:
    def test_zero_total(self):
        assert wilson_ci(0, 0) == (0.0, 0.0)

    def test_extreme_proportions_stay_in_unit_interval(self):
        lo, hi = wilson_ci(0, 200)
        assert lo == 0.0 and 0 <= hi < 0.03
        lo, hi = wilson_ci(200, 200)
        assert 0.97 < lo <= 1.0 and hi == 1.0

    def test_brackets_point_estimate(self):
        n, k = 200, 19  # 9.5%
        lo, hi = wilson_ci(k, n)
        p = k / n
        assert lo < p < hi
        assert hi - lo < 0.1

    def test_matches_closed_form(self):
        n, k, z = 100, 50, 1.96
        p = k / n
        denom = 1 + z * z / n
        center = (p + z * z / (2 * n)) / denom
        half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
        assert wilson_ci(k, n, z) == pytest.approx((center - half, center + half))


def record(**overrides) -> dict:
    row = {
        "task_id": 1,
        "reward": 1.0,
        "reward_detail": {"r_type": 1.0, "r_att": 1.0, "r_option": 1.0, "r_price": True},
        "purchase_asin": "A1",
        "goal_asin": "A1",
        "model_turns": 7,
        "tool_calls": 9,
        "termination_reason": "environment_done",
        "category": "运动服",
        "attribute_count": 4,
        "option_count": 2,
    }
    row.update(overrides)
    return row


class TestCandidateOutcome:
    def test_perfect_outcome(self):
        outcome = _candidate_outcome(record())
        assert outcome["done"] == 1
        assert outcome["r_success"] == 1
        assert outcome["right_product"] == 1
        assert outcome["r_hard"] == 1.0  # all four sub-scores are 1 (True -> 1.0)

    def test_r_option_missing_defaults_to_one(self):
        # The deliberate default: optionless tasks have no r_option key, and a
        # missing r_option must NOT zero out r_hard (regression guard).
        detail = {"r_type": 1.0, "r_att": 1.0, "r_price": True}
        outcome = _candidate_outcome(record(reward_detail=detail))
        assert outcome["sub_scores"]["r_option"] == 1.0
        assert outcome["r_hard"] == 1.0
        assert outcome["r_success"] == 1

    def test_other_missing_sub_scores_default_to_zero(self):
        detail = {"r_option": 1.0}
        outcome = _candidate_outcome(record(reward_detail=detail))
        assert outcome["sub_scores"]["r_type"] == 0.0
        assert outcome["r_hard"] == 0.0
        assert outcome["r_success"] == 0

    def test_unscored_candidate(self):
        outcome = _candidate_outcome(record(reward_detail={}, reward=0.0))
        assert outcome["done"] == 0
        assert outcome["r_hard"] == 0.0
        for name in SUB_SCORES:
            assert outcome["sub_scores"][name] == 0.0

    def test_wrong_product(self):
        outcome = _candidate_outcome(record(purchase_asin="A2"))
        assert outcome["right_product"] == 0


class TestOfficialMetrics:
    def test_empty(self):
        assert official_metrics([]) == {"samples": 0}

    def test_base_metrics_and_efficiency(self):
        metrics = official_metrics([record(), record(reward=0.0, reward_detail={})])
        assert metrics["samples"] == 2
        assert metrics["done_rate"] == 0.5
        assert metrics["r_loose"] == 0.5
        assert metrics["r_success"] == 0.5
        assert metrics["mean_model_turns"] == 7.0
        assert metrics["mean_model_turns_done"] == 7.0  # only the done candidate
        assert metrics["mean_tool_calls"] == 9.0
        assert metrics["turn_limit_rate"] == 0.0

    def test_turn_limit_rate(self):
        rows = [record(termination_reason="turn_limit"), record()]
        assert official_metrics(rows)["turn_limit_rate"] == 0.5

    def test_confidence_intervals_match_wilson(self):
        rows = [record()] + [record(reward=0.0, reward_detail={}) for _ in range(9)]
        metrics = official_metrics(rows)
        lo, hi = wilson_ci(1, 10)
        assert metrics["r_success_ci95"] == [round(lo, 6), round(hi, 6)]
        lo, hi = wilson_ci(1, 10)  # one positive-reward row
        assert metrics["pass_positive_reward_ci95"] == [round(lo, 6), round(hi, 6)]

    def test_slice_by_category(self):
        rows = [
            record(category="A"),
            record(category="A", reward=0.0, reward_detail={}),
            record(category="B"),
        ]
        metrics = official_metrics(rows)
        assert metrics["by_category"]["A"]["samples"] == 2
        assert metrics["by_category"]["A"]["r_success"] == 0.5
        assert metrics["by_category"]["B"]["samples"] == 1
        assert metrics["by_category"]["B"]["r_success"] == 1.0

    def test_single_group_slice_is_skipped(self):
        metrics = official_metrics([record(category="A"), record(category="A")])
        assert "by_category" not in metrics

    def test_missing_slice_dimension_is_skipped(self):
        metrics = official_metrics([record(), record()])
        # All rows share the same attribute/option counts -> single group.
        assert "by_attribute_count" not in metrics

    def test_legacy_records_without_new_fields(self):
        legacy = {
            "task_id": 1,
            "reward": 1.0,
            "reward_detail": {"r_type": 1.0, "r_att": 1.0, "r_option": 1.0, "r_price": 1.0},
            "purchase_asin": "A1",
            "goal_asin": "A1",
        }
        metrics = official_metrics([legacy])
        assert metrics["samples"] == 1
        assert metrics["mean_model_turns"] is None
        assert metrics["mean_tool_calls"] is None
        assert "by_category" not in metrics
        assert metrics["pass_at_k"] is not None

    def test_pass_at_k_grouping(self):
        rows = [
            record(task_id=1, reward=0.0, reward_detail={}),
            record(task_id=1, reward=0.5, reward_detail={}),
            record(task_id=2, reward=0.0, reward_detail={}),
            record(task_id=2, reward=0.0, reward_detail={}),
        ]
        pass_at_k = official_metrics(rows)["pass_at_k"]
        assert pass_at_k["tasks"] == 2
        assert pass_at_k["samples_per_task"] == 2
        assert pass_at_k["reward_variance_task_fraction"] == 0.5
        assert pass_at_k["all_zero_reward_task_fraction"] == 0.5
        assert pass_at_k["pass_positive_reward"] == 0.5

    def test_task_id_none_disables_pass_at_k(self):
        metrics = official_metrics([record(task_id=None)])
        assert metrics["pass_at_k"] is None
        assert metrics["samples"] == 1
