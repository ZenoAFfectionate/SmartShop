"""Tests for examples.ShopSimulator.generate (GRPO normalization, group gate & helpers)."""

from __future__ import annotations

import argparse
import asyncio
import math

import pytest

from examples.ShopSimulator import generate as generate_module
from examples.ShopSimulator.generate import (
    _behavior_bonus,
    _candidate_fragments,
    _capture_rollout_events_enabled,
    _decomposed_group_advantages,
    _sample_prompt,
    _task_id,
    abort_sample,
    finish_candidate_session,
    make_unique_session_id,
    normalize_candidate_group_rewards,
    validate_complete_groups,
)
from examples.ShopSimulator.pi_harness import InfrastructureError, PiRunResult
from slime.utils.types import Sample


def make_args(**overrides) -> argparse.Namespace:
    args = argparse.Namespace(
        reward_key=None,
        n_samples_per_prompt=4,
        grpo_std_normalization=True,
        debug_rollout_only=False,
    )
    args.__dict__.update(overrides)
    return args


def make_sample(
    *,
    group_index: int,
    rollout_id: int,
    index: int,
    reward: float,
    trainable: bool = True,
    remove: bool = False,
    status: Sample.Status = Sample.Status.COMPLETED,
) -> Sample:
    return Sample(
        group_index=group_index,
        index=index,
        rollout_id=rollout_id,
        reward=reward,
        response="r",
        response_length=1,
        loss_mask=[1] if trainable else [0],
        remove_sample=remove,
        status=status,
        metadata={},
    )


def full_group(group_index: int, rewards: list[float]) -> list[Sample]:
    assert len(rewards) == 4
    return [
        make_sample(group_index=group_index, rollout_id=candidate, index=candidate, reward=reward)
        for candidate, reward in enumerate(rewards)
    ]


class TestNormalizeCandidateGroupRewards:
    def test_basic_group_centering_and_std(self):
        samples = full_group(0, [0.0, 0.0, 1.0, 1.0])
        raw, normalized = normalize_candidate_group_rewards(make_args(), samples)
        assert raw == [0.0, 0.0, 1.0, 1.0]
        # mean = 0.5; centered = [-.5, -.5, .5, .5]; std (n-1) = sqrt(1/3)
        expected_unit = 0.5 / (pow(1 / 3, 0.5) + 1e-6)
        for value in normalized:
            assert value == pytest.approx(expected_unit if value > 0 else -expected_unit)
        assert sum(normalized) == pytest.approx(0.0, abs=1e-6)

    def test_advantage_mean_zero_per_group(self):
        samples = full_group(0, [0.1, 0.4, 0.9, 1.0])
        _, normalized = normalize_candidate_group_rewards(make_args(), samples)
        assert sum(normalized) == pytest.approx(0.0, abs=1e-6)

    def test_zero_variance_group_gives_zero_advantage(self, monkeypatch):
        monkeypatch.setenv("SHOP_REQUIRE_NONZERO_VARIANCE_PER_ROLLOUT", "0")
        samples = full_group(0, [1.0, 1.0, 1.0, 1.0])
        raw, normalized = normalize_candidate_group_rewards(make_args(), samples)
        assert normalized == [0.0, 0.0, 0.0, 0.0]

    def test_all_zero_variance_raises_when_required(self, monkeypatch):
        monkeypatch.setenv("SHOP_REQUIRE_NONZERO_VARIANCE_PER_ROLLOUT", "1")
        samples = full_group(0, [1.0, 1.0, 1.0, 1.0])
        with pytest.raises(RuntimeError, match="zero-signal"):
            normalize_candidate_group_rewards(make_args(), samples)

    def test_mixed_variance_groups_do_not_raise(self, monkeypatch):
        monkeypatch.setenv("SHOP_REQUIRE_NONZERO_VARIANCE_PER_ROLLOUT", "1")
        samples = full_group(0, [1.0, 1.0, 1.0, 1.0]) + full_group(1, [0.0, 0.0, 1.0, 1.0])
        _, normalized = normalize_candidate_group_rewards(make_args(), samples)
        assert normalized[:4] == [0.0, 0.0, 0.0, 0.0]
        assert any(value != 0 for value in normalized[4:])

    def test_fanout_fragments_share_advantage(self):
        # Candidate 0 is split into two fragments (index 0 and 1) that both
        # carry the same reward; candidate identity comes from rollout_id.
        samples = [
            make_sample(group_index=0, rollout_id=0, index=0, reward=0.0),
            make_sample(group_index=0, rollout_id=0, index=1, reward=0.0),
            make_sample(group_index=0, rollout_id=1, index=2, reward=1.0),
            make_sample(group_index=0, rollout_id=2, index=3, reward=1.0),
            make_sample(group_index=0, rollout_id=3, index=4, reward=1.0),
        ]
        _, normalized = normalize_candidate_group_rewards(make_args(), samples)
        assert normalized[0] == normalized[1]
        assert normalized[0] < 0 < normalized[2]

    def test_inconsistent_fragment_reward_raises(self):
        samples = [
            make_sample(group_index=0, rollout_id=0, index=0, reward=0.0),
            make_sample(group_index=0, rollout_id=0, index=1, reward=1.0),  # same candidate, other reward
            make_sample(group_index=0, rollout_id=1, index=2, reward=1.0),
            make_sample(group_index=0, rollout_id=2, index=3, reward=1.0),
            make_sample(group_index=0, rollout_id=3, index=4, reward=1.0),
        ]
        with pytest.raises(RuntimeError, match="inconsistent reward"):
            normalize_candidate_group_rewards(make_args(), samples)

    def test_wrong_candidate_count_raises(self):
        samples = full_group(0, [0.0, 1.0, 1.0, 1.0])[:3]
        with pytest.raises(RuntimeError, match="expected 4"):
            normalize_candidate_group_rewards(make_args(), samples)

    def test_missing_group_index_raises(self):
        sample = make_sample(group_index=0, rollout_id=0, index=0, reward=0.0)
        sample.group_index = None
        with pytest.raises(RuntimeError, match="group_index"):
            normalize_candidate_group_rewards(make_args(), [sample])

    def test_fully_filtered_group_is_skipped(self, monkeypatch):
        monkeypatch.setenv("SHOP_REQUIRE_NONZERO_VARIANCE_PER_ROLLOUT", "0")
        samples = [
            make_sample(group_index=0, rollout_id=c, index=c, reward=1.0, remove=True, status=Sample.Status.ABORTED)
            for c in range(4)
        ]
        raw, normalized = normalize_candidate_group_rewards(make_args(), samples)
        assert normalized == [0.0, 0.0, 0.0, 0.0]

    def test_partially_filtered_group_raises(self):
        samples = [
            make_sample(group_index=0, rollout_id=0, index=0, reward=1.0, remove=True, status=Sample.Status.ABORTED),
            make_sample(group_index=0, rollout_id=1, index=1, reward=1.0),
            make_sample(group_index=0, rollout_id=2, index=2, reward=1.0),
            make_sample(group_index=0, rollout_id=3, index=3, reward=1.0),
        ]
        with pytest.raises(RuntimeError, match="partially filtered"):
            normalize_candidate_group_rewards(make_args(), samples)

    def test_std_normalization_disabled(self):
        samples = full_group(0, [0.0, 0.0, 1.0, 1.0])
        args = make_args(grpo_std_normalization=False)
        _, normalized = normalize_candidate_group_rewards(args, samples)
        for value in normalized:
            assert value == pytest.approx(0.5 if value > 0 else -0.5)


class TestValidateCompleteGroups:
    def test_complete_group_passes(self):
        groups = [full_group(0, [0.0, 0.0, 1.0, 1.0])]
        validate_complete_groups(make_args(), groups)
        assert all(not sample.remove_sample for group in groups for sample in group)

    def test_aborted_candidate_filters_whole_group(self):
        group = full_group(0, [0.0, 0.0, 1.0, 1.0])
        group[2].remove_sample = True
        group[2].status = Sample.Status.ABORTED
        validate_complete_groups(make_args(), [group])
        assert all(sample.remove_sample for sample in group)
        assert all(sample.metadata.get("group_filtered") for sample in group)
        assert "aborted" in group[0].metadata["group_filter_reason"]

    def test_fanout_candidate_list_form(self):
        # generate.py may hand each candidate as a list of fragments.
        fragments = [
            [make_sample(group_index=0, rollout_id=0, index=0, reward=1.0)],
            [make_sample(group_index=0, rollout_id=1, index=1, reward=1.0)],
            [make_sample(group_index=0, rollout_id=2, index=2, reward=1.0)],
            [make_sample(group_index=0, rollout_id=3, index=3, reward=1.0)],
        ]
        validate_complete_groups(make_args(), [fragments])

    def test_untrainable_candidate_raises(self):
        group = full_group(0, [0.0, 0.0, 1.0, 1.0])
        group[1].loss_mask = [0]
        with pytest.raises(RuntimeError, match="no trainable action tokens"):
            validate_complete_groups(make_args(), [group])

    def test_duplicate_candidate_id_raises(self):
        group = full_group(0, [0.0, 0.0, 1.0, 1.0])
        group[1].rollout_id = group[0].rollout_id
        with pytest.raises(RuntimeError, match="duplicate"):
            validate_complete_groups(make_args(), [group])

    def test_wrong_group_size_raises(self):
        group = full_group(0, [0.0, 0.0, 1.0, 1.0])[:3]
        with pytest.raises(RuntimeError, match="expected 4"):
            validate_complete_groups(make_args(), [group])

    def test_empty_candidate_fragments_raise(self):
        group = full_group(0, [0.0, 0.0, 1.0, 1.0])
        group[0].rollout_id = None
        group[0].index = None
        with pytest.raises(RuntimeError, match="candidate id"):
            validate_complete_groups(make_args(), [group])


def _sample(**kwargs):
    defaults = dict(
        group_index=0,
        index=0,
        rollout_id=0,
        reward=0.0,
        response="r",
        response_length=1,
        loss_mask=[1],
        metadata={},
    )
    defaults.update(kwargs)
    return Sample(**defaults)


def _act_events(actions):
    return [
        {"type": "tool_execution_start", "toolName": "shop_act", "args": {"action": action}}
        for action in actions
    ]


def _scored_sample(*, group_index, rollout_id, index, reward, detail=None, events=None, goal_asin=None, behavior_bonus=0.0):
    sample = make_sample(group_index=group_index, rollout_id=rollout_id, index=index, reward=reward)
    sample.metadata = {
        "reward_detail": detail or {},
        "events": events or [],
        "goal_asin": goal_asin,
        "behavior_bonus": behavior_bonus,
    }
    return sample


class TestCandidateFragments:
    def test_single_sample_wrapped(self):
        sample = _sample()
        assert _candidate_fragments(sample) == [sample]

    def test_list_returned_as_is(self):
        samples = [_sample(), _sample(index=1, rollout_id=1)]
        assert _candidate_fragments(samples) == samples


class TestAbortSample:
    def test_marks_aborted_and_removes(self):
        sample = _sample(metadata={"task_id": 7})
        out = abort_sample(sample, "boom", error_kind="infrastructure_error")
        assert out == [sample]
        assert sample.status == Sample.Status.ABORTED
        assert sample.remove_sample is True
        assert sample.reward == 0.0
        assert sample.loss_mask == [0]
        assert sample.response == ""
        assert sample.metadata["abort_reason"] == "boom"
        assert sample.metadata["error_kind"] == "infrastructure_error"
        assert sample.metadata["task_id"] == 7

    def test_default_error_message_is_reason(self):
        sample = _sample()
        abort_sample(sample, "some reason")
        assert sample.metadata["error_message"] == "some reason"

    def test_explicit_error_message_overrides(self):
        sample = _sample()
        abort_sample(sample, "reason", error_message="detailed")
        assert sample.metadata["error_message"] == "detailed"


class TestTaskId:
    def test_valid(self):
        assert _task_id(_sample(metadata={"task_id": 42})) == 42

    def test_string_task_id_coerced(self):
        assert _task_id(_sample(metadata={"task_id": "7"})) == 7

    def test_missing_raises(self):
        with pytest.raises(InfrastructureError, match="missing task_id"):
            _task_id(_sample(metadata={}))

    def test_non_integer_raises(self):
        with pytest.raises(InfrastructureError, match="not an integer"):
            _task_id(_sample(metadata={"task_id": "abc"}))

    def test_negative_raises(self):
        with pytest.raises(InfrastructureError, match="non-negative"):
            _task_id(_sample(metadata={"task_id": -1}))


class TestSamplePrompt:
    def test_extracts_text(self):
        sample = _sample(prompt=[{"role": "user", "content": "hello"}])
        assert _sample_prompt(sample) == "hello"

    def test_unusable_raises(self):
        sample = _sample(prompt=[{"role": "user"}])
        with pytest.raises(InfrastructureError, match="unusable"):
            _sample_prompt(sample)


class TestMakeUniqueSessionId:
    def test_includes_task_and_indices(self):
        sample = _sample(index=3, group_index=2)
        assert make_unique_session_id(sample, 7).startswith("shop-7-3-2-")

    def test_fallback_without_index(self):
        sample = _sample(index=None, group_index=2)
        assert make_unique_session_id(sample, 7).startswith("shop-7-")

    def test_uniqueness(self):
        sample = _sample(index=1, group_index=0)
        assert make_unique_session_id(sample, 1) != make_unique_session_id(sample, 1)


class _FakeAdapter:
    def __init__(self):
        self.captured_metadata = None

    def session_termination_reason(self, session_id):
        return "agent_stop"

    def session_turn_count(self, session_id):
        return 3

    async def finish_session(self, session_id, *, base_sample, reward, extra_metadata):
        self.captured_metadata = extra_metadata
        base_sample.metadata = dict(extra_metadata)
        return [base_sample]


class _FakeState:
    def __init__(self):
        self.adapter = _FakeAdapter()
        self.max_model_turns = 40


def _done_result_with_traces() -> PiRunResult:
    result = PiRunResult(exit_code=0, done=True, reward=1.0)
    result.events = [{"type": "turn_start"}, {"type": "tool_execution_end", "toolName": "shop_act"}]
    result.context_traces = [{"messages": [{"role": "user", "content": "go"}]}]
    return result


class TestFinishCandidateSessionTraces:
    def test_traces_persisted_when_enabled(self, monkeypatch):
        monkeypatch.setattr(generate_module, "CAPTURE_ROLLOUT_EVENTS", True)
        state = _FakeState()
        result = _done_result_with_traces()
        base_sample = _sample(metadata={"task_id": 1})
        samples = asyncio.run(
            finish_candidate_session(
                state, "sid", base_sample=base_sample, task_id=1, result=result
            )
        )
        assert samples[0].metadata["events"] == result.events
        assert samples[0].metadata["context_traces"] == result.context_traces

    def test_traces_empty_when_disabled(self, monkeypatch):
        monkeypatch.setattr(generate_module, "CAPTURE_ROLLOUT_EVENTS", False)
        state = _FakeState()
        result = _done_result_with_traces()
        base_sample = _sample(metadata={"task_id": 1})
        samples = asyncio.run(
            finish_candidate_session(
                state, "sid", base_sample=base_sample, task_id=1, result=result
            )
        )
        assert samples[0].metadata["events"] == []
        assert samples[0].metadata["context_traces"] == []


class TestCaptureRolloutEventsFlag:
    def test_defaults_enabled(self, monkeypatch):
        monkeypatch.delenv("SHOP_CAPTURE_ROLLOUT_EVENTS", raising=False)
        assert _capture_rollout_events_enabled() is True

    def test_disabled_via_zero(self, monkeypatch):
        monkeypatch.setenv("SHOP_CAPTURE_ROLLOUT_EVENTS", "0")
        assert _capture_rollout_events_enabled() is False

    def test_disabled_via_false_like_strings(self, monkeypatch):
        for value in ("false", "False", "no", "off"):
            monkeypatch.setenv("SHOP_CAPTURE_ROLLOUT_EVENTS", value)
            assert _capture_rollout_events_enabled() is False, value

    def test_enabled_via_arbitrary_truthy(self, monkeypatch):
        monkeypatch.setenv("SHOP_CAPTURE_ROLLOUT_EVENTS", "yes")
        assert _capture_rollout_events_enabled() is True



class TestBehaviorBonus:
    """R3-2 behaviour process reward from the action stream."""

    def test_visit_goal_before_buy_now(self):
        events = _act_events([
            "search[pink dress]",
            "click[B0123GOAL]",
            "click[features]",
            "click[buy now]",
        ])
        assert _behavior_bonus(events, "B0123GOAL", 0.05) == pytest.approx(0.05)

    def test_no_goal_visit_before_buy_now(self):
        events = _act_events(["search[pink dress]", "click[B0OTHER00]", "click[buy now]"])
        assert _behavior_bonus(events, "B0123GOAL", 0.05) == pytest.approx(0.0)

    def test_goal_visit_after_buy_now_does_not_count(self):
        events = _act_events(["click[buy now]", "click[B0123GOAL]"])
        assert _behavior_bonus(events, "B0123GOAL", 0.05) == pytest.approx(0.0)

    def test_repeated_actions_penalized(self):
        events = _act_events([
            "search[a]",
            "click[B0123GOAL]",
            "search[a]",      # global search repeat: -delta
            "search[a]",      # consecutive (also global): -delta once
            "click[buy now]",
        ])
        # +delta for goal visit, -2*delta for two repeats.
        assert _behavior_bonus(events, "B0123GOAL", 0.05) == pytest.approx(-0.05)

    def test_repeats_penalized_even_without_buy_now(self):
        events = _act_events(["search[a]", "search[a]", "search[a]"])
        assert _behavior_bonus(events, "B0123GOAL", 0.05) == pytest.approx(-0.10)

    def test_search_repeat_across_gap_still_penalized(self):
        events = _act_events(["search[a]", "click[B0123GOAL]", "search[a]"])
        assert _behavior_bonus(events, "B0123GOAL", 0.05) == pytest.approx(-0.05)

    def test_tab_reuse_across_products_not_penalized(self):
        # Same tab verb on different products = different state (legit exploration).
        events = _act_events([
            "click[B0AAA000001]",
            "click[features]",
            "click[B0BBB000002]",
            "click[features]",
        ])
        assert _behavior_bonus(events, "B0AAA000001", 0.05) == pytest.approx(0.0)

    def test_reclicking_product_for_comparison_not_penalized(self):
        # A->B->A comparison is good shopping behaviour, not a loop.
        events = _act_events([
            "click[B0AAA000001]",
            "click[B0BBB000002]",
            "click[B0AAA000001]",
        ])
        assert _behavior_bonus(events, "B0AAA000001", 0.05) == pytest.approx(0.0)

    def test_navigation_not_penalized(self):
        events = _act_events([
            "search[a]",
            "click[B0AAA000001]",
            "click[back to search]",
            "click[next >]",
            "click[< prev]",
        ])
        assert _behavior_bonus(events, "B0AAA000001", 0.05) == pytest.approx(0.0)

    def test_consecutive_identical_actions_penalized(self):
        # Stuck-in-place: same action twice in a row, whatever the verb.
        events = _act_events(["click[back]", "click[back]"])
        assert _behavior_bonus(events, "B0AAA000001", 0.05) == pytest.approx(-0.05)

    def test_no_buy_now_no_goal_bonus(self):
        events = _act_events(["search[a]", "click[B0123GOAL]"])
        assert _behavior_bonus(events, "B0123GOAL", 0.05) == pytest.approx(0.0)

    def test_empty_events_zero(self):
        assert _behavior_bonus([], "B0123GOAL", 0.05) == 0.0

    def test_zero_delta_noop(self):
        events = _act_events(["click[B0123GOAL]", "click[buy now]", "search[a]", "search[a]"])
        assert _behavior_bonus(events, "B0123GOAL", 0.0) == 0.0

    def test_none_goal_asin_no_goal_bonus(self):
        events = _act_events(["click[B0123GOAL]", "click[buy now]"])
        assert _behavior_bonus(events, None, 0.05) == pytest.approx(0.0)

    def test_arguments_field_fallback(self):
        # Some event producers nest under `arguments` instead of `args`.
        events = [
            {"type": "tool_execution_start", "toolName": "shop_act", "arguments": {"action": "click[B0123GOAL]"}},
            {"type": "tool_execution_start", "toolName": "shop_act", "arguments": {"action": "click[buy now]"}},
        ]
        assert _behavior_bonus(events, "B0123GOAL", 0.05) == pytest.approx(0.05)

    def test_non_shop_act_events_ignored(self):
        events = [
            {"type": "tool_execution_start", "toolName": "shop_reset", "args": {"action": "x"}},
            {"type": "tool_execution_start", "toolName": "shop_act", "args": {"action": "click[B0123GOAL]"}},
            {"type": "tool_execution_start", "toolName": "shop_act", "args": {"action": "click[buy now]"}},
        ]
        assert _behavior_bonus(events, "B0123GOAL", 0.05) == pytest.approx(0.05)


class TestDecomposedGroupAdvantages:
    """R1-2 per-dimension normalization on the scored candidate subset."""

    DETAIL_BASE = {"r_type": 1.0, "r_att": 1.0, "r_price": 1.0}

    def _candidates(self, r_options):
        samples = []
        candidate_rewards = []
        for i, r_option in enumerate(r_options):
            detail = dict(self.DETAIL_BASE, r_option=r_option) if r_option is not None else {}
            samples.append(_scored_sample(
                group_index=0, rollout_id=i, index=i, reward=1.0, detail=detail,
            ))
            candidate_rewards.append((i, 1.0, [i]))
        return candidate_rewards, samples

    def test_uniform_dims_only_r_option_varies(self):
        candidate_rewards, samples = self._candidates([0.2, 0.4, 0.6, 0.8])
        out = _decomposed_group_advantages(candidate_rewards, samples, use_std=True)
        # r_type/r_att/r_price identical -> z=0; r_option mean=0.5, z-scores
        # symmetric; decomposed = z/4 averaged over the four dimensions.
        std = math.sqrt((0.09 + 0.01 + 0.01 + 0.09) / 3)
        expected = [((r - 0.5) / (std + 1e-6)) / 4.0 for r in (0.2, 0.4, 0.6, 0.8)]
        for got, want in zip(out, expected, strict=True):
            assert got == pytest.approx(want, abs=1e-6)
        assert sum(out) == pytest.approx(0.0, abs=1e-6)

    def test_unscored_candidates_get_zero_and_scored_subset_normalized(self):
        candidate_rewards, samples = self._candidates([0.2, 0.8, None, None])
        out = _decomposed_group_advantages(candidate_rewards, samples, use_std=True)
        assert out[2] == 0.0 and out[3] == 0.0
        # scored subset of two: sample std (n-1) -> z = ±1/sqrt(2), /4 dims
        z = (1.0 / math.sqrt(2)) / 4.0
        assert out[0] == pytest.approx(-z, abs=1e-6)
        assert out[1] == pytest.approx(z, abs=1e-6)

    def test_single_scored_candidate_degenerates_to_zero(self):
        candidate_rewards, samples = self._candidates([0.2, None, None, None])
        assert _decomposed_group_advantages(candidate_rewards, samples, use_std=True) == [0.0] * 4

    def test_all_unscored_returns_zero(self):
        candidate_rewards, samples = self._candidates([None, None, None, None])
        assert _decomposed_group_advantages(candidate_rewards, samples, use_std=True) == [0.0] * 4

    def test_partial_detail_counts_as_unscored(self):
        # Missing r_price -> not a valid scored candidate.
        samples = [
            _scored_sample(group_index=0, rollout_id=0, index=0, reward=1.0,
                           detail={"r_type": 1.0, "r_att": 1.0, "r_option": 0.5}),
            _scored_sample(group_index=0, rollout_id=1, index=1, reward=1.0,
                           detail=dict(self.DETAIL_BASE, r_option=0.2)),
        ]
        candidate_rewards = [(0, 1.0, [0]), (1, 1.0, [1])]
        assert _decomposed_group_advantages(candidate_rewards, samples, use_std=True) == [0.0, 0.0]

    def test_mean_centering_without_std(self):
        candidate_rewards, samples = self._candidates([0.2, 0.4, 0.6, 0.8])
        out = _decomposed_group_advantages(candidate_rewards, samples, use_std=False)
        expected = [(r - 0.5) / 4.0 for r in (0.2, 0.4, 0.6, 0.8)]
        for got, want in zip(out, expected, strict=True):
            assert got == pytest.approx(want, abs=1e-6)


class TestNormalizeWithR12R32:
    """Integration: R1-2 blend + R3-2 bonus inside normalize_candidate_group_rewards."""

    def test_decomposed_rescues_scalar_zero_variance_group(self, monkeypatch):
        # The R1-2 motivation case: identical scalar rewards (zero variance)
        # but differing r_option -> gradient signal exists after decomposition.
        monkeypatch.setattr(generate_module, "DECOMPOSED_ADVANTAGE_WEIGHT", 1.0)
        monkeypatch.setenv("SHOP_REQUIRE_NONZERO_VARIANCE_PER_ROLLOUT", "1")
        samples = [
            _scored_sample(group_index=0, rollout_id=i, index=i, reward=0.9,
                           detail={"r_type": 1.0, "r_att": 1.0, "r_option": r, "r_price": 1.0})
            for i, r in enumerate((0.2, 0.4, 0.6, 0.8))
        ]
        raw, normalized = normalize_candidate_group_rewards(make_args(), samples)
        assert raw == [0.9, 0.9, 0.9, 0.9]  # env reward untouched
        assert normalized[0] < normalized[1] < normalized[2] < normalized[3]
        assert normalized[0] < 0 < normalized[3]

    def test_blend_weight_halves_contribution(self, monkeypatch):
        monkeypatch.setattr(generate_module, "DECOMPOSED_ADVANTAGE_WEIGHT", 0.5)
        monkeypatch.setenv("SHOP_REQUIRE_NONZERO_VARIANCE_PER_ROLLOUT", "0")
        samples = [
            _scored_sample(group_index=0, rollout_id=i, index=i, reward=r,
                           detail={"r_type": 1.0, "r_att": 1.0, "r_option": r, "r_price": 1.0})
            for i, r in enumerate((0.0, 0.0, 1.0, 1.0))
        ]
        # scalar part for rewards [0,0,1,1]: mean=.5, std(n-1)=sqrt(1/3)
        unit = 0.5 / (math.sqrt(1 / 3) + 1e-6)
        _, normalized = normalize_candidate_group_rewards(make_args(), samples)
        # weight .5: advantage must lie between pure-scalar (unit) and the
        # decomposed-only value, i.e. strictly smaller in magnitude than unit.
        assert abs(normalized[3]) < unit
        assert normalized[3] > 0

    def test_behavior_bonus_shifts_advantage_not_raw(self, monkeypatch):
        # normalize consumes the precomputed metadata["behavior_bonus"] (written
        # by finish_candidate_session) — it must never recompute from events.
        monkeypatch.setattr(generate_module, "BEHAVIOR_DELTA", 0.05)
        monkeypatch.setenv("SHOP_REQUIRE_NONZERO_VARIANCE_PER_ROLLOUT", "0")
        good = _scored_sample(
            group_index=0, rollout_id=0, index=0, reward=0.8, behavior_bonus=0.05,
        )
        bad = _scored_sample(
            group_index=0, rollout_id=1, index=1, reward=0.8, behavior_bonus=-0.05,
        )
        filler = [
            _scored_sample(group_index=0, rollout_id=i, index=i, reward=0.8)
            for i in (2, 3)
        ]
        raw, normalized = normalize_candidate_group_rewards(make_args(), [good, bad, *filler])
        assert all(r == 0.8 for r in raw)  # raw stays pure env reward
        assert normalized[0] > normalized[1]
        assert normalized[0] > 0

    def test_bonus_variance_counts_as_signal(self, monkeypatch):
        # Zero scalar variance but differing bonuses -> a real gradient signal;
        # the zero-signal guard must not fire.
        monkeypatch.setattr(generate_module, "BEHAVIOR_DELTA", 0.05)
        monkeypatch.setenv("SHOP_REQUIRE_NONZERO_VARIANCE_PER_ROLLOUT", "1")
        samples = [
            _scored_sample(group_index=0, rollout_id=0, index=0, reward=0.8, behavior_bonus=0.05),
            _scored_sample(group_index=0, rollout_id=1, index=1, reward=0.8, behavior_bonus=-0.05),
            _scored_sample(group_index=0, rollout_id=2, index=2, reward=0.8),
            _scored_sample(group_index=0, rollout_id=3, index=3, reward=0.8),
        ]
        raw, normalized = normalize_candidate_group_rewards(make_args(), samples)
        assert raw == [0.8] * 4
        assert normalized[0] > 0 > normalized[1]

    def test_group_sum_still_zero_with_all_features(self, monkeypatch):
        monkeypatch.setattr(generate_module, "DECOMPOSED_ADVANTAGE_WEIGHT", 0.5)
        monkeypatch.setattr(generate_module, "BEHAVIOR_DELTA", 0.05)
        monkeypatch.setenv("SHOP_REQUIRE_NONZERO_VARIANCE_PER_ROLLOUT", "0")
        samples = [
            _scored_sample(group_index=0, rollout_id=0, index=0, reward=0.8, goal_asin="G",
                           detail={"r_type": 1, "r_att": 1, "r_option": 0.2, "r_price": 1},
                           events=_act_events(["click[G]", "click[buy now]"])),
            _scored_sample(group_index=0, rollout_id=1, index=1, reward=0.4, goal_asin="G",
                           detail={"r_type": 1, "r_att": 1, "r_option": 0.8, "r_price": 1},
                           events=_act_events(["search[a]", "search[a]"])),
            _scored_sample(group_index=0, rollout_id=2, index=2, reward=0.0),
            _scored_sample(group_index=0, rollout_id=3, index=3, reward=0.0),
        ]
        _, normalized = normalize_candidate_group_rewards(make_args(), samples)
        assert sum(normalized) == pytest.approx(0.0, abs=1e-6)

    def test_defaults_fully_off_match_legacy(self, monkeypatch):
        monkeypatch.setattr(generate_module, "DECOMPOSED_ADVANTAGE_WEIGHT", 0.0)
        monkeypatch.setattr(generate_module, "BEHAVIOR_DELTA", 0.0)
        samples = full_group(0, [0.0, 0.0, 1.0, 1.0])
        raw, normalized = normalize_candidate_group_rewards(make_args(), samples)
        unit = 0.5 / (math.sqrt(1 / 3) + 1e-6)
        assert normalized[0] == pytest.approx(-unit, abs=1e-6)
        assert normalized[3] == pytest.approx(unit, abs=1e-6)


class TestBehaviorBonusInMetadata:
    """finish_candidate_session writes the audit-trail behavior_bonus into metadata."""

    def _run(self, monkeypatch, events, delta=0.05, capture=True):
        monkeypatch.setattr(generate_module, "CAPTURE_ROLLOUT_EVENTS", capture)
        monkeypatch.setattr(generate_module, "BEHAVIOR_DELTA", delta)
        state = _FakeState()
        result = PiRunResult(exit_code=0, done=True, reward=1.0)
        result.goal_asin = "B0GOAL"
        result.events = events
        return asyncio.run(
            finish_candidate_session(
                state, "sid", base_sample=_sample(metadata={"task_id": 1}), task_id=1, result=result
            )
        )

    def test_goal_visit_bonus_written(self, monkeypatch):
        events = _act_events(["click[B0GOAL]", "click[features]", "click[buy now]"])
        samples = self._run(monkeypatch, events)
        assert samples[0].metadata["behavior_bonus"] == pytest.approx(0.05)

    def test_repeat_penalty_written(self, monkeypatch):
        events = _act_events(["search[a]", "search[a]", "click[B0GOAL]", "click[buy now]"])
        samples = self._run(monkeypatch, events)
        # +delta (goal visit before buy) -delta (search repeat) = 0.0 net
        assert samples[0].metadata["behavior_bonus"] == pytest.approx(0.0)

    def test_pure_penalty_written(self, monkeypatch):
        events = _act_events(["search[a]", "search[a]"])
        samples = self._run(monkeypatch, events)
        assert samples[0].metadata["behavior_bonus"] == pytest.approx(-0.05)

    def test_zero_when_capture_disabled(self, monkeypatch):
        events = _act_events(["click[B0GOAL]", "click[buy now]"])
        samples = self._run(monkeypatch, events, capture=False)
        assert samples[0].metadata["behavior_bonus"] == 0.0

    def test_zero_when_delta_disabled(self, monkeypatch):
        events = _act_events(["click[B0GOAL]", "click[buy now]"])
        samples = self._run(monkeypatch, events, delta=0.0)
        assert samples[0].metadata["behavior_bonus"] == 0.0
