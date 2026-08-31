"""Tests for examples.ShopSimulator.generate (GRPO normalization & group gate)."""

from __future__ import annotations

import argparse

import pytest

from examples.ShopSimulator.generate import (
    normalize_candidate_group_rewards,
    validate_complete_groups,
)
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

