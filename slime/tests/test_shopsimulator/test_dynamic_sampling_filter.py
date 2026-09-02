"""Tests for the DAPO dynamic-sampling filter (utils.check_reward_nonzero_std_grouped).

The ShopSimulator agent rollout returns ``list[list[Sample]]`` (each candidate is a
multi-turn trajectory split into fragments), which the stock slime filter cannot
handle. The unified implementation in ``examples.ShopSimulator.utils`` must handle
both the nested agent layout and the flat standard layout.

(The former standalone ``examples.ShopSimulator.dynamic_sampling_filter.py`` module
has been merged into ``utils.py`` and deleted.)
"""

from __future__ import annotations

import argparse

from examples.ShopSimulator.utils import check_reward_nonzero_std_grouped as filter_utils
from slime.utils.types import Sample


def _args() -> argparse.Namespace:
    return argparse.Namespace(reward_key=None)


def _sample(rollout_id, index, reward, group_index=0) -> Sample:
    return Sample(
        group_index=group_index,
        index=index,
        rollout_id=rollout_id,
        reward=reward,
        response="r",
        response_length=1,
        loss_mask=[1],
        metadata={},
    )


class TestNestedAgentRollout:
    """list[list[Sample]]: each candidate's reward is the max over its fragments."""

    def test_nonzero_std_keeps(self):
        samples = [
            [_sample(0, 0, 0.0), _sample(0, 1, 1.0)],  # candidate reward 1.0
            [_sample(1, 2, 0.0)],  # candidate reward 0.0
        ]
        output = filter_utils(_args(), samples)
        assert bool(output.keep) is True
        assert output.reason is None

    def test_zero_std_drops(self):
        samples = [
            [_sample(0, 0, 0.0), _sample(0, 1, 1.0)],  # candidate reward 1.0
            [_sample(1, 2, 0.0), _sample(1, 3, 1.0)],  # candidate reward 1.0
        ]
        output = filter_utils(_args(), samples)
        assert bool(output.keep) is False
        assert output.reason == "zero_std_1.0"

    def test_candidate_reward_is_max_over_fragments(self):
        # rewards must be [1.0, 1.0, 0.0] -> nonzero std, proving max() is used.
        samples = [
            [_sample(0, 0, 1.0), _sample(0, 1, 0.0)],  # max 1.0
            [_sample(1, 2, 1.0), _sample(1, 3, 0.0)],  # max 1.0
            [_sample(2, 4, 0.0)],  # 0.0
        ]
        output = filter_utils(_args(), samples)
        assert bool(output.keep) is True


class TestFlatStandardRollout:
    """Flat list[Sample]: the stock single-turn layout must still work."""

    def test_nonzero_std_keeps(self):
        samples = [_sample(0, 0, 0.0), _sample(1, 1, 1.0)]
        assert bool(filter_utils(_args(), samples).keep) is True

    def test_zero_std_drops(self):
        samples = [_sample(0, 0, 1.0), _sample(1, 1, 1.0)]
        output = filter_utils(_args(), samples)
        assert bool(output.keep) is False
        assert output.reason == "zero_std_1.0"
