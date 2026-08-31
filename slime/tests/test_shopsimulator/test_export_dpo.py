"""Tests for export_dpo_pairs (C4: preference-pair extraction)."""

from __future__ import annotations

import pytest

from examples.ShopSimulator.export_dpo_pairs import (
    aggregate_dump_samples,
    build_preference_pairs,
)


class TestBuildPreferencePairs:
    @staticmethod
    def record(group, candidate, reward, response="r", prompt="p"):
        return {"group": group, "candidate": candidate, "reward": reward, "response": response, "prompt": prompt}

    def test_pairs_best_against_worst(self):
        records = [
            self.record("g1", 0, 0.0, "bad"),
            self.record("g1", 1, 0.2, "mid"),
            self.record("g1", 2, 1.0, "good"),
            self.record("g1", 3, 0.4, "ok"),
        ]
        pairs, stats = build_preference_pairs(records)
        assert stats["pairs"] == 1
        assert pairs[0]["chosen"] == "good"
        assert pairs[0]["rejected"] == "bad"
        assert pairs[0]["prompt"] == "p"
        assert pairs[0]["metadata"]["chosen_reward"] == 1.0
        assert pairs[0]["metadata"]["rejected_reward"] == 0.0

    def test_min_reward_gap_filters_flat_groups(self):
        records = [self.record("g1", 0, 0.5), self.record("g1", 1, 0.5)]
        pairs, stats = build_preference_pairs(records, min_reward_gap=0.3)
        assert pairs == []
        assert stats["below_gap_groups"] == 1

    def test_gap_equal_to_threshold_is_filtered(self):
        records = [self.record("g1", 0, 0.0), self.record("g1", 1, 0.3)]
        pairs, _ = build_preference_pairs(records, min_reward_gap=0.3)
        assert pairs == []  # gap must be strictly greater than the threshold

    def test_single_candidate_group_skipped(self):
        _, stats = build_preference_pairs([self.record("g1", 0, 1.0)])
        assert stats["single_candidate_groups"] == 1

    def test_multiple_groups(self):
        records = [
            self.record("g1", 0, 0.0), self.record("g1", 1, 1.0),
            self.record("g2", 0, 0.2), self.record("g2", 1, 0.9),
        ]
        pairs, stats = build_preference_pairs(records)
        assert stats["groups"] == 2
        assert {pair["metadata"]["group"] for pair in pairs} == {"g1", "g2"}

    def test_prompt_falls_back_to_rejected_record(self):
        records = [
            self.record("g1", 0, 0.0, prompt="shared"),
            self.record("g1", 1, 1.0, prompt=None),
        ]
        pairs, _ = build_preference_pairs(records)
        assert pairs[0]["prompt"] == "shared"


class TestAggregateDumpSamples:
    def test_fragments_are_merged_in_order(self):
        samples = [
            {"group_index": 0, "rollout_id": 7, "index": 1, "reward": 1.0, "response": "B", "prompt": [{"role": "user", "content": "go"}]},
            {"group_index": 0, "rollout_id": 7, "index": 0, "reward": 1.0, "response": "A", "prompt": [{"role": "user", "content": "go"}]},
        ]
        records = aggregate_dump_samples(samples)
        assert len(records) == 1
        assert records[0]["response"] == "AB"
        assert records[0]["prompt"] == "go"
        assert records[0]["reward"] == 1.0

    def test_inconsistent_fragment_rewards_raise(self):
        samples = [
            {"group_index": 0, "rollout_id": 7, "index": 0, "reward": 1.0, "response": "A"},
            {"group_index": 0, "rollout_id": 7, "index": 1, "reward": 0.0, "response": "B"},
        ]
        with pytest.raises(ValueError, match="inconsistent rewards"):
            aggregate_dump_samples(samples)

    def test_missing_identity_raises(self):
        with pytest.raises(ValueError, match="identity"):
            aggregate_dump_samples([{"reward": 1.0, "response": "x"}])

    def test_index_fallback_for_candidate_id(self):
        samples = [
            {"group_index": 0, "index": 3, "reward": 1.0, "response": "x"},
            {"group_index": 0, "index": 4, "reward": 0.0, "response": "y"},
        ]
        records = aggregate_dump_samples(samples)
        assert {record["candidate"] for record in records} == {3, 4}

    def test_load_dump_roundtrip(self, tmp_path):
        torch = pytest.importorskip("torch")
        samples = [
            {"group_index": 0, "rollout_id": 0, "index": 0, "reward": 0.0, "response": "bad", "prompt": [{"role": "user", "content": "go"}]},
            {"group_index": 0, "rollout_id": 1, "index": 1, "reward": 1.0, "response": "good", "prompt": [{"role": "user", "content": "go"}]},
        ]
        dump = tmp_path / "rollout_0.pt"
        torch.save({"samples": samples}, dump)
        from examples.ShopSimulator.export_dpo_pairs import load_dump
        records = load_dump(dump)
        pairs, _ = build_preference_pairs(records)
        assert pairs[0]["chosen"] == "good"
        assert pairs[0]["rejected"] == "bad"
