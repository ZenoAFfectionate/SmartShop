"""Tests for the summarize_eval CLI --dump mode (B2 mid-training inspection)."""

from __future__ import annotations

import json

import pytest

from examples.ShopSimulator.summarize_eval import main


@pytest.fixture()
def dump_file(tmp_path):
    torch = pytest.importorskip("torch")

    def sample(rollout_id, reward, done, category, metadata_extra=None):
        base = {
            "group_index": rollout_id,
            "rollout_id": rollout_id,
            "index": rollout_id,
            "reward": reward,
            "response": "a",
            "prompt": [{"role": "user", "content": "go"}],
            "metadata": {
                "task_id": 7 + rollout_id,
                "reward_detail": (
                    {"r_type": 1.0, "r_att": 1.0, "r_option": 1.0, "r_price": True}
                    if done else {}
                ),
                "purchase_asin": "A1" if done else None,
                "goal_asin": "A1",
                "env_done": done,
                "model_turns": 5,
                "pi_tool_calls": 6,
                "termination_reason": "environment_done" if done else "turn_limit",
                "error_kind": "environment_terminal" if done else "turn_limit",
                "category": category,
            },
        }
        base["metadata"].update(metadata_extra or {})
        return base

    samples = [
        sample(0, 1.0, True, "玩具"),
        sample(1, 0.0, False, "图书"),
    ]
    path = tmp_path / "rollout_eval_0001.pt"
    torch.save({"samples": samples}, path)
    return path


class TestDumpMode:
    def test_dump_prints_summary_without_writing_results(self, dump_file, capsys, monkeypatch, tmp_path):
        monkeypatch.setattr("sys.argv", ["summarize_eval.py", "--dump", str(dump_file)])
        main()
        out = capsys.readouterr().out
        payload = json.loads(out)
        metrics = payload["metrics"]
        assert metrics["samples"] == 2
        assert metrics["r_success"] == 0.5
        assert metrics["by_category"]["玩具"]["r_success"] == 1.0
        assert metrics["by_category"]["图书"]["r_success"] == 0.0
        assert metrics["turn_limit_rate"] == 0.5
        assert not (tmp_path / "eval_results.json").exists()

    def test_run_root_and_dump_are_mutually_exclusive(self, dump_file, monkeypatch):
        monkeypatch.setattr(
            "sys.argv",
            ["summarize_eval.py", "--dump", str(dump_file), "--run-root", "somehere"],
        )
        with pytest.raises(SystemExit):
            main()

    def test_one_of_the_two_is_required(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["summarize_eval.py"])
        with pytest.raises(SystemExit):
            main()
