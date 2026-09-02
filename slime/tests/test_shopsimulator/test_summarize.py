"""Tests for ShopSimulator evaluation summarization.

Covers two layers of the same module (examples.ShopSimulator.utils):
- unit tests for the underlying helpers: summarize_dump / summarize_run / _candidate_key;
- CLI integration tests for the `summarize` subcommand (--dump mode).

Merged from the former test_utils_summarize.py and test_summarize_eval_cli.py.
"""

from __future__ import annotations

import json

import pytest

from examples.ShopSimulator.utils import (
    _candidate_key,
    main,
    summarize_dump,
    summarize_run,
)

torch = pytest.importorskip("torch")


def _sample(rollout_id, index, reward, task_id=7, **meta):
    metadata = {
        "task_id": task_id,
        "env_done": bool(reward > 0),
        "env_over": False,
        "model_turns": 5,
        "pi_tool_calls": 6,
        "reward_detail": {},
        "purchase_asin": None,
        "goal_asin": None,
        "termination_reason": "environment_done" if reward > 0 else "turn_limit",
        "error_kind": "environment_terminal" if reward > 0 else "turn_limit",
        "error_message": None,
        "category": None,
        "attribute_count": None,
        "option_count": None,
    }
    metadata.update(meta)
    return {
        "group_index": rollout_id,
        "rollout_id": rollout_id,
        "index": index,
        "reward": reward,
        "response": "a",
        "prompt": [{"role": "user", "content": "go"}],
        "metadata": metadata,
    }


def _write_dump(tmp_path, samples, name="rollout_eval_0001.pt"):
    path = tmp_path / name
    torch.save({"samples": samples}, path)
    return path


@pytest.fixture()
def dump_file(tmp_path):
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


class TestCandidateKey:
    def test_uses_rollout_id(self):
        assert _candidate_key(_sample(9, 1, 1.0, task_id=3)) == (3, 9)

    def test_falls_back_to_index(self):
        sample = _sample(9, 1, 1.0, task_id=3)
        sample["rollout_id"] = None
        assert _candidate_key(sample) == (3, 1)

    def test_missing_metadata(self):
        assert _candidate_key({"rollout_id": 5}) == (None, 5)


class TestSummarizeDump:
    def test_single_candidate(self, tmp_path):
        summary = summarize_dump(_write_dump(tmp_path, [_sample(0, 0, 1.0)]))
        assert summary["tasks"] == 1
        assert summary["candidates"] == 1
        assert summary["fragments"] == 1
        assert summary["metrics"]["samples"] == 1
        assert summary["records"][0]["task_id"] == 7
        assert summary["records"][0]["reward"] == 1.0

    def test_multiple_fragments_one_candidate(self, tmp_path):
        samples = [_sample(0, 0, 0.0, task_id=7), _sample(0, 1, 0.0, task_id=7)]
        summary = summarize_dump(_write_dump(tmp_path, samples))
        assert summary["candidates"] == 1
        assert summary["fragments"] == 2
        assert summary["records"][0]["fragments"] == 2

    def test_empty_dump_raises(self, tmp_path):
        with pytest.raises(ValueError, match="no samples"):
            summarize_dump(_write_dump(tmp_path, []))

    def test_non_object_sample_raises(self, tmp_path):
        with pytest.raises(ValueError, match="non-object"):
            summarize_dump(_write_dump(tmp_path, ["not-a-dict"]))

    def test_inconsistent_fragment_reward_raises(self, tmp_path):
        samples = [_sample(0, 0, 0.0, task_id=7), _sample(0, 1, 1.0, task_id=7)]
        with pytest.raises(ValueError, match="inconsistent reward"):
            summarize_dump(_write_dump(tmp_path, samples))


class TestSummarizeRun:
    def test_writes_eval_results(self, tmp_path):
        run_root = tmp_path / "run"
        dump_dir = run_root / "rollout_dumps"
        dump_dir.mkdir(parents=True)
        torch.save({"samples": [_sample(0, 0, 1.0)]}, dump_dir / "rollout_eval_0001.pt")
        result = summarize_run(run_root)
        assert result["status"] == "eval-complete"
        assert result["metrics"]["samples"] == 1
        payload = json.loads((run_root / "eval_results.json").read_text())
        assert payload["metrics"]["samples"] == 1
        assert payload["records"][0]["task_id"] == 7

    def test_falls_back_to_any_pt(self, tmp_path):
        run_root = tmp_path / "run"
        dump_dir = run_root / "rollout_dumps"
        dump_dir.mkdir(parents=True)
        torch.save({"samples": [_sample(0, 0, 1.0)]}, dump_dir / "eval_0.pt")
        result = summarize_run(run_root)
        assert result["metrics"]["samples"] == 1

    def test_no_dump_raises(self, tmp_path):
        run_root = tmp_path / "run"
        (run_root / "rollout_dumps").mkdir(parents=True)
        with pytest.raises(ValueError, match="no evaluation dump"):
            summarize_run(run_root)


class TestDumpMode:
    def test_dump_prints_summary_without_writing_results(self, dump_file, capsys, monkeypatch, tmp_path):
        monkeypatch.setattr("sys.argv", ["utils", "summarize", "--dump", str(dump_file)])
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
            ["utils", "summarize", "--dump", str(dump_file), "--run-root", "somehere"],
        )
        with pytest.raises(SystemExit):
            main()

    def test_one_of_the_two_is_required(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["utils", "summarize"])
        with pytest.raises(SystemExit):
            main()
