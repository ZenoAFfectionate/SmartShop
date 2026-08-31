"""Tests for the P1 tooling: system-prompt override (D5), self_improve (C5), aggregate_seeds (D4)."""

from __future__ import annotations

import json

import pytest

from examples.ShopSimulator.aggregate_seeds import aggregate
from examples.ShopSimulator.pi_harness import DEFAULT_SYSTEM_PROMPT, effective_system_prompt
from examples.ShopSimulator.self_improve import build_plan, weak_task_ids_from_eval


class TestEffectiveSystemPrompt:
    def test_default_without_env(self, monkeypatch):
        monkeypatch.delenv("SHOP_SYSTEM_PROMPT", raising=False)
        monkeypatch.delenv("SHOP_SYSTEM_PROMPT_FILE", raising=False)
        assert effective_system_prompt() == DEFAULT_SYSTEM_PROMPT

    def test_literal_env_override(self, monkeypatch):
        monkeypatch.setenv("SHOP_SYSTEM_PROMPT", "你是 ReAct 购物 Agent。")
        monkeypatch.delenv("SHOP_SYSTEM_PROMPT_FILE", raising=False)
        assert effective_system_prompt() == "你是 ReAct 购物 Agent。"

    def test_file_env_override(self, monkeypatch, tmp_path):
        monkeypatch.delenv("SHOP_SYSTEM_PROMPT", raising=False)
        prompt_file = tmp_path / "react.txt"
        prompt_file.write_text("  ReAct prompt from file. \n", encoding="utf-8")
        monkeypatch.setenv("SHOP_SYSTEM_PROMPT_FILE", str(prompt_file))
        assert effective_system_prompt() == "ReAct prompt from file."

    def test_literal_takes_precedence_over_file(self, monkeypatch, tmp_path):
        monkeypatch.setenv("SHOP_SYSTEM_PROMPT", "literal wins")
        monkeypatch.setenv("SHOP_SYSTEM_PROMPT_FILE", str(tmp_path / "nonexistent.txt"))
        assert effective_system_prompt() == "literal wins"


class TestWeakTaskIds:
    @staticmethod
    def record(task_id, reward, done=True):
        return {"task_id": task_id, "reward": reward, "done": done}

    def test_below_threshold_and_not_done(self):
        records = [
            self.record(1, 1.0),          # strong, done -> not weak
            self.record(2, 0.4),          # below 0.5 -> weak
            self.record(3, 0.8, done=False),  # not done -> weak
            self.record(4, 0.5),          # exactly at threshold -> not weak
        ]
        assert weak_task_ids_from_eval(records, 0.5) == {2, 3}

    def test_missing_task_id_ignored(self):
        assert weak_task_ids_from_eval([{"reward": 0.0, "done": False}], 0.5) == set()


class TestSelfImprovePlan:
    @staticmethod
    def eval_results(tmp_path, records):
        path = tmp_path / "eval_results.json"
        path.write_text(json.dumps({"metrics": {"samples": len(records)}, "records": records}), encoding="utf-8")
        return path

    def test_plan_merges_weak_and_uncovered(self, tmp_path):
        collect = tmp_path / "collect"
        collect.mkdir()
        (collect / "uncovered_tasks.jsonl").write_text(
            json.dumps({"task_id": 99, "samples": []}) + "\n", encoding="utf-8"
        )
        eval_path = self.eval_results(tmp_path, [
            {"task_id": 1, "reward": 1.0, "done": True},
            {"task_id": 2, "reward": 0.2, "done": True},
        ])
        plan = build_plan(eval_path, 0.5, collect, samples_per_task=2, sglang_port=30000,
                          served_model_name="qwen3.5-0.8b-rl")
        assert plan["target_tasks"] == 2          # task 2 + uncovered 99
        task_ids_file = collect / "self_improve_task_ids.txt"
        assert task_ids_file.read_text(encoding="utf-8") == "2\n99\n"
        assert plan["weak_tasks"] == 1
        assert plan["uncovered_tasks"] == 1
        # Command sequence sanity: serve -> collect -> prepare -> retrain.
        assert "sglang.launch_server" in plan["commands"][0]
        assert "--teacher-provider sglang-rl" in plan["commands"][1]
        assert "--min-reward 1e-12" in plan["commands"][1]
        assert "prepare_sft" in plan["commands"][2]
        assert "run_sft.sh" in plan["commands"][3]

    def test_plan_without_collect_root_uses_eval_dir(self, tmp_path):
        eval_path = self.eval_results(tmp_path, [{"task_id": 5, "reward": 0.0, "done": False}])
        plan = build_plan(eval_path, 0.5, None, 2, 30000, "m")
        assert plan["target_tasks"] == 1
        assert (tmp_path / "self_improve_task_ids.txt").read_text(encoding="utf-8") == "5\n"

    def test_empty_records_exit(self, tmp_path):
        eval_path = self.eval_results(tmp_path, [])
        with pytest.raises(SystemExit, match="no records"):
            build_plan(eval_path, 0.5, None, 2, 30000, "m")


class TestAggregateSeeds:
    @staticmethod
    def write_run(tmp_path, name, records):
        run_root = tmp_path / name
        run_root.mkdir()
        (run_root / "eval_results.json").write_text(
            json.dumps({"records": records, "metrics": {"samples": len(records)}}),
            encoding="utf-8",
        )
        return run_root

    @staticmethod
    def record(task_id, reward, success):
        detail = (
            {"r_type": 1.0, "r_att": 1.0, "r_option": 1.0, "r_price": 1.0}
            if success else {"r_type": 1.0, "r_att": 0.0, "r_option": 1.0, "r_price": 0.0}
        )
        return {"task_id": task_id, "reward": reward, "reward_detail": detail,
                "purchase_asin": "A" if success else "B", "goal_asin": "A"}

    def test_aggregate_pools_and_summarizes(self, tmp_path):
        seed42 = self.write_run(tmp_path, "seed_42", [
            self.record(1, 1.0, True), self.record(2, 0.0, False),
        ])
        seed43 = self.write_run(tmp_path, "seed_43", [
            self.record(1, 0.0, False), self.record(2, 1.0, True),
        ])
        result = aggregate([seed42, seed43])
        assert result["seeds"] == ["42", "43"]   # dir-name fallback
        summary = result["per_seed_summary"]["r_success"]
        assert summary["per_seed"] == [0.5, 0.5]
        assert summary["mean"] == 0.5
        pooled = result["pooled"]
        assert pooled["samples"] == 4
        assert pooled["r_success"] == 0.5
        # Wilson CI on 2/4 successes must bracket 0.5.
        assert pooled["r_success_ci95"][0] < 0.5 < pooled["r_success_ci95"][1]

    def test_missing_results_exit(self, tmp_path):
        with pytest.raises(SystemExit, match="no eval_results.json"):
            aggregate([tmp_path / "nowhere"])
