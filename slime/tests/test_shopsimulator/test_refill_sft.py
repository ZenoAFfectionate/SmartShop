"""Tests for refill_sft (C1: uncovered-task refill planning)."""

from __future__ import annotations

import json

import pytest

from examples.ShopSimulator.refill_sft import build_plan, classify_uncovered


def uncovered_row(task_id: int, tool_calls: list):
    return {
        "task_id": task_id,
        "attempted_samples": len(tool_calls),
        "samples": [
            {"sample_id": i, "tool_calls": calls, "reward": 0.0, "done": False}
            for i, calls in enumerate(tool_calls)
        ],
    }


class TestClassifyUncovered:
    def test_separates_budget_bound(self):
        rows = [
            uncovered_row(1, [12, 30]),   # below budget -> refillable
            uncovered_row(2, [40]),       # hit the cap -> budget bound
            uncovered_row(3, [40, 38]),   # one sample hit the cap -> budget bound
            uncovered_row(4, []),         # no samples -> refillable
        ]
        refillable, budget_bound = classify_uncovered(rows, max_turns=40)
        assert [row["task_id"] for row in refillable] == [1, 4]
        assert [row["task_id"] for row in budget_bound] == [2, 3]


class TestBuildPlan:
    def test_writes_task_ids_and_command(self, tmp_path):
        rows = [uncovered_row(10, [12]), uncovered_row(11, [40])]
        (tmp_path / "uncovered_tasks.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        (tmp_path / "summary.json").write_text(
            json.dumps({"max_turns": 40, "candidates": 2}), encoding="utf-8"
        )
        plan = build_plan(tmp_path, max_turns=None, samples_per_task=2)
        assert plan["refillable_tasks"] == 1
        assert plan["budget_bound_tasks"] == 1
        assert plan["refill_task_ids"] == [10]
        assert plan["budget_bound_task_ids"] == [11]
        task_ids_file = tmp_path / "refill_task_ids.txt"
        assert task_ids_file.read_text(encoding="utf-8") == "10\n"
        assert "--task-ids-file" in plan["recommended_command"]
        assert str(task_ids_file) in plan["recommended_command"]

    def test_requires_uncovered_file(self, tmp_path):
        with pytest.raises(SystemExit, match="uncovered_tasks.jsonl"):
            build_plan(tmp_path, max_turns=40, samples_per_task=2)

    def test_all_budget_bound_recommends_raising_turns(self, tmp_path):
        (tmp_path / "uncovered_tasks.jsonl").write_text(
            json.dumps(uncovered_row(7, [40])) + "\n", encoding="utf-8"
        )
        plan = build_plan(tmp_path, max_turns=40, samples_per_task=2)
        assert plan["refillable_tasks"] == 0
        assert plan["recommended_command"] is None
        assert "max-turns" in plan["recommendation"]

    def test_max_turns_defaults_from_summary(self, tmp_path):
        (tmp_path / "uncovered_tasks.jsonl").write_text(
            json.dumps(uncovered_row(1, [5])) + "\n", encoding="utf-8"
        )
        (tmp_path / "summary.json").write_text(
            json.dumps({"max_turns": 24}), encoding="utf-8"
        )
        plan = build_plan(tmp_path, max_turns=None, samples_per_task=2)
        assert plan["max_turns"] == 24
        assert plan["refillable_tasks"] == 1  # 5 < 24

    def test_unusable_summary_requires_explicit_max_turns(self, tmp_path):
        (tmp_path / "uncovered_tasks.jsonl").write_text(
            json.dumps(uncovered_row(1, [5])) + "\n", encoding="utf-8"
        )
        (tmp_path / "summary.json").write_text(json.dumps({}), encoding="utf-8")
        with pytest.raises(SystemExit, match="max-turns"):
            build_plan(tmp_path, max_turns=None, samples_per_task=2)
