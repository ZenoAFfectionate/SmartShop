"""Tests for select_checkpoint (C2: dev-based model selection)."""

from __future__ import annotations

from examples.ShopSimulator.select_checkpoint import (
    discover_results,
    load_run,
    rank_runs,
)


class TestSelectCheckpoint:
    @staticmethod
    def write_run(tmp_path, name, r_success, r_loose, command=""):
        run_root = tmp_path / name
        run_root.mkdir()
        payload = {
            "command": command or "python train.py",
            "metrics": {
                "samples": 100,
                "r_success": r_success,
                "r_loose": r_loose,
                "right_product": r_success,
                "r_success_ci95": [0.0, 1.0],
            },
        }
        (run_root / "eval_results.json").write_text(
            __import__("json").dumps(payload), encoding="utf-8"
        )
        return run_root

    def test_discover_recursive(self, tmp_path):
        self.write_run(tmp_path, "run_a", 0.1, 0.3)
        self.write_run(tmp_path, "run_b", 0.2, 0.4)
        paths = discover_results([tmp_path])
        assert len(paths) == 2

    def test_discover_accepts_direct_file(self, tmp_path):
        run_root = self.write_run(tmp_path, "run_a", 0.1, 0.3)
        paths = discover_results([run_root / "eval_results.json"])
        assert len(paths) == 1

    def test_rank_orders_by_metric(self, tmp_path):
        self.write_run(tmp_path, "run_a", 0.10, 0.30)
        self.write_run(tmp_path, "run_b", 0.31, 0.62)
        self.write_run(tmp_path, "run_c", 0.20, 0.40)
        ranked = rank_runs([load_run(path) for path in discover_results([tmp_path])], "r_success")
        assert [run["checkpoint"] for run in ranked] == ["run_b", "run_c", "run_a"]

    def test_rank_ties_break_on_r_loose(self):
        runs = [
            {"checkpoint": "x", "r_success": 0.5, "r_loose": 0.4, "samples": 10},
            {"checkpoint": "y", "r_success": 0.5, "r_loose": 0.6, "samples": 10},
        ]
        ranked = rank_runs(runs, "r_success")
        assert ranked[0]["checkpoint"] == "y"

    def test_load_run_parses_command_checkpoint(self, tmp_path):
        run_root = self.write_run(
            tmp_path, "run_x", 0.1, 0.3,
            command="python train.py --save-hf /runs/rl/hf/rollout_75",
        )
        run = load_run(run_root / "eval_results.json")
        assert run["checkpoint"] == "rollout_75"
        assert run["samples"] == 100
