"""Tests for format_eval_table (G1: full README table row rendering)."""

from __future__ import annotations

import json

import pytest

from examples.ShopSimulator.format_eval_table import (
    TABLE_COLUMNS,
    format_header,
    format_row,
    load_metrics,
)


def sample_metrics():
    return {
        "samples": 200,
        "done_rate": 0.95,
        "r_loose": 0.627786,
        "r_hard": 0.35453,
        "r_success": 0.31,
        "right_product": 0.62,
        "r_type": 0.9,
        "r_att": 0.8,
        "r_option": 0.85,
        "r_price": 0.7,
        "mean_model_turns": 12.4,
        "turn_limit_rate": 0.05,
        "pass_at_k": {
            "tasks": 200,
            "samples_per_task": 1,
            "pass_positive_reward": 0.905,
            "pass_success": 0.31,
        },
    }


class TestFormatRow:
    def test_full_row_renders_every_column(self):
        row = format_row("RL", sample_metrics())
        cells = [cell.strip() for cell in row.strip("|").split("|")]
        assert len(cells) == len(TABLE_COLUMNS)
        assert cells[0] == "RL"
        assert cells[1] == "90.5%"       # positive pass@1
        assert cells[2] == "31.0%"       # strict success
        assert cells[3] == "0.627786"    # r_loose
        assert cells[4] == "0.354530"    # r_hard
        assert cells[5] == "0.900000"    # r_type
        assert cells[9] == "95.0%"       # done rate
        assert cells[10] == "62.0%"      # right product
        assert cells[11] == "5.0%"       # turn_limit rate
        assert cells[12] == "12.4"       # mean turns

    def test_missing_metrics_render_as_placeholder(self):
        row = format_row("Base", {"samples": 200})
        cells = [cell.strip() for cell in row.strip("|").split("|")]
        assert cells[0] == "Base"
        for cell in cells[1:]:
            assert cell == "—"

    def test_positive_pass1_falls_back_to_records(self):
        metrics = {
            "samples": 4,
            "pass_at_k": None,
        }
        row = format_row("X", metrics)
        # no records either -> placeholder
        assert "| — |" in row

    def test_header_shape(self):
        header = format_header().splitlines()
        assert header[0].count("|") == len(TABLE_COLUMNS) + 1
        # First column (model label) is left-aligned, the rest right-aligned.
        separators = [cell.strip() for cell in header[1].strip("|").split("|")]
        assert separators[0] == "---"
        assert separators[1:] == ["---:"] * (len(TABLE_COLUMNS) - 1)


class TestLoadMetrics:
    @staticmethod
    def write_results(tmp_path, metrics, records=None, name="eval_results.json"):
        path = tmp_path / name
        payload = {"metrics": metrics, "records": records or [], "command": "x"}
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_loads_from_file_and_directory(self, tmp_path):
        metrics = sample_metrics()
        path = self.write_results(tmp_path, metrics)
        assert load_metrics(path)["_positive_pass1"] == 0.905
        assert load_metrics(tmp_path)["_positive_pass1"] == 0.905  # directory form

    def test_positive_pass1_falls_back_to_records_count(self, tmp_path):
        metrics = {"samples": 4, "pass_at_k": None}
        records = [
            {"reward": 0.0}, {"reward": 0.5}, {"reward": 1.0}, {"reward": 0.0},
        ]
        path = self.write_results(tmp_path, metrics, records)
        assert load_metrics(path)["_positive_pass1"] == 0.5

    def test_missing_file_exits(self, tmp_path):
        with pytest.raises(SystemExit, match="no eval_results.json"):
            load_metrics(tmp_path / "nowhere")


class TestCliEndToEnd:
    def test_main_renders_rows(self, tmp_path, capsys, monkeypatch):
        from examples.ShopSimulator import format_eval_table
        run_root = tmp_path / "run_rl"
        run_root.mkdir()
        (run_root / "eval_results.json").write_text(
            json.dumps({"metrics": sample_metrics(), "records": []}), encoding="utf-8"
        )
        monkeypatch.setattr(
            "sys.argv",
            ["format_eval_table.py", "--header", "--label", "RL", "--results", str(run_root)],
        )
        code = format_eval_table.main()
        assert code == 0
        out = capsys.readouterr().out
        assert "| 模型 |" in out            # header printed
        assert "| RL | 90.5% |" in out      # row rendered
