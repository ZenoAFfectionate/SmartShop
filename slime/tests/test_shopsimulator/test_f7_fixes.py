"""Tests for the F7 small fixes: api-key required, rate-limit default,
explicit adapter_url=None, README metric documentation."""

from __future__ import annotations

import asyncio
import json

import pytest

from examples.ShopSimulator.collect_sft import async_main, build_parser
from examples.ShopSimulator.pi_harness import run_pi


class TestParserDefaults:
    def test_api_key_file_has_no_default(self):
        # F7-#1: the old /root/api.txt default silently picked up stale keys.
        args = build_parser().parse_args(["--output-dir", "/tmp/x"])
        assert args.api_key_file is None

    def test_launches_per_minute_defaults_to_30(self):
        # F7-#4: a conservative default instead of unlimited (0 still disables).
        args = build_parser().parse_args(["--output-dir", "/tmp/x"])
        assert args.launches_per_minute == 30

    def test_teacher_provider_defaults_to_deepseek(self):
        args = build_parser().parse_args(["--output-dir", "/tmp/x"])
        assert args.teacher_provider == "deepseek"

    def test_keep_act_results_defaults_to_none(self):
        args = build_parser().parse_args(["--output-dir", "/tmp/x"])
        assert args.keep_act_results is None


def write_tasks(tmp_path) -> object:
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text(
        json.dumps({"metadata": {"task_id": 1, "split": "sft"},
                    "prompt": [{"role": "user", "content": "go"}]}) + "\n",
        encoding="utf-8",
    )
    return tasks


def make_args(tmp_path, **overrides):
    argv = [
        "--output-dir", str(tmp_path / "out"),
        "--tasks", str(write_tasks(tmp_path)),
    ]
    parser = build_parser()
    args = parser.parse_args(argv)
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


class TestApiKeyRequired:
    def test_missing_api_key_aborts_unless_dry_run(self, tmp_path):
        args = make_args(tmp_path)  # api_key_file=None, dry_run=False
        with pytest.raises(SystemExit, match="--api-key-file is required"):
            asyncio.run(async_main(args))

    def test_dry_run_never_needs_api_key(self, tmp_path):
        args = make_args(tmp_path, dry_run=True)
        summary = asyncio.run(async_main(args))
        assert summary["selected_tasks"] == 1

    def test_empty_key_file_rejected(self, tmp_path):
        key_file = tmp_path / "key.txt"
        key_file.write_text("\n", encoding="utf-8")
        args = make_args(tmp_path, api_key_file=key_file)
        with pytest.raises(SystemExit, match="first line is empty"):
            asyncio.run(async_main(args))


class TestRunPiAdapterUrlContract:
    async def _run(self, **kwargs):
        base = dict(
            session_id="s", task_id=1, env_url="http://127.0.0.1:5000",
            prompt="go", timeout_sec=1.0,
        )
        base.update(kwargs)
        return await run_pi(**base)

    def test_teacher_mode_without_adapter_url_is_valid(self):
        # F7-#2: adapter_url=None plus an explicit model_base_url must be the
        # supported teacher path (no fake address needed). The call proceeds
        # past argument validation; it fails later at pi startup, which we do
        # not reach because pi_bin does not exist here.
        with pytest.raises(Exception) as excinfo:
            asyncio.run(self._run(
                adapter_url=None,
                model_base_url="https://api.deepseek.com",
                pi_bin="/nonexistent-pi",
            ))
        assert "neither was provided" not in str(excinfo.value)

    def test_neither_url_rejected_early(self):
        # Neither teacher base-url nor adapter -> contract violation before
        # any subprocess or tempdir is created.
        with pytest.raises(ValueError, match="neither was provided"):
            asyncio.run(self._run(adapter_url=None, model_base_url=None))

    def test_student_mode_still_works_via_adapter_url(self):
        with pytest.raises(Exception) as excinfo:
            asyncio.run(self._run(
                adapter_url="http://127.0.0.1:18080",
                pi_bin="/nonexistent-pi",
            ))
        assert "neither was provided" not in str(excinfo.value)


class TestReadmeDocumentsMetrics:
    """F7-#6: the reward formulas must be documented, not folklore."""

    @staticmethod
    def readme() -> str:
        from pathlib import Path

        return (Path(__file__).resolve().parents[3] / "README.md").read_text(encoding="utf-8")

    def test_loose_formula_present(self):
        text = self.readme()
        assert "r_loose" in text
        assert "|U_att∩Y_att|" in text  # additive formula skeleton
        assert "arXiv:2601.18225" in text

    def test_hard_formula_present(self):
        text = self.readme()
        assert "r_type × r_att × r_option × r_price" in text

    def test_subscore_table_present(self):
        text = self.readme()
        for name in ("r_type", "r_att", "r_option", "r_price"):
            assert name in text

    def test_environment_requirements_banner_present(self):
        # F7-#5: Linux + NVIDIA GPU must be stated up front.
        text = self.readme()
        assert "环境要求" in text
        assert "NVIDIA GPU" in text
        assert "macOS" in text
