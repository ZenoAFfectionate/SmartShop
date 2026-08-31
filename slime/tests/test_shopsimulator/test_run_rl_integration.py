"""Bash-level integration test for run_rl.sh startup guards (A3/C2).

Drives run_rl.sh with CHECK_ONLY=1 and stub inputs far enough to prove:
1. an SFT dataset whose pruning depth mismatches the RL config aborts with
   exit 2 *before* any training launch (A3 acceptance);
2. a matching dataset passes the guard and the script reaches CHECK_ONLY
   output;
3. SAVE_INTERVAL (C2 periodic checkpointing) is wired into the train args.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

SLIME_ROOT = Path(__file__).resolve().parents[2]
RUN_RL = SLIME_ROOT / "examples/ShopSimulator/run_rl.sh"


def write_sft_data(path: Path, keep: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"messages": [], "metadata": {"context_keep_act_results": keep}}) + "\n",
        encoding="utf-8",
    )
    return path


def run_script(tmp_path: Path, sft_data: Path | None) -> subprocess.CompletedProcess:
    import shutil

    stub = tmp_path / "stub"
    stub.write_text("stub", encoding="utf-8")
    env = {
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "SLIME_PYTHON": shutil.which("python3") or "/usr/bin/python3",  # must be an existing path
        "MEGATRON_DIR": str(SLIME_ROOT),          # any existing directory
        "HF_CHECKPOINT": str(stub),
        "REF_MODEL_PATH": str(stub),
        "PI_BIN": "/usr/bin/env",                 # any executable
        "RUN_ROOT": str(tmp_path / "run"),
        "RAY_TEMP_DIR": str(tmp_path / "ray"),
        "CHECK_ONLY": "1",
        "HOME": str(tmp_path),
    }
    if sft_data is not None:
        env["SFT_TURN_DATA"] = str(sft_data)
    return subprocess.run(
        ["bash", str(RUN_RL)],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


@pytest.fixture(scope="module")
def consistent_run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("consistent")
    data = write_sft_data(tmp / "turn_examples.jsonl", 3)
    result = run_script(tmp, data)
    return result


class TestRunRlStartupGuards:
    def test_mismatched_pruning_aborts_before_launch(self, tmp_path):
        data = write_sft_data(tmp_path / "turn_examples.jsonl", 5)  # config says 3
        result = run_script(tmp_path, data)
        assert result.returncode == 2, result.stderr + result.stdout
        assert "context_keep_act_results" in (result.stdout + result.stderr)

    def test_matching_pruning_passes_guard(self, consistent_run):
        assert consistent_run.returncode == 0, consistent_run.stderr + consistent_run.stdout
        assert "NUM_ROLLOUTS=100" in consistent_run.stdout

    def test_legacy_dataset_without_field_aborts(self, tmp_path):
        data = tmp_path / "turn_examples.jsonl"
        data.write_text(json.dumps({"messages": [], "metadata": {}}) + "\n", encoding="utf-8")
        result = run_script(tmp_path, data)
        assert result.returncode == 2
        assert "predates" in (result.stdout + result.stderr)

    def test_save_interval_is_wired_into_train_args(self):
        # C2: the launcher must pass SAVE_INTERVAL (default 25) to --save-interval
        # instead of the historical "only save at the very end" behaviour.
        text = RUN_RL.read_text(encoding="utf-8")
        assert 'SAVE_INTERVAL="${SAVE_INTERVAL:-25}"' in text
        assert '--save-interval "${SAVE_INTERVAL}"' in text
        assert '--save-interval "${TOTAL_ROLLOUTS}"' not in text


class TestRunEvalDatasetOverride:
    """PLAN.md stage A depends on PROMPT_DATA being overridable in run_eval.sh
    (dev100 validation must not silently evaluate official_test_200)."""

    RUN_EVAL = SLIME_ROOT / "examples/ShopSimulator/run_eval.sh"

    def test_prompt_data_is_env_overridable(self):
        text = self.RUN_EVAL.read_text(encoding="utf-8")
        assert 'PROMPT_DATA="${PROMPT_DATA:-' in text
        # The default dataset is still official_test_200, but only as a fallback
        # inside the ${VAR:-default} form.
        assert 'data/tasks_v2/official_test_200.jsonl}"' in text

    def test_eval_config_is_env_overridable(self):
        text = self.RUN_EVAL.read_text(encoding="utf-8")
        assert 'EVAL_CONFIG="${EVAL_CONFIG:-' in text

    def test_system_prompt_override_is_forwarded(self):
        # D5: SHOP_SYSTEM_PROMPT[_FILE] must be exported into the Ray runtime env.
        text = self.RUN_EVAL.read_text(encoding="utf-8")
        assert '"SHOP_SYSTEM_PROMPT"' in text
        assert '"SHOP_SYSTEM_PROMPT_FILE"' in text
