"""Tests for pi_harness event parsing, error-prefix parity, prompt override & run_pi URL contract."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

import pytest

from examples.ShopSimulator import pi_harness as pi_harness_module
from examples.ShopSimulator.pi_harness import (
    AGENT_ERROR_PREFIX,
    AUTHORITATIVE_EVENT_TYPES,
    DEFAULT_SYSTEM_PROMPT,
    INFRASTRUCTURE_ERROR_PREFIX,
    PiRunResult,
    _classify_tool_error,
    _effective_turns,
    _release_env_slot,
    effective_system_prompt,
    parse_pi_event,
    run_pi,
)


EXAMPLE_DIR = Path(__file__).resolve().parents[2] / "examples/ShopSimulator"
TS_SOURCE = (EXAMPLE_DIR / "shop_extension.ts").read_text(encoding="utf-8")


def ts_constant(name: str) -> str:
    match = re.search(r'const\s+' + re.escape(name) + r'\s*=\s*"((?:[^"\\]|\\.)*)"', TS_SOURCE)
    assert match, f"shop_extension.ts no longer declares {name}"
    return match.group(1).encode().decode("unicode_escape")


def result() -> PiRunResult:
    return PiRunResult(exit_code=-1)


class TestModelTurnCounting:
    def test_turn_start_increments_model_turns(self):
        parsed = result()
        parse_pi_event(parsed, {"type": "turn_start"})
        parse_pi_event(parsed, {"type": "turn_start"})
        assert parsed.model_turns == 2
        assert parsed.tool_calls == 0

    def test_tool_execution_end_increments_tool_calls_only(self):
        parsed = result()
        parse_pi_event(parsed, {"type": "tool_execution_end", "toolName": "shop_act", "result": {}})
        assert parsed.tool_calls == 1
        assert parsed.model_turns == 0

    def test_one_turn_with_parallel_tool_calls(self):
        # A single model turn may issue several tool calls; the budget unit
        # must stay at one turn, matching the adapter-side turn cap.
        parsed = result()
        parse_pi_event(parsed, {"type": "turn_start"})
        for _ in range(3):
            parse_pi_event(parsed, {"type": "tool_execution_end", "toolName": "shop_act", "result": {}})
        assert parsed.model_turns == 1
        assert parsed.tool_calls == 3
        assert _effective_turns(parsed) == 1

    def test_effective_turns_falls_back_without_turn_events(self):
        parsed = result()
        for _ in range(5):
            parse_pi_event(parsed, {"type": "tool_execution_end", "toolName": "shop_act", "result": {}})
        assert parsed.model_turns == 0
        assert _effective_turns(parsed) == 5

    def test_shop_act_done_extracted_from_details(self):
        parsed = result()
        parse_pi_event(parsed, {
            "type": "tool_execution_end",
            "toolName": "shop_act",
            "result": {"details": {
                "done": True,
                "reward": 0.75,
                "reward_detail": {"r_type": 1.0, "r_att": 0.5, "r_price": True},
                "purchase_asin": "A1",
                "goal_asin": "A2",
                "env_idx": 3,
            }},
        })
        assert parsed.done
        assert parsed.reward == 0.75
        assert parsed.reward_detail == {"r_type": 1.0, "r_att": 0.5, "r_price": 1.0}
        assert parsed.purchase_asin == "A1"
        assert parsed.goal_asin == "A2"
        assert parsed.env_idx == 3

    def test_over_flag_sets_terminal(self):
        parsed = result()
        parse_pi_event(parsed, {
            "type": "tool_execution_end",
            "toolName": "shop_act",
            "result": {"details": {"over": True}},
        })
        assert parsed.over and not parsed.done

    def test_terminal_clears_errors(self):
        parsed = result()
        parse_pi_event(parsed, {
            "type": "tool_execution_end",
            "toolName": "shop_act",
            "isError": True,
            "result": {"content": [{"type": "text", "text": "[shop_infrastructure] HTTP 503"}]},
        })
        assert parsed.error == "HTTP 503"
        assert parsed.error_kind == "infrastructure_error"
        parse_pi_event(parsed, {
            "type": "tool_execution_end",
            "toolName": "shop_act",
            "result": {"details": {"done": True, "reward": 1.0}},
        })
        assert parsed.done
        assert parsed.error is None  # terminal result dominates sibling errors

    def test_infrastructure_error_classified(self):
        parsed = result()
        parse_pi_event(parsed, {
            "type": "tool_execution_end",
            "toolName": "shop_act",
            "isError": True,
            "result": {"content": [{"type": "text", "text": "[shop_infrastructure] HTTP 500"}]},
        })
        assert parsed.error_kind == "infrastructure_error"
        assert parsed.error == "HTTP 500"

    def test_agent_tool_error_classified(self):
        parsed = result()
        parse_pi_event(parsed, {
            "type": "tool_execution_end",
            "toolName": "shop_act",
            "isError": True,
            "result": {"content": [{"type": "text", "text": "[shop_agent] call reset first"}]},
        })
        assert parsed.tool_errors[-1]["kind"] == "agent_tool_error"


class TestPrefixParity:
    def test_infrastructure_prefix_matches(self):
        assert ts_constant("INFRASTRUCTURE_ERROR_PREFIX") == INFRASTRUCTURE_ERROR_PREFIX

    def test_agent_prefix_matches(self):
        assert ts_constant("AGENT_ERROR_PREFIX") == AGENT_ERROR_PREFIX

    def test_prefixes_are_distinct_and_bracketed(self):
        # A shared shape keeps _classify_tool_error's startswith() ordering safe.
        for prefix in (INFRASTRUCTURE_ERROR_PREFIX, AGENT_ERROR_PREFIX):
            assert prefix.startswith("[") and prefix.endswith("]")
        assert INFRASTRUCTURE_ERROR_PREFIX != AGENT_ERROR_PREFIX

    def test_classification_round_trip(self):
        # The classification the Python side performs on TS-tagged messages.
        kind, message = _classify_tool_error(f"{INFRASTRUCTURE_ERROR_PREFIX} HTTP 500")
        assert kind == "infrastructure_error" and message == "HTTP 500"
        kind, message = _classify_tool_error(f"{AGENT_ERROR_PREFIX} call reset first")
        assert kind == "agent_tool_error" and message == "call reset first"

    def test_python_comment_references_the_guard_test(self):
        # Keep the cross-reference discoverable from the Python side too.
        py_source = (EXAMPLE_DIR / "pi_harness.py").read_text(encoding="utf-8")
        assert "test_pi_harness" in py_source


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


class TestRunPiAdapterUrlContract:
    async def _run(self, **kwargs):
        base = dict(
            session_id="s", task_id=1, env_url="http://127.0.0.1:5000",
            prompt="go", timeout_sec=1.0,
        )
        base.update(kwargs)
        return await run_pi(**base)

    def test_teacher_mode_without_adapter_url_is_valid(self):
        # adapter_url=None plus an explicit model_base_url must be the supported
        # teacher path (no fake address needed). The call proceeds past argument
        # validation; it fails later at pi startup, which we do not reach because
        # pi_bin does not exist here.
        with pytest.raises(Exception) as excinfo:
            asyncio.run(self._run(
                adapter_url=None,
                model_base_url="https://api.deepseek.com",
                pi_bin="/nonexistent-pi",
            ))
        assert "neither was provided" not in str(excinfo.value)

    def test_neither_url_rejected_early(self):
        # Neither teacher base-url nor adapter -> contract violation before any
        # subprocess or tempdir is created.
        with pytest.raises(ValueError, match="neither was provided"):
            asyncio.run(self._run(adapter_url=None, model_base_url=None))

    def test_student_mode_still_works_via_adapter_url(self):
        with pytest.raises(Exception) as excinfo:
            asyncio.run(self._run(
                adapter_url="http://127.0.0.1:18080",
                pi_bin="/nonexistent-pi",
            ))
        assert "neither was provided" not in str(excinfo.value)


class TestAuthoritativeEventTypes:
    def test_non_empty_set(self):
        assert AUTHORITATIVE_EVENT_TYPES
        assert isinstance(AUTHORITATIVE_EVENT_TYPES, set)

    def test_covers_event_types_parse_pi_event_needs(self):
        # parse_pi_event relies on these to compute model_turns and the reward,
        # so they must always be part of the captured stream.
        assert "turn_start" in AUTHORITATIVE_EVENT_TYPES
        assert "tool_execution_end" in AUTHORITATIVE_EVENT_TYPES

    def test_collect_sft_shares_the_same_constant(self):
        # collect_sft imports this from pi_harness instead of keeping a private
        # copy, so the SFT and RL capture filters can never drift apart.
        from examples.ShopSimulator.collect_sft import AUTHORITATIVE_EVENT_TYPES as sft_types

        assert sft_types is AUTHORITATIVE_EVENT_TYPES


class TestReleaseEnvSlot:
    """Belt-and-braces slot return: abnormal pi exits must free the env slot.

    Regression guard for the 2026-09-01 incident where SIGKILLed pi processes
    leaked all 20 ShopSimulator slots and the training silently degraded to
    zero-reward rollouts.
    """

    DEAD_ENV_URL = "http://127.0.0.1:59999/api/shop_agent"

    async def _run(self, **kwargs):
        base = dict(
            session_id="sid-release-test",
            task_id=1,
            env_url=self.DEAD_ENV_URL,
            adapter_url="http://127.0.0.1:59998",
            prompt="go",
            timeout_sec=5.0,
        )
        base.update(kwargs)
        return await run_pi(**base)

    def test_release_survives_dead_env(self):
        # Best-effort by design: an unreachable env must not raise.
        asyncio.run(_release_env_slot(self.DEAD_ENV_URL, "sid-x"))

    def test_release_skipped_without_env_url(self):
        asyncio.run(_release_env_slot("", "sid-x"))

    def test_run_pi_releases_slot_on_nonzero_exit(self, monkeypatch):
        calls = []

        async def fake_release(env_url, rollout_session_id):
            calls.append((env_url, rollout_session_id))

        monkeypatch.setattr(pi_harness_module, "_release_env_slot", fake_release)
        parsed = asyncio.run(self._run(pi_bin="/bin/false"))
        assert parsed.exit_code != 0
        assert calls == [(self.DEAD_ENV_URL, "sid-release-test")]

    def test_run_pi_releases_slot_even_on_clean_exit(self, monkeypatch):
        # The release is unconditional (idempotent server-side): even a clean
        # pi exit triggers it, closing every leak path with one mechanism.
        calls = []

        async def fake_release(env_url, rollout_session_id):
            calls.append((env_url, rollout_session_id))

        monkeypatch.setattr(pi_harness_module, "_release_env_slot", fake_release)
        parsed = asyncio.run(self._run(pi_bin="/bin/true"))
        assert parsed.exit_code == 0
        assert calls == [(self.DEAD_ENV_URL, "sid-release-test")]
