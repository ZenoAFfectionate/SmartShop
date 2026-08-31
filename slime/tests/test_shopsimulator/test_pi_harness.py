"""Tests for pi_harness event parsing (A4: model-turn budget semantics)."""

from __future__ import annotations

from examples.ShopSimulator.pi_harness import (
    PiRunResult,
    _effective_turns,
    parse_pi_event,
)


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
