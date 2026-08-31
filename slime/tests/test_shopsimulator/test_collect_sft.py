"""Tests for examples.ShopSimulator.collect_sft (candidate gate & traces)."""

from __future__ import annotations

import json

import pytest

from examples.ShopSimulator.collect_sft import (
    context_snapshots,
    context_trace_matches,
    evaluate_candidate,
    normalize_context_traces,
    read_index,
)


def tool_end(name: str, call_id: str = "c1", is_error: bool = False) -> dict:
    return {
        "type": "tool_execution_end",
        "toolName": name,
        "toolCallId": call_id,
        "isError": is_error,
        "result": {"content": [{"type": "text", "text": "ok"}]},
    }


def tool_start(name: str, call_id: str = "c1") -> dict:
    return {"type": "tool_execution_start", "toolName": name, "toolCallId": call_id}


def good_events(turns: int = 3) -> list[dict]:
    events = [tool_start("shop_reset", "r"), tool_end("shop_reset", "r")]
    for i in range(turns):
        events.append(tool_start("shop_act", f"a{i}"))
        events.append(tool_end("shop_act", f"a{i}"))
    return events


def candidate(**overrides) -> dict:
    row = {
        "context_trace_valid": True,
        "split": "sft",
        "error": None,
        "done": True,
        "reward": 1.0,
        "events": good_events(),
    }
    row.update(overrides)
    return row


class TestEvaluateCandidate:
    def test_good_trajectory_is_accepted(self):
        accepted, reasons = evaluate_candidate(candidate(), min_reward=1e-12)
        assert accepted
        assert reasons == []

    @pytest.mark.parametrize(
        "overrides, expected_reason",
        [
            ({"context_trace_valid": False}, "context_trace_mismatch"),
            ({"split": "dev"}, "wrong_split"),
            ({"error": "boom"}, "runtime_error"),
            ({"done": False}, "not_done"),
            ({"reward": 0.0}, "reward_below_threshold"),
        ],
    )
    def test_scalar_rejection_reasons(self, overrides, expected_reason):
        accepted, reasons = evaluate_candidate(candidate(**overrides), min_reward=1e-12)
        assert not accepted
        assert expected_reason in reasons

    def test_tool_error_flag(self):
        events = good_events()
        events[1]["isError"] = True
        accepted, reasons = evaluate_candidate(candidate(events=events), min_reward=1e-12)
        assert "tool_error" in reasons

    def test_illegal_tool(self):
        events = good_events() + [tool_start("bash", "b"), tool_end("bash", "b")]
        _, reasons = evaluate_candidate(candidate(events=events), min_reward=1e-12)
        assert "illegal_tool" in reasons
        assert "non_shop_act_after_reset" in reasons

    def test_reset_count_zero_and_twice(self):
        events = [tool_start("shop_act", "a0"), tool_end("shop_act", "a0")]
        _, reasons = evaluate_candidate(candidate(events=events), min_reward=1e-12)
        assert "reset_count" in reasons
        assert "reset_not_first" in reasons

        events = good_events()[:2] + good_events()[:2] + good_events()[2:]
        _, reasons = evaluate_candidate(candidate(events=events), min_reward=1e-12)
        assert "reset_count" in reasons

    def test_reset_not_first(self):
        events = [tool_start("shop_act", "a0"), tool_end("shop_act", "a0")] + good_events()
        _, reasons = evaluate_candidate(candidate(events=events), min_reward=1e-12)
        assert "reset_not_first" in reasons

    def test_no_tool_events(self):
        _, reasons = evaluate_candidate(candidate(events=[]), min_reward=1e-12)
        assert "no_tool_events" in reasons

    def test_tool_pair_mismatch(self):
        events = good_events()
        events.append(tool_start("shop_act", "a99"))  # start without matching end
        _, reasons = evaluate_candidate(candidate(events=events), min_reward=1e-12)
        assert "tool_pair_mismatch" in reasons

    def test_pairing_ignored_when_call_ids_absent(self):
        # Older pi builds do not echo call ids onto execution events; the
        # strict pairing check must not fire when ids are unavailable.
        events = good_events()
        for event in events:
            event.pop("toolCallId", None)
        accepted, reasons = evaluate_candidate(candidate(events=events), min_reward=1e-12)
        assert accepted
        assert "tool_pair_mismatch" not in reasons

    def test_min_reward_threshold_boundary(self):
        # reward exactly at the threshold is accepted (check is strict <).
        accepted, _ = evaluate_candidate(candidate(reward=0.5), min_reward=0.5)
        assert accepted
        accepted, reasons = evaluate_candidate(candidate(reward=0.49), min_reward=0.5)
        assert not accepted
        assert "reward_below_threshold" in reasons

    def test_reasons_are_deduplicated_and_sorted(self):
        row = candidate(done=False, error="x", reward=0.0)
        _, reasons = evaluate_candidate(row, min_reward=1e-12)
        assert reasons == sorted(set(reasons))


class TestContextSnapshots:
    @staticmethod
    def messages_with_acts(n: int) -> list[dict]:
        messages: list[dict] = [{"role": "system", "content": "s"}]
        for i in range(n):
            messages.append({"role": "assistant", "content": [{"type": "text", "text": f"t{i}"}]})
            messages.append({
                "role": "toolResult",
                "toolName": "shop_act",
                "toolCallId": f"a{i}",
                "content": [{"type": "text", "text": f"act-result-{i}"}],
            })
        messages.append({"role": "assistant", "content": [{"type": "text", "text": "final"}]})
        return messages

    def test_old_act_results_are_pruned(self):
        messages = self.messages_with_acts(5)
        snapshots = context_snapshots(messages, keep_act_results=2)
        assert len(snapshots) == 6  # one snapshot per assistant message
        final_snapshot = snapshots[-1]["messages"]
        act_texts = [
            part["text"]
            for message in final_snapshot
            if message.get("role") == "toolResult" and message.get("toolName") == "shop_act"
            for part in message["content"]
        ]
        # Only the last two act results survive verbatim; the rest are replaced.
        assert "act-result-3" in act_texts and "act-result-4" in act_texts
        assert "act-result-0" not in act_texts
        assert sum("已裁剪" in text for text in act_texts) == 3

    def test_keep_all_when_fewer_than_budget(self):
        messages = self.messages_with_acts(1)
        snapshots = context_snapshots(messages, keep_act_results=3)
        act_texts = [
            part["text"]
            for message in snapshots[-1]["messages"]
            if message.get("role") == "toolResult"
            for part in message["content"]
        ]
        assert act_texts == ["act-result-0"]

    def test_input_messages_not_mutated(self):
        messages = self.messages_with_acts(4)
        before = json.dumps(messages)
        context_snapshots(messages, keep_act_results=1)
        assert json.dumps(messages) == before


class TestContextTraces:
    def test_normalize_inserts_system_prompt(self):
        traces = [{"messages": [{"role": "user", "content": "u"}], "request_index": 0}]
        normalized = normalize_context_traces(traces, system_prompt="SYS")
        assert normalized[0]["messages"][0] == {"role": "system", "content": "SYS"}

    def test_matches_equal_length(self):
        reconstructed = [{"messages": [{"role": "user", "content": "u"}]}]
        actual = [{"messages": [{"role": "user", "content": "u"}]}]
        assert context_trace_matches(reconstructed, actual)

    def test_matches_one_extra_terminal_tool_result(self):
        reconstructed = [{"messages": [{"role": "user", "content": "u"}]}]
        actual = reconstructed + [{"messages": [{"role": "toolResult", "content": "x"}]}]
        assert context_trace_matches(reconstructed, actual)

    def test_rejects_one_extra_non_tool_result(self):
        reconstructed = [{"messages": [{"role": "user", "content": "u"}]}]
        actual = reconstructed + [{"messages": [{"role": "assistant", "content": "x"}]}]
        assert not context_trace_matches(reconstructed, actual)

    def test_rejects_content_mismatch(self):
        reconstructed = [{"messages": [{"role": "user", "content": "u"}]}]
        actual = [{"messages": [{"role": "user", "content": "different"}]}]
        assert not context_trace_matches(reconstructed, actual)

    def test_rejects_wrong_length(self):
        reconstructed = [{"messages": []}]
        actual = reconstructed + reconstructed + [{"messages": []}]
        assert not context_trace_matches(reconstructed, actual)


class TestReadIndex:
    @staticmethod
    def write(tmp_path, rows):
        path = tmp_path / "tasks.jsonl"
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        return path

    @staticmethod
    def row(task_id: int, split: str = "sft") -> dict:
        return {"metadata": {"task_id": task_id, "split": split}, "prompt": [{"role": "user", "content": "go"}]}

    def test_valid_index(self, tmp_path):
        rows = read_index(self.write(tmp_path, [self.row(1), self.row(2)]))
        assert [row["metadata"]["task_id"] for row in rows] == [1, 2]

    def test_duplicate_task_id_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="duplicate task_id"):
            read_index(self.write(tmp_path, [self.row(1), self.row(1)]))

    def test_non_sft_split_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="split=sft"):
            read_index(self.write(tmp_path, [self.row(1, split="dev")]))

    def test_invalid_json_rejected(self, tmp_path):
        path = tmp_path / "tasks.jsonl"
        path.write_text("{not json}\n", encoding="utf-8")
        with pytest.raises(ValueError, match="invalid JSON"):
            read_index(path)
