"""Tests for prepare_sft helpers and the SFT/RL consistency verifier (A3)."""

from __future__ import annotations

import json

import pytest

from examples.ShopSimulator.prepare_sft import percentile, text_content, to_qwen_message
from examples.ShopSimulator.verify_consistency import verify


class TestPrepareSftHelpers:
    def test_text_content_variants(self):
        assert text_content("plain") == "plain"
        assert text_content(None) == ""
        assert text_content(123) == "123"
        assert text_content([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]) == "ab"
        assert text_content([{"type": "image"}]) == ""

    def test_to_qwen_message_system_user(self):
        assert to_qwen_message({"role": "system", "content": "s"}, trainable=True) == {
            "role": "system", "content": "s", "step_loss_mask": 0,
        }
        user = to_qwen_message({"role": "user", "content": [{"type": "text", "text": "u"}]}, trainable=True)
        assert user["step_loss_mask"] == 0

    def test_to_qwen_message_tool_result(self):
        message = {"role": "toolResult", "content": [{"type": "text", "text": "res"}], "toolCallId": "c1"}
        converted = to_qwen_message(message, trainable=True)
        assert converted == {"role": "tool", "content": "res", "tool_call_id": "c1", "step_loss_mask": 0}

    def test_to_qwen_message_assistant_with_tool_calls(self):
        message = {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "acting"},
                {"type": "toolCall", "id": "t1", "name": "shop_act", "arguments": {"action": "search[x]"}},
            ],
        }
        converted = to_qwen_message(message, trainable=True)
        assert converted["role"] == "assistant"
        assert converted["content"] == "acting"
        assert converted["step_loss_mask"] == 1
        assert converted["tool_calls"] == [{
            "id": "t1", "type": "function",
            "function": {"name": "shop_act", "arguments": {"action": "search[x]"}},
        }]

    def test_to_qwen_message_assistant_untrainable(self):
        message = {"role": "assistant", "content": [{"type": "text", "text": "x"}]}
        assert to_qwen_message(message, trainable=False)["step_loss_mask"] == 0

    def test_to_qwen_message_rejects_thinking(self):
        message = {"role": "assistant", "content": [{"type": "thinking", "text": "hmm"}]}
        with pytest.raises(ValueError, match="thinking"):
            to_qwen_message(message, trainable=True)

    def test_to_qwen_message_rejects_unknown_role(self):
        with pytest.raises(ValueError, match="unsupported pi role"):
            to_qwen_message({"role": "weird", "content": ""}, trainable=True)

    def test_percentile(self):
        values = [1, 2, 3, 4, 5]
        assert percentile(values, 0.0) == 1.0
        assert percentile(values, 0.5) == 3.0
        assert percentile(values, 1.0) == 5.0
        assert percentile([], 0.5) == 0.0


class TestVerifyConsistency:
    @staticmethod
    def write_rows(tmp_path, rows, name="turn_examples.jsonl"):
        path = tmp_path / name
        path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        return path

    @staticmethod
    def row(keep):
        metadata = {"example_id": "e", "task_id": 1}
        if keep is not None:
            metadata["context_keep_act_results"] = keep
        return {"messages": [], "metadata": metadata}

    def test_consistent(self, tmp_path):
        path = self.write_rows(tmp_path, [self.row(3), self.row(3)])
        ok, message = verify(path, 3)
        assert ok and "consistent" in message

    def test_inconsistent(self, tmp_path):
        path = self.write_rows(tmp_path, [self.row(3), self.row(5)])
        ok, message = verify(path, 3)
        assert not ok and "5" in message

    def test_missing_field_reports_legacy_dataset(self, tmp_path):
        path = self.write_rows(tmp_path, [self.row(None)])
        ok, message = verify(path, 3)
        assert not ok and "predates" in message

    def test_empty_dataset(self, tmp_path):
        path = self.write_rows(tmp_path, [])
        ok, message = verify(path, 3)
        assert not ok and "no rows" in message

    def test_invalid_json_raises(self, tmp_path):
        path = tmp_path / "turn_examples.jsonl"
        path.write_text("{broken\n", encoding="utf-8")
        with pytest.raises(ValueError, match="invalid JSON"):
            verify(path, 3)
