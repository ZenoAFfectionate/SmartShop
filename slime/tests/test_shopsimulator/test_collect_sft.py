"""Tests for examples.ShopSimulator.collect_sft (candidate gate, traces, I/O & CLI)."""

from __future__ import annotations

import argparse
import asyncio
import json

import pytest

from examples.ShopSimulator.collect_sft import (
    PRUNED_SHOP_ACT_RESULT,
    _event_call_id,
    _event_tool_name,
    _retryable,
    async_main,
    atomic_write_json,
    atomic_write_jsonl,
    authoritative_messages,
    build_parser,
    collect_one,
    context_snapshots,
    context_trace_matches,
    evaluate_candidate,
    export_results,
    lineage_conflicts,
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

    def test_old_act_results_are_pruned(self, monkeypatch):
        monkeypatch.setenv("SHOP_CONTEXT_STRUCTURED_MEMORY", "0")
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

    def test_old_act_results_become_memory_lines_by_default(self, monkeypatch):
        monkeypatch.setenv("SHOP_CONTEXT_STRUCTURED_MEMORY", "1")
        messages = self.messages_with_acts(5)
        snapshots = context_snapshots(messages, keep_act_results=2)
        act_texts = [
            part["text"]
            for message in snapshots[-1]["messages"]
            if message.get("role") == "toolResult" and message.get("toolName") == "shop_act"
            for part in message["content"]
        ]
        assert "act-result-3" in act_texts and "act-result-4" in act_texts
        # raw text is replaced by a memory line (which may quote it as content)
        assert not any(text == "act-result-0" for text in act_texts)
        memory = [text for text in act_texts if text.startswith("[记忆]")]
        assert len(memory) == 3
        assert "[记忆] act: act-result-0" in memory

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

    def test_thinking_parts_are_stripped_like_the_extension(self, monkeypatch):
        # shop_extension.ts drops assistant `thinking` parts before sending, so
        # the Python rebuild must drop them too — otherwise a reasoning teacher
        # would make SFT contexts diverge from RL rollout contexts.
        monkeypatch.setenv("SHOP_CONTEXT_STRUCTURED_MEMORY", "1")
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "任务"}]},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "text": "内部推理"},
                    {"type": "text", "text": "可见回答"},
                ],
            },
            {
                "role": "toolResult",
                "toolName": "shop_act",
                "toolCallId": "c1",
                "content": [{"type": "text", "text": "act-result-0"}],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "收尾"}]},
        ]
        snapshots = context_snapshots(messages, keep_act_results=1)
        assistants = [m for m in snapshots[-1]["messages"] if m.get("role") == "assistant"]
        assert assistants, "assistant messages must survive the rebuild"
        assert [part["type"] for part in assistants[0]["content"]] == ["text"]
        assert all(
            part.get("type") != "thinking"
            for message in assistants
            for part in message["content"]
        )

    def test_every_assistant_thinking_part_is_stripped(self, monkeypatch):
        monkeypatch.setenv("SHOP_CONTEXT_STRUCTURED_MEMORY", "1")
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "任务"}]},
            {"role": "assistant", "content": [{"type": "thinking", "text": "t1"}, {"type": "text", "text": "a1"}]},
            {"role": "toolResult", "toolName": "shop_act", "toolCallId": "c1", "content": [{"type": "text", "text": "r1"}]},
            {"role": "assistant", "content": [{"type": "thinking", "text": "t2"}]},
            {"role": "toolResult", "toolName": "shop_act", "toolCallId": "c2", "content": [{"type": "text", "text": "r2"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "a3"}]},
        ]
        rebuilt = context_snapshots(messages, keep_act_results=1)[-1]["messages"]
        assert all(
            part.get("type") != "thinking"
            for message in rebuilt
            if message.get("role") == "assistant"
            for part in message["content"]
        )

    def test_keep_zero_compresses_every_act_result(self, monkeypatch):
        # keep_act_results=0 means nothing survives verbatim — every result must
        # become a memory line (structured) instead of leaking raw text.
        monkeypatch.setenv("SHOP_CONTEXT_STRUCTURED_MEMORY", "1")
        messages = self.messages_with_acts(3)
        snapshots = context_snapshots(messages, keep_act_results=0)
        texts = [
            part["text"]
            for message in snapshots[-1]["messages"]
            if message.get("role") == "toolResult"
            for part in message["content"]
        ]
        assert texts, "the fixture must contain act results"
        assert all(text.startswith("[记忆]") for text in texts)

    def test_legacy_mode_marks_every_pruned_result(self, monkeypatch):
        monkeypatch.setenv("SHOP_CONTEXT_STRUCTURED_MEMORY", "0")
        messages = self.messages_with_acts(3)
        snapshots = context_snapshots(messages, keep_act_results=0)
        texts = [
            part["text"]
            for message in snapshots[-1]["messages"]
            if message.get("role") == "toolResult"
            for part in message["content"]
        ]
        assert all(text == PRUNED_SHOP_ACT_RESULT for text in texts)


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


class TestEventHelpers:
    def test_tool_name_variants(self):
        assert _event_tool_name({"toolName": "a"}) == "a"
        assert _event_tool_name({"tool_name": "b"}) == "b"
        assert _event_tool_name({}) is None

    def test_call_id_variants(self):
        assert _event_call_id({"toolCallId": 1}) == "1"
        assert _event_call_id({"tool_call_id": "x"}) == "x"
        assert _event_call_id({"id": "y"}) == "y"
        assert _event_call_id({}) is None


class TestAuthoritativeMessages:
    def test_only_message_end_events(self):
        events = [
            {"type": "turn_start"},
            {"type": "message_end", "message": {"role": "user", "content": "hi"}},
            {"type": "tool_execution_end"},
        ]
        assert authoritative_messages(events) == [{"role": "user", "content": "hi"}]

    def test_skips_message_without_role(self):
        events = [{"type": "message_end", "message": {"content": "no role"}}]
        assert authoritative_messages(events) == []

    def test_strips_assistant_thinking_parts(self):
        events = [{
            "type": "message_end",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "text": "hmm"},
                    {"type": "text", "text": "act"},
                    {"type": "toolCall", "id": "t1", "name": "shop_act", "arguments": {}},
                ],
            },
        }]
        messages = authoritative_messages(events)
        content = messages[0]["content"]
        assert [part.get("type") for part in content] == ["text", "toolCall"]

    def test_does_not_mutate_input(self):
        event = {"type": "message_end", "message": {"role": "assistant", "content": [{"type": "thinking", "text": "x"}]}}
        authoritative_messages([event])
        assert event["message"]["content"][0]["type"] == "thinking"


class TestAtomicWrites:
    def test_atomic_write_json(self, tmp_path):
        path = tmp_path / "sub" / "out.json"
        atomic_write_json(path, {"b": 1, "a": 2})
        assert json.loads(path.read_text()) == {"a": 2, "b": 1}

    def test_atomic_write_jsonl(self, tmp_path):
        path = tmp_path / "out.jsonl"
        atomic_write_jsonl(path, [{"b": 1}, {"a": 2}])
        lines = path.read_text().strip().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0]) == {"b": 1}
        assert json.loads(lines[1]) == {"a": 2}

    def test_atomic_write_overwrites(self, tmp_path):
        path = tmp_path / "out.json"
        atomic_write_json(path, {"v": 1})
        atomic_write_json(path, {"v": 2})
        assert json.loads(path.read_text()) == {"v": 2}


class TestRetryable:
    def test_retryable_markers(self):
        for text in [
            "429",
            "rate limit",
            "too many requests",
            "connection reset",
            "timed out",
            "timeout",
            "502",
            "503",
            "504",
            "temporarily unavailable",
            # turn exhaustion is a sampling-luck failure: ~half of retries finish
            "pi reached max model turns (40)",
        ]:
            assert _retryable(text), text

    def test_non_retryable(self):
        assert not _retryable("invalid api key")
        assert not _retryable("deterministic failure")


class TestExportResults:
    def _candidate(self, task_id, sample_id, accepted, reward=1.0, tool_calls=5, done=True):
        return {
            "trajectory_id": f"sft-{task_id:06d}-{sample_id:03d}",
            "task_id": task_id,
            "sample_id": sample_id,
            "split": "sft",
            "accepted": accepted,
            "reward": reward,
            "done": done,
            "error": None,
            "tool_calls": tool_calls,
            "messages": [{"role": "system", "content": "s"}],
            "rejection_reasons": [] if accepted else ["not_done"],
            "attempts": [],
        }

    def test_export_writes_accepted_and_summary(self, tmp_path):
        args = argparse.Namespace(
            model="deepseek-flash", max_turns=40,
            teacher_provider="deepseek", base_url="https://api.deepseek.com",
        )
        candidates = [
            self._candidate(1, 0, True),
            self._candidate(1, 1, False, done=False),
            self._candidate(2, 0, True),
        ]
        summary = export_results(tmp_path, candidates, args)
        assert summary["candidates"] == 3
        assert summary["accepted"] == 2
        assert summary["tasks_attempted"] == 2
        assert summary["tasks_covered"] == 2
        assert summary["task_coverage"] == 1.0
        assert summary["uncovered_tasks"] == 0
        accepted_lines = (tmp_path / "accepted.jsonl").read_text().strip().splitlines()
        assert len(accepted_lines) == 2
        assert json.loads(accepted_lines[0])["metadata"]["task_id"] == 1
        assert (tmp_path / "summary.json").is_file()

    def test_summary_records_every_teacher_model_seen(self, tmp_path):
        # With --allow-mixed-teacher a resumed run can reuse older trajectories;
        # the summary must show both models so the mixture stays auditable.
        args = argparse.Namespace(
            model="deepseek-flash", max_turns=40,
            teacher_provider="deepseek", base_url="https://api.deepseek.com",
        )
        candidates = [self._candidate(1, 0, True), self._candidate(2, 0, True)]
        candidates[0]["teacher"] = {"model": "deepseek-flash"}
        candidates[1]["teacher"] = {"model": "legacy-teacher"}
        summary = export_results(tmp_path, candidates, args)
        assert summary["teacher_models_seen"] == ["deepseek-flash", "legacy-teacher"]
        assert summary["teacher_model"] == "deepseek-flash"

    def test_export_uncovered_tasks(self, tmp_path):
        args = argparse.Namespace(
            model="deepseek-flash", max_turns=40,
            teacher_provider="deepseek", base_url="https://api.deepseek.com",
        )
        candidates = [
            self._candidate(1, 0, False, done=False, tool_calls=40),
            self._candidate(2, 0, True),
        ]
        summary = export_results(tmp_path, candidates, args)
        assert summary["tasks_covered"] == 1
        assert summary["uncovered_tasks"] == 1
        uncovered_lines = (tmp_path / "uncovered_tasks.jsonl").read_text().strip().splitlines()
        assert len(uncovered_lines) == 1
        assert json.loads(uncovered_lines[0])["task_id"] == 1


class TestParserDefaults:
    def test_api_key_file_has_no_default(self):
        # The old /root/api.txt default silently picked up stale keys.
        args = build_parser().parse_args(["--output-dir", "/tmp/x"])
        assert args.api_key_file is None

    def test_launches_per_minute_defaults_to_30(self):
        # A conservative default instead of unlimited (0 still disables).
        args = build_parser().parse_args(["--output-dir", "/tmp/x"])
        assert args.launches_per_minute == 30

    def test_teacher_provider_defaults_to_deepseek(self):
        args = build_parser().parse_args(["--output-dir", "/tmp/x"])
        assert args.teacher_provider == "deepseek"

    def test_keep_act_results_defaults_to_none(self):
        args = build_parser().parse_args(["--output-dir", "/tmp/x"])
        assert args.keep_act_results is None

    def test_default_teacher_model(self):
        args = build_parser().parse_args(["--output-dir", "/tmp/x"])
        assert args.model == "deepseek-flash"

    def test_mixed_teacher_is_rejected_by_default(self):
        args = build_parser().parse_args(["--output-dir", "/tmp/x"])
        assert args.allow_mixed_teacher is False

    def test_mixed_teacher_flag_enables_reuse(self):
        args = build_parser().parse_args(["--output-dir", "/tmp/x", "--allow-mixed-teacher"])
        assert args.allow_mixed_teacher is True


class TestLineageGuard:
    """Resumable collection must never silently mix teacher models or context
    formats into one dataset (trajectories are reused verbatim)."""

    @staticmethod
    def _args(tmp_path, **overrides):
        values = dict(
            output_dir=tmp_path,
            model="deepseek-flash",
            max_turns=40,
            teacher_provider="deepseek",
            base_url="https://api.deepseek.com",
            context_window=262144,
            max_tokens=32768,
            keep_act_results=None,
            allow_mixed_teacher=False,
        )
        values.update(overrides)
        return argparse.Namespace(**values)

    @staticmethod
    def _candidate(model="deepseek-flash"):
        harness = {
            "context_keep_act_results": 3,
            "context_structured_memory": True,
            "max_turns": 40,
            "system_prompt": "s",
        }
        return {
            "teacher": {
                "provider": "deepseek", "model": model, "base_url": "https://api.deepseek.com",
                "context_window": 262144, "max_tokens": 32768, "thinking": False,
            },
            "harness": harness,
        }

    def test_matching_candidate_is_reusable(self, tmp_path):
        conflicts = lineage_conflicts(
            self._candidate(), args=self._args(tmp_path), keep_act_results=3, structured_memory=True
        )
        assert conflicts == []

    @pytest.mark.parametrize("group,field,stale_value", [
        ("teacher", "provider", "other-provider"),
        ("teacher", "model", "legacy-teacher"),
        ("teacher", "base_url", "https://example.invalid"),
        ("teacher", "context_window", 8192),
        ("teacher", "max_tokens", 1024),
        ("harness", "context_keep_act_results", 1),
        ("harness", "max_turns", 20),
        ("harness", "context_structured_memory", False),
    ])
    def test_every_fingerprint_field_is_checked(self, tmp_path, group, field, stale_value):
        candidate = self._candidate()
        candidate[group][field] = stale_value
        conflicts = lineage_conflicts(
            candidate, args=self._args(tmp_path), keep_act_results=3, structured_memory=True
        )
        assert any(item.startswith(f"{group}.{field}:") for item in conflicts), conflicts

    def test_reported_conflict_shows_both_values(self, tmp_path):
        conflicts = lineage_conflicts(
            self._candidate(model="legacy-teacher"),
            args=self._args(tmp_path), keep_act_results=3, structured_memory=True,
        )
        assert conflicts == ["teacher.model: 'legacy-teacher' != 'deepseek-flash'"]

    def test_legacy_candidate_matches_legacy_run(self, tmp_path):
        # Trajectories predating the structured-memory switch carry no field.
        harness = {"context_keep_act_results": 3, "max_turns": 40, "system_prompt": "s"}
        candidate = {
            "teacher": self._candidate()["teacher"],
            "harness": harness,
        }
        assert lineage_conflicts(
            candidate, args=self._args(tmp_path), keep_act_results=3, structured_memory=False
        ) == []
        assert lineage_conflicts(
            candidate, args=self._args(tmp_path), keep_act_results=3, structured_memory=True
        ) != []

    def test_keep_act_results_change_is_reported(self, tmp_path):
        conflicts = lineage_conflicts(
            self._candidate(), args=self._args(tmp_path), keep_act_results=1, structured_memory=True
        )
        assert any("harness.context_keep_act_results" in item for item in conflicts)

    def test_collect_one_aborts_then_reuses_with_flag(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SHOP_CONTEXT_KEEP_ACT_RESULTS", "3")
        monkeypatch.setenv("SHOP_CONTEXT_STRUCTURED_MEMORY", "1")
        raw = tmp_path / "raw" / "000001"
        raw.mkdir(parents=True)
        stale = self._candidate(model="legacy-teacher")
        (raw / "000.json").write_text(json.dumps(stale), encoding="utf-8")
        row = {"metadata": {"task_id": 1}}

        args = self._args(tmp_path)
        with pytest.raises(ValueError, match="different teacher/harness"):
            asyncio.run(collect_one(row=row, sample_id=0, args=args, api_key="k", limiter=None))

        args.allow_mixed_teacher = True
        reused = asyncio.run(collect_one(row=row, sample_id=0, args=args, api_key="k", limiter=None))
        assert reused["teacher"]["model"] == "legacy-teacher"

    def test_collect_one_reuses_matching_candidate(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SHOP_CONTEXT_KEEP_ACT_RESULTS", "3")
        monkeypatch.setenv("SHOP_CONTEXT_STRUCTURED_MEMORY", "1")
        raw = tmp_path / "raw" / "000001"
        raw.mkdir(parents=True)
        (raw / "000.json").write_text(json.dumps(self._candidate()), encoding="utf-8")
        reused = asyncio.run(collect_one(
            row={"metadata": {"task_id": 1}}, sample_id=0,
            args=self._args(tmp_path), api_key="k", limiter=None,
        ))
        assert reused["teacher"]["model"] == "deepseek-flash"


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
