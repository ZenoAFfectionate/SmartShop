"""Tests for structured context memory (R5): Python twin + TS parity + wiring.

``shop_memory.py`` mirrors ``shop_memory.ts`` for the SFT data pipeline. Any
drift between the two would desynchronise SFT contexts from RL rollout
contexts, so the central test here runs the TypeScript module through Node on a
shared fixture set and compares every character.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from examples.ShopSimulator import shop_memory as memory
from examples.ShopSimulator.collect_sft import authoritative_messages, context_snapshots
from examples.ShopSimulator.shop_memory import (
    MAX_MEMORY_BUDGET_CHARS,
    MAX_SUMMARY_CHARS,
    STRUCTURED_MEMORY_PLACEHOLDER,
    build_memory_lines,
    clamp_text,
    parse_shop_action,
    structured_memory_enabled,
    summarize_shop_act_result,
)

SLIME_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_DIR = SLIME_ROOT / "examples/ShopSimulator"
TS_MODULE = EXAMPLE_DIR / "shop_memory.ts"
TS_EXTENSION = EXAMPLE_DIR / "shop_extension.ts"
FIXTURE = Path(__file__).parent / "fixtures/shop_memory_samples.json"
TS_TEST = Path(__file__).parent / "shop_memory.test.ts"
HOOK_DRIVER = Path(__file__).parent / "fixtures/run_extension_hook.mjs"
NODE = shutil.which("node")

FIXTURES = json.loads(FIXTURE.read_text(encoding="utf-8"))["samples"]


def fixture(name: str) -> dict:
    for item in FIXTURES:
        if item["name"] == name:
            return item
    raise AssertionError(f"fixture {name} is missing")


class TestSummarizeShopActResult:
    def test_search_page_uses_action_query_and_caps_candidates(self):
        item = fixture("search_page")
        line = summarize_shop_act_result(item["text"], item["action"])
        assert line.startswith("[记忆] search[美容镜 充电 黑灯!] 150件: ")
        assert "801801854113|" in line
        assert "738295250701|" in line  # MAX_SEARCH_ITEMS=4
        assert "774272652940" not in line  # 5th candidate dropped
        assert len(line) <= MAX_SUMMARY_CHARS

    def test_search_page_without_action_omits_query(self):
        line = summarize_shop_act_result(fixture("search_page")["text"], "")
        assert line.startswith("[记忆] search 150件: ")

    def test_detail_page_reports_asin_title_price_shop_options(self):
        item = fixture("detail_page")
        line = summarize_shop_act_result(item["text"], item["action"])
        assert line.startswith("[记忆] view[824148419972] | ")
        assert "价格: 288.0 to 338.0" in line
        assert "店铺: 格瑞达家居旗舰店" in line
        assert "颜色分类: " in line
        assert len(line) <= MAX_SUMMARY_CHARS

    def test_detail_page_without_action_falls_back_to_title(self):
        line = summarize_shop_act_result(fixture("detail_page")["text"], "")
        assert line.startswith("[记忆] view[")
        assert "view[824148419972]" not in line
        assert "价格: 288.0 to 338.0" in line

    def test_purchase_page_reports_asin(self):
        line = summarize_shop_act_result(fixture("done_page")["text"], "")
        assert "asin=617584252607" in line

    def test_welcome_page_falls_back_to_first_line(self):
        assert summarize_shop_act_result(fixture("welcome_page")["text"], "").startswith("[记忆] act: ")

    def test_empty_input_marker(self):
        assert summarize_shop_act_result("") == "[记忆] (空结果)"
        assert summarize_shop_act_result("   ") == "[记忆] (空结果)"

    def test_malformed_action_is_ignored(self):
        item = fixture("bad_action")
        line = summarize_shop_act_result(item["text"], item["action"])
        assert line.startswith("[记忆] search 2件: ")
        assert "buy now" not in line

    def test_all_fixture_summaries_respect_cap(self):
        for item in FIXTURES:
            line = summarize_shop_act_result(item["text"], item["action"])
            assert len(line) <= MAX_SUMMARY_CHARS, item["name"]


class TestHelpers:
    def test_clamp_text_uses_code_points(self):
        assert clamp_text("abc", 5) == "abc"
        assert clamp_text("  abc  ", 3) == "abc"
        assert clamp_text("abcdef", 3) == "abc…"
        assert clamp_text("🛒🛒🛒", 2) == "🛒🛒…"

    def test_parse_shop_action(self):
        assert parse_shop_action("search[美容镜]") == ("search", "美容镜")
        assert parse_shop_action("click[123456789]") == ("click", "123456789")
        assert parse_shop_action(" click[ back to search ] ") == ("click", "back to search")
        assert parse_shop_action("buy now") == ("", "")
        assert parse_shop_action(None) == ("", "")

    def test_build_memory_lines_budget_from_newest(self):
        summaries = ["a" * 100, "b" * 100, "c" * 100]
        assert build_memory_lines(summaries, 250) == [STRUCTURED_MEMORY_PLACEHOLDER, summaries[1], summaries[2]]

    def test_build_memory_lines_collapses_oversized_single_line(self):
        oversized = "x" * (MAX_MEMORY_BUDGET_CHARS + 10)
        assert build_memory_lines([oversized]) == [STRUCTURED_MEMORY_PLACEHOLDER]

    def test_build_memory_lines_exact_budget_boundary(self):
        # cost = len(line) + 1, so a 99-char line exactly fills a 100 budget and
        # one character less must collapse it.
        line = "x" * 99
        assert build_memory_lines([line], 100) == [line]
        assert build_memory_lines([line], 99) == [STRUCTURED_MEMORY_PLACEHOLDER]

    def test_build_memory_lines_empty_input(self):
        assert build_memory_lines([]) == []

    def test_structured_memory_enabled_values(self):
        for value in (None, "", "1", "TRUE"):
            assert structured_memory_enabled(value) is True
        for value in ("0", "false", "no", "off", "OFF", " false "):
            assert structured_memory_enabled(value) is False


def _run_ts(expression: str) -> list[str]:
    """Execute an expression against shop_memory.ts via Node and parse the JSON out."""
    script = (
        f"import {{ readFileSync }} from 'node:fs';\n"
        f"import * as m from '{TS_MODULE.as_posix()}';\n"
        f"const data = JSON.parse(readFileSync('{FIXTURE.as_posix()}', 'utf8'));\n"
        f"console.log(JSON.stringify({expression}));\n"
    )
    result = subprocess.run(
        [NODE, "--experimental-strip-types", "--input-type=module", "-e", script],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


class TestJavaScriptSemantics:
    """The option decoder mirrors JavaScript ``String()`` / ``Array#toString`` /
    ``Object.values()`` semantics: SFT contexts are rebuilt in Python and must
    match what the TS extension actually sends during RL rollout, byte for byte.
    End-to-end coverage lives in TestTypeScriptParity (shared fixtures)."""

    def test_js_string_renders_boundary_types_like_javascript(self):
        assert memory._js_string(None) == "null"
        assert memory._js_string(True) == "true"
        assert memory._js_string(False) == "false"
        assert memory._js_string(2.0) == "2"
        assert memory._js_string(0.5) == "0.5"
        assert memory._js_string("红") == "红"
        assert memory._js_string({"x": 1}) == "[object Object]"

    def test_js_array_to_string_matches_javascript(self):
        assert memory._js_array_to_string([]) == ""
        assert memory._js_array_to_string(["红", "蓝"]) == "红,蓝"
        assert memory._js_array_to_string([1, [2, 3]]) == "1,2,3"  # nested flattened
        assert memory._js_array_to_string([1, None]) == "1,"  # null renders empty
        assert memory._js_array_to_string([{}, "x"]) == "[object Object],x"

    def test_js_object_values_reorders_array_index_keys(self):
        # Object.values(): array-index-like keys first in ascending order, the
        # remaining keys keep insertion order.
        assert memory._js_object_values({"2": "b", "1": "a"}) == ["a", "b"]
        assert memory._js_object_values({"b": 1, "a": 2}) == [1, 2]
        assert memory._js_object_values({"10": "x", "2": "y", "k": "z"}) == ["y", "x", "z"]
        # 2**32-1 is beyond the array-index bound, so insertion order is kept.
        assert memory._js_object_values({str(2**32 - 1): "big", "1": "one"}) == ["one", "big"]

    def test_decode_options_covers_json_boundaries(self):
        assert memory._decode_options('{"颜色分类": null}') == "null"
        assert memory._decode_options('{"a": ["红", "蓝"]}') == "红,蓝"
        assert memory._decode_options('{"a": {"x": 1}}') == "[object Object]"
        assert memory._decode_options('{"a": [], "b": {}}') == ", [object Object]"
        assert memory._decode_options('{"a": [1, 2.5, null]}') == "1,2.5,"
        assert memory._decode_options("not json") == ""
        assert memory._decode_options("[1, 2]") == ""  # non-object payload


@pytest.mark.skipif(NODE is None, reason="node is required for parity checks")
class TestTypeScriptParity:
    """Byte-for-byte equality between shop_memory.py and shop_memory.ts."""

    def test_summaries_match_on_shared_fixtures(self):
        python_lines = [
            summarize_shop_act_result(item["text"], item["action"]) for item in FIXTURES
        ]
        ts_lines = _run_ts("data.samples.map(s => m.summarizeShopActResult(s.text, s.action))")
        assert python_lines == ts_lines

    def test_budget_folding_matches(self):
        for payload, budget in (
            (["a" * 100, "b" * 100, "c" * 100], 250),
            (["x" * 99], 100),  # exactly fills the budget
            (["x" * 99], 99),  # one short -> collapses to the placeholder
            ([], 250),  # empty input
        ):
            ts_lines = _run_ts(f"m.buildMemoryLines({json.dumps(payload)}, {budget})")
            assert build_memory_lines(payload, budget) == ts_lines

    def test_clamp_text_matches(self):
        cases = ["abc", "  abc  ", "abcdef", "🛒🛒🛒", "中文标题截断测试"]
        ts_values = _run_ts(f"({json.dumps(cases)}).map(v => m.clampText(v, 3))")
        assert [clamp_text(value, 3) for value in cases] == ts_values

    def test_constants_match_python(self):
        source = TS_MODULE.read_text(encoding="utf-8")
        for name, expected in {
            "STRUCTURED_MEMORY_PREFIX": memory.STRUCTURED_MEMORY_PREFIX,
            "STRUCTURED_MEMORY_PLACEHOLDER": memory.STRUCTURED_MEMORY_PLACEHOLDER,
        }.items():
            match = re.search(rf'export const {name} = "((?:[^"\\]|\\.)*)"', source)
            assert match, f"{name} missing from shop_memory.ts"
            assert json.loads(f'"{match.group(1)}"') == expected
        for name, expected in {
            "MAX_SUMMARY_CHARS": MAX_SUMMARY_CHARS,
            "MAX_MEMORY_BUDGET_CHARS": MAX_MEMORY_BUDGET_CHARS,
            "TITLE_MAX_CHARS": memory.TITLE_MAX_CHARS,
            "MAX_SEARCH_ITEMS": memory.MAX_SEARCH_ITEMS,
            "OPTION_MAX_ITEMS": memory.OPTION_MAX_ITEMS,
        }.items():
            match = re.search(rf"export const {name} = ([0-9]+);", source)
            assert match, f"{name} missing from shop_memory.ts"
            assert int(match.group(1)) == expected

    def test_node_test_suite_passes(self):
        result = subprocess.run(
            [NODE, "--experimental-strip-types", "--test", str(TS_TEST)],
            capture_output=True, text=True, timeout=300,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "# fail 0" in result.stdout

    def test_extension_hook_integration_suite_passes(self):
        """The hook-level suite loads the real extension through jiti (as Pi
        does) with only @earendil-works/pi-ai stubbed."""
        result = subprocess.run(
            [NODE, "--test", str(Path(__file__).parent / "shop_extension.test.mjs")],
            capture_output=True, text=True, timeout=300,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "# fail 0" in result.stdout


class TestContextSnapshotsIntegration:
    """collect_sft.context_snapshots must rebuild what the Pi extension sends."""

    @staticmethod
    def _messages(count: int) -> list[dict]:
        messages: list[dict] = [{"role": "system", "content": "sys"}]
        for index in range(count):
            messages.append({
                "role": "assistant",
                "content": [{"type": "toolCall", "id": f"call_{index}", "name": "shop_act",
                             "arguments": {"action": f"search[词{index}]"}}],
            })
            messages.append({
                "role": "toolResult",
                "toolCallId": f"call_{index}",
                "toolName": "shop_act",
                "content": [{"type": "text", "text": fixture("search_page")["text"]}],
            })
        messages.append({"role": "assistant", "content": [{"type": "text", "text": "final"}]})
        return messages

    def test_structured_memory_replaces_old_results_with_summaries(self, monkeypatch):
        monkeypatch.setenv("SHOP_CONTEXT_STRUCTURED_MEMORY", "1")
        snapshots = context_snapshots(self._messages(5), keep_act_results=3)
        assert len(snapshots) == 6  # one per assistant message
        rebuilt = snapshots[-1]["messages"]
        tool_results = [m for m in rebuilt if m.get("role") == "toolResult"]
        assert len(tool_results) == 5
        assert tool_results[0]["content"][0]["text"].startswith("[记忆] search[词0]")
        assert tool_results[1]["content"][0]["text"].startswith("[记忆] search[词1]")
        # the newest three stay untouched
        assert tool_results[2]["content"][0]["text"] == fixture("search_page")["text"]

    def test_legacy_placeholder_when_disabled(self, monkeypatch):
        monkeypatch.setenv("SHOP_CONTEXT_STRUCTURED_MEMORY", "0")
        snapshots = context_snapshots(self._messages(5), keep_act_results=3)
        rebuilt = snapshots[-1]["messages"]
        tool_results = [m for m in rebuilt if m.get("role") == "toolResult"]
        assert tool_results[0]["content"][0]["text"] == "[旧的 shop_act 工具结果已裁剪；done=false]"

    def test_short_history_is_untouched(self, monkeypatch):
        monkeypatch.setenv("SHOP_CONTEXT_STRUCTURED_MEMORY", "1")
        snapshots = context_snapshots(self._messages(2), keep_act_results=3)
        rebuilt = snapshots[-1]["messages"]
        tool_results = [m for m in rebuilt if m.get("role") == "toolResult"]
        assert all(m["content"][0]["text"] == fixture("search_page")["text"] for m in tool_results)


@pytest.mark.skipif(NODE is None, reason="node is required for the hook driver")
class TestCrossImplementationContexts:
    """End-to-end: the real extension hook (TS) and the SFT rebuild (Python)
    must produce identical contexts for the same history. collect_sft rejects
    samples whose rebuilt context does not match the observed trace, so any
    drift here would silently drop every training sample."""

    RAW_MESSAGES = [
        {"role": "system", "content": "sys"},
        {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "t0"},
            {"type": "toolCall", "id": "call_0", "name": "shop_act", "arguments": {"action": "search[美容镜]"}},
        ]},
        {"role": "toolResult", "toolCallId": "call_0", "toolName": "shop_act",
         "content": [{"type": "text", "text": fixture("search_page")["text"]}]},
        {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "t1"},
            {"type": "toolCall", "id": "call_1", "name": "shop_act",
             "arguments": {"action": "click[824148419972]"}},
        ]},
        {"role": "toolResult", "toolCallId": "call_1", "toolName": "shop_act",
         "content": [{"type": "text", "text": fixture("detail_page")["text"]}]},
        {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "t2"},
            {"type": "toolCall", "id": "call_2", "name": "shop_act", "arguments": {"action": "click[buy now]"}},
        ]},
    ]

    def _run_hook(self, tmp_path: Path, messages: list[dict], keep: int, structured: str) -> list[dict]:
        payload = tmp_path / "messages.json"
        payload.write_text(json.dumps(messages, ensure_ascii=False), encoding="utf-8")
        result = subprocess.run(
            [NODE, str(HOOK_DRIVER), str(payload), str(keep), structured],
            capture_output=True, text=True, timeout=180,
        )
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout)["messages"]

    def test_rebuilt_context_matches_extension_hook(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SHOP_CONTEXT_STRUCTURED_MEMORY", "1")
        events = [{"type": "message_end", "message": message} for message in self.RAW_MESSAGES]
        rebuilt = context_snapshots(authoritative_messages(events), keep_act_results=1)
        last = rebuilt[-1]
        prefix = self.RAW_MESSAGES[: last["assistant_message_index"]]
        from_hook = self._run_hook(tmp_path, prefix, keep=1, structured="1")
        assert from_hook == last["messages"]

    def test_rebuilt_context_matches_extension_hook_legacy(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SHOP_CONTEXT_STRUCTURED_MEMORY", "0")
        events = [{"type": "message_end", "message": message} for message in self.RAW_MESSAGES]
        rebuilt = context_snapshots(authoritative_messages(events), keep_act_results=1)
        last = rebuilt[-1]
        prefix = self.RAW_MESSAGES[: last["assistant_message_index"]]
        from_hook = self._run_hook(tmp_path, prefix, keep=1, structured="0")
        assert from_hook == last["messages"]

    def test_driver_goes_through_the_extension(self, tmp_path):
        # sanity: two act results with keep=1 must leave one memory line, which
        # only the real extension (not the Python twin) can produce here.
        from_hook = self._run_hook(tmp_path, self.RAW_MESSAGES[:5], keep=1, structured="1")
        memory = [
            part["text"]
            for message in from_hook
            if message.get("role") == "toolResult"
            for part in message["content"]
            if part["text"].startswith("[记忆]")
        ]
        assert len(memory) == 1
        assert "search[美容镜]" in memory[0]


class TestWiring:
    def test_extension_uses_shared_memory_module(self):
        source = TS_EXTENSION.read_text(encoding="utf-8")
        assert 'from "./shop_memory"' in source
        assert "buildMemoryLines(" in source
        assert "summarizeShopActResult(" in source
        assert "SHOP_CONTEXT_STRUCTURED_MEMORY" in source
        assert "v2-structured" in source  # trace compression version

    def test_scripts_export_and_propagate_the_switch(self):
        for name in ("run_rl.sh", "run_eval.sh"):
            source = (EXAMPLE_DIR / name).read_text(encoding="utf-8")
            assert 'SHOP_CONTEXT_STRUCTURED_MEMORY="${SHOP_CONTEXT_STRUCTURED_MEMORY:-1}"' in source
            assert '"SHOP_CONTEXT_STRUCTURED_MEMORY"' in source, f"{name} does not propagate the switch into Ray"
