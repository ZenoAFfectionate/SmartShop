/**
 * Unit tests for the structured context memory (R5) TypeScript implementation.
 *
 * Run directly by Node's built-in test runner:
 *   node --experimental-strip-types --test tests/test_shopsimulator/shop_memory.test.ts
 *
 * The Python twin (shop_memory.py) is exercised on the same fixture set by
 * tests/test_shopsimulator/test_shop_memory.py, which also runs this file to
 * enforce byte-for-byte parity between the two implementations.
 */

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

import {
	MAX_MEMORY_BUDGET_CHARS,
	MAX_SUMMARY_CHARS,
	STRUCTURED_MEMORY_PLACEHOLDER,
	buildMemoryLines,
	clampText,
	parseShopAction,
	structuredMemoryEnabled,
	summarizeShopActResult,
} from "../../examples/ShopSimulator/shop_memory.ts";

const here = dirname(fileURLToPath(import.meta.url));
const fixture = JSON.parse(readFileSync(join(here, "fixtures/shop_memory_samples.json"), "utf8")) as {
	samples: { name: string; action: string; text: string }[];
};

function sample(name: string): { name: string; action: string; text: string } {
	const found = fixture.samples.find((item) => item.name === name);
	assert.ok(found, `fixture sample ${name} is missing`);
	return found;
}

test("search page keeps the real query from the action and top candidates", () => {
	const { text, action } = sample("search_page");
	const line = summarizeShopActResult(text, action);
	assert.match(line, /^\[记忆\] search\[美容镜 充电 黑灯!\] 150件: /);
	// newest-agnostic: the first four candidates of page 1 are kept
	assert.ok(line.includes("801801854113|"));
	assert.ok(line.includes("625397120514|"));
	assert.ok(line.includes("568498403099|"));
	assert.ok(line.includes("738295250701|"), "MAX_SEARCH_ITEMS=4 should include the 4th item");
	assert.ok(!line.includes("774272652940"), "the 5th candidate must be dropped");
	assert.ok(line.length <= MAX_SUMMARY_CHARS);
});

test("search page without an action omits the query bracket", () => {
	const { text } = sample("search_page");
	const line = summarizeShopActResult(text, "");
	assert.match(line, /^\[记忆\] search 150件: /);
});

test("detail page reports asin, title, price, shop and first options", () => {
	const { text, action } = sample("detail_page");
	const line = summarizeShopActResult(text, action);
	assert.match(line, /^\[记忆\] view\[824148419972\] \| /);
	assert.ok(line.includes("价格: 288.0 to 338.0"));
	assert.ok(line.includes("店铺: 格瑞达家居旗舰店"));
	assert.ok(line.includes("颜色分类: "));
	assert.ok(line.length <= MAX_SUMMARY_CHARS);
});

test("detail page falls back to the title when the action is unavailable", () => {
	const { text } = sample("detail_page");
	const line = summarizeShopActResult(text, "");
	assert.ok(line.startsWith("[记忆] view[") && !line.includes("view[824148419972]"));
	assert.ok(line.includes("价格: 288.0 to 338.0"));
});

test("purchase page reports the bought asin", () => {
	const { text } = sample("done_page");
	const line = summarizeShopActResult(text, "");
	assert.match(line, /^\[记忆\] buy✓( asin=[0-9]+)?/);
	assert.ok(line.includes("asin=617584252607"));
});

test("option decoding follows JavaScript String()/Array/Object semantics", () => {
	// These exact strings are what RL rollouts emit; the Python twin in
	// shop_memory.py must reproduce them byte-for-byte, so pin the reference
	// behaviour here as well (see test_shop_memory.py::TestJavaScriptSemantics).
	const page = (options: string) =>
		`X [SEP] Thank you for shopping [SEP] asin [SEP] 123456789 [SEP] options [SEP] ${options}`;
	assert.equal(summarizeShopActResult(page('{"颜色分类": null}')), "[记忆] buy✓ asin=123456789 options=null");
	assert.equal(summarizeShopActResult(page('{"颜色分类": ["红","蓝"]}')), "[记忆] buy✓ asin=123456789 options=红,蓝");
	assert.equal(summarizeShopActResult(page('{"a": {"x": 1}}')), "[记忆] buy✓ asin=123456789 options=[object Object]");
	assert.equal(summarizeShopActResult(page('{"2": "b", "1": "a"}')), "[记忆] buy✓ asin=123456789 options=a, b");
	assert.equal(summarizeShopActResult(page('{"a": [1, null]}')), "[记忆] buy✓ asin=123456789 options=1,");
	assert.equal(summarizeShopActResult(page('{"a": [], "b": {}}')), "[记忆] buy✓ asin=123456789 options=, [object Object]");
	assert.equal(summarizeShopActResult(page("not-json")), "[记忆] buy✓ asin=123456789");
});

test("welcome page falls back to its first meaningful line", () => {
	const { text } = sample("welcome_page");
	const line = summarizeShopActResult(text, "");
	assert.match(line, /^\[记忆\] act: /);
});

test("empty input produces an explicit empty marker", () => {
	assert.equal(summarizeShopActResult(""), "[记忆] (空结果)");
	assert.equal(summarizeShopActResult("   "), "[记忆] (空结果)");
});

test("text without separators still yields a bounded fallback line", () => {
	const { text, action } = sample("no_separator");
	const line = summarizeShopActResult(text, action);
	assert.match(line, /^\[记忆\] act: /);
	assert.ok(line.length <= MAX_SUMMARY_CHARS);
});

test("a malformed action never leaks into the summary", () => {
	const { text, action } = sample("bad_action");
	const line = summarizeShopActResult(text, action);
	assert.match(line, /^\[记忆\] search 2件: /);
	assert.ok(!line.includes("buy now"));
});

test("clampText truncates by code points and appends an ellipsis", () => {
	assert.equal(clampText("abc", 5), "abc");
	assert.equal(clampText("  abc  ", 3), "abc");
	assert.equal(clampText("abcdef", 3), "abc…");
	// surrogate pairs must count as one character
	assert.equal(clampText("🛒🛒🛒", 2), "🛒🛒…");
});

test("parseShopAction understands search/click and rejects everything else", () => {
	assert.deepEqual(parseShopAction("search[美容镜]"), { kind: "search", value: "美容镜" });
	assert.deepEqual(parseShopAction("click[123456789]"), { kind: "click", value: "123456789" });
	assert.deepEqual(parseShopAction(" click[ back to search ] "), { kind: "click", value: "back to search" });
	assert.deepEqual(parseShopAction("buy now"), { kind: "", value: "" });
	assert.deepEqual(parseShopAction(undefined), { kind: "", value: "" });
});

test("buildMemoryLines spends the budget from the newest entry backwards", () => {
	const summaries = ["a".repeat(100), "b".repeat(100), "c".repeat(100)];
	const lines = buildMemoryLines(summaries, 250);
	// newest two fit (201 chars <= 250), oldest collapses
	assert.deepEqual(lines, [STRUCTURED_MEMORY_PLACEHOLDER, summaries[1], summaries[2]]);
});

test("buildMemoryLines collapses everything when even one line exceeds the budget", () => {
	const lines = buildMemoryLines(["x".repeat(MAX_MEMORY_BUDGET_CHARS + 10)], MAX_MEMORY_BUDGET_CHARS);
	assert.deepEqual(lines, [STRUCTURED_MEMORY_PLACEHOLDER]);
});

test("buildMemoryLines keeps every line when the budget is generous", () => {
	const summaries = ["one", "two", "three"];
	assert.deepEqual(buildMemoryLines(summaries, MAX_MEMORY_BUDGET_CHARS), summaries);
});

test("structuredMemoryEnabled defaults on and accepts the usual off values", () => {
	assert.equal(structuredMemoryEnabled(undefined), true);
	assert.equal(structuredMemoryEnabled(""), true);
	assert.equal(structuredMemoryEnabled("1"), true);
	assert.equal(structuredMemoryEnabled("TRUE"), true);
	for (const off of ["0", "false", "no", "off", "OFF", " false "]) {
		assert.equal(structuredMemoryEnabled(off), false, `${off} should disable`);
	}
});

test("all fixture summaries respect the per-line character cap", () => {
	for (const item of fixture.samples) {
		const line = summarizeShopActResult(item.text, item.action);
		assert.ok(line.length <= MAX_SUMMARY_CHARS, `${item.name}: ${line.length} chars`);
	}
});
