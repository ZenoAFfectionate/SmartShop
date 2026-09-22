/**
 * Integration tests for shop_extension.ts's context hook.
 *
 * The unit tests (shop_memory.test.ts) cover the summarizer in isolation; these
 * tests load the real extension the same way Pi does (jiti, with only the
 * `@earendil-works/pi-ai` import aliased to a stub), drive the registered
 * `context` handler with realistic message histories, and assert the exact
 * messages sent to the model — including tool-call pairing, thinking stripping,
 * the structured-memory switch, budgeting, and the context-trace contract.
 *
 * Run:
 *   node tests/test_shopsimulator/shop_extension.test.mjs
 */

import assert from "node:assert/strict";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

import { loadExtension } from "./fixtures/extension_loader.mjs";

const here = dirname(fileURLToPath(import.meta.url));
const FIXTURE = join(here, "fixtures/shop_memory_samples.json");

const FIXTURES = JSON.parse(readFileSync(FIXTURE, "utf8")).samples;
const SEARCH_TEXT = FIXTURES.find((s) => s.name === "search_page").text;
const PRUNED_LEGACY = "[旧的 shop_act 工具结果已裁剪；done=false]";

function makeMessages(actCount, { thinking = true } = {}) {
	const messages = [{ role: "system", content: "sys" }];
	for (let index = 0; index < actCount; index += 1) {
		messages.push({
			role: "assistant",
			content: [
				...(thinking ? [{ type: "thinking", thinking: "internal" }] : []),
				{ type: "toolCall", id: `call_${index}`, name: "shop_act", arguments: { action: `search[词${index}]` } },
			],
		});
		messages.push({
			role: "toolResult",
			toolCallId: `call_${index}`,
			toolName: "shop_act",
			content: [{ type: "text", text: SEARCH_TEXT }],
		});
	}
	messages.push({ role: "assistant", content: [{ type: "text", text: "final" }] });
	return { messages };
}

const baseEnv = { SHOP_ROLLOUT_SESSION_ID: "test-session", SHOP_TASK_ID: "0" };
const actTexts = (messages) => messages
	.filter((message) => message.role === "toolResult" && message.toolName === "shop_act")
	.map((message) => message.content[0].text);

test("registered tools cover shop_reset and shop_act", () => {
	const { tools } = loadExtension(baseEnv);
	assert.deepEqual(tools.map((tool) => tool.name).sort(), ["shop_act", "shop_reset"]);
});

test("structured memory replaces old results and keeps the newest verbatim", async () => {
	const { handlers } = loadExtension({ ...baseEnv, SHOP_CONTEXT_KEEP_ACT_RESULTS: "1", SHOP_CONTEXT_STRUCTURED_MEMORY: "1" });
	const { messages } = makeMessages(4);
	const result = await handlers.context({ messages });
	assert.ok(result && Array.isArray(result.messages));
	const texts = actTexts(result.messages);
	assert.equal(texts.length, 4);
	assert.equal(texts[3], SEARCH_TEXT, "newest result must survive verbatim");
	for (let index = 0; index < 3; index += 1) {
		assert.ok(texts[index].startsWith("[记忆] search[词" + index + "]"), texts[index]);
		assert.ok(texts[index].length <= 240, "summaries stay bounded");
	}
});

test("actions are paired through toolCallId, not message order", async () => {
	const { handlers } = loadExtension({ ...baseEnv, SHOP_CONTEXT_KEEP_ACT_RESULTS: "1" });
	const { messages } = makeMessages(3);
	// Swap the first two toolResults: their actions must still follow their
	// own toolCallId, which only works if pairing is id-based.
	[messages[2], messages[4]] = [messages[4], messages[2]];
	const memory = actTexts((await handlers.context({ messages })).messages)
		.filter((text) => text.startsWith("[记忆]"));
	assert.ok(memory.some((text) => text.includes("search[词0]")), "call_0 keeps its own action");
	assert.ok(memory.some((text) => text.includes("search[词1]")), "call_1 keeps its own action");
});

test("thinking parts are stripped from assistant messages", async () => {
	const { handlers } = loadExtension({ ...baseEnv, SHOP_CONTEXT_KEEP_ACT_RESULTS: "4" });
	const { messages } = makeMessages(2);
	const result = await handlers.context({ messages });
	const assistant = result.messages.find((message) => message.role === "assistant" && Array.isArray(message.content));
	assert.equal(assistant.content.length, 1);
	assert.equal(assistant.content[0].type, "toolCall");
});

test("no relevant messages yields undefined instead of a rewrite", async () => {
	// keep=5 with only 2 acts and no thinking parts: nothing to rewrite.
	const { handlers } = loadExtension({ ...baseEnv, SHOP_CONTEXT_KEEP_ACT_RESULTS: "5" });
	const { messages } = makeMessages(2, { thinking: false });
	assert.equal(await handlers.context({ messages }), undefined);
});

test("legacy placeholder mode keeps the pre-R5 contract", async () => {
	const { handlers } = loadExtension({ ...baseEnv, SHOP_CONTEXT_KEEP_ACT_RESULTS: "1", SHOP_CONTEXT_STRUCTURED_MEMORY: "0" });
	const { messages } = makeMessages(3);
	const texts = actTexts((await handlers.context({ messages })).messages);
	assert.equal(texts[0], PRUNED_LEGACY);
	assert.equal(texts[1], PRUNED_LEGACY);
	assert.equal(texts[2], SEARCH_TEXT);
});

test("context trace records the compression version and the sent messages", async () => {
	const traceDir = mkdtempSync(join(tmpdir(), "shop-trace-"));
	const traceFile = join(traceDir, "context-trace.jsonl");
	try {
		const { handlers } = loadExtension({
			...baseEnv,
			SHOP_CONTEXT_KEEP_ACT_RESULTS: "1",
			SHOP_CONTEXT_STRUCTURED_MEMORY: "1",
			SHOP_CONTEXT_TRACE_PATH: traceFile,
		});
		await handlers.context(makeMessages(3));
		const lines = readFileSync(traceFile, "utf8").trim().split("\n").map((line) => JSON.parse(line));
		assert.equal(lines.length, 1);
		assert.equal(lines[0].schema_version, 1);
		assert.equal(lines[0].compression_version, "keep-last-1-shop-act-results-v2-structured");
		assert.ok(Array.isArray(lines[0].messages));
	} finally {
		rmSync(traceDir, { recursive: true, force: true });
	}
});

test("legacy trace version is recorded when structured memory is off", async () => {
	const traceDir = mkdtempSync(join(tmpdir(), "shop-trace-"));
	const traceFile = join(traceDir, "context-trace.jsonl");
	try {
		const { handlers } = loadExtension({
			...baseEnv,
			SHOP_CONTEXT_KEEP_ACT_RESULTS: "2",
			SHOP_CONTEXT_STRUCTURED_MEMORY: "0",
			SHOP_CONTEXT_TRACE_PATH: traceFile,
		});
		await handlers.context(makeMessages(3));
		const [line] = readFileSync(traceFile, "utf8").trim().split("\n").map((item) => JSON.parse(item));
		assert.equal(line.compression_version, "keep-last-2-shop-act-results-v1");
	} finally {
		rmSync(traceDir, { recursive: true, force: true });
	}
});

test("repeated calls on the same history are stable and budget-bounded", async () => {
	const { handlers } = loadExtension({ ...baseEnv, SHOP_CONTEXT_KEEP_ACT_RESULTS: "1" });
	const { messages } = makeMessages(30);
	const first = await handlers.context({ messages });
	const second = await handlers.context({ messages });
	assert.deepEqual(actTexts(second.messages), actTexts(first.messages), "cache must not change results");
	const memoryChars = actTexts(first.messages)
		.filter((text) => text.startsWith("[记忆]"))
		.reduce((total, text) => total + text.length, 0);
	assert.ok(memoryChars <= 2400, `memory budget exceeded: ${memoryChars}`);
});

test("input message objects are not mutated", async () => {
	const { handlers } = loadExtension({ ...baseEnv, SHOP_CONTEXT_KEEP_ACT_RESULTS: "1" });
	const { messages } = makeMessages(3);
	const before = JSON.stringify(messages);
	await handlers.context({ messages });
	assert.equal(JSON.stringify(messages), before);
});
