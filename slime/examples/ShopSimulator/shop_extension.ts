import { appendFileSync } from "node:fs";
import { Type } from "@earendil-works/pi-ai";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { buildMemoryLines, structuredMemoryEnabled, summarizeShopActResult } from "./shop_memory";

const SHOP_ENV_URL = process.env.SHOP_ENV_URL || "http://127.0.0.1:5000/api/shop_agent";
const parsedKeepActResults = Number.parseInt(process.env.SHOP_CONTEXT_KEEP_ACT_RESULTS || "3", 10);
const parsedTaskId = Number.parseInt(process.env.SHOP_TASK_ID || "", 10);
const rolloutSessionId = process.env.SHOP_ROLLOUT_SESSION_ID || "";
const contextTracePath = process.env.SHOP_CONTEXT_TRACE_PATH || "";
// R5 structured memory: older shop_act results are replaced by a compact
// candidate summary instead of a blank placeholder, so the model remembers
// what it already searched/viewed (repeated-action loops are the direct cause
// of every turn_limit trajectory). Set SHOP_CONTEXT_STRUCTURED_MEMORY=0 to
// fall back to the legacy placeholder.
const structuredMemory = structuredMemoryEnabled(process.env.SHOP_CONTEXT_STRUCTURED_MEMORY);
if (!rolloutSessionId) {
	throw new Error("SHOP_ROLLOUT_SESSION_ID must be a non-empty string");
}
if (!Number.isInteger(parsedTaskId) || parsedTaskId < 0) {
	throw new Error("SHOP_TASK_ID must be a non-negative integer");
}
if (!Number.isInteger(parsedKeepActResults) || parsedKeepActResults < 1) {
	throw new Error("SHOP_CONTEXT_KEEP_ACT_RESULTS must be a positive integer");
}

const PRUNED_SHOP_ACT_RESULT = "[旧的 shop_act 工具结果已裁剪；done=false]";
const COMPRESSION_VERSION = structuredMemory
	? `keep-last-${parsedKeepActResults}-shop-act-results-v2-structured`
	: `keep-last-${parsedKeepActResults}-shop-act-results-v1`;

/** Concatenated text of a toolResult message (provider-agnostic). */
function messageText(message: { content?: unknown }): string {
	if (!Array.isArray(message.content)) return "";
	return message.content
		.map((part) => (part && typeof part === "object" && (part as { type?: string }).type === "text"
			? String((part as { text?: unknown }).text ?? "")
			: ""))
		.join("");
}
// Error-classification prefixes. These MUST stay byte-identical to the Python
// side in slime/examples/ShopSimulator/pi_harness.py (INFRASTRUCTURE_ERROR_PREFIX
// / AGENT_ERROR_PREFIX, used by _classify_tool_error): the harness decides
// "retry vs reject" based on these prefixes. A silent drift on either side
// misclassifies agent errors as infrastructure errors (or vice versa).
// Parity is enforced by tests/test_shopsimulator/test_pi_harness.py.
const INFRASTRUCTURE_ERROR_PREFIX = "[shop_infrastructure]";
const AGENT_ERROR_PREFIX = "[shop_agent]";

type EnvResult = {
	env_idx?: number;
	done?: boolean;
	reward?: number;
	over?: boolean;
	instruction?: string;
	error?: string;
	// Populated by ShopSimulator only when done=true.
	reward_detail?: Record<string, unknown>;
	purchase?: Record<string, unknown>;
	goal?: Record<string, unknown>;
	[key: string]: unknown;
};

async function callEnv(payload: Record<string, unknown>, signal?: AbortSignal): Promise<EnvResult> {
	try {
		const response = await fetch(SHOP_ENV_URL, {
			method: "POST",
			headers: { "content-type": "application/json" },
			body: JSON.stringify(payload),
			signal,
		});
		if (!response.ok) throw new Error(`ShopSimulator HTTP ${response.status}: ${await response.text()}`);
		const body = (await response.json()) as { result?: EnvResult; error?: string };
		if (body.error) throw new Error(`ShopSimulator error: ${body.error}`);
		if (!body.result || typeof body.result !== "object") throw new Error("ShopSimulator response is missing result");
		if (body.result.error) throw new Error(`ShopSimulator error: ${body.result.error}`);
		return body.result;
	} catch (error) {
		const message = error instanceof Error ? error.message : String(error);
		if (message.startsWith(INFRASTRUCTURE_ERROR_PREFIX)) throw error;
		throw new Error(`${INFRASTRUCTURE_ERROR_PREFIX} ${message}`);
	}
}

function record(value: unknown): Record<string, unknown> | null {
	return value && typeof value === "object" && !Array.isArray(value)
		? (value as Record<string, unknown>)
		: null;
}

function details(result: EnvResult) {
	return {
		env_idx: typeof result.env_idx === "number" ? result.env_idx : null,
		done: Boolean(result.done),
		reward: typeof result.reward === "number" ? result.reward : 0,
		over: Boolean(result.over),
		// ShopSimulator scores four sub-dimensions (type/attribute/option/price).
		// Keeping them lets the finalizer report the same metrics as the upstream
		// scorer instead of only the scalar reward.
		reward_detail: record(result.reward_detail),
		purchase_asin: record(result.purchase)?.asin ?? null,
		goal_asin: record(result.goal)?.asin ?? null,
	};
}

export default function (pi: ExtensionAPI) {
	let envIdx: number | undefined;
	let terminalResult: EnvResult | undefined;
	let contextTraceIndex = 0;
	// Single-result summary cache: the same history is re-processed on every
	// model turn, and the page texts are kilobytes long, so re-parsing them
	// each time would burn CPU on the adapter's single event loop.
	const summaryCache = new Map<string, string>();
	const SUMMARY_CACHE_LIMIT = 512;

	const cachedSummary = (text: string, action: string): string => {
		const key = `${action}\u0000${text}`;
		const hit = summaryCache.get(key);
		if (hit !== undefined) return hit;
		const line = summarizeShopActResult(text, action);
		if (summaryCache.size >= SUMMARY_CACHE_LIMIT) summaryCache.clear();
		summaryCache.set(key, line);
		return line;
	};

	/** Map toolCallId → native action string for the messages in one request. */
	const collectCallActions = (messages: readonly { role?: string; content?: unknown }[]): Map<string, string> => {
		const actions = new Map<string, string>();
		for (const message of messages) {
			if (message.role !== "assistant" || !Array.isArray(message.content)) continue;
			for (const part of message.content) {
				if (!part || typeof part !== "object") continue;
				const candidate = part as { type?: string; id?: unknown; arguments?: unknown };
				if (candidate.type !== "toolCall" || typeof candidate.id !== "string") continue;
				const args = candidate.arguments;
				if (args && typeof args === "object" && typeof (args as { action?: unknown }).action === "string") {
					actions.set(candidate.id, (args as { action: string }).action);
				}
			}
		}
		return actions;
	};

	// Deterministic, non-destructive request-time pruning. Pi keeps the full
	// in-process history and JSON events; only the copy sent to the model is
	// shortened. Preserve toolResult messages and their call IDs so provider
	// tool-call pairing remains valid. Older shop_act results are replaced by
	// compact memory lines (structured memory) or a placeholder when disabled.
	pi.on("context", async (event) => {
		const shopActResultIndices = event.messages
			.map((message, index) => message.role === "toolResult" && message.toolName === "shop_act" ? index : -1)
			.filter((index) => index >= 0);
		const pruneBefore = Math.max(0, shopActResultIndices.length - parsedKeepActResults);
		const prunedIndices = shopActResultIndices.slice(0, pruneBefore);
		const replacements = new Map<number, string>();
		if (structuredMemory) {
			const callActions = collectCallActions(event.messages);
			const summaries = prunedIndices.map((index) => {
				const message = event.messages[index] as { toolCallId?: unknown };
				const action = typeof message.toolCallId === "string" ? callActions.get(message.toolCallId) ?? "" : "";
				return cachedSummary(messageText(message), action);
			});
			buildMemoryLines(summaries).forEach((line, position) => {
				replacements.set(prunedIndices[position], line);
			});
		} else {
			prunedIndices.forEach((index) => replacements.set(index, PRUNED_SHOP_ACT_RESULT));
		}
		let changed = false;
		const messages = event.messages.map((message, index) => {
			if (message.role === "assistant" && Array.isArray(message.content)) {
				const content = message.content.filter((part) => part.type !== "thinking");
				if (content.length !== message.content.length) {
					changed = true;
					return { ...message, content };
				}
			}
			if (replacements.has(index) && message.role === "toolResult") {
				changed = true;
				return {
					...message,
					content: [{ type: "text", text: replacements.get(index) as string }],
				};
			}
			return message;
		});
		if (contextTracePath) {
			appendFileSync(contextTracePath, `${JSON.stringify({
				schema_version: 1,
				request_index: contextTraceIndex++,
				compression_version: COMPRESSION_VERSION,
				messages,
			})}\n`, { encoding: "utf8", mode: 0o600 });
		}
		return changed ? { messages } : undefined;
	});

	pi.registerTool({
		name: "shop_reset",
		label: "Reset shopping task",
		description: "Start the assigned shopping task. Call exactly once before shop_act.",
		executionMode: "sequential",
		parameters: Type.Object({}, { additionalProperties: false }),
		async execute(_id, _params, signal) {
			if (envIdx !== undefined || terminalResult !== undefined) {
				throw new Error(`${AGENT_ERROR_PREFIX} shop_reset has already been called`);
			}
			const result = await callEnv({ action: "reset", idx: parsedTaskId, rollout_session_id: rolloutSessionId }, signal);
			if (typeof result.env_idx !== "number") {
				throw new Error(`${INFRASTRUCTURE_ERROR_PREFIX} ShopSimulator reset did not return env_idx`);
			}
			envIdx = result.env_idx;
			return {
				content: [{ type: "text", text: String(result.instruction || result.message || "Task started") }],
				details: details(result),
			};
		},
	});

	pi.registerTool({
		name: "shop_act",
		label: "Act in shop",
		description: "Send one native ShopSimulator action, for example search[...] or click[...].",
		executionMode: "sequential",
		parameters: Type.Object({ action: Type.String({ minLength: 1 }) }, { additionalProperties: false }),
		async execute(_id, params, signal) {
			if (terminalResult !== undefined) {
				return {
					content: [{ type: "text", text: "[environment already terminal; action ignored]" }],
					details: { ...details(terminalResult), terminal_noop: true },
				};
			}
			if (envIdx === undefined) throw new Error(`${AGENT_ERROR_PREFIX} Call shop_reset before shop_act`);
			const result = await callEnv({ action: "interact", env_idx: envIdx, response: params.action, rollout_session_id: rolloutSessionId }, signal);
			const resultDetails = details(result);
			if (result.done || result.over) {
				terminalResult = result;
				envIdx = undefined; // the server auto-releases on over=true
			}
			return {
				content: [{ type: "text", text: String(result.instruction || result.message || "") }],
				details: resultDetails,
			};
		},
	});

	pi.on("session_shutdown", async () => {
		const allocated = envIdx;
		envIdx = undefined;
		if (allocated === undefined) return;
		try {
			await callEnv({ action: "release_one", env_idx: allocated, rollout_session_id: rolloutSessionId });
		} catch (error) {
			console.error(`[shop_extension] failed to release env ${allocated}:`, error);
		}
	});
}
