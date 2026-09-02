import { appendFileSync } from "node:fs";
import { Type } from "@earendil-works/pi-ai";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

const SHOP_ENV_URL = process.env.SHOP_ENV_URL || "http://127.0.0.1:5000/api/shop_agent";
const parsedKeepActResults = Number.parseInt(process.env.SHOP_CONTEXT_KEEP_ACT_RESULTS || "3", 10);
const parsedTaskId = Number.parseInt(process.env.SHOP_TASK_ID || "", 10);
const rolloutSessionId = process.env.SHOP_ROLLOUT_SESSION_ID || "";
const contextTracePath = process.env.SHOP_CONTEXT_TRACE_PATH || "";
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

	// Deterministic, non-destructive request-time pruning. Pi keeps the full
	// in-process history and JSON events; only the copy sent to the model is
	// shortened. Preserve toolResult messages and their call IDs so provider
	// tool-call pairing remains valid.
	pi.on("context", async (event) => {
		const shopActResultIndices = event.messages
			.map((message, index) => message.role === "toolResult" && message.toolName === "shop_act" ? index : -1)
			.filter((index) => index >= 0);
		const pruneBefore = Math.max(0, shopActResultIndices.length - parsedKeepActResults);
		const prunedIndices = new Set(shopActResultIndices.slice(0, pruneBefore));
		let changed = false;
		const messages = event.messages.map((message, index) => {
			if (message.role === "assistant" && Array.isArray(message.content)) {
				const content = message.content.filter((part) => part.type !== "thinking");
				if (content.length !== message.content.length) {
					changed = true;
					return { ...message, content };
				}
			}
			if (prunedIndices.has(index) && message.role === "toolResult") {
				changed = true;
				return {
					...message,
					content: [{ type: "text", text: PRUNED_SHOP_ACT_RESULT }],
				};
			}
			return message;
		});
		if (contextTracePath) {
			appendFileSync(contextTracePath, `${JSON.stringify({
				schema_version: 1,
				request_index: contextTraceIndex++,
				compression_version: `keep-last-${parsedKeepActResults}-shop-act-results-v1`,
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
