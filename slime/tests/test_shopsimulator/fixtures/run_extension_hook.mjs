/**
 * Drive the real shop_extension context hook over a message list supplied as
 * JSON, printing the messages that would be sent to the model.
 *
 * Usage: node run_extension_hook.mjs <messages.json> [keepActResults] [structured]
 *
 * The Python side (test_shop_memory.py) feeds the same raw messages through
 * collect_sft.context_snapshots; both results must be identical, which is what
 * keeps SFT-rebuilt contexts and RL rollout contexts in sync.
 */

import { readFileSync } from "node:fs";

import { loadExtension } from "./extension_loader.mjs";

const [inputPath, keep = "3", structured = "1"] = process.argv.slice(2);
if (!inputPath) {
	console.error("usage: node run_extension_hook.mjs <messages.json> [keepActResults] [structured]");
	process.exit(2);
}

const messages = JSON.parse(readFileSync(inputPath, "utf8"));
const { handlers } = loadExtension({
	SHOP_ROLLOUT_SESSION_ID: "parity-session",
	SHOP_TASK_ID: "0",
	SHOP_CONTEXT_KEEP_ACT_RESULTS: keep,
	SHOP_CONTEXT_STRUCTURED_MEMORY: structured,
});

const result = await handlers.context({ messages });
// No rewrite means "send the history unchanged" — mirror that for the caller.
console.log(JSON.stringify({ messages: result ? result.messages : messages }));
