/**
 * Shared loader that reproduces how Pi loads shop_extension.ts: jiti resolves
 * the TypeScript entry point, and only `@earendil-works/pi-ai` (whose `Type`
 * value is needed to declare tool schemas) is aliased to a stub. Everything
 * else in the pi packages is type-only and erased before execution.
 *
 * Used by shop_extension.test.mjs (hook behaviour) and run_extension_hook.mjs
 * (cross-language context parity checks driven from Python).
 */

import { execSync } from "node:child_process";
import { realpathSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);

export const SLIME_ROOT = join(here, "..", "..", "..");
export const EXTENSION = join(SLIME_ROOT, "examples/ShopSimulator/shop_extension.ts");
export const MOCK_PI_AI = join(here, "mock_pi_ai.cjs");

/** Locate the jiti copy that ships with the installed Pi package. */
export function loadJiti() {
	const piBin = execSync("command -v pi", { encoding: "utf8" }).trim();
	const packageRoot = join(dirname(realpathSync(piBin)), "..");
	return require(join(packageRoot, "node_modules", "jiti"));
}

const createJiti = loadJiti();

/**
 * Load a fresh copy of the extension (cache disabled, so module-level env
 * reads happen again) and capture the handlers/tools it registers.
 */
export function loadExtension(env = {}) {
	const saved = {};
	for (const [key, value] of Object.entries(env)) {
		saved[key] = process.env[key];
		process.env[key] = value;
	}
	const jiti = createJiti(fileURLToPath(import.meta.url), {
		alias: { "@earendil-works/pi-ai": MOCK_PI_AI },
		moduleCache: false, // each load must re-read module-level env
		fsCache: false,
	});
	const handlers = {};
	const tools = [];
	try {
		const module = jiti(EXTENSION);
		module.default({
			on: (name, handler) => { handlers[name] = handler; },
			registerTool: (definition) => { tools.push(definition); },
		});
	} finally {
		for (const [key, value] of Object.entries(saved)) {
			if (value === undefined) delete process.env[key];
			else process.env[key] = value;
		}
	}
	return { handlers, tools };
}
