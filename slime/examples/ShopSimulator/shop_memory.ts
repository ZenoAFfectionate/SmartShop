/**
 * Structured context memory for ShopSimulator tool results (R5).
 *
 * When the agentic history grows, older `shop_act` tool results are replaced by
 * a compact summary line instead of a "pruned" placeholder, so the model still
 * remembers which products it searched/viewed and at what price. Losing that
 * memory is what drives repeated-search loops (and the turn-limit failures
 * they cause), which a behaviour penalty alone can only punish, not prevent.
 *
 * IMPORTANT — cross-language parity: `shop_memory.py` re-implements this module
 * byte-for-byte for the SFT data pipeline (collect_sft.py rebuilds contexts in
 * Python). Both implementations MUST produce identical strings; parity is
 * enforced by tests/test_shopsimulator/test_shop_memory.py, which runs this
 * module through Node and compares outputs on a shared fixture set.
 *
 * This file must stay dependency-free (no pi imports) so it can be executed by
 * bare Node (`node --experimental-strip-types`) in tests.
 */

export const STRUCTURED_MEMORY_PREFIX = "[记忆]";
export const STRUCTURED_MEMORY_PLACEHOLDER = "[该结果已折叠；详见上方记忆条目]";
export const MAX_SUMMARY_CHARS = 240;
// Total character budget for all memory lines in one request. 40-turn traces
// would otherwise accumulate ~10k chars of summaries on top of the 3 full
// results and crowd the 16k context window; the oldest lines collapse first.
export const MAX_MEMORY_BUDGET_CHARS = 2400;
export const MAX_SEARCH_ITEMS = 4;
export const TITLE_MAX_CHARS = 18;
export const QUERY_MAX_CHARS = 20;
export const OPTION_MAX_ITEMS = 2;
export const OPTION_MAX_CHARS = 14;
export const FALLBACK_MAX_CHARS = 60;

const SEPARATOR = " [SEP] ";

/** Truncate by Unicode code points (mirrors Python slicing on str). */
export function clampText(value: string, limit: number): string {
	const chars = Array.from(value.trim());
	if (chars.length <= limit) return chars.join("");
	return `${chars.slice(0, limit).join("")}…`;
}

function splitSegments(text: string): string[] {
	return text.split(SEPARATOR).map((segment) => segment.trim());
}

function collapse(value: string): string {
	return value.replace(/\s+/g, " ").trim();
}

/** Parse a native ShopSimulator action (`search[镜子]`, `click[123...]`). */
export function parseShopAction(action: string | undefined): { kind: string; value: string } {
	const match = /^\s*(search|click)\s*\[(.*)\]\s*$/.exec(action ?? "");
	if (!match) return { kind: "", value: "" };
	return { kind: match[1], value: match[2].trim() };
}

/** Search-results page: navigation header + repeating (asin, title, price). */
function summarizeSearch(segments: string[], total: string, action: { kind: string; value: string }): string {
	const items: string[] = [];
	for (let index = 0; index + 2 < segments.length; index += 1) {
		const asin = segments[index];
		if (!/^\d{9,14}$/.test(asin)) continue;
		const title = clampText(segments[index + 1], TITLE_MAX_CHARS);
		const price = clampText(collapse(segments[index + 2]), 20);
		items.push(`${asin}|${title}|${price}`);
		if (items.length >= MAX_SEARCH_ITEMS) break;
		index += 2;
	}
	// The page itself only echoes the task description; the *actual* search
	// term lives in the tool call arguments, so prefer that when available.
	const query = action.kind === "search" ? clampText(action.value, QUERY_MAX_CHARS) : "";
	const head = query
		? `${STRUCTURED_MEMORY_PREFIX} search[${query}] ${total}件`
		: `${STRUCTURED_MEMORY_PREFIX} search ${total}件`;
	if (items.length === 0) return head;
	return `${head}: ${items.join("; ")}`;
}

// Navigation/affordance labels that must never be mistaken for a product title.
const DETAIL_NAVIGATION = new Set([
	"Back to Search", "< Prev", "Next >", "Description", "Features", "Reviews", "Buy Now",
]);

/** Product-detail page: title + price + shop + the first option group. */
function summarizeDetail(segments: string[], action: { kind: string; value: string }): string {
	let price = "";
	let shop = "";
	let priceIndex = -1;
	for (let index = 0; index < segments.length; index += 1) {
		const segment = segments[index];
		if (segment.startsWith("价格")) {
			price = clampText(collapse(segment), 24);
			if (priceIndex < 0) priceIndex = index;
		} else if (segment.startsWith("店铺")) {
			shop = clampText(collapse(segment), 18);
		}
	}
	// The title sits directly above the price line, but navigation tabs
	// (Reviews/Description/...) may appear between them after the agent clicks
	// around the page, so walk back to the first non-navigation segment.
	let title = "";
	let titleIndex = -1;
	for (let index = (priceIndex > 0 ? priceIndex - 1 : segments.length - 1); index > 0; index -= 1) {
		const segment = segments[index];
		if (!segment || DETAIL_NAVIGATION.has(segment) || segment.startsWith("价格") || segment.startsWith("店铺")) continue;
		title = clampText(segment, TITLE_MAX_CHARS);
		titleIndex = index;
		break;
	}
	// Options live between the "颜色分类" label and the title line.
	let optionLabel = "";
	const options: string[] = [];
	const labelIndex = segments.indexOf("颜色分类");
	if (labelIndex >= 0 && titleIndex > labelIndex) {
		optionLabel = "颜色分类";
		for (let index = labelIndex + 1; index < titleIndex && options.length < OPTION_MAX_ITEMS; index += 1) {
			const option = segments[index];
			if (!option || option === "Back to Search" || option === "< Prev") continue;
			options.push(clampText(option, OPTION_MAX_CHARS));
		}
	}
	// Prefer the clicked asin as the identifier; clicks on option/review labels
	// fall back to the product title so the line still names a product.
	const clicked = action.kind === "click" ? action.value : "";
	const identifier = /^\d{9,14}$/.test(clicked) ? clicked : title;
	const parts = [`${STRUCTURED_MEMORY_PREFIX} view[${identifier}]`];
	if (title && identifier !== title) parts.push(title);
	if (price) parts.push(price);
	if (shop) parts.push(shop);
	if (options.length > 0) parts.push(`${optionLabel}: ${options.join(", ")}`);
	return clampText(parts.join(" | "), MAX_SUMMARY_CHARS);
}

/** Terminal purchase page: extract the purchased asin and selected options. */
function summarizePurchase(segments: string[]): string {
	let asin = "";
	let options = "";
	for (let index = 0; index < segments.length; index += 1) {
		if (segments[index] === "asin" && index + 1 < segments.length) {
			const value = segments[index + 1];
			if (/^\d{9,14}$/.test(value)) asin = value;
		} else if (segments[index] === "options" && index + 1 < segments.length) {
			options = decodeOptions(segments[index + 1]);
		}
	}
	const parts = [`${STRUCTURED_MEMORY_PREFIX} buy✓`];
	if (asin) parts.push(`asin=${asin}`);
	if (options) parts.push(`options=${clampText(options, 60)}`);
	return clampText(parts.join(" "), MAX_SUMMARY_CHARS);
}

function decodeOptions(raw: string): string {
	try {
		// JSON.parse decodes \uXXXX escapes itself; json.loads does the same on
		// the Python side, keeping the two implementations byte-identical.
		const parsed = JSON.parse(raw) as Record<string, unknown>;
		if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return "";
		return Object.values(parsed)
			.map((value) => collapse(String(value)))
			.join(", ");
	} catch {
		return "";
	}
}

/** Fallback for welcome/error pages: first meaningful line. */
function summarizeFallback(segments: string[]): string {
	const skip = new Set(["Instruction:", "WebShop", "Back to Search", "< Prev", "Next >"]);
	const line = segments.find((segment) => segment && !skip.has(segment)) ?? "";
	return `${STRUCTURED_MEMORY_PREFIX} act: ${clampText(line, FALLBACK_MAX_CHARS)}`;
}

export function summarizeShopActResult(text: string, action?: string): string {
	if (!text || !text.trim()) return `${STRUCTURED_MEMORY_PREFIX} (空结果)`;
	const parsedAction = parseShopAction(action);
	const collapsed = collapse(text);
	if (collapsed.includes("Thank you for shopping")) {
		return summarizePurchase(splitSegments(text));
	}
	const total = /Total results:\s*(\d+)/.exec(collapsed);
	if (total) {
		return summarizeSearch(splitSegments(text), total[1], parsedAction);
	}
	if (/价格[:：]/.test(collapsed) && /Buy Now/i.test(collapsed)) {
		return summarizeDetail(splitSegments(text), parsedAction);
	}
	return summarizeFallback(splitSegments(text));
}

export function structuredMemoryEnabled(value: string | undefined): boolean {
	if (value === undefined || value.trim() === "") return true; // default on
	return !["0", "false", "no", "off"].includes(value.trim().toLowerCase());
}

/**
 * Apply the character budget to already-summarized lines (chronological
 * order), spending it from the newest entry backwards. Entries that do not fit
 * collapse to STRUCTURED_MEMORY_PLACEHOLDER, so total context growth stays
 * bounded no matter how long the trajectory is.
 *
 * Taking pre-computed summaries keeps the caller free to cache per-result
 * summaries (the extension sees the same history on every turn).
 */
export function buildMemoryLines(
	summaries: string[],
	budget: number = MAX_MEMORY_BUDGET_CHARS,
): string[] {
	const lines = summaries.map(() => STRUCTURED_MEMORY_PLACEHOLDER);
	let used = 0;
	for (let index = summaries.length - 1; index >= 0; index -= 1) {
		const cost = Array.from(summaries[index]).length + 1;
		if (used + cost > budget) break;
		lines[index] = summaries[index];
		used += cost;
	}
	return lines;
}
