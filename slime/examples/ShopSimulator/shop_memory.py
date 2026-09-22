"""Structured context memory for ShopSimulator tool results (R5) — Python twin.

`shop_memory.ts` compresses older ``shop_act`` tool results at rollout time
(inside the Pi extension); this module does the identical job for the SFT data
pipeline, where ``collect_sft.context_snapshots`` rebuilds the same contexts in
Python. Both sides MUST emit byte-identical strings — otherwise SFT samples and
RL rollouts would see different contexts and the train/deploy distribution
alignment (a core quality gate of this project) breaks.

Parity is enforced by ``tests/test_shopsimulator/test_shop_memory.py``: it runs
the TypeScript module through Node on a shared fixture set and compares every
output character-by-character. Keep the algorithms and constants in lockstep
with ``shop_memory.ts`` when editing either file.
"""

from __future__ import annotations

import json
import re

STRUCTURED_MEMORY_PREFIX = "[记忆]"
STRUCTURED_MEMORY_PLACEHOLDER = "[该结果已折叠；详见上方记忆条目]"
MAX_SUMMARY_CHARS = 240
# Total character budget for all memory lines in one request (see shop_memory.ts).
MAX_MEMORY_BUDGET_CHARS = 2400
MAX_SEARCH_ITEMS = 4
TITLE_MAX_CHARS = 18
QUERY_MAX_CHARS = 20
OPTION_MAX_ITEMS = 2
OPTION_MAX_CHARS = 14
FALLBACK_MAX_CHARS = 60

_SEPARATOR = " [SEP] "
_WHITESPACE = re.compile(r"\s+")
_ASIN = re.compile(r"^[0-9]{9,14}$")
_TOTAL_RESULTS = re.compile(r"Total results:\s*([0-9]+)")
_PRICE_LABEL = re.compile(r"价格[:：]")
_BUY_NOW = re.compile(r"Buy Now", re.IGNORECASE)

_SKIP_FALLBACK = {"Instruction:", "WebShop", "Back to Search", "< Prev", "Next >"}
_ACTION = re.compile(r"^\s*(search|click)\s*\[(.*)\]\s*$")


def clamp_text(value: str, limit: int) -> str:
    """Truncate by Unicode code points (mirrors ``Array.from`` on the TS side)."""
    stripped = value.strip()
    if len(stripped) <= limit:
        return stripped
    return stripped[:limit] + "…"


def _collapse(value: str) -> str:
    return _WHITESPACE.sub(" ", value).strip()


def _segments(text: str) -> list[str]:
    return [segment.strip() for segment in text.split(_SEPARATOR)]


def _seg(segments: list[str], index: int) -> str:
    if 0 <= index < len(segments):
        return segments[index]
    return ""


def _index_of(segments: list[str], needle: str) -> int:
    try:
        return segments.index(needle)
    except ValueError:
        return -1


_JS_INT_KEY = re.compile(r"^(?:0|[1-9][0-9]*)$")
# JS "array index" upper bound (2**32 - 2): only array-index-like keys are
# re-ordered by Object.values(); bigger integer-like keys keep insertion order.
_JS_MAX_ARRAY_INDEX = 2**32 - 2


def _js_array_to_string(items: list) -> str:
    """Mimic JS ``Array.prototype.toString()``: elements joined by "," with
    null/undefined rendering as empty strings and nested arrays flattened."""
    return ",".join(
        ""
        if item is None
        else (_js_array_to_string(item) if isinstance(item, list) else _js_string(item))
        for item in items
    )


def _js_string(value: object) -> str:
    """Mimic JavaScript ``String(value)`` so option decoding stays identical.

    Covers the JSON boundary types that a plain ``str()`` renders differently
    (null/arrays/objects) — a silent divergence here would make SFT contexts
    differ from RL rollout contexts.
    """
    if value is None:
        return "null"  # JS String(null) === "null" (not "")
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        return _js_array_to_string(value)
    if isinstance(value, dict):
        return "[object Object]"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _js_object_values(parsed: dict) -> list:
    """Mimic JS ``Object.values()`` ordering: array-index-like keys come first in
    ascending numeric order, then the remaining keys in insertion order."""
    indexed: list[tuple[int, str]] = []
    rest: list[str] = []
    for key in parsed:
        if _JS_INT_KEY.match(key) and int(key) <= _JS_MAX_ARRAY_INDEX:
            indexed.append((int(key), key))
        else:
            rest.append(key)
    indexed.sort(key=lambda pair: pair[0])
    return [parsed[key] for _, key in indexed] + [parsed[key] for key in rest]


def _decode_options(raw: str) -> str:
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return ""
    if not isinstance(parsed, dict):
        return ""
    return ", ".join(_collapse(_js_string(value)) for value in _js_object_values(parsed))


def parse_shop_action(action: str | None) -> tuple[str, str]:
    """Parse a native ShopSimulator action (`search[镜子]`, `click[123...]`)."""
    match = _ACTION.match(action or "")
    if not match:
        return "", ""
    return match.group(1), match.group(2).strip()


def _summarize_search(segments: list[str], total: str, action: tuple[str, str]) -> str:
    items: list[str] = []
    index = 0
    while index + 2 < len(segments):
        asin = segments[index]
        if not _ASIN.match(asin):
            index += 1
            continue
        title = clamp_text(segments[index + 1], TITLE_MAX_CHARS)
        price = clamp_text(_collapse(segments[index + 2]), 20)
        items.append(f"{asin}|{title}|{price}")
        if len(items) >= MAX_SEARCH_ITEMS:
            break
        index += 3
    # The page echoes the task description; the real search term comes from the
    # tool-call arguments (see shop_extension.ts).
    query = clamp_text(action[1], QUERY_MAX_CHARS) if action[0] == "search" else ""
    head = (
        f"{STRUCTURED_MEMORY_PREFIX} search[{query}] {total}件"
        if query
        else f"{STRUCTURED_MEMORY_PREFIX} search {total}件"
    )
    if not items:
        return head
    return f"{head}: " + "; ".join(items)


# Navigation/affordance labels that must never be mistaken for a product title.
_DETAIL_NAVIGATION = {
    "Back to Search", "< Prev", "Next >", "Description", "Features", "Reviews", "Buy Now",
}


def _summarize_detail(segments: list[str], action: tuple[str, str]) -> str:
    price = ""
    shop = ""
    price_index = -1
    for index, segment in enumerate(segments):
        if segment.startswith("价格"):
            price = clamp_text(_collapse(segment), 24)
            if price_index < 0:
                price_index = index
        elif segment.startswith("店铺"):
            shop = clamp_text(_collapse(segment), 18)
    # The title sits directly above the price line, but navigation tabs may
    # appear in between after the agent clicks around the page.
    title = ""
    title_index = -1
    start = price_index - 1 if price_index > 0 else len(segments) - 1
    for index in range(start, 0, -1):
        segment = segments[index]
        if (
            not segment
            or segment in _DETAIL_NAVIGATION
            or segment.startswith("价格")
            or segment.startswith("店铺")
        ):
            continue
        title = clamp_text(segment, TITLE_MAX_CHARS)
        title_index = index
        break
    option_label = ""
    options: list[str] = []
    label_index = _index_of(segments, "颜色分类")
    if label_index >= 0 and title_index > label_index:
        option_label = "颜色分类"
        index = label_index + 1
        while index < title_index and len(options) < OPTION_MAX_ITEMS:
            option = segments[index]
            if option and option not in {"Back to Search", "< Prev"}:
                options.append(clamp_text(option, OPTION_MAX_CHARS))
            index += 1
    clicked = action[1] if action[0] == "click" else ""
    identifier = clicked if _ASIN.match(clicked) else title
    parts = [f"{STRUCTURED_MEMORY_PREFIX} view[{identifier}]"]
    if title and identifier != title:
        parts.append(title)
    if price:
        parts.append(price)
    if shop:
        parts.append(shop)
    if options:
        parts.append(f"{option_label}: " + ", ".join(options))
    return clamp_text(" | ".join(parts), MAX_SUMMARY_CHARS)


def _summarize_purchase(segments: list[str]) -> str:
    asin = ""
    options = ""
    for index, segment in enumerate(segments):
        if segment == "asin" and index + 1 < len(segments):
            value = segments[index + 1]
            if _ASIN.match(value):
                asin = value
        elif segment == "options" and index + 1 < len(segments):
            options = _decode_options(segments[index + 1])
    parts = [f"{STRUCTURED_MEMORY_PREFIX} buy✓"]
    if asin:
        parts.append(f"asin={asin}")
    if options:
        parts.append(f"options={clamp_text(options, 60)}")
    return clamp_text(" ".join(parts), MAX_SUMMARY_CHARS)


def _summarize_fallback(segments: list[str]) -> str:
    line = ""
    for segment in segments:
        if segment and segment not in _SKIP_FALLBACK:
            line = segment
            break
    return f"{STRUCTURED_MEMORY_PREFIX} act: {clamp_text(line, FALLBACK_MAX_CHARS)}"


def summarize_shop_act_result(text: str, action: str | None = None) -> str:
    """Compress one ``shop_act`` result into a single memory line."""
    if not text or not text.strip():
        return f"{STRUCTURED_MEMORY_PREFIX} (空结果)"
    parsed_action = parse_shop_action(action)
    collapsed = _collapse(text)
    if "Thank you for shopping" in collapsed:
        return _summarize_purchase(_segments(text))
    total = _TOTAL_RESULTS.search(collapsed)
    if total:
        return _summarize_search(_segments(text), total.group(1), parsed_action)
    if _PRICE_LABEL.search(collapsed) and _BUY_NOW.search(collapsed):
        return _summarize_detail(_segments(text), parsed_action)
    return _summarize_fallback(_segments(text))


def structured_memory_enabled(value: str | None) -> bool:
    """Mirror of the TS-side env switch (default on; 0/false/no/off disable)."""
    if value is None or value.strip() == "":
        return True
    return value.strip().lower() not in {"0", "false", "no", "off"}


def build_memory_lines(summaries: list[str], budget: int = MAX_MEMORY_BUDGET_CHARS) -> list[str]:
    """Apply the character budget to already-summarized lines (chronological).

    The budget is spent from the newest entry backwards; entries that do not
    fit collapse to STRUCTURED_MEMORY_PLACEHOLDER (mirrors buildMemoryLines).
    """
    lines = [STRUCTURED_MEMORY_PLACEHOLDER] * len(summaries)
    used = 0
    for index in range(len(summaries) - 1, -1, -1):
        cost = len(summaries[index]) + 1
        if used + cost > budget:
            break
        lines[index] = summaries[index]
        used += cost
    return lines
