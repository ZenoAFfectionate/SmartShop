"""Small helpers shared by ShopSimulator rollout and evaluation."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable

SUB_SCORES = ("r_type", "r_att", "r_option", "r_price")
# Dataset metadata dimensions used for per-slice metric breakdowns (D1).
SLICE_KEYS = ("category", "attribute_count", "option_count")


def prompt_text(prompt: str | list[dict[str, Any]]) -> str:
    """Flatten a dataset prompt into the string passed to Pi."""
    if isinstance(prompt, str):
        return prompt
    parts: list[str] = []
    for message in prompt or []:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content", "")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            parts.extend(
                str(block.get("text", ""))
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
    text = "\n".join(part for part in parts if part)
    if not text:
        raise ValueError("prompt has no user text")
    return text


def wilson_ci(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Preferred over the normal approximation because it stays inside [0, 1] and
    behaves well for small samples and extreme proportions (p near 0 or 1),
    which is exactly the Base-model regime (e.g. 0% strict success).
    """
    if total <= 0 or successes < 0:
        return (0.0, 0.0)
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _candidate_outcome(row: dict[str, Any]) -> dict[str, Any]:
    detail = row.get("reward_detail") or {}
    scored = bool(detail)
    values = {
        name: float(detail.get(name, 1.0 if name == "r_option" else 0.0)) if scored else 0.0
        for name in SUB_SCORES
    }
    hard_reward = 1.0
    for value in values.values():
        hard_reward *= value
    purchase, goal = row.get("purchase_asin"), row.get("goal_asin")
    turns = row.get("model_turns")
    tool_calls = row.get("tool_calls")
    return {
        "task_id": row.get("task_id"),
        "done": 1 if scored else 0,
        "r_loose": float(row.get("reward", 0.0) or 0.0),
        "r_hard": hard_reward,
        "r_success": 1 if scored and all(value == 1 for value in values.values()) else 0,
        "right_product": 1 if scored and purchase is not None and purchase == goal else 0,
        "sub_scores": values,
        # Efficiency fields (D2): absent on legacy records, tolerated here.
        "model_turns": int(turns) if isinstance(turns, (int, float)) else None,
        "tool_calls": int(tool_calls) if isinstance(tool_calls, (int, float)) else None,
        "termination_reason": row.get("termination_reason"),
        # Slice dimensions (D1): straight from the dataset row metadata.
        "category": row.get("category"),
        "attribute_count": row.get("attribute_count"),
        "option_count": row.get("option_count"),
    }


def _mean_or_none(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 6) if values else None


def official_metrics(candidates: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Compute mean@k-style averages, per-task pass@k and dimension statistics."""
    outcomes = [_candidate_outcome(row) for row in candidates]
    if not outcomes:
        return {"samples": 0}

    def mean(values: list[float]) -> float:
        return round(sum(values) / len(values), 6)

    metrics: dict[str, Any] = {
        "samples": len(outcomes),
        "done_rate": mean([outcome["done"] for outcome in outcomes]),
        "r_loose": mean([outcome["r_loose"] for outcome in outcomes]),
        "r_hard": mean([outcome["r_hard"] for outcome in outcomes]),
        "r_success": mean([outcome["r_success"] for outcome in outcomes]),
        "right_product": mean([outcome["right_product"] for outcome in outcomes]),
        **{
            name: mean([outcome["sub_scores"][name] for outcome in outcomes])
            for name in SUB_SCORES
        },
    }

    # Efficiency metrics (D2): how many turns/actions a candidate needed, and
    # how often the run was killed by the adapter turn cap.
    all_turns = [o["model_turns"] for o in outcomes if o["model_turns"] is not None]
    done_turns = [
        o["model_turns"] for o in outcomes
        if o["model_turns"] is not None and o["done"]
    ]
    all_actions = [o["tool_calls"] for o in outcomes if o["tool_calls"] is not None]
    metrics["mean_model_turns"] = _mean_or_none([float(v) for v in all_turns])
    metrics["mean_model_turns_done"] = _mean_or_none([float(v) for v in done_turns])
    metrics["mean_tool_calls"] = _mean_or_none([float(v) for v in all_actions])
    metrics["turn_limit_rate"] = mean([
        1 if o["termination_reason"] == "turn_limit" else 0 for o in outcomes
    ])

    # Confidence intervals (D4): Wilson intervals for the headline proportions.
    strict_successes = sum(o["r_success"] for o in outcomes)
    positive_rewards = sum(1 for o in outcomes if o["r_loose"] > 0)
    lo, hi = wilson_ci(strict_successes, len(outcomes))
    metrics["r_success_ci95"] = [round(lo, 6), round(hi, 6)]
    lo, hi = wilson_ci(positive_rewards, len(outcomes))
    metrics["pass_positive_reward_ci95"] = [round(lo, 6), round(hi, 6)]

    # Per-dimension slices (D1): locate weak categories / difficulty bands.
    # Slices with a single group or more than 50 groups are skipped as
    # non-informative for a 200-sample evaluation.
    for key in SLICE_KEYS:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for outcome in outcomes:
            value = outcome.get(key)
            if value is None:
                continue
            groups[str(value)].append(outcome)
        if 1 < len(groups) <= 50:
            metrics[f"by_{key}"] = {
                value: {
                    "samples": len(group),
                    "r_loose": mean([o["r_loose"] for o in group]),
                    "r_success": mean([o["r_success"] for o in group]),
                    "right_product": mean([o["right_product"] for o in group]),
                }
                for value, group in sorted(groups.items())
            }

    if any(outcome["task_id"] is None for outcome in outcomes):
        metrics["pass_at_k"] = None
        return metrics

    by_task: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for outcome in outcomes:
        by_task[outcome["task_id"]].append(outcome)
    counts = {len(group) for group in by_task.values()}
    metrics["pass_at_k"] = {
        "tasks": len(by_task),
        "samples_per_task": (
            min(counts) if len(counts) == 1 else {"min": min(counts), "max": max(counts)}
        ),
        "reward_variance_task_fraction": mean([
            1 if max(item["r_loose"] for item in group) != min(item["r_loose"] for item in group) else 0
            for group in by_task.values()
        ]),
        "all_zero_reward_task_fraction": mean([
            1 if max(item["r_loose"] for item in group) == 0 else 0
            for group in by_task.values()
        ]),
        "pass_success": mean([
            1 if any(item["r_success"] for item in group) else 0
            for group in by_task.values()
        ]),
        "pass_positive_reward": mean([
            1 if any(item["r_loose"] > 0 for item in group) else 0
            for group in by_task.values()
        ]),
        "pass_done": mean([
            1 if any(item["done"] for item in group) else 0
            for group in by_task.values()
        ]),
        "pass_right_product": mean([
            1 if any(item["right_product"] for item in group) else 0
            for group in by_task.values()
        ]),
    }
    return metrics
