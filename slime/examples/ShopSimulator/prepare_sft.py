#!/usr/bin/env python3
"""Convert accepted pi trajectories into independently trainable turn examples."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from slime.utils.mask_utils import MultiTurnLossMaskGenerator
from slime.utils.processing_utils import load_tokenizer

TOOLS = [
    {"type": "function", "function": {"name": "shop_reset", "description": "Start the assigned shopping task.", "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}},
    {"type": "function", "function": {"name": "shop_act", "description": "Send one native ShopSimulator action.", "parameters": {"type": "object", "properties": {"action": {"type": "string"}}, "required": ["action"], "additionalProperties": False}}},
]


def text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return "" if content is None else str(content)
    return "".join(str(part.get("text", "")) for part in content if isinstance(part, dict) and part.get("type") == "text")


def to_qwen_message(message: dict, *, trainable: bool) -> dict:
    role = message["role"]
    if role in {"system", "user"}:
        return {"role": role, "content": text_content(message.get("content")), "step_loss_mask": 0}
    if role == "toolResult":
        return {"role": "tool", "content": text_content(message.get("content")), "tool_call_id": str(message.get("toolCallId", "")), "step_loss_mask": 0}
    if role != "assistant":
        raise ValueError(f"unsupported pi role: {role}")
    parts = message.get("content") or []
    if any(isinstance(part, dict) and part.get("type") == "thinking" for part in parts):
        raise ValueError("thinking block reached SFT conversion")
    result: dict[str, Any] = {"role": "assistant", "content": text_content(parts), "step_loss_mask": 1 if trainable else 0}
    calls = []
    for part in parts if isinstance(parts, list) else []:
        if isinstance(part, dict) and part.get("type") == "toolCall":
            calls.append({"id": str(part.get("id", "")), "type": "function", "function": {"name": part["name"], "arguments": part.get("arguments") or {}}})
    if calls:
        result["tool_calls"] = calls
    return result


def percentile(values: list[int], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return float(ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))])


def prepare(
    input_dir: Path,
    output_dir: Path,
    tokenizer_path: str,
    max_tokens: int,
    min_reward: float = 0.0,
) -> dict:
    """Expand every accepted trajectory into one example per decision turn."""
    raw_files = sorted((input_dir / "raw").rglob("*.json"))
    candidates = [json.loads(path.read_text(encoding="utf-8")) for path in raw_files]
    accepted = [row for row in candidates if row.get("accepted")]
    tokenizer = load_tokenizer(tokenizer_path, trust_remote_code=True)
    masker = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type="qwen3_5")
    examples = []
    rejected = []
    reward_filtered = []
    for row in accepted:
        if min_reward > 0 and float(row.get("reward", 0.0) or 0.0) < min_reward:
            reward_filtered.append({
                "trajectory_id": row["trajectory_id"],
                "reason": "reward_below_min",
                "reward": float(row.get("reward", 0.0) or 0.0),
            })
            continue
        traces = row.get("paired_context_traces") or []
        messages = row["messages"]
        if not row.get("context_trace_valid") or len(traces) != len(row.get("context_snapshots") or []):
            rejected.append({"trajectory_id": row["trajectory_id"], "reason": "invalid_context_trace"})
            continue
        turn_count = len(traces)
        keep_act_results = row.get("harness", {}).get("context_keep_act_results")
        for turn_index, trace in enumerate(traces):
            target_index = int(row["context_snapshots"][turn_index]["assistant_message_index"])
            target = messages[target_index]
            qwen_messages = [to_qwen_message(message, trainable=False) for message in trace["messages"]]
            qwen_messages.append(to_qwen_message(target, trainable=True))
            token_ids, loss_mask = masker.get_loss_mask(qwen_messages, tools=TOOLS)
            if len(token_ids) != len(loss_mask) or not any(loss_mask):
                raise ValueError(f"invalid loss mask for {row['trajectory_id']} turn {turn_index}")
            example_id = f"{row['trajectory_id']}:turn-{turn_index}"
            item = {
                "messages": qwen_messages,
                "tools": TOOLS,
                "metadata": {
                    "example_id": example_id,
                    "trajectory_id": row["trajectory_id"],
                    "task_id": row["task_id"],
                    "assistant_turn_index": turn_index,
                    "compression_version": trace.get("compression_version"),
                    "trajectory_turn_count": turn_count,
                    "reward": row["reward"],
                    "token_count": len(token_ids),
                    "target_token_count": sum(loss_mask),
                    # A3: record the context pruning depth so RL startup can
                    # verify SFT/RL context-distribution consistency.
                    "context_keep_act_results": keep_act_results,
                },
            }
            if len(token_ids) > max_tokens:
                rejected.append({"example_id": example_id, "trajectory_id": row["trajectory_id"], "reason": "over_token_limit", "token_count": len(token_ids)})
            else:
                examples.append(item)
    ids = [row["metadata"]["example_id"] for row in examples]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate turn example id")
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "turn_examples.jsonl").open("w", encoding="utf-8") as handle:
        for row in examples:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    with (output_dir / "accepted_trajectories.jsonl").open("w", encoding="utf-8") as handle:
        for row in accepted:
            handle.write(json.dumps({"trajectory_id": row["trajectory_id"], "task_id": row["task_id"], "reward": row["reward"], "raw_path": str(Path("raw") / f"{int(row['task_id']):06d}" / f"{int(row['sample_id']):03d}.json")}, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    token_counts = [row["metadata"]["token_count"] for row in examples]
    target_counts = [row["metadata"]["target_token_count"] for row in examples]
    accepted_rewards = sorted(float(row.get("reward", 0.0) or 0.0) for row in accepted)
    keep_values = sorted({
        row["metadata"]["context_keep_act_results"]
        for row in examples
        if row["metadata"]["context_keep_act_results"] is not None
    })
    over_limit = sum(1 for row in rejected if row["reason"] == "over_token_limit")
    summary = {
        "schema_version": 1,
        "tokenizer": tokenizer_path,
        "max_tokens": max_tokens,
        "min_reward": min_reward,
        "candidates": len(candidates),
        "accepted_trajectories": len(accepted),
        "reward_filtered_trajectories": len(reward_filtered),
        "turn_examples": len(examples),
        "rejected_examples": len(rejected),
        "rejection_reasons": {reason: sum(1 for row in rejected if row["reason"] == reason) for reason in sorted({row["reason"] for row in rejected})},
        "over_token_limit_rate": round(over_limit / (len(examples) + over_limit), 6) if examples or over_limit else 0.0,
        "trace_mismatches": sum(1 for row in accepted if not row.get("context_trace_valid")),
        "thinking_blocks": 0,
        # C3: reward stratification of the accepted teacher trajectories.
        "reward_min": accepted_rewards[0] if accepted_rewards else 0.0,
        "reward_p25": percentile([int(r * 1000) for r in accepted_rewards], 0.25) / 1000 if accepted_rewards else 0.0,
        "reward_p50": percentile([int(r * 1000) for r in accepted_rewards], 0.50) / 1000 if accepted_rewards else 0.0,
        "reward_p75": percentile([int(r * 1000) for r in accepted_rewards], 0.75) / 1000 if accepted_rewards else 0.0,
        "reward_max": accepted_rewards[-1] if accepted_rewards else 0.0,
        # A3: the context pruning depth embedded in this dataset. RL startup
        # must be configured with the same value.
        "context_keep_act_results_values": keep_values,
        "token_count_mean": statistics.mean(token_counts) if token_counts else 0,
        "token_count_p95": percentile(token_counts, 0.95),
        "token_count_max": max(token_counts, default=0),
        "target_token_count_mean": statistics.mean(target_counts) if target_counts else 0,
    }
    (output_dir / "rejected_turn_examples.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rejected + reward_filtered), encoding="utf-8")
    (output_dir / "turn_examples_summary.json").write_text(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--tokenizer", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument(
        "--min-reward",
        type=float,
        default=0.0,
        help="Drop accepted teacher trajectories whose reward is below this "
             "threshold (C3 quality gate). 0 keeps every accepted trajectory.",
    )
    args = parser.parse_args()
    summary = prepare(
        args.input_dir,
        args.output_dir or args.input_dir / "prepared",
        args.tokenizer,
        args.max_tokens,
        args.min_reward,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
