#!/usr/bin/env python3
"""Render a full README evaluation-table row from eval_results.json (G1).

The README result table is being extended beyond the five headline numbers to
the full metric set that ``official_metrics`` already computes: the four
sub-scores (r_type / r_att / r_option / r_price), done rate, right-product
rate, turn-limit rate and mean model turns. This tool formats one markdown
row per run so new evaluations can be pasted into the README table without
hand-editing numbers.

Usage:
    python -m examples.ShopSimulator.format_eval_table \
        --label "RL" --results /path/to/eval_run_root

Accepts either an eval RUN_ROOT (containing eval_results.json) or the json
file itself. ``--header`` prints the table header for a fresh table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

TABLE_COLUMNS = (
    "模型", "正奖励 pass@1", "严格成功", "r_loose", "r_hard",
    "r_type", "r_att", "r_option", "r_price",
    "done 率", "买对商品", "turn_limit 率", "平均 turn 数",
)


def _fmt_ratio(value: Any) -> str:
    if value is None:
        return "—"
    return f"{100.0 * float(value):.1f}%"


def _fmt_mean(value: Any) -> str:
    if value is None:
        return "—"
    return f"{float(value):.6f}"


def _fmt_turns(value: Any) -> str:
    if value is None:
        return "—"
    return f"{float(value):.1f}"


def load_metrics(path: Path) -> dict[str, Any]:
    if path.is_dir():
        path = path / "eval_results.json"
    if not path.is_file():
        raise SystemExit(f"no eval_results.json at {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload.get("metrics") or {}
    records = payload.get("records") or []
    # Positive-reward pass@1: prefer the per-task pass_at_k value; fall back to
    # a direct count over records when task ids were unavailable.
    positive = None
    pass_at_k = metrics.get("pass_at_k")
    if isinstance(pass_at_k, dict):
        positive = pass_at_k.get("pass_positive_reward")
    elif records:
        positive = sum(1 for row in records if float(row.get("reward", 0.0) or 0.0) > 0) / len(records)
    metrics = dict(metrics)
    metrics["_positive_pass1"] = positive
    return metrics


def _positive_pass1(metrics: dict[str, Any]) -> Any:
    """Positive-reward pass@1 with the same fallback chain as load_metrics."""
    if metrics.get("_positive_pass1") is not None:
        return metrics["_positive_pass1"]
    pass_at_k = metrics.get("pass_at_k")
    if isinstance(pass_at_k, dict):
        return pass_at_k.get("pass_positive_reward")
    return None


def format_row(label: str, metrics: dict[str, Any]) -> str:
    cells = [
        label,
        _fmt_ratio(_positive_pass1(metrics)),
        _fmt_ratio(metrics.get("r_success")),
        _fmt_mean(metrics.get("r_loose")),
        _fmt_mean(metrics.get("r_hard")),
        _fmt_mean(metrics.get("r_type")),
        _fmt_mean(metrics.get("r_att")),
        _fmt_mean(metrics.get("r_option")),
        _fmt_mean(metrics.get("r_price")),
        _fmt_ratio(metrics.get("done_rate")),
        _fmt_ratio(metrics.get("right_product")),
        _fmt_ratio(metrics.get("turn_limit_rate")),
        _fmt_turns(metrics.get("mean_model_turns")),
    ]
    return "| " + " | ".join(cells) + " |"


def format_header() -> str:
    lines = [
        "| " + " | ".join(TABLE_COLUMNS) + " |",
        "| " + " | ".join(["---"] + ["---:" for _ in TABLE_COLUMNS[1:]]) + " |",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", action="append", default=[],
                        help="row label; repeat together with --results, or once for all")
    parser.add_argument("--results", action="append", type=Path, required=True,
                        help="eval RUN_ROOT or eval_results.json; repeatable")
    parser.add_argument("--header", action="store_true",
                        help="print the markdown table header first")
    args = parser.parse_args()

    labels = args.label or [f"run_{i + 1}" for i in range(len(args.results))]
    if len(labels) != len(args.results):
        if len(labels) == 1:
            labels = labels * len(args.results)
        else:
            raise SystemExit("provide one --label per --results, or exactly one --label")

    if args.header:
        print(format_header())
    for label, path in zip(labels, args.results, strict=True):
        print(format_row(label, load_metrics(path)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
