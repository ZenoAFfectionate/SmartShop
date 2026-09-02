#!/usr/bin/env python3
"""Shared utilities for the ShopSimulator example.

Three功能段合入本模块，避免散落的辅助脚本：

1. DAPO dynamic-sampling filter — agent 多轮 rollout 的零方差组过滤
   （slime ``--dynamic-sampling-filter-path`` 动态加载）；
2. Evaluation summarization — 汇总评测 rollout dump 为 eval_results.json；
3. README table formatting — 把 eval_results.json 渲染为 README 结果表行。

统一 CLI::

    python -m examples.ShopSimulator.utils summarize --run-root RUN_ROOT
    python -m examples.ShopSimulator.utils summarize --dump DUMP.pt
    python -m examples.ShopSimulator.utils table --label "RL" --results RUN_ROOT [--header]

历史薄壳 ``summarize_eval.py`` / ``format_eval_table.py`` 已并入本模块并删除。
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any

from .common import official_metrics

__all__ = [
    "check_reward_nonzero_std_grouped",
    "summarize_dump",
    "summarize_run",
    "load_metrics",
    "format_row",
    "format_header",
    "TABLE_COLUMNS",
]


# =====================================================================
# 1) DAPO dynamic-sampling filter（agent 嵌套组结构适配版）
# =====================================================================
#
# slime 官方 ``check_reward_nonzero_std`` 只适配标准单轮 rollout 的平铺
# ``list[Sample]`` 组；本项目的 agent rollout 返回 ``list[list[Sample]]``
# （每个 candidate 是一条多轮轨迹拆成的多个分段）。此实现两种结构都兼容：
# candidate 的 reward 取其所有分段的 max（任务奖励写在终止分段上）。


def _candidate_reward(args, candidate: list) -> float:
    return max(sample.get_reward_value(args) for sample in candidate)


def check_reward_nonzero_std_grouped(args, samples, **kwargs):
    """Keep only groups whose candidate rewards have non-zero std (DAPO dynamic sampling)."""
    import torch

    if samples and isinstance(samples[0], list):
        # Agent rollout: group of candidates, each candidate a list of fragments.
        rewards = [_candidate_reward(args, candidate) for candidate in samples]
    else:
        # Standard rollout: flat list[Sample].
        rewards = [sample.get_reward_value(args) for sample in samples]
    keep = torch.tensor(rewards, dtype=torch.float64).std() > 1e-6

    from slime.rollout.filter_hub.base_types import DynamicFilterOutput

    return DynamicFilterOutput(
        keep=keep,
        reason=None if keep else f"zero_std_{round(float(rewards[0]), 1)}",
    )


# =====================================================================
# 2) Evaluation summarization
# =====================================================================


def _candidate_key(sample: dict[str, Any]) -> tuple[Any, Any]:
    metadata = sample.get("metadata") or {}
    identity = sample.get("rollout_id")
    if identity is None:
        identity = sample.get("index")
    return metadata.get("task_id"), identity


def summarize_dump(path: Path) -> dict[str, Any]:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    samples = payload.get("samples") if isinstance(payload, dict) else None
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"evaluation dump has no samples: {path}")

    by_candidate: dict[tuple[Any, Any], list[dict[str, Any]]] = collections.defaultdict(list)
    for sample in samples:
        if not isinstance(sample, dict):
            raise ValueError(f"evaluation dump contains a non-object sample: {path}")
        by_candidate[_candidate_key(sample)].append(sample)

    records: list[dict[str, Any]] = []
    termination = collections.Counter()
    error_kinds = collections.Counter()
    for key, fragments in sorted(by_candidate.items(), key=lambda item: repr(item[0])):
        metadata = fragments[0].get("metadata") or {}
        rewards = {float(fragment.get("reward", 0.0) or 0.0) for fragment in fragments}
        if len(rewards) != 1:
            raise ValueError(f"candidate {key} has inconsistent reward across fragments")
        termination[str(metadata.get("termination_reason") or "unspecified")] += 1
        error_kinds[str(metadata.get("error_kind") or "none")] += 1
        records.append({
            "task_id": metadata.get("task_id"),
            "candidate": key[1],
            "fragments": len(fragments),
            "reward": next(iter(rewards)),
            "reward_detail": metadata.get("reward_detail") or {},
            "purchase_asin": metadata.get("purchase_asin"),
            "goal_asin": metadata.get("goal_asin"),
            "done": bool(metadata.get("env_done")),
            "over": bool(metadata.get("env_over")),
            "model_turns": metadata.get("model_turns"),
            "tool_calls": metadata.get("pi_tool_calls"),
            "termination_reason": metadata.get("termination_reason"),
            "error_kind": metadata.get("error_kind"),
            "error_message": metadata.get("error_message"),
            # Slice dimensions for per-category / difficulty breakdowns (D1).
            "category": metadata.get("category"),
            "attribute_count": metadata.get("attribute_count"),
            "option_count": metadata.get("option_count"),
        })

    return {
        "path": str(path),
        "tasks": len({record["task_id"] for record in records}),
        "candidates": len(records),
        "fragments": len(samples),
        "termination_reason_counts": dict(sorted(termination.items())),
        "error_kind_counts": dict(sorted(error_kinds.items())),
        "metrics": official_metrics(records),
        "records": records,
    }


def summarize_run(run_root: Path) -> dict[str, Any]:
    dump_dir = run_root / "rollout_dumps"
    dumps = sorted(dump_dir.glob("rollout_eval_*.pt"))
    if not dumps:
        dumps = sorted(dump_dir.glob("*.pt"))
    if not dumps:
        raise ValueError(f"no evaluation dump below {dump_dir}")

    summaries = [summarize_dump(path) for path in dumps]
    records = [record for summary in summaries for record in summary["records"]]
    result = {
        "status": "eval-complete",
        "command": (
            (run_root / "eval_command.txt").read_text(encoding="utf-8").strip()
            if (run_root / "eval_command.txt").is_file()
            else None
        ),
        "metrics": official_metrics(records),
        "dumps": [
            {key: value for key, value in summary.items() if key != "records"}
            for summary in summaries
        ],
        "records": records,
    }
    output = run_root / "eval_results.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    return result


# =====================================================================
# 3) README table formatting
# =====================================================================

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


# =====================================================================
# Unified CLI
# =====================================================================


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p_sum = sub.add_parser("summarize", help="summarize eval dumps into eval_results.json")
    p_sum.add_argument("--run-root", type=Path, help="summarize every eval dump below this run root")
    p_sum.add_argument("--dump", type=Path, help="summarize a single rollout dump (prints, no file)")

    p_tab = sub.add_parser("table", help="render a README evaluation-table row")
    p_tab.add_argument("--label", action="append", default=[],
                       help="row label; repeat together with --results, or once for all")
    p_tab.add_argument("--results", action="append", type=Path, required=True,
                       help="eval RUN_ROOT or eval_results.json; repeatable")
    p_tab.add_argument("--header", action="store_true",
                       help="print the markdown table header first")

    args = parser.parse_args(argv)

    if args.command == "summarize":
        if args.dump is not None:
            if args.run_root is not None:
                parser.error("use either --run-root or --dump, not both")
            summary = summarize_dump(args.dump)
            print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))
            return 0
        if args.run_root is None:
            parser.error("either --run-root or --dump is required")
        result = summarize_run(args.run_root)
        print(json.dumps(
            {key: value for key, value in result.items() if key != "records"},
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        ))
        return 0

    # table
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
