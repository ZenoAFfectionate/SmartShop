#!/usr/bin/env python3
"""Summarize Slime evaluation dumps without enforcing dataset identity."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any

from .common import official_metrics


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, help="summarize every eval dump below this run root")
    parser.add_argument(
        "--dump",
        type=Path,
        help="summarize a single rollout dump file (B2 mid-training inspection); "
             "prints the summary without writing eval_results.json",
    )
    args = parser.parse_args()
    if args.dump is not None:
        if args.run_root is not None:
            parser.error("use either --run-root or --dump, not both")
        summary = summarize_dump(args.dump)
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))
        return
    if args.run_root is None:
        parser.error("either --run-root or --dump is required")
    result = summarize_run(args.run_root)
    print(json.dumps(
        {key: value for key, value in result.items() if key != "records"},
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    ))


if __name__ == "__main__":
    main()
