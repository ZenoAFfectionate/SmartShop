"""Parallel, resumable teacher-data collector for the bundled sft_512 tasks."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import tempfile
import time
from collections import Counter, deque
from pathlib import Path
from typing import Any

from .pi_harness import (
    AUTHORITATIVE_EVENT_TYPES,
    DEFAULT_SYSTEM_PROMPT,
    InfrastructureError,
    effective_system_prompt,
    run_pi,
)
from .shop_memory import build_memory_lines, structured_memory_enabled, summarize_shop_act_result

SCHEMA_VERSION = 1
# Legacy placeholder, kept for SHOP_CONTEXT_STRUCTURED_MEMORY=0 (must stay
# byte-identical to the TS constant in shop_extension.ts).
PRUNED_SHOP_ACT_RESULT = "[旧的 shop_act 工具结果已裁剪；done=false]"
DEFAULT_PROMPT = "完成给定的购物任务。先调用 shop_reset，然后只使用 shop_act 与环境交互。"

def read_index(path: Path) -> list[dict]:
    rows: list[dict] = []
    seen: set[int] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
        metadata = row.get("metadata") or {}
        task_id = int(metadata["task_id"])
        if metadata.get("split") != "sft":
            raise ValueError(f"{path}:{line_number}: collector only accepts split=sft")
        if task_id in seen:
            raise ValueError(f"{path}:{line_number}: duplicate task_id {task_id}")
        seen.add(task_id)
        rows.append(row)
    return rows


def _event_tool_name(event: dict) -> str | None:
    return event.get("toolName") or event.get("tool_name")


def _event_call_id(event: dict) -> str | None:
    value = event.get("toolCallId") or event.get("tool_call_id") or event.get("id")
    return str(value) if value is not None else None


def evaluate_candidate(candidate: dict, min_reward: float) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if candidate.get("context_trace_valid") is False:
        reasons.append("context_trace_mismatch")
    if candidate.get("split") != "sft":
        reasons.append("wrong_split")
    if candidate.get("error"):
        reasons.append("runtime_error")
    if not candidate.get("done"):
        reasons.append("not_done")
    if float(candidate.get("reward", 0.0)) < min_reward:
        reasons.append("reward_below_threshold")

    starts: list[tuple[str | None, str | None]] = []
    ends: list[tuple[str | None, str | None]] = []
    for event in candidate.get("events") or []:
        event_type = event.get("type")
        if event_type == "tool_execution_start":
            starts.append((_event_call_id(event), _event_tool_name(event)))
        elif event_type == "tool_execution_end":
            name = _event_tool_name(event)
            ends.append((_event_call_id(event), name))
            if event.get("isError"):
                reasons.append("tool_error")
            if name not in {"shop_reset", "shop_act"}:
                reasons.append("illegal_tool")

    names = [name for _, name in ends]
    if names.count("shop_reset") != 1:
        reasons.append("reset_count")
    if names and names[0] != "shop_reset":
        reasons.append("reset_not_first")
    if any(name != "shop_act" for name in names[1:]):
        reasons.append("non_shop_act_after_reset")
    if not ends:
        reasons.append("no_tool_events")

    # Pi versions differ in whether a call id is copied onto execution events.
    # Enforce exact pairing whenever ids are available on every start/end.
    if starts and ends and all(call_id for call_id, _ in starts + ends):
        if starts != ends:
            reasons.append("tool_pair_mismatch")
    return not reasons, sorted(set(reasons))


def authoritative_messages(events: list[dict]) -> list[dict]:
    messages = []
    for event in events:
        if event.get("type") != "message_end":
            continue
        message = event.get("message")
        if not isinstance(message, dict) or not message.get("role"):
            continue
        cleaned = json.loads(json.dumps(message, ensure_ascii=False))
        # shop_extension.ts never sends assistant `thinking` parts to the model,
        # so drop them here too. `context_snapshots` applies the same filter on
        # purpose: it is also invoked directly (tests, ad-hoc rebuilds) and must
        # keep the "identical to what the model actually saw" contract alone.
        if cleaned.get("role") == "assistant" and isinstance(cleaned.get("content"), list):
            cleaned["content"] = [
                part for part in cleaned["content"]
                if not isinstance(part, dict) or part.get("type") != "thinking"
            ]
        messages.append(cleaned)
    return messages


def context_snapshots(messages: list[dict], keep_act_results: int) -> list[dict]:
    """Rebuild per-assistant contexts exactly like the Pi extension does at
    rollout time (shop_extension.ts). Older shop_act results become compact
    memory lines (R5 structured memory) so SFT samples and RL rollouts see the
    same context — the train/deploy distribution alignment this project
    enforces. shop_memory.py mirrors the TS implementation byte-for-byte.
    """
    structured = structured_memory_enabled(os.environ.get("SHOP_CONTEXT_STRUCTURED_MEMORY"))
    snapshots = []
    for message_index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        context = json.loads(json.dumps(messages[:message_index], ensure_ascii=False))
        act_results = [
            index for index, item in enumerate(context)
            if item.get("role") == "toolResult" and item.get("toolName") == "shop_act"
        ]
        pruned = act_results[:max(0, len(act_results) - keep_act_results)]
        if structured:
            call_actions = _collect_call_actions(context)
            summaries = [
                summarize_shop_act_result(
                    _message_text(context[index]),
                    call_actions.get(str(context[index].get("toolCallId") or ""), ""),
                )
                for index in pruned
            ]
            replacements = build_memory_lines(summaries)
        else:
            replacements = [PRUNED_SHOP_ACT_RESULT] * len(pruned)
        for index, text in zip(pruned, replacements, strict=True):
            context[index]["content"] = [{"type": "text", "text": text}]
        # Mirror shop_extension.ts: assistant `thinking` parts never reach the
        # model, so the rebuilt context must drop them too. Today's teacher runs
        # with thinking disabled (thinking_blocks=0), but a reasoning teacher
        # would otherwise make SFT contexts diverge from RL rollouts.
        for item in context:
            if item.get("role") != "assistant" or not isinstance(item.get("content"), list):
                continue
            filtered = [
                part
                for part in item["content"]
                if not (isinstance(part, dict) and part.get("type") == "thinking")
            ]
            if len(filtered) != len(item["content"]):
                item["content"] = filtered
        snapshots.append({"assistant_message_index": message_index, "messages": context})
    return snapshots


def _collect_call_actions(messages: list[dict]) -> dict[str, str]:
    """Map toolCallId → native action string (mirrors collectCallActions in TS)."""
    actions: dict[str, str] = {}
    for message in messages:
        if message.get("role") != "assistant" or not isinstance(message.get("content"), list):
            continue
        for part in message["content"]:
            if not isinstance(part, dict) or part.get("type") != "toolCall":
                continue
            call_id, args = part.get("id"), part.get("arguments")
            if isinstance(call_id, str) and isinstance(args, dict) and isinstance(args.get("action"), str):
                actions[call_id] = args["action"]
    return actions


def _message_text(message: dict) -> str:
    content = message.get("content")
    if not isinstance(content, list):
        return ""
    return "".join(
        str(part.get("text") or "")
        for part in content
        if isinstance(part, dict) and part.get("type") == "text"
    )



def normalize_context_traces(traces: list[dict], system_prompt: str) -> list[dict]:
    snapshots = []
    for trace in traces:
        messages = json.loads(json.dumps(trace.get("messages") or [], ensure_ascii=False))
        if not messages or messages[0].get("role") != "system":
            messages.insert(0, {"role": "system", "content": system_prompt})
        snapshots.append({
            "assistant_message_index": len(messages),
            "request_index": int(trace.get("request_index", len(snapshots))),
            "compression_version": trace.get("compression_version"),
            "messages": messages,
        })
    return snapshots


def context_trace_matches(reconstructed: list[dict], actual: list[dict]) -> bool:
    if len(actual) not in {len(reconstructed), len(reconstructed) + 1}:
        return False
    if not all(
        expected["messages"] == observed["messages"]
        for expected, observed in zip(reconstructed, actual[:len(reconstructed)], strict=True)
    ):
        return False
    if len(actual) == len(reconstructed) + 1:
        messages = actual[-1]["messages"]
        return bool(messages and messages[-1].get("role") == "toolResult")
    return True

def atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def atomic_write_jsonl(path: Path, rows: list[dict]) -> None:
    text = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


class LaunchRateLimiter:
    """Limit rollout launches; pi may make several provider requests per rollout."""

    def __init__(self, launches_per_minute: int) -> None:
        self.limit = launches_per_minute
        self.timestamps: deque[float] = deque()
        self.lock = asyncio.Lock()

    async def acquire(self) -> None:
        if self.limit <= 0:
            return
        while True:
            async with self.lock:
                now = time.monotonic()
                while self.timestamps and now - self.timestamps[0] >= 60:
                    self.timestamps.popleft()
                if len(self.timestamps) < self.limit:
                    self.timestamps.append(now)
                    return
                delay = 60 - (now - self.timestamps[0])
            await asyncio.sleep(max(0.05, delay))


def _retryable(message: str) -> bool:
    lowered = message.lower()
    return any(marker in lowered for marker in (
        "429", "rate limit", "too many requests", "connection", "timed out",
        "timeout", "502", "503", "504", "temporarily unavailable",
        # 轮次耗尽是采样运气问题而非确定性失败：同配置重采约半数能完成
        # （2026-09-22 补齐 512 任务时实测 7/14）。纳入重试后由 --max-attempts
        # 控制次数，否则未完成的任务会以 attempts=1 静默落选。
        "max model turns",
    ))


def lineage_conflicts(
    candidate: dict,
    *,
    args: argparse.Namespace,
    keep_act_results: int,
    structured_memory: bool,
) -> list[str]:
    """Report teacher/harness fields that differ from the current run.

    Resumable collection reuses an existing trajectory verbatim, so a changed
    teacher model or context format would silently mix two populations into one
    dataset.
    """
    expected = {
        "teacher": {
            "provider": args.teacher_provider,
            "model": args.model,
            "base_url": args.base_url,
            "context_window": args.context_window,
            "max_tokens": args.max_tokens,
        },
        "harness": {
            "context_keep_act_results": keep_act_results,
            "max_turns": args.max_turns,
            "context_structured_memory": structured_memory,
        },
    }
    # Trajectories collected before the structured-memory switch carry no such
    # field; they used the legacy placeholder context, i.e. False.
    observed = {
        "teacher": candidate.get("teacher") or {},
        "harness": {"context_structured_memory": False, **(candidate.get("harness") or {})},
    }
    return [
        f"{group}.{field}: {observed[group].get(field)!r} != {value!r}"
        for group, fields in expected.items()
        for field, value in fields.items()
        if observed[group].get(field) != value
    ]


async def collect_one(
    *,
    row: dict,
    sample_id: int,
    args: argparse.Namespace,
    api_key: str,
    limiter: LaunchRateLimiter,
) -> dict:
    task_id = int(row["metadata"]["task_id"])
    trajectory_id = f"sft-{task_id:06d}-{sample_id:03d}"
    destination = args.output_dir / "raw" / f"{task_id:06d}" / f"{sample_id:03d}.json"
    keep_act_results = (
        args.keep_act_results
        if args.keep_act_results is not None
        else int(os.environ.get("SHOP_CONTEXT_KEEP_ACT_RESULTS", "3"))
    )
    structured_memory = structured_memory_enabled(os.environ.get("SHOP_CONTEXT_STRUCTURED_MEMORY"))
    if destination.exists():
        candidate = json.loads(destination.read_text(encoding="utf-8"))
        conflicts = lineage_conflicts(
            candidate, args=args, keep_act_results=keep_act_results, structured_memory=structured_memory
        )
        if conflicts and not args.allow_mixed_teacher:
            raise ValueError(
                f"{destination} was collected with a different teacher/harness configuration: "
                + "; ".join(conflicts)
                + ". Re-run with a fresh --output-dir, or pass --allow-mixed-teacher to reuse it anyway."
            )
        return candidate
    # Resolve once per candidate so messages, traces and the recorded harness
    # metadata all reflect the same prompt (SHOP_SYSTEM_PROMPT[_FILE] aware).
    system_prompt = effective_system_prompt()

    attempts: list[dict] = []
    final: dict[str, Any] | None = None
    for attempt in range(1, args.max_attempts + 1):
        await limiter.acquire()
        started = time.monotonic()
        rollout_session_id = f"deepseek-sft-{trajectory_id}-a{attempt}-{random.getrandbits(32):08x}"
        try:
            result = await run_pi(
                session_id=rollout_session_id,
                task_id=task_id,
                # Teacher mode: pi talks straight to the provider base-url, so
                # no Slime adapter exists to point at (None is explicit).
                adapter_url=None,
                env_url=args.env_url,
                pi_bin=os.environ.get("PI_BIN", "pi"),
                prompt=DEFAULT_PROMPT,
                timeout_sec=args.timeout,
                provider_id=args.teacher_provider,
                model_id=args.model,
                model_name=args.model,
                model_base_url=args.base_url,
                model_api_key=api_key,
                system_prompt=system_prompt,
                context_window=args.context_window,
                max_tokens=args.max_tokens,
                capture_events=True,
                capture_event_types=AUTHORITATIVE_EVENT_TYPES,
                max_model_turns=args.max_turns,
                context_keep_act_results=keep_act_results,
            )
            final = {
                "exit_code": result.exit_code,
                "done": result.done,
                "reward": result.reward,
                "env_idx": result.env_idx,
                "tool_calls": result.tool_calls,
                "error": result.error,
                "events": result.events,
                "context_traces": result.context_traces,
            }
            error_text = result.error or ""
        except InfrastructureError as exc:
            result = exc.result
            final = {
                "exit_code": result.exit_code if result else -1,
                "done": result.done if result else False,
                "reward": result.reward if result else 0.0,
                "env_idx": result.env_idx if result else None,
                "tool_calls": result.tool_calls if result else 0,
                "error": str(exc),
                "events": result.events if result else [],
                "context_traces": result.context_traces if result else [],
            }
            error_text = str(exc)
        attempts.append({
            "attempt": attempt,
            "rollout_session_id": rollout_session_id,
            "elapsed_sec": round(time.monotonic() - started, 3),
            "error": final["error"],
            "done": final["done"],
            "reward": final["reward"],
        })
        if final["done"] or not _retryable(error_text) or attempt == args.max_attempts:
            break
        delay = min(args.retry_max_delay, args.retry_base_delay * (2 ** (attempt - 1)))
        await asyncio.sleep(delay * random.uniform(0.75, 1.25))

    assert final is not None
    candidate = {
        "schema_version": SCHEMA_VERSION,
        "trajectory_id": trajectory_id,
        "task_id": task_id,
        "sample_id": sample_id,
        "split": "sft",
        "teacher": {
            "provider": args.teacher_provider,
            "model": args.model,
            "base_url": args.base_url,
            "thinking": False,
            "context_window": args.context_window,
            "max_tokens": args.max_tokens,
        },
        "harness": {
            "system_prompt": system_prompt,
            "context_keep_act_results": keep_act_results,
            "context_structured_memory": structured_memory,
            "max_turns": args.max_turns,
            "timeout_sec": args.timeout,
        },
        "attempts": attempts,
        **final,
    }
    candidate["messages"] = [{"role": "system", "content": system_prompt}] + authoritative_messages(
        candidate["events"]
    )
    reconstructed = context_snapshots(
        candidate["messages"], keep_act_results
    )
    candidate["context_traces"] = normalize_context_traces(
        candidate.get("context_traces") or [], system_prompt
    )
    candidate["context_snapshots"] = reconstructed
    candidate["context_trace_valid"] = context_trace_matches(
        reconstructed, candidate["context_traces"]
    )
    candidate["paired_context_traces"] = candidate["context_traces"][:len(reconstructed)]
    candidate["unpaired_terminal_traces"] = len(candidate["context_traces"]) - len(reconstructed)
    accepted, reasons = evaluate_candidate(candidate, args.min_reward)
    candidate["accepted"] = accepted
    candidate["rejection_reasons"] = reasons
    await asyncio.to_thread(atomic_write_json, destination, candidate)
    return candidate


def export_results(output_dir: Path, candidates: list[dict], args: argparse.Namespace) -> dict:
    ordered = sorted(candidates, key=lambda row: (int(row["task_id"]), int(row["sample_id"])))
    accepted = [row for row in ordered if row.get("accepted")]
    atomic_write_jsonl(output_dir / "accepted.jsonl", [{
        "messages": row["messages"],
        "metadata": {
            "trajectory_id": row["trajectory_id"],
            "task_id": row["task_id"],
            "split": "sft",
            "reward": row["reward"],
            "raw_path": str(Path("raw") / f"{int(row['task_id']):06d}" / f"{int(row['sample_id']):03d}.json"),
        },
    } for row in accepted])

    task_ids = {int(row["task_id"]) for row in ordered}
    accepted_task_ids = {int(row["task_id"]) for row in accepted}

    # The refill decision keys off the tool_calls distribution of the tasks that
    # no sample covered: still clustered at --max-turns means the budget bound,
    # and retrying at the same budget fails deterministically the same way.
    by_task: dict[int, list[dict]] = {}
    for row in ordered:
        by_task.setdefault(int(row["task_id"]), []).append(row)
    uncovered = [
        {
            "task_id": task_id,
            "attempted_samples": len(rows),
            "samples": [
                {
                    "sample_id": int(row["sample_id"]),
                    "trajectory_id": row["trajectory_id"],
                    "tool_calls": int(row.get("tool_calls", 0)),
                    "reward": float(row.get("reward", 0.0) or 0.0),
                    "done": bool(row.get("done")),
                    "error": row.get("error"),
                    "rejection_reasons": row.get("rejection_reasons", []),
                }
                for row in sorted(rows, key=lambda row: int(row["sample_id"]))
            ],
        }
        for task_id, rows in sorted(by_task.items())
        if task_id not in accepted_task_ids
    ]
    atomic_write_jsonl(output_dir / "uncovered_tasks.jsonl", uncovered)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "teacher_model": args.model,
        "teacher_base_url": args.base_url,
        "teacher_provider": args.teacher_provider,
        # With --allow-mixed-teacher the reuse path can pull in older
        # trajectories; surfacing the full set keeps the mixture auditable.
        "teacher_models_seen": sorted({
            (row.get("teacher") or {}).get("model")
            for row in ordered
            if (row.get("teacher") or {}).get("model")
        }),
        "candidates": len(ordered),
        "accepted": len(accepted),
        "tasks_attempted": len(task_ids),
        "tasks_covered": len(accepted_task_ids),
        "task_coverage": len(accepted_task_ids) / len(task_ids) if task_ids else 0.0,
        "reward_distribution": dict(Counter(str(row.get("reward", 0.0)) for row in ordered)),
        "rejection_reasons": dict(Counter(
            reason for row in ordered for reason in row.get("rejection_reasons", [])
        )),
        "uncovered_tasks": len(uncovered),
        "uncovered_tool_calls": dict(sorted(Counter(
            sample["tool_calls"] for row in uncovered for sample in row["samples"]
        ).items())),
        "max_turns": args.max_turns,
        "elapsed_sec_total": round(sum(
            float(attempt.get("elapsed_sec", 0.0))
            for row in ordered for attempt in row.get("attempts", [])
        ), 3),
    }
    atomic_write_json(output_dir / "summary.json", summary)
    return summary


async def async_main(args: argparse.Namespace) -> dict:
    rows = read_index(args.tasks)
    if args.task_ids_file:
        requested = {int(line.strip()) for line in args.task_ids_file.read_text(encoding="utf-8").splitlines() if line.strip()}
        rows = [row for row in rows if int(row["metadata"]["task_id"]) in requested]
        found = {int(row["metadata"]["task_id"]) for row in rows}
        if found != requested:
            raise SystemExit(f"task id file contains ids outside {args.tasks}: {sorted(requested - found)}")
    if args.task_offset < 0:
        raise SystemExit("--task-offset must be non-negative")
    rows = rows[args.task_offset :]
    if args.task_limit > 0:
        rows = rows[: args.task_limit]
    if args.concurrency < 1 or args.concurrency > 20:
        raise SystemExit("--concurrency must be between 1 and the default 20-slot environment capacity")
    if args.dry_run:
        return {
            "tasks": str(args.tasks),
            "selected_tasks": len(rows),
            "samples_per_task": args.samples_per_task,
            "planned_trajectories": len(rows) * args.samples_per_task,
        }

    if args.api_key_file is None:
        raise SystemExit(
            "--api-key-file is required (no default; it previously pointed at "
            "/root/api.txt, which silently picked up stale keys on root boxes)"
        )
    api_key = args.api_key_file.read_text(encoding="utf-8").splitlines()[0].strip()
    if not api_key:
        raise SystemExit("API key file first line is empty")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    limiter = LaunchRateLimiter(args.launches_per_minute)
    semaphore = asyncio.Semaphore(args.concurrency)

    async def guarded(row: dict, sample_id: int) -> dict:
        async with semaphore:
            return await collect_one(
                row=row, sample_id=sample_id, args=args,
                api_key=api_key, limiter=limiter,
            )

    jobs = [
        guarded(row, sample_id)
        for row in rows
        for sample_id in range(args.sample_id_start, args.sample_id_start + args.samples_per_task)
    ]
    await asyncio.gather(*jobs)
    candidates = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((args.output_dir / "raw").rglob("*.json"))
    ]
    return export_results(args.output_dir, candidates, args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tasks",
        type=Path,
        default=Path(__file__).resolve().parent / "data/tasks_v2/sft_512.jsonl",
        help="Task JSONL to collect; defaults to the bundled 512-task SFT slice.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--api-key-file",
        type=Path,
        default=None,
        help="File whose first line is the teacher API key. Required unless --dry-run.",
    )
    parser.add_argument(
        "--model",
        default="deepseek-flash",
        help="Teacher model id as accepted by the provider API. The DeepSeek "
             "endpoint supports 'deepseek-flash' (default, fast) and "
             "'deepseek-v4-pro' (stronger, pricier). Any differing id counts as "
             "a different teacher for the lineage check, so keep it stable "
             "across a single dataset.",
    )
    parser.add_argument(
        "--teacher-provider",
        default="deepseek",
        help="Pi provider id for the teacher endpoint. Any OpenAI-compatible "
             "base-url works, e.g. a local SGLang server for C5 "
             "self-improvement sampling.",
    )
    parser.add_argument(
        "--allow-mixed-teacher",
        action="store_true",
        help="Reuse existing trajectories whose teacher model or harness settings "
             "differ from this run. Default is to abort, so one dataset can never "
             "silently mix teacher models or context formats.",
    )
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--env-url", default="http://127.0.0.1:5000/api/shop_agent")
    parser.add_argument("--task-ids-file", type=Path)
    parser.add_argument("--sample-id-start", type=int, default=0)
    parser.add_argument("--task-offset", type=int, default=0)
    parser.add_argument("--task-limit", type=int, default=0, help="0 means all rows in --tasks")
    parser.add_argument("--samples-per-task", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument(
        "--launches-per-minute",
        type=int,
        default=30,
        help="Rate limit for rollout launches (pi may make several provider "
             "requests per rollout). Default 30 avoids hammering the teacher "
             "API; pass 0 to disable.",
    )
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--retry-base-delay", type=float, default=2.0)
    parser.add_argument("--retry-max-delay", type=float, default=60.0)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument(
        "--max-turns",
        type=int,
        default=40,
        help="Teacher model-turn budget, counted from turn_start events so it matches "
             "the RL-side max_model_turns semantics (one unit per model request, not per "
             "tool call). Falls back to counting tool calls if turn_start events are "
             "unavailable. The earlier value of 24 was the sole cause of every uncovered "
             "task in the v1 collection.",
    )
    parser.add_argument(
        "--keep-act-results",
        type=int,
        default=None,
        help="Number of trailing shop_act tool results kept in the model context "
             "(context pruning). Must match the RL config's "
             "context_keep_shop_act_results or SFT/RL context distributions diverge. "
             "Defaults to the SHOP_CONTEXT_KEEP_ACT_RESULTS environment variable or 3.",
    )
    parser.add_argument("--min-reward", type=float, default=1e-12)
    parser.add_argument("--context-window", type=int, default=1_000_000)
    parser.add_argument("--max-tokens", type=int, default=32_768)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = asyncio.run(async_main(args))
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
