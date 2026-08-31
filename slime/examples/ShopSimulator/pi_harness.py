"""Launch pi as a JSONL subprocess and extract ShopSimulator outcomes."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import tempfile
from dataclasses import dataclass, field
from pathlib import Path


class InfrastructureError(RuntimeError):
    """The harness could not produce a trustworthy rollout."""

    def __init__(self, message: str, *, result: "PiRunResult | None" = None) -> None:
        super().__init__(message)
        self.result = result


@dataclass
class PiRunResult:
    exit_code: int
    done: bool = False
    over: bool = False
    reward: float = 0.0
    env_idx: int | None = None
    # ShopSimulator's four sub-scores (r_type/r_att/r_option/r_price) plus the
    # purchased vs goal asin, all populated only on a done terminal.
    reward_detail: dict[str, float] = field(default_factory=dict)
    purchase_asin: str | None = None
    goal_asin: str | None = None
    tool_calls: int = 0
    # Model turns (one ``turn_start`` event per request to the model). This is
    # the budget unit that matches the student-side adapter turn cap
    # (``max_turns_per_sid`` in slime/agent/adapters/common.py): both count
    # model requests, not individual tool executions.
    model_turns: int = 0
    error: str | None = None
    error_kind: str | None = None
    tool_errors: list[dict[str, str]] = field(default_factory=list)
    events: list[dict] = field(default_factory=list, repr=False)
    context_traces: list[dict] = field(default_factory=list, repr=False)


DEFAULT_EXTENSION = Path(__file__).with_name("shop_extension.ts")
# Error-classification prefixes. These MUST stay byte-identical to the
# TypeScript side in shop_extension.ts (INFRASTRUCTURE_ERROR_PREFIX /
# AGENT_ERROR_PREFIX): the extension tags tool errors with them and
# _classify_tool_error() maps the prefix to retry-vs-reject semantics here.
# Parity is enforced by tests/test_shopsimulator/test_f6_prefix_parity.py.
INFRASTRUCTURE_ERROR_PREFIX = "[shop_infrastructure]"
AGENT_ERROR_PREFIX = "[shop_agent]"
DEFAULT_SYSTEM_PROMPT = (
    "你是购物 Agent。先调用一次 shop_reset 获取任务，然后只使用 shop_act，"
    "以 ShopSimulator 原生动作 DSL 与环境交互，直到任务结束。"
    "一旦找到与任务要求最匹配的商品，立即选择所需规格并点击购买；不要反复搜索。"
    "任务以成功下单为结束标志；在完成购买前不要停止交互。"
)


def effective_system_prompt() -> str:
    """Resolve the agent system prompt (D5 non-RL baseline support).

    The default prompt is used unless overridden via environment:
    - ``SHOP_SYSTEM_PROMPT``: the literal prompt text, or
    - ``SHOP_SYSTEM_PROMPT_FILE``: path to a file whose contents are the prompt
      (trailing whitespace stripped).

    Overriding the prompt changes only the agent's instructions, not the
    environment or the reward; this is how a ReAct-style non-RL baseline can
    run against the same checkpoint with identical tooling.
    """
    text = os.environ.get("SHOP_SYSTEM_PROMPT")
    if text:
        return text
    path = os.environ.get("SHOP_SYSTEM_PROMPT_FILE")
    if path:
        return Path(path).read_text(encoding="utf-8").strip()
    return DEFAULT_SYSTEM_PROMPT

# Keep the rollout harness independent from pi's user/project defaults.  CLI
# discovery switches below are the primary guard; these settings also pin
# features whose defaults are enabled and which do not have a CLI disable flag.
MINIMAL_PI_SETTINGS = {
    "compaction": {"enabled": False},
    "packages": [],
    "extensions": [],
    "skills": [],
    "prompts": [],
    "themes": [],
    "enableSkillCommands": False,
    "enableInstallTelemetry": False,
    "quietStartup": True,
}


def _event_details(event: dict) -> tuple[str | None, dict]:
    tool_name = event.get("toolName") or event.get("tool_name")
    result = event.get("result") or {}
    details = result.get("details") if isinstance(result, dict) else {}
    return tool_name, details if isinstance(details, dict) else {}


def _event_error_text(event: dict, tool_name: str | None) -> str:
    event_result = event.get("result") or {}
    content = event_result.get("content") if isinstance(event_result, dict) else None
    if isinstance(content, list):
        texts = [
            str(block.get("text"))
            for block in content
            if isinstance(block, dict) and block.get("text")
        ]
        if texts:
            return "\n".join(texts)
    return f"{tool_name or 'tool'} failed"


def _classify_tool_error(message: str) -> tuple[str, str]:
    if message.startswith(INFRASTRUCTURE_ERROR_PREFIX):
        return "infrastructure_error", message.removeprefix(INFRASTRUCTURE_ERROR_PREFIX).strip()
    if message.startswith(AGENT_ERROR_PREFIX):
        message = message.removeprefix(AGENT_ERROR_PREFIX).strip()
    return "agent_tool_error", message


def _is_terminal(result: PiRunResult) -> bool:
    return result.done or result.over


def _effective_turns(result: PiRunResult) -> int:
    """Turn-budget unit used by the harness cap.

    Prefers model turns (``turn_start`` events), which match the student-side
    adapter's ``max_turns_per_sid`` semantics exactly. Falls back to the raw
    tool-call count when a pi build does not emit ``turn_start`` events, so the
    cap still fires instead of relying on the wall-clock timeout alone.
    """
    return result.model_turns if result.model_turns > 0 else result.tool_calls


def parse_pi_event(result: PiRunResult, event: dict) -> None:
    event_type = event.get("type")
    if event_type == "turn_start":
        # One turn_start per model request; this is the turn-budget unit that
        # matches the student-side adapter's max_turns_per_sid semantics.
        result.model_turns += 1
    elif event_type == "tool_execution_end":
        result.tool_calls += 1
        tool_name, details = _event_details(event)
        if isinstance(details.get("env_idx"), int):
            result.env_idx = details["env_idx"]
        if tool_name == "shop_act":
            event_done = bool(details.get("done", False))
            event_over = bool(details.get("over", False))
            result.done = result.done or event_done
            result.over = result.over or event_over
            if event_done:
                result.reward = float(details.get("reward", 0.0) or 0.0)
                detail = details.get("reward_detail")
                if isinstance(detail, dict):
                    # ShopSimulator reports some sub-scores as booleans --
                    # r_price is `price <= goal['price_upper']` -- so bools must
                    # be kept and coerced, not filtered out as non-numeric.
                    result.reward_detail = {
                        key: float(value)
                        for key, value in detail.items()
                        if isinstance(value, (int, float, bool))
                    }
                for attribute in ("purchase_asin", "goal_asin"):
                    value = details.get(attribute)
                    if isinstance(value, str) and value:
                        setattr(result, attribute, value)
        if event.get("isError"):
            kind, message = _classify_tool_error(_event_error_text(event, tool_name))
            result.tool_errors.append({
                "tool_name": tool_name or "tool",
                "kind": kind,
                "message": message,
            })
            if kind == "infrastructure_error" and not _is_terminal(result):
                result.error = message
                result.error_kind = kind
        if _is_terminal(result):
            # A trustworthy environment terminal result dominates errors from
            # sibling calls and process cancellation during intentional stop.
            result.error = None
            result.error_kind = None
    elif event_type == "message_end":
        message = event.get("message") or {}
        stop_reason = event.get("stopReason") or (message.get("stopReason") if isinstance(message, dict) else None)
        if stop_reason in {"error", "aborted"} and not _is_terminal(result):
            result.error = f"pi message ended with {stop_reason}"
            result.error_kind = "model_or_process_error"
    elif event_type == "agent_settled" and event.get("error") and not _is_terminal(result):
        result.error = str(event["error"])
        result.error_kind = "model_or_process_error"


async def _terminate_process_group(proc: asyncio.subprocess.Process) -> None:
    if proc.returncode is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        await asyncio.wait_for(proc.wait(), timeout=5)
    except asyncio.TimeoutError:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        await proc.wait()


async def run_pi(
    *,
    session_id: str,
    task_id: int,
    adapter_url: str | None,
    env_url: str,
    prompt: str,
    timeout_sec: float,
    pi_bin: str = "pi",
    extension: str | Path = DEFAULT_EXTENSION,
    system_prompt: str | None = None,
    provider_id: str = "slime-adapter",
    model_id: str = "qwen3.5-0.8b",
    model_name: str = "Qwen3.5-0.8B (Slime/SGLang)",
    model_base_url: str | None = None,
    model_api_key: str | None = None,
    context_window: int = 262144,
    max_tokens: int = 32768,
    capture_events: bool = False,
    capture_event_types: set[str] | None = None,
    max_model_turns: int | None = None,
    context_keep_act_results: int | None = None,
) -> PiRunResult:
    if system_prompt is None:
        system_prompt = effective_system_prompt()
    if model_base_url is None and adapter_url is None:
        raise ValueError(
            "run_pi needs either model_base_url (teacher mode, e.g. a DeepSeek-"
            "compatible endpoint) or adapter_url (student mode, the Slime "
            "OpenAI adapter); neither was provided"
        )
    env = os.environ.copy()
    env.update({
        "SHOP_ENV_URL": env_url,
        "SHOP_TASK_ID": str(task_id),
        "SHOP_ROLLOUT_SESSION_ID": session_id,
        "SHOP_CONTEXT_KEEP_ACT_RESULTS": str(
            context_keep_act_results
            if context_keep_act_results is not None
            else os.environ.get("SHOP_CONTEXT_KEEP_ACT_RESULTS", "3")
        ),
    })
    # Each worker may expose a different adapter host/port. Give pi an isolated
    # config so it cannot accidentally use the global ~/.pi baseUrl.
    agent_dir = tempfile.mkdtemp(prefix="shop-pi-agent-")
    context_trace_path = Path(agent_dir, "context-trace.jsonl")
    env["SHOP_CONTEXT_TRACE_PATH"] = str(context_trace_path)
    base_url = model_base_url or f"{adapter_url.rstrip('/')}/v1"
    api_key = model_api_key or "overridden-by-cli"
    models = {
        "providers": {
            provider_id: {
                "name": provider_id,
                "baseUrl": base_url,
                "apiKey": api_key,
                "api": "openai-completions",
                "compat": {
                    "supportsDeveloperRole": False,
                    "supportsReasoningEffort": False,
                    "supportsStore": False,
                    "maxTokensField": "max_tokens",
                },
                "models": [{
                    "id": model_id, "name": model_name,
                    "reasoning": False, "input": ["text"], "contextWindow": context_window,
                    "maxTokens": max_tokens,
                    "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                }],
            }
        }
    }
    Path(agent_dir, "models.json").write_text(json.dumps(models), encoding="utf-8")
    Path(agent_dir, "settings.json").write_text(
        json.dumps(MINIMAL_PI_SETTINGS), encoding="utf-8"
    )
    env["PI_CODING_AGENT_DIR"] = agent_dir
    env["PI_OFFLINE"] = "1"
    env["PI_SKIP_VERSION_CHECK"] = "1"
    env["PI_TELEMETRY"] = "0"
    local_hosts = {"127.0.0.1", "localhost"}
    for url in (adapter_url, env_url):
        if url is None:
            continue
        try:
            from urllib.parse import urlparse

            if host := urlparse(url).hostname:
                local_hosts.add(host)
        except ValueError:
            pass
    existing = env.get("NO_PROXY") or env.get("no_proxy") or ""
    no_proxy = ",".join(sorted(local_hosts | {v for v in existing.split(",") if v}))
    env["NO_PROXY"] = env["no_proxy"] = no_proxy

    command = [
        pi_bin, "--mode", "json", "-p", prompt, "--no-session", "--offline",
        "--no-approve", "--no-builtin-tools", "--no-extensions", "--extension", str(extension),
        "--no-skills", "--no-prompt-templates", "--no-themes", "--no-context-files",
        "--tools", "shop_reset,shop_act",
        "--system-prompt", system_prompt, "--model", f"{provider_id}/{model_id}",
    ]
    if model_api_key is None:
        command.extend(["--api-key", session_id])
    command.extend(["--thinking", "off"])
    try:
        proc = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            start_new_session=True,
            # ShopSimulator observations can make one pi JSONL event much
            # larger than asyncio's default 64 KiB StreamReader limit.
            limit=16 * 1024 * 1024,
        )
    except (OSError, ValueError) as exc:
        shutil.rmtree(agent_dir, ignore_errors=True)
        raise InfrastructureError(f"cannot start pi: {exc}") from exc

    parsed = PiRunResult(exit_code=-1)
    terminal_event = asyncio.Event()

    async def consume_stdout() -> None:
        assert proc.stdout is not None
        async for raw_line in proc.stdout:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                if (
                    capture_events
                    and (capture_event_types is None or event.get("type") in capture_event_types)
                ):
                    parsed.events.append(event)
                parse_pi_event(parsed, event)
                if _is_terminal(parsed):
                    terminal_event.set()
                elif max_model_turns is not None and _effective_turns(parsed) >= max_model_turns:
                    parsed.error = f"pi reached max model turns ({max_model_turns})"
                    parsed.error_kind = "model_or_process_error"
                    terminal_event.set()

    stdout_task = asyncio.create_task(consume_stdout())
    assert proc.stderr is not None
    stderr_task = asyncio.create_task(proc.stderr.read())
    proc_wait_task = asyncio.create_task(proc.wait())
    done_wait_task = asyncio.create_task(terminal_event.wait())
    stderr_bytes = b""
    try:
        finished, _ = await asyncio.wait(
            {proc_wait_task, done_wait_task},
            timeout=timeout_sec,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if not finished:
            raise asyncio.TimeoutError
        if done_wait_task in finished and proc.returncode is None:
            await _terminate_process_group(proc)
        else:
            await proc_wait_task
        await stdout_task
        stderr_bytes = await stderr_task
    except asyncio.TimeoutError as exc:
        stdout_task.cancel()
        await _terminate_process_group(proc)
        await asyncio.gather(stdout_task, return_exceptions=True)
        stderr_bytes = await stderr_task
        parsed.error = f"pi exceeded wall-clock timeout ({timeout_sec}s)"
        parsed.error_kind = "infrastructure_error"
        raise InfrastructureError(
            parsed.error, result=parsed
        ) from exc
    finally:
        if context_trace_path.exists():
            try:
                parsed.context_traces = [
                    json.loads(line)
                    for line in context_trace_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
            except (OSError, json.JSONDecodeError) as exc:
                if not _is_terminal(parsed):
                    parsed.error = parsed.error or f"invalid context trace: {exc}"
                    parsed.error_kind = parsed.error_kind or "infrastructure_error"
        for task in (proc_wait_task, done_wait_task):
            if not task.done():
                task.cancel()
        shutil.rmtree(agent_dir, ignore_errors=True)

    # Once ShopSimulator reports done, the harness intentionally stops pi's
    # process group instead of waiting for another model turn.  Node may expose
    # that expected SIGTERM as 143; it is not a rollout failure.
    parsed.exit_code = 0 if _is_terminal(parsed) else int(proc.returncode or 0)
    stderr = stderr_bytes.decode("utf-8", errors="replace").strip()
    if parsed.exit_code != 0 and not _is_terminal(parsed):
        parsed.error = parsed.error or f"pi exited {parsed.exit_code}: {stderr[-500:]}"
        parsed.error_kind = parsed.error_kind or "model_or_process_error"
    return parsed
