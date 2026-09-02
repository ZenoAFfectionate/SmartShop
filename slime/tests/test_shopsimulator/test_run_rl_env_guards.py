"""Offline tests for the run_rl.sh env preflight & watchdog guards.

Validates the shell functions added after the 2026-09-01 slot-leak incident
(``ensure_env_ready`` / ``check_gpu_free`` / ``start_env_watchdog`` /
``stop_env_watchdog`` / ``env_call`` / ``env_field`` / ``start_env_server``)
against a local stub HTTP env server and a stubbed nvidia-smi — no real
ShopSimulator env or free GPU is needed, so these run on any dev box.

The shell functions are extracted verbatim from run_rl.sh (single source of
truth) via sed and evaluated in a bash subprocess driven by pytest.
"""

from __future__ import annotations

import json
import socket
import subprocess
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

SLIME_ROOT = Path(__file__).resolve().parents[2]
RUN_RL = SLIME_ROOT / "examples/ShopSimulator/run_rl.sh"

FUNC_BLOCK_START = "# ── ShopSimulator env 预检与守护"
FUNC_BLOCK_END = "# ── 训练单个算法"


class StubEnvState:
    """Mutable state shared by the stub env server (records release calls)."""

    def __init__(self, capacity: int = 20) -> None:
        self.capacity = capacity
        self.free = capacity
        self.active = 0
        self.requests: list[dict] = []


def make_stub_env_server(state: StubEnvState) -> tuple[ThreadingHTTPServer, int]:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 — http.server API
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length) or b"{}")
            state.requests.append(payload)
            action = payload.get("action")
            if action == "status":
                result = {
                    "capacity": state.capacity,
                    "free": state.free,
                    "active": state.active,
                    "active_sessions": {},
                }
            elif action == "release_all":
                state.free = state.capacity
                state.active = 0
                result = {"message": "released all"}
            elif action == "release_stale":
                result = {"message": "released_stale", "env_idx_list": []}
            elif action == "release_session":
                result = {"message": "released", "env_idx_list": []}
            else:
                result = {"error": f"unknown action {action!r}"}
            body = json.dumps({"result": result}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args) -> None:  # silence
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, server.server_address[1]


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def extract_functions() -> str:
    """Pull the guard-function block verbatim out of run_rl.sh."""
    text = RUN_RL.read_text(encoding="utf-8")
    start = text.index(FUNC_BLOCK_START)
    end = text.index(FUNC_BLOCK_END, start)
    return text[start:end]


def write_nvidia_smi_stub(path: Path, used_mib: str) -> None:
    path.write_text(f"#!/bin/sh\necho '{used_mib}'\n", encoding="utf-8")
    path.chmod(0o755)


def run_bash(script: str, timeout: float = 60.0) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", "-c", script],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


PREFLIGHT_DEFS = extract_functions()

# The guard functions call `say` (defined elsewhere in run_rl.sh) and expect
# SLIME_PYTHON / env_field to exist; inject minimal stand-ins so the block is
# self-contained for offline testing.
SAY_STUB = 'say() { echo "[say] $*"; }\n'


class TestEnvCallAndEnvField:
    def test_env_call_parses_status(self):
        state = StubEnvState()
        server, port = make_stub_env_server(state)
        try:
            script = f"""
            SHOP_ENV_URL=http://127.0.0.1:{port}
            SLIME_PYTHON=/usr/bin/python3
            {SAY_STUB}
            {PREFLIGHT_DEFS}
            out=$(env_call '{{"action":"status"}}')
            echo "$out"
            """
            result = run_bash(script)
            assert result.returncode == 0, result.stderr
            assert '"capacity"' in result.stdout
        finally:
            server.shutdown()

    def test_env_field_extracts_capacity_and_free(self):
        state = StubEnvState(capacity=20)
        server, port = make_stub_env_server(state)
        try:
            script = f"""
            SHOP_ENV_URL=http://127.0.0.1:{port}
            SLIME_PYTHON=/usr/bin/python3
            {SAY_STUB}
            {PREFLIGHT_DEFS}
            out=$(env_call '{{"action":"status"}}')
            echo "cap=$(env_field "$out" capacity)"
            echo "free=$(env_field "$out" free)"
            """
            result = run_bash(script)
            assert "cap=20" in result.stdout, result.stdout + result.stderr
            assert "free=20" in result.stdout, result.stdout + result.stderr
        finally:
            server.shutdown()


class TestEnsureEnvReady:
    def _script(self, port: int, stub_python: str = "/usr/bin/python3") -> str:
        return f"""
        SHOP_ENV_URL=http://127.0.0.1:{port}
        SLIME_PYTHON={stub_python}
        SHOPSIM_PYTHON=/nonexistent/python
        SHOP_ENV_DIR=/nonexistent/dir
        SHOP_ENV_LOG=/dev/null
        {SAY_STUB}
        {PREFLIGHT_DEFS}
        if ensure_env_ready; then echo "READY"; else echo "REFUSED"; fi
        """

    def test_passes_when_pool_healthy(self):
        state = StubEnvState()
        server, port = make_stub_env_server(state)
        try:
            result = run_bash(self._script(port))
            combined = result.stdout + result.stderr
            assert "READY" in result.stdout, combined
            assert "env 预检通过: 20/20" in combined
        finally:
            server.shutdown()

    def test_auto_cleans_leftover_sessions(self):
        state = StubEnvState()
        state.free = 5
        state.active = 15  # leftover sessions from a crashed run
        server, port = make_stub_env_server(state)
        try:
            result = run_bash(self._script(port))
            combined = result.stdout + result.stderr
            assert "READY" in result.stdout, combined
            assert "残留会话" in combined
            # release_all was actually invoked against the stub server
            assert any(r.get("action") == "release_all" for r in state.requests)
        finally:
            server.shutdown()

    def test_refuses_when_env_is_down(self):
        port = free_port()  # nothing listens there
        result = run_bash(self._script(port))
        combined = result.stdout + result.stderr
        assert "REFUSED" in result.stdout, combined
        assert "无法自动启动" in combined


class TestCheckGpuFree:
    def _script(self, bin_dir: Path) -> str:
        return f"""
        PATH="{bin_dir}:/usr/bin:/bin"
        GPU_INDEX=0
        {PREFLIGHT_DEFS}
        if check_gpu_free; then echo "GPUOK"; else echo "GPUBUSY"; fi
        """

    def test_accepts_idle_gpu(self, tmp_path):
        stub_dir = tmp_path / "bin"
        stub_dir.mkdir()
        write_nvidia_smi_stub(stub_dir / "nvidia-smi", "15")
        result = run_bash(self._script(stub_dir))
        assert "GPUOK" in result.stdout, result.stdout + result.stderr

    def test_refuses_busy_gpu(self, tmp_path):
        stub_dir = tmp_path / "bin"
        stub_dir.mkdir()
        write_nvidia_smi_stub(stub_dir / "nvidia-smi", "41618")
        result = run_bash(self._script(stub_dir))
        assert "GPUBUSY" in result.stdout, result.stdout + result.stderr

    def test_refuses_on_unreadable_gpu_state(self, tmp_path):
        # nvidia-smi producing garbage (driver failure / not installed) must be
        # treated as unreadable -> fail-fast, never silently train on a busy GPU.
        stub_dir = tmp_path / "bin"
        stub_dir.mkdir()
        (stub_dir / "nvidia-smi").write_text("#!/bin/sh\necho 'N/A'\nexit 1\n", encoding="utf-8")
        (stub_dir / "nvidia-smi").chmod(0o755)
        result = run_bash(self._script(stub_dir))
        assert "GPUBUSY" in result.stdout, result.stdout + result.stderr


class TestEnvWatchdog:
    def test_watchdog_calls_release_stale_periodically(self):
        state = StubEnvState()
        server, port = make_stub_env_server(state)
        try:
            script = f"""
            SHOP_ENV_URL=http://127.0.0.1:{port}
            SLIME_PYTHON=/usr/bin/python3
            ENV_WATCHDOG=1
            ENV_WATCHDOG_INTERVAL=1
            ROLLOUT_TIMEOUT_SEC=600
            {SAY_STUB}
            {PREFLIGHT_DEFS}
            start_env_watchdog
            SAVED_PID=$ENV_WATCHDOG_PID
            echo "pid=$SAVED_PID"
            sleep 2.5
            stop_env_watchdog
            sleep 0.3
            if kill -0 "$SAVED_PID" 2>/dev/null; then echo "STILL_ALIVE"; else echo "STOPPED"; fi
            """
            result = run_bash(script, timeout=30)
            combined = result.stdout + result.stderr
            assert result.returncode == 0, combined
            assert "STOPPED" in result.stdout, combined
            stale_calls = [r for r in state.requests if r.get("action") == "release_stale"]
            assert len(stale_calls) >= 2, f"watchdog fired {len(stale_calls)} times"
            # The threshold must be rollout-timeout + 300s grace, not a hardcode.
            assert all(r.get("max_idle_seconds") == 900 for r in stale_calls)
        finally:
            server.shutdown()

    def test_disabled_watchdog_is_a_noop(self):
        state = StubEnvState()
        server, port = make_stub_env_server(state)
        try:
            script = f"""
            SHOP_ENV_URL=http://127.0.0.1:{port}
            SLIME_PYTHON=/usr/bin/python3
            ENV_WATCHDOG=0
            ENV_WATCHDOG_INTERVAL=1
            ROLLOUT_TIMEOUT_SEC=600
            {SAY_STUB}
            {PREFLIGHT_DEFS}
            start_env_watchdog
            echo "pid=${{ENV_WATCHDOG_PID:-unset}}"
            """
            result = run_bash(script, timeout=15)
            combined = result.stdout + result.stderr
            assert "pid=unset" in result.stdout, combined
            assert not state.requests
        finally:
            server.shutdown()


class TestStartEnvServer:
    def test_refuses_invalid_config(self):
        script = f"""
        SHOPSIM_PYTHON=/nonexistent/python
        SHOP_ENV_DIR=/nonexistent/dir
        {SAY_STUB}
        {PREFLIGHT_DEFS}
        if start_env_server; then echo "STARTED"; else echo "REFUSED"; fi
        """
        result = run_bash(script)
        assert "REFUSED" in result.stdout, result.stdout + result.stderr


class TestRunRlWiring:
    """Static wiring checks: the guards must be invoked in train_one."""

    def test_preflight_is_called_before_ray_start(self):
        text = RUN_RL.read_text(encoding="utf-8")
        preflight_pos = text.index("ensure_env_ready || return 2")
        gpu_pos = text.index("check_gpu_free || return 2")
        ray_pos = text.index('"${RAY_BIN}" start --head')
        assert preflight_pos < ray_pos
        assert gpu_pos < ray_pos

    def test_watchdog_starts_with_ray_and_stops_in_cleanup(self):
        text = RUN_RL.read_text(encoding="utf-8")
        assert "start_env_watchdog" in text
        cleanup_start = text.index("cleanup() {")
        cleanup_end = text.index("}", cleanup_start)
        cleanup_body = text[cleanup_start:cleanup_end]
        assert "stop_env_watchdog" in cleanup_body

    def test_release_stale_threshold_uses_rollout_timeout(self):
        text = RUN_RL.read_text(encoding="utf-8")
        assert "ROLLOUT_TIMEOUT_SEC + 300" in text
