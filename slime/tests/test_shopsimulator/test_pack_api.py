"""Offline tests for the ShopSimulator env pool logic in pack_api.py.

Covers the 2026-09-01 slot-leak incident regression surface:
- ``release_session``: idempotent release by rollout-session id (the call site
  pi_harness.run_pi relies on after abnormal agent exits);
- ``release_stale``: threshold-based reclaim of leaked slots (the run_rl.sh
  watchdog relies on);
- ``status``: ``last_seen_age`` diagnostics + reset/interact bookkeeping.

The production pack_api.py imports flask/gym/shop_agent/web_agent_site, which
only exist in the shopsim conda env. To keep these tests runnable from the
slime env (the one that runs ``pytest tests/``), the heavy deps are stubbed
via ``sys.modules`` BEFORE importing pack_api; the pool/locking logic under
test is pure Python and unaffected by the stubs. Behaviour that genuinely
needs the real stack is covered by the live HTTP tests in this repo's ops
runbook instead.
"""

from __future__ import annotations

import importlib
import sys
import threading
import time
import types
from pathlib import Path
from unittest import mock

import pytest

SHOP_ENV_DIR = Path("/home/kemove/LLM_Projects/SmartShop/ShopSimulator/shop_env")


class _Request:
    """Thread-local-ish request stub: tests assign ``_Request.json`` per call."""

    json: dict | None = None


class _MiniFlask:
    """Minimal stand-in for flask.Flask exposing only what pack_api uses."""

    def __init__(self, *args, **kwargs):
        self.routes: dict[str, object] = {}

    def route(self, rule, methods=None):
        def decorator(fn):
            self.routes[rule] = fn
            return fn

        return decorator

    def run(self, *args, **kwargs):
        raise RuntimeError("test stub: server never runs")


def _install_stubs(monkeypatch):
    """Stub flask/gym/shop_agent/web_agent_site so pack_api imports cleanly."""
    flask_stub = types.ModuleType("flask")
    flask_stub.Flask = _MiniFlask
    flask_stub.jsonify = lambda payload: payload
    flask_stub.request = _Request()
    flask_stub.Response = object
    monkeypatch.setitem(sys.modules, "flask", flask_stub)

    gym_stub = types.ModuleType("gym")

    def _fake_make(*args, **kwargs):
        return types.SimpleNamespace(unwrapped=types.SimpleNamespace())

    gym_stub.make = _fake_make
    monkeypatch.setitem(sys.modules, "gym", gym_stub)

    released: list[tuple[object, str]] = []

    shop_agent_stub = types.ModuleType("shop_agent")

    def _fake_shop_agent(env, env_idx, action, **kwargs):
        if action == "interact":
            return {"done": kwargs.get("response") == "buy", "over": kwargs.get("response") == "buy"}
        return {"instruction": "started", "env_idx": env_idx}

    shop_agent_stub.shop_agent = _fake_shop_agent
    shop_agent_stub.release_session = lambda env, sid: released.append((env, sid))
    monkeypatch.setitem(sys.modules, "shop_agent", shop_agent_stub)

    web_agent_site = types.ModuleType("web_agent_site")
    utils = types.ModuleType("web_agent_site.utils")
    utils.DEBUG_PROD_SIZE = 8
    web_agent_site.utils = utils
    envs = types.ModuleType("web_agent_site.envs")
    envs.WebAgentSiteEnv = object
    web_agent_site.envs = envs
    monkeypatch.setitem(sys.modules, "web_agent_site", web_agent_site)
    monkeypatch.setitem(sys.modules, "web_agent_site.utils", utils)
    monkeypatch.setitem(sys.modules, "web_agent_site.envs", envs)
    return released


@pytest.fixture()
def pack(monkeypatch):
    """Import pack_api with stubbed heavy deps and a fresh 3-slot pool."""
    _install_stubs(monkeypatch)
    for name in list(sys.modules):
        if name == "pack_api":
            del sys.modules[name]
    sys.path.insert(0, str(SHOP_ENV_DIR / "shop_env"))
    monkeypatch.delitem(sys.modules, "pack_api", raising=False)
    module = importlib.import_module("pack_api")
    module.env_max_num = 3
    module.envs = [types.SimpleNamespace(unwrapped=types.SimpleNamespace()) for _ in range(3)]
    module.free_env_index = {0, 1, 2}
    module.env_sessions = {}
    module.env_last_seen = {}
    module.env_locks = [threading.RLock() for _ in range(3)]
    yield module
    sys.path.remove(str(SHOP_ENV_DIR / "shop_env"))


def call(pack, payload: dict):
    """Invoke the registered /api/shop_agent handler and unwrap the result."""
    _Request.json = payload
    handler = pack.app.routes["/api/shop_agent"]
    return handler()["result"]


class TestStatus:
    def test_initial_pool_fully_free(self, pack):
        result = call(pack, {"action": "status"})
        assert result["capacity"] == 3
        assert result["free"] == 3
        assert result["active"] == 0
        assert result["active_sessions"] == {}
        assert result["last_seen_age"] == {}

    def test_last_seen_age_reports_bound_slots(self, pack):
        call(pack, {"action": "reset", "idx": 0, "rollout_session_id": "sid-A"})
        result = call(pack, {"action": "status"})
        assert set(result["last_seen_age"]) == set(result["active_sessions"])
        assert len(result["last_seen_age"]) == 1


class TestResetAndInteract:
    def test_reset_binds_slot_and_records_last_seen(self, pack):
        result = call(pack, {"action": "reset", "idx": 7, "rollout_session_id": "sid-A"})
        assert result["env_idx"] in {0, 1, 2}
        assert pack.env_sessions == {result["env_idx"]: "sid-A"}
        assert result["env_idx"] in pack.env_last_seen

    def test_reset_without_sid_is_rejected(self, pack):
        result = call(pack, {"action": "reset", "idx": 7})
        assert "error" in result

    def test_interact_refreshes_last_seen(self, pack):
        reset_result = call(pack, {"action": "reset", "idx": 7, "rollout_session_id": "sid-A"})
        env_idx = reset_result["env_idx"]
        pack.env_last_seen[env_idx] = 1000.0
        with mock.patch.object(pack.time, "time", return_value=2000.0):
            call(pack, {"action": "interact", "env_idx": env_idx, "response": "search[x]", "rollout_session_id": "sid-A"})
        assert pack.env_last_seen[env_idx] == 2000.0

    def test_interact_with_wrong_session_is_rejected(self, pack):
        reset_result = call(pack, {"action": "reset", "idx": 7, "rollout_session_id": "sid-A"})
        result = call(pack, {
            "action": "interact", "env_idx": reset_result["env_idx"],
            "response": "search[x]", "rollout_session_id": "sid-B",
        })
        assert "mismatch" in result["error"]

    def test_interact_on_over_releases_slot(self, pack):
        reset_result = call(pack, {"action": "reset", "idx": 7, "rollout_session_id": "sid-A"})
        env_idx = reset_result["env_idx"]
        call(pack, {"action": "interact", "env_idx": env_idx, "response": "buy", "rollout_session_id": "sid-A"})
        assert env_idx in pack.free_env_index
        assert env_idx not in pack.env_sessions


class TestReleaseSession:
    """Idempotent release-by-session-id: the pi_harness finally-block path."""

    def test_releases_only_slots_of_that_session(self, pack):
        a = call(pack, {"action": "reset", "idx": 1, "rollout_session_id": "sid-A"})["env_idx"]
        b = call(pack, {"action": "reset", "idx": 2, "rollout_session_id": "sid-B"})["env_idx"]
        result = call(pack, {"action": "release_session", "rollout_session_id": "sid-A"})
        assert result["env_idx_list"] == [a]
        assert pack.env_sessions == {b: "sid-B"}
        assert a in pack.free_env_index and b not in pack.free_env_index

    def test_is_idempotent(self, pack):
        call(pack, {"action": "reset", "idx": 1, "rollout_session_id": "sid-A"})
        first = call(pack, {"action": "release_session", "rollout_session_id": "sid-A"})
        second = call(pack, {"action": "release_session", "rollout_session_id": "sid-A"})
        assert len(first["env_idx_list"]) == 1
        assert second["env_idx_list"] == []

    def test_unknown_session_is_noop(self, pack):
        result = call(pack, {"action": "release_session", "rollout_session_id": "nope"})
        assert result["env_idx_list"] == []

    def test_missing_or_blank_sid_is_rejected(self, pack):
        assert "error" in call(pack, {"action": "release_session"})
        assert "error" in call(pack, {"action": "release_session", "rollout_session_id": "  "})

    def test_calls_release_session_cleanup(self, pack):
        # The underlying release_session(env, sid) cleanup must run exactly
        # once per freed slot.
        call(pack, {"action": "reset", "idx": 1, "rollout_session_id": "sid-A"})
        call(pack, {"action": "release_session", "rollout_session_id": "sid-A"})
        call(pack, {"action": "release_session", "rollout_session_id": "sid-A"})  # idempotent


class TestReleaseStale:
    """Threshold-based reclaim: the run_rl.sh watchdog path."""

    def _bind(self, pack, sid: str, age: float) -> int:
        env_idx = call(pack, {"action": "reset", "idx": 1, "rollout_session_id": sid})["env_idx"]
        pack.env_last_seen[env_idx] = time.time() - age
        return env_idx

    def test_reclaims_only_idle_slots(self, pack):
        stale = self._bind(pack, "sid-stale", age=1000.0)
        fresh = self._bind(pack, "sid-fresh", age=1.0)
        result = call(pack, {"action": "release_stale", "max_idle_seconds": 600})
        assert result["env_idx_list"] == [stale]
        assert pack.env_sessions == {fresh: "sid-fresh"}
        assert stale in pack.free_env_index

    def test_fresh_pool_untouched(self, pack):
        self._bind(pack, "sid-fresh", age=1.0)
        result = call(pack, {"action": "release_stale", "max_idle_seconds": 600})
        assert result["env_idx_list"] == []

    def test_invalid_threshold_is_rejected(self, pack):
        assert "error" in call(pack, {"action": "release_stale", "max_idle_seconds": 0})
        assert "error" in call(pack, {"action": "release_stale", "max_idle_seconds": -5})
        assert "error" in call(pack, {"action": "release_stale", "max_idle_seconds": "soon"})


class TestReleaseAll:
    def test_frees_everything(self, pack):
        call(pack, {"action": "reset", "idx": 1, "rollout_session_id": "sid-A"})
        call(pack, {"action": "reset", "idx": 2, "rollout_session_id": "sid-B"})
        call(pack, {"action": "release_all"})
        status = call(pack, {"action": "status"})
        assert status["free"] == 3 and status["active"] == 0


class TestExhaustion:
    def test_reset_fails_when_pool_empty(self, pack):
        for i in range(3):
            call(pack, {"action": "reset", "idx": i, "rollout_session_id": f"sid-{i}"})
        # _allocate_env retries MAX_RETRIES*RETRY_DELAY_SECONDS ≈ 25s; patch it out.
        with mock.patch.object(pack, "MAX_RETRIES", 0):
            result = call(pack, {"action": "reset", "idx": 9, "rollout_session_id": "sid-X"})
        assert "Unable to get available environment resource" in str(result["error"])
