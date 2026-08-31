"""F6: the TS/Python error-prefix contract must not drift silently.

``shop_extension.ts`` tags tool errors with ``[shop_infrastructure]`` /
``[shop_agent]``; ``pi_harness.py::_classify_tool_error`` maps those prefixes
to retry-vs-reject semantics. If one side is edited without the other, agent
errors get misclassified (spurious retries or wrong rejections) with no test
failing -- this module pins the parity by parsing the TS source text.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from examples.ShopSimulator.pi_harness import (
    AGENT_ERROR_PREFIX,
    INFRASTRUCTURE_ERROR_PREFIX,
    _classify_tool_error,
)

EXAMPLE_DIR = Path(__file__).resolve().parents[2] / "examples/ShopSimulator"
TS_SOURCE = (EXAMPLE_DIR / "shop_extension.ts").read_text(encoding="utf-8")


def ts_constant(name: str) -> str:
    match = re.search(r'const\s+' + re.escape(name) + r'\s*=\s*"((?:[^"\\]|\\.)*)"', TS_SOURCE)
    assert match, f"shop_extension.ts no longer declares {name}"
    return match.group(1).encode().decode("unicode_escape")


class TestPrefixParity:
    def test_infrastructure_prefix_matches(self):
        assert ts_constant("INFRASTRUCTURE_ERROR_PREFIX") == INFRASTRUCTURE_ERROR_PREFIX

    def test_agent_prefix_matches(self):
        assert ts_constant("AGENT_ERROR_PREFIX") == AGENT_ERROR_PREFIX

    def test_prefixes_are_distinct_and_bracketed(self):
        # A shared shape keeps _classify_tool_error's startswith() ordering safe.
        for prefix in (INFRASTRUCTURE_ERROR_PREFIX, AGENT_ERROR_PREFIX):
            assert prefix.startswith("[") and prefix.endswith("]")
        assert INFRASTRUCTURE_ERROR_PREFIX != AGENT_ERROR_PREFIX

    def test_classification_round_trip(self):
        # The classification the Python side performs on TS-tagged messages.
        kind, message = _classify_tool_error(f"{INFRASTRUCTURE_ERROR_PREFIX} HTTP 500")
        assert kind == "infrastructure_error" and message == "HTTP 500"
        kind, message = _classify_tool_error(f"{AGENT_ERROR_PREFIX} call reset first")
        assert kind == "agent_tool_error" and message == "call reset first"

    def test_python_comment_references_the_guard_test(self):
        # Keep the cross-reference discoverable from the Python side too.
        py_source = (EXAMPLE_DIR / "pi_harness.py").read_text(encoding="utf-8")
        assert "test_f6_prefix_parity" in py_source
