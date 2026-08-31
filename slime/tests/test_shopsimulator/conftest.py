"""Ensure the slime repo root is importable for the ShopSimulator example tests."""

from __future__ import annotations

import sys
from pathlib import Path

SLIME_ROOT = Path(__file__).resolve().parents[2]
if str(SLIME_ROOT) not in sys.path:
    sys.path.insert(0, str(SLIME_ROOT))
