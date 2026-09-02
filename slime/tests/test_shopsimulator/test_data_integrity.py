"""Data-integrity guards for the bundled task pools (F4 as pytest guardrails).

These tests固化 the one-shot overlap verification performed on 2026-08-26:
the seven task pools must stay pairwise disjoint so future data additions
cannot silently leak training tasks into the evaluation sets. They also pin
the structural invariants of the experiment slices (row counts, split labels,
subset relations) and of the dev100 evaluation slice (C2).
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import pytest

TASKS_DIR = Path(__file__).resolve().parents[2] / "examples/ShopSimulator/data/tasks_v2"

POOL_FILES = [
    "sft.jsonl", "sft_512.jsonl", "rl.jsonl", "rl_500.jsonl",
    "dev.jsonl", "official_test.jsonl", "official_test_200.jsonl", "dev100.jsonl",
]


def load_pool(name: str) -> list[dict]:
    path = TASKS_DIR / name
    assert path.is_file(), f"missing task pool: {path}"
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def task_ids(name: str) -> set[int]:
    return {row["metadata"]["task_id"] for row in load_pool(name)}


class TestPoolDisjointness:
    @pytest.mark.parametrize("a,b", [
        ("sft_512.jsonl", "rl_500.jsonl"),
        ("sft_512.jsonl", "dev.jsonl"),
        ("sft_512.jsonl", "official_test_200.jsonl"),
        ("rl_500.jsonl", "dev.jsonl"),
        ("rl_500.jsonl", "official_test_200.jsonl"),
        ("dev.jsonl", "official_test_200.jsonl"),
        ("sft.jsonl", "official_test.jsonl"),
        ("rl.jsonl", "official_test.jsonl"),
        ("sft.jsonl", "dev.jsonl"),
        ("rl.jsonl", "dev.jsonl"),
        ("dev.jsonl", "official_test.jsonl"),
    ])
    def test_experiment_pools_are_pairwise_disjoint(self, a, b):
        overlap = task_ids(a) & task_ids(b)
        assert not overlap, f"{a} and {b} share task_ids: {sorted(overlap)[:10]}"

    def test_all_eight_pools_pairwise_disjoint(self):
        # Parent pools and their own slices naturally overlap (sft_512 ⊆ sft);
        # every other pair must be disjoint.
        parent_child = {
            ("sft.jsonl", "sft_512.jsonl"),
            ("rl.jsonl", "rl_500.jsonl"),
            ("official_test.jsonl", "official_test_200.jsonl"),
            ("dev.jsonl", "dev100.jsonl"),
        }
        pools = ["sft.jsonl", "sft_512.jsonl", "rl.jsonl", "rl_500.jsonl",
                 "dev.jsonl", "dev100.jsonl",
                 "official_test.jsonl", "official_test_200.jsonl"]
        for a, b in combinations(pools, 2):
            if (a, b) in parent_child or (b, a) in parent_child:
                continue
            overlap = task_ids(a) & task_ids(b)
            assert not overlap, f"{a} ∩ {b} = {sorted(overlap)[:10]}"


class TestPoolStructure:
    def test_subset_relations(self):
        assert task_ids("sft_512.jsonl") <= task_ids("sft.jsonl")
        assert task_ids("rl_500.jsonl") <= task_ids("rl.jsonl")
        assert task_ids("official_test_200.jsonl") <= task_ids("official_test.jsonl")
        assert task_ids("dev100.jsonl") <= task_ids("dev.jsonl")

    @pytest.mark.parametrize("name, expected_rows", [
        ("sft_512.jsonl", 512),
        ("rl_500.jsonl", 500),
        ("official_test_200.jsonl", 200),
        ("dev100.jsonl", 100),
    ])
    def test_slice_row_counts(self, name, expected_rows):
        rows = load_pool(name)
        assert len(rows) == expected_rows
        ids = [row["metadata"]["task_id"] for row in rows]
        assert len(set(ids)) == expected_rows  # no duplicates

    @pytest.mark.parametrize("name, expected_split", [
        ("sft_512.jsonl", "sft"),
        ("rl_500.jsonl", "rl"),
        ("dev.jsonl", "dev"),
        ("dev100.jsonl", "dev"),
        ("official_test_200.jsonl", "official_test"),
    ])
    def test_split_labels(self, name, expected_split):
        for row in load_pool(name):
            assert row["metadata"]["split"] == expected_split, (
                f"{name}: task {row['metadata']['task_id']} has split "
                f"{row['metadata']['split']!r}, expected {expected_split!r}"
            )

    def test_every_row_has_user_prompt(self):
        from examples.ShopSimulator.common import prompt_text
        for name in POOL_FILES:
            for row in load_pool(name):
                prompt_text(row["prompt"])  # raises if unusable


class TestDev100Slice:
    """C2: the dev100 slice must be a faithful, reproducible dev subsample."""

    def test_dev100_is_deterministic_resample(self):
        # Re-drawing with seed 42 must reproduce the committed file exactly.
        import random

        rows = load_pool("dev.jsonl")
        rng = random.Random(42)
        picked = sorted(rng.sample(rows, 100), key=lambda r: r["metadata"]["task_id"])
        committed = load_pool("dev100.jsonl")
        assert [r["metadata"]["task_id"] for r in picked] == [
            r["metadata"]["task_id"] for r in committed
        ]

    def test_dev100_category_coverage(self):
        categories = {row["metadata"]["category"] for row in load_pool("dev100.jsonl")}
        dev_categories = {row["metadata"]["category"] for row in load_pool("dev.jsonl")}
        assert len(categories) >= 50  # broad category coverage (56 at creation)
        assert categories <= dev_categories


class TestReadmeDocumentsMetrics:
    """The reward formulas must be documented, not folklore."""

    @staticmethod
    def readme() -> str:
        return (Path(__file__).resolve().parents[3] / "README.md").read_text(encoding="utf-8")

    def test_loose_formula_present(self):
        text = self.readme()
        assert "r_loose" in text
        assert "|U_att∩Y_att|" in text  # additive formula skeleton
        assert "arXiv:2601.18225" in text

    def test_hard_formula_present(self):
        text = self.readme()
        assert "r_type × r_att × r_option × r_price" in text

    def test_subscore_table_present(self):
        text = self.readme()
        for name in ("r_type", "r_att", "r_option", "r_price"):
            assert name in text

    def test_environment_requirements_banner_present(self):
        # Linux + NVIDIA GPU must be stated up front.
        text = self.readme()
        assert "环境要求" in text
        assert "NVIDIA GPU" in text
        assert "macOS" in text
