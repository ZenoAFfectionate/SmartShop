"""Config-guard tests: pin the semantics of the four eval yaml files (D3/C2).

These yamls encode deliberate experimental contrasts (sampling k=1 vs k=4,
temperature 1.0 vs greedy 0.0, official test vs dev). A silent edit to the
wrong key would invalidate published comparisons, so the contrasts themselves
are pinned here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

CONFIG_DIR = Path(__file__).resolve().parents[2] / "examples/ShopSimulator/config"


def load_eval_config(name: str) -> dict:
    path = CONFIG_DIR / name
    assert path.is_file(), f"missing config: {path}"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def defaults_of(name: str) -> dict:
    config = load_eval_config(name)
    assert set(config) == {"eval"}, f"{name}: top-level keys {set(config)}"
    return config["eval"]["defaults"]


def dataset_of(name: str) -> dict:
    datasets = load_eval_config(name)["eval"]["datasets"]
    assert len(datasets) == 1
    return datasets[0]


class TestOfficialK1Baseline:
    def test_sampling_contrast(self):
        defaults = defaults_of("shop_eval_official_k1.yaml")
        assert defaults["n_samples_per_eval_prompt"] == 1
        assert defaults["temperature"] == 1.0
        assert defaults["top_p"] == 1.0

    def test_dataset_path(self):
        dataset = dataset_of("shop_eval_official_k1.yaml")
        assert dataset["name"] == "shop_official_test_200"
        assert dataset["path"].endswith("official_test_200.jsonl")
        assert dataset["custom_generate_function_path"] == "examples.ShopSimulator.generate.generate"


class TestOfficialK4:
    def test_only_k_differs_from_k1(self):
        k1 = defaults_of("shop_eval_official_k1.yaml")
        k4 = defaults_of("shop_eval_official_k4.yaml")
        assert k4["n_samples_per_eval_prompt"] == 4
        assert k4["temperature"] == k1["temperature"]
        assert dataset_of("shop_eval_official_k4.yaml")["path"] == dataset_of("shop_eval_official_k1.yaml")["path"]


class TestOfficialK1Greedy:
    def test_only_temperature_differs_from_k1(self):
        k1 = defaults_of("shop_eval_official_k1.yaml")
        greedy = defaults_of("shop_eval_official_k1_greedy.yaml")
        assert greedy["n_samples_per_eval_prompt"] == 1
        assert greedy["temperature"] == 0.0
        assert greedy["top_p"] == k1["top_p"]
        assert dataset_of("shop_eval_official_k1_greedy.yaml")["path"] == dataset_of("shop_eval_official_k1.yaml")["path"]


class TestDev100:
    def test_dev_slice_points_at_dev100(self):
        defaults = defaults_of("shop_dev100.yaml")
        assert defaults["n_samples_per_eval_prompt"] == 1
        assert defaults["temperature"] == 1.0
        dataset = dataset_of("shop_dev100.yaml")
        assert dataset["name"] == "shop_dev100"
        assert dataset["path"].endswith("dev100.jsonl")

    def test_referenced_datasets_exist(self):
        slime_root = CONFIG_DIR.parents[2]  # .../slime (config -> ShopSimulator -> examples -> slime)
        for name in (
            "shop_eval_official_k1.yaml",
            "shop_eval_official_k4.yaml",
            "shop_eval_official_k1_greedy.yaml",
            "shop_dev100.yaml",
        ):
            path = dataset_of(name)["path"]
            assert (slime_root / path).is_file(), f"{name} references missing dataset {path}"
