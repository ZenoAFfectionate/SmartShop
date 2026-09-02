"""Config-guard tests: pin the semantics of the primary eval yaml (D3/C2).

The published main table is produced with ``shop_eval_official_k1.yaml``
(sampling k=1, temperature 1.0, top_p 1.0). A silent edit to the wrong key —
e.g. flipping temperature to 0.0 or changing the dataset — would invalidate
every published comparison, so the k1 semantics are pinned here.

The contrast configs that used to be guarded here (k4 / greedy / dev100 for
the X3/X7 ablations) were removed together with their experiments on
2026-09-02: those experiments are not being run, so the guards had nothing
to protect and only produced xfail noise.
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
