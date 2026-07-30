"""Acceptance tests for the Tinker pilot task-identity readout.

Two properties, both checked on synthetic PEFT adapters so they run without
network access, without a GPU, and without the (paid) real bank:

1.  DETECTION — a planted per-task direction is recovered: every same-task pair
    is closer than every cross-task pair, and 1-NN task accuracy is 1.0.
2.  GAUGE INVARIANCE — the headline property the readout rests on. Replacing
    (B, A) with (B G, G^-1 A) for G in GL(r) leaves the effective update
    DW = B @ A mathematically unchanged, so the CANONICAL features must not
    move while the RAW features must.

Tolerances are relative at 1e-5, matching the invariance tolerance
``scripts/asset1_canonicalize.py`` documents: the stored factors are float32,
so the G/inv(G) round-trip carries ~1e-6 relative error by construction.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
for _p in (str(SCRIPTS_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tinker_pilot_signal import (  # noqa: E402
    canonical_feature, cosine_distances, evaluate, load_bridgeless_adapter,
    raw_feature, scaling_from_config)

RANK = 8
D_OUT, D_IN = 64, 48
MODULES = [f"base_model.model.model.layers.{L}.self_attn.{p}_proj"
           for L in range(3) for p in ("q", "k")]
TASKS = ("alpaca", "math", "agnews")
SEEDS = (0, 1)
TOL = 1e-5


def build_tensors(task_index: int, seed: int) -> dict[str, torch.Tensor]:
    """Base (A, B) per module with a planted per-task direction in B."""
    gen = torch.Generator().manual_seed(seed)
    task_vec = torch.zeros(D_OUT)
    task_vec[task_index * 10:(task_index + 1) * 10] = 1.0
    tensors: dict[str, torch.Tensor] = {}
    for module in MODULES:
        A = torch.randn(RANK, D_IN, generator=gen) * 0.02
        B = torch.randn(D_OUT, RANK, generator=gen) * 0.02
        tensors[f"{module}.lora_A.weight"] = A
        tensors[f"{module}.lora_B.weight"] = B + task_vec[:, None] * 0.5
    return tensors


def apply_gauge(tensors: dict, seed: int) -> dict[str, torch.Tensor]:
    """(B, A) -> (B G, G^-1 A): DW is preserved, the factors are not.

    Uses its own generator so the base weights are bit-identical.
    """
    gen = torch.Generator().manual_seed(seed)
    out: dict[str, torch.Tensor] = {}
    for module in MODULES:
        A = tensors[f"{module}.lora_A.weight"]
        B = tensors[f"{module}.lora_B.weight"]
        # +3I keeps G well away from singular, so inv(G) is numerically sane.
        G = torch.randn(RANK, RANK, generator=gen) + torch.eye(RANK) * 3.0
        out[f"{module}.lora_A.weight"] = torch.linalg.inv(G) @ A
        out[f"{module}.lora_B.weight"] = B @ G
    return out


def write_adapter(dirpath: Path, tensors: dict) -> None:
    dirpath.mkdir(parents=True, exist_ok=True)
    save_file(dict(tensors), str(dirpath / "adapter_model.safetensors"))
    (dirpath / "adapter_config.json").write_text(
        json.dumps({"r": RANK, "lora_alpha": 16, "peft_type": "LORA"}),
        encoding="utf-8")


def features_for(dirpath: Path) -> tuple[np.ndarray, np.ndarray]:
    modules, cfg = load_bridgeless_adapter(dirpath)
    scaling, _ = scaling_from_config(cfg, RANK)
    return (raw_feature(modules),
            canonical_feature(modules, scaling, proj_dim=16, proj_seed=0))


def rel_change(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-30))


@pytest.fixture(scope="module")
def synthetic_bank(tmp_path_factory) -> Path:
    bank = tmp_path_factory.mktemp("tinker_pilot_bank")
    for task_index, task in enumerate(TASKS):
        for seed in SEEDS:
            write_adapter(bank / f"{task}_{seed}",
                          build_tensors(task_index, seed=100 * task_index + seed))
    return bank


def test_loader_reads_standard_peft_names(synthetic_bank):
    modules, cfg = load_bridgeless_adapter(synthetic_bank / "alpaca_0")
    assert set(modules) == set(MODULES)
    for entry in modules.values():
        assert entry["lora_A"].shape == (RANK, D_IN)
        assert entry["lora_B"].shape == (D_OUT, RANK)
    scaling, meta = scaling_from_config(cfg, RANK)
    assert scaling == pytest.approx(16 / RANK)      # alpha / r, bridgeless
    assert meta["scaling_mode"] == "alpha/r"


@pytest.mark.parametrize("space", ["raw", "canonical"])
def test_planted_task_signal_is_detected(synthetic_bank, space):
    feats, task_of = {}, {}
    for run_dir in sorted(d for d in synthetic_bank.iterdir() if d.is_dir()):
        run_id = run_dir.name
        task_of[run_id] = run_id.rsplit("_", 1)[0]
        raw, can = features_for(run_dir)
        feats[run_id] = raw if space == "raw" else can

    ev = evaluate(cosine_distances(feats), task_of)
    assert ev["n_pairs"] == 15
    assert ev["n_within_task_pairs"] == 3
    assert ev["n_cross_task_pairs"] == 12
    assert ev["separated"], (
        f"{space}: max_within={ev['max_within_task']:.6f} "
        f"!< min_cross={ev['min_cross_task']:.6f}")
    assert ev["nn_task_accuracy"] == 1.0


def test_gauge_transform_preserves_the_effective_update(tmp_path):
    base = build_tensors(0, seed=7)
    gauged = apply_gauge(base, seed=999)
    worst = 0.0
    for module in MODULES:
        dw = (base[f"{module}.lora_B.weight"].double()
              @ base[f"{module}.lora_A.weight"].double())
        dw_g = (gauged[f"{module}.lora_B.weight"].double()
                @ gauged[f"{module}.lora_A.weight"].double())
        worst = max(worst, float((dw - dw_g).norm() / dw.norm()))
    assert worst < TOL, f"gauge did not preserve DW (rel {worst:.3e})"


def test_canonical_features_are_gauge_invariant_and_raw_are_not(tmp_path):
    """The load-bearing claim: canonicalization removes the GL(r) gauge."""
    base = build_tensors(0, seed=7)
    plain_dir, gauged_dir = tmp_path / "plain", tmp_path / "gauged"
    write_adapter(plain_dir, base)
    write_adapter(gauged_dir, apply_gauge(base, seed=999))

    raw_p, can_p = features_for(plain_dir)
    raw_g, can_g = features_for(gauged_dir)

    assert rel_change(can_p, can_g) < TOL, (
        f"canonical features moved under a pure gauge change "
        f"({rel_change(can_p, can_g):.3e})")
    assert rel_change(raw_p, raw_g) > 1e-2, (
        "raw features did NOT move under a gauge change — the synthetic "
        "gauge is too weak to make this test meaningful")
