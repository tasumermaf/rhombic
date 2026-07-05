"""Tests for the W2T canonicalization pipeline (Asset 1 D1 amendment).

Covers the pre-registered properties:
  1. GL(r)-invariance — canonicalize(B @ G, G^{-1} @ A) matches
     canonicalize(B, A) at atol 1e-5, tested at r=6 and r=24.
  2. Exact reconstruction — U @ diag(S) @ V^T == B @ A.
  3. Determinism of the sign convention (idempotency, bitwise repeatability,
     and invariance to injected column sign flips).
  4. Bridge-absorption path — canonicalize_adapter on real smoke adapter
     states reconstructs the effective DW = scaling * B @ E @ A.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
for _p in (str(SCRIPTS_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

torch = pytest.importorskip("torch")

import asset1_canonicalize as canon  # noqa: E402
from rhombic.nn.absorb import _expand_bridge  # noqa: E402

SMOKE_ADAPTERS = sorted(
    (REPO_ROOT / "results" / "asset1-smoke").glob("*/*/run_*/adapter_state.pt"))

ATOL = 1e-5  # pre-registered invariance tolerance


# ── Helpers ─────────────────────────────────────────────────────────


def _random_factors(d_out: int, r: int, d_in: int, seed: int):
    gen = torch.Generator().manual_seed(seed)
    B = torch.randn(d_out, r, generator=gen, dtype=torch.float64)
    A = torch.randn(r, d_in, generator=gen, dtype=torch.float64)
    return B, A


def _random_invertible(r: int, seed: int) -> torch.Tensor:
    """Random G in GL(r); resample the rare ill-conditioned draw so the
    invariance test measures the canonicalization, not float round-off
    through a near-singular G."""
    gen = torch.Generator().manual_seed(seed)
    for _ in range(64):
        G = torch.randn(r, r, generator=gen, dtype=torch.float64)
        if torch.linalg.cond(G).item() < 1e3:
            return G
    raise RuntimeError("could not draw a well-conditioned G")


def _max_diff(c1, c2) -> float:
    return max((c1[k] - c2[k]).abs().max().item() for k in ("U", "S", "V"))


# ── 1. GL(r)-invariance ─────────────────────────────────────────────


@pytest.mark.parametrize("r,d_out,d_in", [(6, 40, 32), (24, 96, 64)])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_gl_invariance(r, d_out, d_in, seed):
    B, A = _random_factors(d_out, r, d_in, seed)
    G = _random_invertible(r, seed + 100)
    c1 = canon.canonicalize_module(B, A)
    c2 = canon.canonicalize_module(B @ G, torch.linalg.inv(G) @ A)
    for key in ("U", "S", "V"):
        assert torch.allclose(c1[key], c2[key], atol=ATOL), (
            f"{key} not GL({r})-invariant: max diff "
            f"{(c1[key] - c2[key]).abs().max().item():.3e}")
    print(f"GL({r}) seed={seed}: max diff {_max_diff(c1, c2):.3e}")


def test_gl_invariance_diagonal_and_permutation():
    """Structured reparameterizations: pure scaling and permutation."""
    B, A = _random_factors(64, 24, 48, seed=7)
    c1 = canon.canonicalize_module(B, A)

    gen = torch.Generator().manual_seed(7)
    d = torch.empty(24, dtype=torch.float64).uniform_(0.5, 2.0, generator=gen)
    G_diag = torch.diag(d)
    perm = torch.randperm(24, generator=gen)
    G_perm = torch.eye(24, dtype=torch.float64)[perm]

    for G in (G_diag, G_perm, G_diag @ G_perm):
        c2 = canon.canonicalize_module(B @ G, torch.linalg.inv(G) @ A)
        assert _max_diff(c1, c2) < ATOL


# ── 2. Reconstruction ───────────────────────────────────────────────


@pytest.mark.parametrize("d_out,r,d_in", [
    (96, 24, 64),    # d_out, d_in > r (the Asset-1 regime)
    (40, 6, 32),
    (16, 24, 48),    # d_out < r — genericity of the thin-QR route
])
def test_reconstruction(d_out, r, d_in):
    B, A = _random_factors(d_out, r, d_in, seed=3)
    c = canon.canonicalize_module(B, A)
    recon = c["U"] @ torch.diag(c["S"]) @ c["V"].T
    ref = B @ A
    assert torch.allclose(recon, ref, atol=1e-8), (
        f"reconstruction error {(recon - ref).abs().max().item():.3e}")
    # Orthonormal columns and descending spectrum
    m = c["S"].numel()
    eye = torch.eye(m, dtype=torch.float64)
    assert torch.allclose(c["U"].T @ c["U"], eye, atol=1e-10)
    assert torch.allclose(c["V"].T @ c["V"], eye, atol=1e-10)
    assert torch.all(c["S"][:-1] >= c["S"][1:] - 1e-12)
    assert torch.all(c["S"] >= 0)


def test_rank_deficient_trailing_zeros():
    """rank(DW) < r gives trailing ~zero slots (W2T Eq. 1 convention)."""
    gen = torch.Generator().manual_seed(11)
    r, true_rank = 24, 5
    B = (torch.randn(64, true_rank, generator=gen, dtype=torch.float64)
         @ torch.randn(true_rank, r, generator=gen, dtype=torch.float64))
    A = torch.randn(r, 48, generator=gen, dtype=torch.float64)
    c = canon.canonicalize_module(B, A)
    assert c["S"].numel() == r
    assert torch.all(c["S"][true_rank:] < 1e-10)
    recon = c["U"] @ torch.diag(c["S"]) @ c["V"].T
    assert torch.allclose(recon, B @ A, atol=1e-8)


# ── 3. Sign-convention determinism ──────────────────────────────────


def test_sign_convention_repeatable():
    B, A = _random_factors(64, 24, 48, seed=5)
    c1 = canon.canonicalize_module(B, A)
    c2 = canon.canonicalize_module(B.clone(), A.clone())
    for key in ("U", "S", "V"):
        assert torch.equal(c1[key], c2[key]), f"{key} not bitwise repeatable"


def test_sign_convention_anchor_positive():
    B, A = _random_factors(64, 24, 48, seed=6)
    c = canon.canonicalize_module(B, A)
    U = c["U"]
    idx = U.abs().argmax(dim=0)
    anchors = U[idx, torch.arange(U.shape[1])]
    assert torch.all(anchors > 0), "max-|.| entry of some U column not positive"


def test_sign_convention_absorbs_column_flips():
    """Rebuilding the factors with injected sign flips D (B' = U D diag(S),
    A' = D V^T — same DW) must return the identical canonical form."""
    B, A = _random_factors(64, 24, 48, seed=8)
    c = canon.canonicalize_module(B, A)
    gen = torch.Generator().manual_seed(8)
    flips = torch.where(torch.rand(24, generator=gen) < 0.5, -1.0, 1.0).to(torch.float64)
    D = torch.diag(flips)
    B2 = c["U"] @ D @ torch.diag(c["S"])
    A2 = D @ c["V"].T
    c2 = canon.canonicalize_module(B2, A2)
    for key in ("U", "S", "V"):
        assert torch.allclose(c[key], c2[key], atol=ATOL), (
            f"{key} changed under injected column sign flips")


# ── 4. Bridge-absorption path (real smoke adapters) ─────────────────


@pytest.mark.skipif(not SMOKE_ADAPTERS,
                    reason="no smoke adapters at results/asset1-smoke/")
def test_bridge_absorption_reconstructs_effective_dw():
    path = SMOKE_ADAPTERS[0]
    modules = canon.load_adapter_modules(path)
    canonical = canon.canonicalize_adapter(path)
    assert set(canonical) == set(modules)

    names = sorted(modules)
    # Two representative modules with different d_out (q_proj vs k_proj on Qwen)
    samples = ([n for n in names if n.endswith("q_proj")][:1]
               + [n for n in names if n.endswith("k_proj")][:1])
    assert samples
    for name in samples:
        entry = modules[name]
        A = entry["lora_A"].to(torch.float64)
        B = entry["lora_B"].to(torch.float64)
        bridge = entry["bridge"].to(torch.float64)
        scaling = float(entry["scaling"])
        rank = int(entry["rank"])
        C = int(entry["n_channels"])
        E = _expand_bridge(bridge, rank // C)
        dw_direct = scaling * (B @ E @ A)              # ground truth composition

        c = canonical[name]
        recon = c["U"] @ torch.diag(c["S"]) @ c["V"].T
        rel = ((recon - dw_direct).norm() / dw_direct.norm().clamp(min=1e-30)).item()
        assert rel < 1e-10, f"{name}: relative Frobenius error {rel:.3e}"
        assert c["S"].numel() == rank
        assert c["U"].shape == (B.shape[0], rank)
        assert c["V"].shape == (A.shape[1], rank)


@pytest.mark.skipif(not SMOKE_ADAPTERS,
                    reason="no smoke adapters at results/asset1-smoke/")
def test_feature_vector_variants_on_smoke_adapter():
    canonical = canon.canonicalize_adapter(SMOKE_ADAPTERS[0])
    n_modules = len(canonical)
    r = next(iter(canonical.values()))["S"].numel()

    sigma = canon.feature_vector(canonical, variant="sigma")
    assert sigma.dtype == torch.float32
    assert sigma.numel() == n_modules * r
    assert torch.all(torch.isfinite(sigma))

    full = canon.feature_vector(canonical, variant="full", proj_dim=16, proj_seed=0)
    assert full.numel() == n_modules * (r + 2 * r * 16)
    assert torch.all(torch.isfinite(full))

    # Determinism of the feature map
    assert torch.equal(sigma, canon.feature_vector(canonical, variant="sigma"))
    assert torch.equal(full, canon.feature_vector(canonical, variant="full"))

    with pytest.raises(ValueError):
        canon.feature_vector(canonical, variant="raw")


def test_feature_vector_gl_invariant_synthetic():
    """Feature vectors built from canonical forms of reparameterized factors
    must agree — the end-to-end property the D1 analysis relies on."""
    B, A = _random_factors(48, 24, 32, seed=9)
    G = _random_invertible(24, seed=109)
    c1 = {"mod": canon.canonicalize_module(B, A)}
    c2 = {"mod": canon.canonicalize_module(B @ G, torch.linalg.inv(G) @ A)}
    for variant in ("sigma", "full"):
        f1 = canon.feature_vector(c1, variant=variant)
        f2 = canon.feature_vector(c2, variant=variant)
        assert torch.allclose(f1, f2, atol=1e-4), (
            f"{variant} features not invariant: "
            f"max diff {(f1 - f2).abs().max().item():.3e}")
