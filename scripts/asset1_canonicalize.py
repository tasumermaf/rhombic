"""Asset 1 — W2T canonicalization of RhombiLoRA adapters (pre-registered D1 amendment).

Implements the GL(r)-invariant canonical representation of LoRA updates from
W2T (arXiv:2603.15990, Section 3, "Identifying Canonical Object of LoRA"):

    A LoRA update DW = B @ A  (B in R^{d_out x r}, A in R^{r x d_in}) is
    non-identifiable: (B @ G, G^{-1} @ A) induces the same DW for any
    G in GL(r). The canonical object is the exact r-slot SVD of DW,
    computed WITHOUT materializing DW (W2T Prop. 3.2):

        B    = Q_B R_B          (thin QR,  Q_B: d_out x k_B)
        A^T  = Q_A R_A          (thin QR,  Q_A: d_in  x k_A)
        M    = R_B @ R_A^T      (small core, k_B x k_A)
        M    = U_hat @ diag(S) @ V_hat^T        (SVD of the core)
        U    = Q_B @ U_hat,   V = Q_A @ V_hat

    giving DW = U @ diag(S) @ V^T exactly, at cost
    O((d_out + d_in) r^2 + r^3) instead of O(d_out d_in r).

Sign convention (determinism)
-----------------------------
SVD is unique only up to per-column sign flips (u_k, v_k) -> (-u_k, -v_k)
(and subspace rotation under degenerate singular values). W2T requires "a
fixed sign and ordering convention" (Prop. 3.1) without prescribing one; we
adopt the standard convention:

    For each column k, locate the entry of maximum |U[:, k]| (ties broken by
    FIRST index, torch.argmax semantics — deterministic). If that entry is
    negative, flip the signs of BOTH U[:, k] and V[:, k].

Ordering follows torch.linalg.svd (singular values descending). Under
repeated singular values the subspace-mixing ambiguity remains — the
standard SVD caveat, acknowledged by W2T ("up to the standard sign and
degeneracy ambiguities"). Trained adapters have generically distinct spectra.

RhombiLoRA bridge absorption
----------------------------
Our adapters are NOT plain (B, A) pairs. Per rhombic/nn/rhombi_lora.py the
forward pass is

    out = scaling * ( B @ E @ A ) @ x        with scaling = alpha / rank,

where E is the (rank x rank) block expansion of the (C x C) bridge — block
(i, j) = bridge[i, j] * I_R, R = rank // C (see rhombic/nn/absorb.py, whose
_expand_bridge we reuse directly). Before canonicalizing we therefore absorb
the bridge with the exact 'into_B' strategy of absorb.py and fold the scalar
alpha/rank scaling into B as well:

    B_eff = scaling * (B @ E),   A_eff = A,
    DW_effective = B_eff @ A_eff             (the TRUE update the model applies)

so the canonical (U, S, V) represents the actual DW, not the raw factors.
The bank saves lora.effective_bridge (asset1_bank.py), so the stored bridge
is already the resolved effective bridge for every bridge mode.

Feature vectors (for the D1 RAW-vs-CANONICAL SVM comparison)
------------------------------------------------------------
Two variants, both deterministic, modules iterated in sorted-name order:

    'sigma' : concat over modules of log1p(sigma) — the W2T stabilizing
              transform (Eq. 3 context). GL(r)-invariant AND rotation-
              invariant; dimension = n_modules * r.
    'full'  : per module [log1p(sigma); (U^T P_u).flatten(); (V^T P_v).flatten()]
              where P_u in R^{d_out x p}, P_v in R^{d_in x p} are FIXED
              seeded Gaussian probes (seed derived from (proj_seed, dim), so
              probes are shared by all modules — and all adapters — with the
              same dimension), scaled 1/sqrt(p). Flattening is rank-slot-major:
              row k holds the p probe responses of singular direction k.

Notes: 'full' features are comparable only across adapters sharing the same
module-name/shape set (true within each Asset-1 family — the D1 setting).
The singular values are deliberately NOT multiplied into the projected
directions; the classifier sees magnitude (sigma block) and direction
(projection blocks) separately, mirroring W2T's separation of modulation
from directional content.

Usage
-----
    python scripts/asset1_canonicalize.py                # smoke sanity pass
    python scripts/asset1_canonicalize.py --bank-root results/asset1-bank
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rhombic.nn.absorb import _expand_bridge  # noqa: E402  (exact into_B absorption)

SMOKE_ROOT = REPO_ROOT / "results" / "asset1-smoke"

# Internal compute dtype. CPU-only analysis; float64 keeps QR/SVD round-off
# far below the 1e-5 invariance tolerance of the pre-registered tests.
_DTYPE = torch.float64


# ── Core canonicalization ───────────────────────────────────────────


def _fix_signs(U: torch.Tensor, V: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic sign convention (documented in the module docstring).

    Force the maximum-|.| entry of each U column positive; apply the same
    flip to the paired V column so U @ diag(S) @ V^T is unchanged.
    """
    idx = U.abs().argmax(dim=0)                      # first max index — deterministic
    anchor = U[idx, torch.arange(U.shape[1])]
    signs = torch.where(anchor < 0,
                        torch.tensor(-1.0, dtype=U.dtype),
                        torch.tensor(1.0, dtype=U.dtype))
    return U * signs, V * signs


def canonicalize_module(B: torch.Tensor, A: torch.Tensor) -> dict[str, torch.Tensor]:
    """W2T QR-SVD canonical representation of DW = B @ A.

    Parameters
    ----------
    B : (d_out, r) tensor — up-projection factor.
    A : (r, d_in) tensor — down-projection factor.

    Returns
    -------
    dict with keys:
        'U' : (d_out, m) float64 — left singular vectors (orthonormal cols)
        'S' : (m,)      float64 — singular values, descending
        'V' : (d_in, m) float64 — right singular vectors (orthonormal cols)
    where m = min(d_out, d_in, r); DW = U @ diag(S) @ V^T exactly. For the
    Asset-1 shapes (d_out, d_in >> r = 24) m = r, with trailing near-zero
    slots when rank(DW) < r (W2T Eq. 1 trailing-zero convention).
    """
    B = torch.as_tensor(B).detach().to(_DTYPE)
    A = torch.as_tensor(A).detach().to(_DTYPE)
    if B.ndim != 2 or A.ndim != 2 or B.shape[1] != A.shape[0]:
        raise ValueError(
            f"expected B (d_out, r) and A (r, d_in) with matching r; "
            f"got B {tuple(B.shape)}, A {tuple(A.shape)}")

    Q_B, R_B = torch.linalg.qr(B, mode="reduced")     # B    = Q_B R_B
    Q_A, R_A = torch.linalg.qr(A.T, mode="reduced")   # A^T  = Q_A R_A
    M = R_B @ R_A.T                                   # core, captures all singular structure
    U_hat, S, V_hat_T = torch.linalg.svd(M, full_matrices=False)
    U = Q_B @ U_hat                                   # lift back to output space
    V = Q_A @ V_hat_T.T                               # lift back to input space
    U, V = _fix_signs(U, V)
    return {"U": U, "S": S, "V": V}


# ── RhombiLoRA adapter handling ─────────────────────────────────────


def load_adapter_modules(adapter_state_path: str | Path) -> dict[str, dict[str, torch.Tensor]]:
    """Load an Asset-1 adapter_state.pt into {module_name: {field: tensor}}.

    Bank format (asset1_bank.py): flat dict with keys
    '{module}.lora_A' (r, d_in), '{module}.lora_B' (d_out, r),
    '{module}.bridge' (C, C — the EFFECTIVE bridge), '{module}.scaling',
    '{module}.n_channels', '{module}.rank'.
    """
    state = torch.load(str(adapter_state_path), map_location="cpu",
                       weights_only=True)
    modules: dict[str, dict[str, torch.Tensor]] = {}
    for key, tensor in state.items():
        module, _, field = key.rpartition(".")
        if not module:
            raise ValueError(f"unrecognized adapter key (no module prefix): {key!r}")
        modules.setdefault(module, {})[field] = tensor
    for name, entry in modules.items():
        for required in ("lora_A", "lora_B"):
            if required not in entry:
                raise ValueError(f"module {name!r} missing {required!r}")
    return modules


def effective_factors(entry: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Absorb bridge + scaling into (B_eff, A_eff) with DW = B_eff @ A_eff.

    Mirrors absorb.py 'into_B' exactly: E = block-expanded bridge,
    B_eff = scaling * (B @ E), A_eff = A. A missing bridge (plain LoRA)
    degrades gracefully to E = I. A missing scaling defaults to 1.0.
    """
    A = entry["lora_A"].detach().to(_DTYPE)          # (r, d_in)
    B = entry["lora_B"].detach().to(_DTYPE)          # (d_out, r)
    rank = B.shape[1]
    scaling = float(entry["scaling"]) if "scaling" in entry else 1.0

    bridge = entry.get("bridge")
    if bridge is not None:
        bridge = bridge.detach().to(_DTYPE)
        C = bridge.shape[0]
        if rank % C != 0:
            raise ValueError(f"rank {rank} not divisible by n_channels {C}")
        E = _expand_bridge(bridge, rank // C)         # (rank, rank) block matrix
        B_eff = scaling * (B @ E)
    else:
        B_eff = scaling * B
    return B_eff, A


def canonicalize_adapter(adapter_state_path: str | Path) -> dict[str, dict[str, torch.Tensor]]:
    """Canonicalize every module of an Asset-1 RhombiLoRA adapter.

    Bridge and alpha/rank scaling are absorbed FIRST (effective_factors), so
    the returned (U, S, V) per module is the canonical decomposition of the
    true effective update DW = scaling * B @ E @ A.

    Returns
    -------
    {module_name: {'U': ..., 'S': ..., 'V': ...}} — sorted iteration order
    is the caller's responsibility (feature_vector sorts by name).
    """
    modules = load_adapter_modules(adapter_state_path)
    out: dict[str, dict[str, torch.Tensor]] = {}
    for name in sorted(modules):
        B_eff, A_eff = effective_factors(modules[name])
        out[name] = canonicalize_module(B_eff, A_eff)
    return out


# ── Feature vectors ─────────────────────────────────────────────────


def _probe(dim: int, proj_dim: int, proj_seed: int) -> torch.Tensor:
    """Fixed Gaussian probe P in R^{dim x proj_dim}, deterministic in
    (dim, proj_dim, proj_seed) so every module/adapter with the same
    dimension shares the same probe. Scaled 1/sqrt(proj_dim) (JL convention).
    """
    gen = torch.Generator()
    gen.manual_seed(proj_seed * 1_000_003 + proj_dim * 10_007 + dim)
    return torch.randn(dim, proj_dim, generator=gen, dtype=_DTYPE) / (proj_dim ** 0.5)


def feature_vector(canonical: dict[str, dict[str, torch.Tensor]],
                   variant: str = "sigma",
                   proj_dim: int = 16,
                   proj_seed: int = 0) -> torch.Tensor:
    """Flatten a canonicalized adapter into a 1-D float32 feature vector.

    Parameters
    ----------
    canonical : output of canonicalize_adapter (module -> {'U','S','V'}).
    variant : 'sigma' — log1p singular values only (fully invariant);
              'full'  — sigma block plus fixed-probe projections of U and V
              columns (documented in module docstring).
    proj_dim, proj_seed : probe geometry for 'full' (ignored for 'sigma').

    Returns
    -------
    1-D torch.float32 tensor. Module order: sorted module names —
    deterministic and identical across adapters with the same module set.
    """
    if variant not in ("sigma", "full"):
        raise ValueError(f"variant must be 'sigma' or 'full', got {variant!r}")
    blocks: list[torch.Tensor] = []
    for name in sorted(canonical):
        entry = canonical[name]
        S, U, V = entry["S"], entry["U"], entry["V"]
        blocks.append(torch.log1p(S))                          # (m,)
        if variant == "full":
            P_u = _probe(U.shape[0], proj_dim, proj_seed)
            P_v = _probe(V.shape[0], proj_dim, proj_seed)
            blocks.append((U.T @ P_u).reshape(-1))             # (m * p,) slot-major
            blocks.append((V.T @ P_v).reshape(-1))             # (m * p,) slot-major
    return torch.cat(blocks).to(torch.float32)


# ── Smoke sanity pass ───────────────────────────────────────────────


def find_adapters(bank_root: Path) -> list[Path]:
    return sorted(bank_root.glob("*/*/run_*/adapter_state.pt"))


def _spectra_lines(name: str, entry: dict[str, torch.Tensor]) -> str:
    top3 = ", ".join(f"{v:.6f}" for v in entry["S"][:3].tolist())
    return f"| `{name}` | {top3} |"


def run_sanity(bank_root: Path, report_path: Path) -> None:
    adapters = find_adapters(bank_root)
    if not adapters:
        print(f"No adapters found under {bank_root}")
        return

    lines = [
        "# Asset 1 — Canonicalization Sanity (smoke adapters)",
        "",
        f"Generated {datetime.now(timezone.utc).isoformat()} by "
        "`scripts/asset1_canonicalize.py` (W2T QR-SVD canonicalization, "
        "arXiv:2603.15990 Section 3; bridge + alpha/rank scaling absorbed via "
        "the exact `into_B` strategy of `rhombic/nn/absorb.py` before "
        "decomposition, so spectra describe the TRUE effective DW).",
        "",
        "Smoke context: 50-step runs with zero-initialized `lora_B`, so "
        "singular values are expected to be small but strictly positive — "
        "the sanity signal is a clean, rapidly decaying spectrum per module.",
        "",
    ]

    for path in adapters:
        rel = path.relative_to(bank_root)
        print(f"Canonicalizing {rel} ...", flush=True)
        canonical = canonicalize_adapter(path)
        names = sorted(canonical)
        sigma_feat = feature_vector(canonical, variant="sigma")
        full_feat = feature_vector(canonical, variant="full")
        all_top = torch.stack([canonical[n]["S"][0] for n in names])

        lines += [
            f"## `{rel.as_posix()}`",
            "",
            f"- modules canonicalized: **{len(names)}**",
            f"- feature dims: sigma-only **{sigma_feat.numel()}**, "
            f"full **{full_feat.numel()}** (proj_dim=16, proj_seed=0)",
            f"- top singular value across modules: "
            f"max **{all_top.max().item():.6f}**, "
            f"median {all_top.median().item():.6f}, "
            f"min {all_top.min().item():.6f}",
            "",
            "| module (sample) | top-3 singular values |",
            "|---|---|",
        ]
        # A couple of representative modules: first q_proj and first k_proj
        # (different d_out on Qwen: 1536 vs 256 — exercises both shapes).
        samples = ([n for n in names if n.endswith("q_proj")][:1]
                   + [n for n in names if n.endswith("k_proj")][:1])
        for n in samples:
            lines.append(_spectra_lines(n, canonical[n]))
            print(f"  {n}: top-3 sigma = "
                  f"{[round(v, 6) for v in canonical[n]['S'][:3].tolist()]}")
        lines.append("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSanity report written to {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="W2T canonicalization of Asset-1 RhombiLoRA adapters")
    parser.add_argument("--bank-root", type=Path, default=SMOKE_ROOT,
                        help="Adapter bank root (default: results/asset1-smoke)")
    parser.add_argument("--report", type=Path, default=None,
                        help="Sanity report path (default: "
                             "<bank-root>/CANONICALIZATION_SANITY.md)")
    args = parser.parse_args()
    report = args.report or (args.bank_root / "CANONICALIZATION_SANITY.md")
    run_sanity(args.bank_root, report)


if __name__ == "__main__":
    main()
