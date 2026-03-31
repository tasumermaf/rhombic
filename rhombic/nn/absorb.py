"""Bridge absorption: fold the RhombiLoRA bridge into standard LoRA A/B matrices.

After training, the learned bridge matrix encodes geometric channel coupling.
For deployment (e.g., Nemotron competition), we need standard LoRA format:
just lora_A and lora_B, no bridge. This module folds the bridge into A and B
so the geometric structure travels invisibly inside a standard adapter.

Mathematical basis
------------------
The RhombiLoRA forward pass computes (ignoring scaling/dropout):

    h = x @ A.T                          # (*, rank)
    h = h.view(*, C, R)                  # (*, n_channels, channel_size)
    h = bridge @ h                       # channel mixing
    h = h.view(*, rank)                  # (*, rank)
    out = h @ B.T                        # (*, out_features)

Define the "expanded bridge" E as a (rank, rank) block matrix where block
(i,j) = bridge[i,j] * I_R. Then the full computation is:

    out = x @ A.T @ E @ B.T = x @ (B @ E @ A).T

To absorb, we decompose B @ E @ A into new_B @ new_A of the same shapes.
Three strategies:

    'into_A':  new_A = E @ A,            new_B = B
    'into_B':  new_A = A,                new_B = B @ E
    'balanced': E = L @ R via SVD,       new_A = R @ A, new_B = B @ L

All three produce mathematically identical outputs and are exact (no
decomposition error). 'balanced' distributes the bridge's effect
symmetrically: for SPD bridges via true matrix sqrt, for general
bridges via SVD factorization (L @ R = E, exact in both cases).
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn


def _expand_bridge(bridge: torch.Tensor, channel_size: int) -> torch.Tensor:
    """Expand (C, C) bridge to (rank, rank) block matrix.

    Block (i, j) = bridge[i, j] * I_{channel_size}.

    Parameters
    ----------
    bridge : (C, C) tensor
    channel_size : int
        R = rank // C. Each channel spans R contiguous dimensions.

    Returns
    -------
    expanded : (rank, rank) tensor where rank = C * channel_size
    """
    C = bridge.shape[0]
    R = channel_size
    rank = C * R
    expanded = torch.zeros(rank, rank, dtype=bridge.dtype, device=bridge.device)
    for i in range(C):
        for j in range(C):
            expanded[i * R:(i + 1) * R, j * R:(j + 1) * R] = (
                bridge[i, j] * torch.eye(R, dtype=bridge.dtype, device=bridge.device)
            )
    return expanded


def _balanced_factors(M: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Decompose M into L, R such that L @ R = M and the norms are balanced.

    For symmetric positive definite M, uses eigendecomposition:
        M = Q @ diag(evals) @ Q.T
        L = R = Q @ diag(sqrt(evals)) @ Q.T  (true matrix square root)

    For general M, uses SVD:
        M = U @ diag(S) @ Vh
        L = U @ diag(sqrt(S))
        R = diag(sqrt(S)) @ Vh

    In both cases, L @ R = M exactly (up to floating point).

    Returns
    -------
    L, R : tensors of same shape as M, with L @ R = M.
    """
    # Try eigendecomposition for symmetric positive definite case
    sym_err = torch.norm(M - M.T).item()
    if sym_err < 1e-6 * torch.norm(M).item():
        try:
            evals, evecs = torch.linalg.eigh(M)
            if torch.all(evals > 0):
                sqrt_diag = torch.diag(torch.sqrt(evals))
                S = evecs @ sqrt_diag @ evecs.T
                return S, S  # true matrix sqrt: S @ S = M
        except Exception:
            pass

    # General case: SVD factorization (exact for any matrix)
    U, S, Vh = torch.linalg.svd(M)
    sqrt_S = torch.sqrt(S)
    L = U @ torch.diag(sqrt_S)       # (n, n)
    R = torch.diag(sqrt_S) @ Vh      # (n, n)
    # L @ R = U @ diag(S) @ Vh = M  (exact)
    return L, R


def absorb_bridge(
    module: nn.Module,
    strategy: Literal['into_A', 'into_B', 'balanced'] = 'into_B',
    verify: bool = True,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> dict[str, torch.Tensor]:
    """Fold the bridge matrix into lora_A and lora_B, producing standard LoRA weights.

    Parameters
    ----------
    module : RhombiLoRALinear
        A trained RhombiLoRA module. The effective bridge (which may come
        from a local bridge, emanation master+proj, or external function)
        is resolved via module.effective_bridge.
    strategy : {'into_A', 'into_B', 'balanced'}
        How to distribute the bridge:
        - 'into_A':  new_A = E @ A, new_B = B. Exact.
        - 'into_B':  new_A = A, new_B = B @ E. Exact.
        - 'balanced': Symmetric distribution via matrix sqrt. Exact for
          positive definite bridges; approximate otherwise.
    verify : bool
        If True, run a numerical verification pass: generate random input
        and assert the absorbed adapter matches the original within tolerance.
        Raises AssertionError on failure. Default True.
    atol : float
        Absolute tolerance for verification. Default 1e-5.
    rtol : float
        Relative tolerance for verification. Default 1e-5.

    Returns
    -------
    dict with keys:
        'lora_A': (rank, in_features) tensor — new down-projection
        'lora_B': (out_features, rank) tensor — new up-projection
        'scaling': float — the alpha/rank scaling factor (unchanged)
        'rank': int
        'in_features': int
        'out_features': int
        'bridge_absorbed': True  — marker flag for downstream consumers
        'bridge_deviation': float — how far the bridge was from identity
        'strategy': str — which absorption strategy was used

    Raises
    ------
    AssertionError
        If verify=True and the absorbed adapter output diverges from the
        original beyond tolerance.
    ValueError
        If strategy is not one of the three valid options.
    """
    if strategy not in ('into_A', 'into_B', 'balanced'):
        raise ValueError(
            f"strategy must be 'into_A', 'into_B', or 'balanced', got {strategy!r}"
        )

    # Resolve the effective bridge (handles standard, emanation, and external modes)
    with torch.no_grad():
        bridge = module.effective_bridge.detach().clone()
        A = module.lora_A.detach().clone()   # (rank, in_features)
        B = module.lora_B.detach().clone()   # (out_features, rank)
        scaling = module.scaling
        C = module.n_channels
        R = module.channel_size

    # Expand bridge to (rank, rank) block matrix
    E = _expand_bridge(bridge, R)

    # Compute absorbed matrices
    if strategy == 'into_A':
        new_A = E @ A           # (rank, in_features)
        new_B = B.clone()       # (out_features, rank)
    elif strategy == 'into_B':
        new_A = A.clone()       # (rank, in_features)
        new_B = B @ E           # (out_features, rank)
    elif strategy == 'balanced':
        # Balanced decomposition: E = L @ R with balanced norms.
        # For SPD bridges: L = R = sqrt(E) (true matrix square root).
        # For general bridges: SVD factorization L @ R = E (exact).
        L, R = _balanced_factors(E)
        new_A = R @ A           # (rank, in_features)
        new_B = B @ L           # (out_features, rank)

    # Compute bridge deviation for diagnostics
    eye = torch.eye(C, dtype=bridge.dtype, device=bridge.device)
    bridge_deviation = float(torch.norm(bridge - eye).item())

    # Verification: original vs. absorbed must produce same output
    if verify:
        _verify_absorption(
            module=module,
            new_A=new_A,
            new_B=new_B,
            scaling=scaling,
            atol=atol,
            rtol=rtol,
        )

    return {
        'lora_A': new_A,
        'lora_B': new_B,
        'scaling': scaling,
        'rank': module.rank,
        'in_features': module.in_features,
        'out_features': module.out_features,
        'bridge_absorbed': True,
        'bridge_deviation': bridge_deviation,
        'strategy': strategy,
    }


def _verify_absorption(
    module: nn.Module,
    new_A: torch.Tensor,
    new_B: torch.Tensor,
    scaling: float,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    n_samples: int = 8,
    seq_len: int = 16,
) -> None:
    """Verify absorbed adapter matches original output.

    Generates random input and compares the full forward pass of the
    original module against the manual standard-LoRA computation with
    the absorbed weights.

    Raises AssertionError with detailed diagnostics on failure.
    """
    device = module.lora_A.device
    dtype = module.lora_A.dtype
    was_training = module.training
    module.eval()  # disable dropout for deterministic comparison

    with torch.no_grad():
        x = torch.randn(n_samples, seq_len, module.in_features,
                         device=device, dtype=dtype)

        # Original module output
        y_original = module(x)

        # Absorbed standard-LoRA output: x @ new_A.T @ new_B.T * scaling
        h = x @ new_A.T
        y_absorbed = h @ new_B.T * scaling

        # Compute error metrics
        abs_err = (y_original - y_absorbed).abs()
        max_abs_err = abs_err.max().item()
        mean_abs_err = abs_err.mean().item()

        # Relative error (avoid division by zero)
        denom = y_original.abs().clamp(min=1e-8)
        max_rel_err = (abs_err / denom).max().item()

    # Restore training mode
    if was_training:
        module.train()

    # Assert within tolerance
    close = torch.allclose(y_original, y_absorbed, atol=atol, rtol=rtol)
    if not close:
        raise AssertionError(
            f"Bridge absorption verification FAILED.\n"
            f"  Max absolute error: {max_abs_err:.2e}\n"
            f"  Mean absolute error: {mean_abs_err:.2e}\n"
            f"  Max relative error: {max_rel_err:.2e}\n"
            f"  Tolerance: atol={atol}, rtol={rtol}\n"
            f"  Output shape: {y_original.shape}\n"
            f"  Output range: [{y_original.min().item():.4f}, "
            f"{y_original.max().item():.4f}]"
        )


def absorb_bridge_into_state_dict(
    module: nn.Module,
    strategy: Literal['into_A', 'into_B', 'balanced'] = 'into_B',
    verify: bool = True,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    key_prefix: str = '',
) -> dict[str, torch.Tensor]:
    """Absorb bridge and return a standard LoRA state_dict (just A and B).

    This is the function you call when exporting for submission. The returned
    dict contains only 'lora_A.weight' and 'lora_B.weight' keys — compatible
    with HuggingFace PEFT's LoraLayer format.

    Parameters
    ----------
    module : RhombiLoRALinear
        Trained module.
    strategy : str
        Absorption strategy. See absorb_bridge().
    verify : bool
        Run numerical verification. Default True.
    atol, rtol : float
        Verification tolerances.
    key_prefix : str
        Optional prefix for state_dict keys (e.g., 'base_model.model.').

    Returns
    -------
    state_dict : dict[str, Tensor]
        Keys: '{prefix}lora_A.weight', '{prefix}lora_B.weight'
    """
    result = absorb_bridge(module, strategy=strategy, verify=verify,
                           atol=atol, rtol=rtol)
    return {
        f'{key_prefix}lora_A.weight': result['lora_A'],
        f'{key_prefix}lora_B.weight': result['lora_B'],
    }


# ---------------------------------------------------------------------------
# Self-test: round-trip correctness
# ---------------------------------------------------------------------------

def test_absorb_roundtrip() -> None:
    """Verify all three absorption strategies on multiple bridge configurations.

    This function is both a standalone self-test and a pytest-discoverable test.
    Run directly: python -m rhombic.nn.absorb
    Run via pytest: pytest rhombic/nn/absorb.py -v
    """
    from rhombic.nn.rhombi_lora import RhombiLoRALinear
    from rhombic.nn.topology import create_emanation_bridge

    configs = [
        # (description, kwargs for RhombiLoRALinear, extra_setup)
        (
            'identity bridge',
            dict(in_features=64, out_features=128, rank=24,
                 n_channels=6, bridge_mode='identity'),
            None,
        ),
        (
            'geometric bridge',
            dict(in_features=64, out_features=128, rank=24,
                 n_channels=6, bridge_mode='geometric'),
            None,
        ),
        (
            'trained bridge (random perturbation)',
            dict(in_features=64, out_features=128, rank=24,
                 n_channels=6, bridge_mode='identity'),
            'perturb_bridge',
        ),
        (
            'non-square channels (n=3)',
            dict(in_features=32, out_features=48, rank=12,
                 n_channels=3, bridge_mode='identity'),
            'perturb_bridge',
        ),
        (
            'large rank',
            dict(in_features=128, out_features=256, rank=48,
                 n_channels=6, bridge_mode='identity'),
            'perturb_bridge',
        ),
        (
            'emanation mode',
            dict(in_features=64, out_features=128, rank=24,
                 n_channels=6),
            'emanation',
        ),
    ]

    strategies = ['into_A', 'into_B', 'balanced']
    passed = 0
    failed = 0
    errors: list[str] = []

    for desc, kwargs, setup in configs:
        for strategy in strategies:
            test_name = f"{desc} / {strategy}"
            try:
                torch.manual_seed(42)

                if setup == 'emanation':
                    master, _ = create_emanation_bridge(
                        kwargs.get('n_channels', 6), 'geometric'
                    )
                    kwargs_copy = {k: v for k, v in kwargs.items()}
                    kwargs_copy['master_bridge'] = master
                    m = RhombiLoRALinear(**kwargs_copy)
                    # Simulate training: perturb layer_proj
                    with torch.no_grad():
                        m.layer_proj.data += torch.randn_like(m.layer_proj) * 0.5
                else:
                    m = RhombiLoRALinear(**kwargs)

                # lora_B must be nonzero for meaningful test
                nn.init.normal_(m.lora_B, std=0.01)

                if setup == 'perturb_bridge':
                    with torch.no_grad():
                        m.bridge.data += torch.randn_like(m.bridge) * 0.1

                m.eval()

                # Absorb with verification (this is the actual test)
                result = absorb_bridge(m, strategy=strategy, verify=True,
                                       atol=1e-4, rtol=1e-4)

                # Additional checks: shapes preserved
                assert result['lora_A'].shape == m.lora_A.shape, \
                    f"A shape mismatch: {result['lora_A'].shape} vs {m.lora_A.shape}"
                assert result['lora_B'].shape == m.lora_B.shape, \
                    f"B shape mismatch: {result['lora_B'].shape} vs {m.lora_B.shape}"
                assert result['bridge_absorbed'] is True
                assert result['strategy'] == strategy

                # Verify state_dict export format
                sd = absorb_bridge_into_state_dict(
                    m, strategy=strategy, verify=True,
                    atol=1e-4, rtol=1e-4,
                )
                assert 'lora_A.weight' in sd
                assert 'lora_B.weight' in sd
                assert sd['lora_A.weight'].shape == m.lora_A.shape
                assert sd['lora_B.weight'].shape == m.lora_B.shape

                passed += 1

            except Exception as e:
                failed += 1
                errors.append(f"  FAIL: {test_name}: {e}")

    # Report
    total = passed + failed
    print(f"\nabsorb_bridge round-trip verification: {passed}/{total} passed")
    if errors:
        print("\nFailures:")
        for err in errors:
            print(err)

    assert failed == 0, f"{failed} absorption tests failed"
    print("All absorption tests passed.")


if __name__ == '__main__':
    test_absorb_roundtrip()
