"""TeLoRA integration for ostris/ai-toolkit.

Adds the 'telora' network_type to AI Toolkit's LoRA training pipeline.
The bridge (C×C matrix between lora_down and lora_up) uses AI Toolkit's
existing `lora_mid` dispatch in `_call_forward`.

Usage (RunPod / any environment with ai-toolkit installed):

    # In your training script, before running:
    import aitoolkit_telora
    aitoolkit_telora.patch()

    # Then in your YAML config:
    network:
      type: telora          # <-- triggers the bridge
      linear: <model_path>
      linear_alpha: 42
      network_kwargs:
        rank: 42
        n_channels: 6       # RD direction pairs (default)
        bridge_mode: identity  # or 'geometric' (requires n_channels=6)

Or run this file directly to patch and launch training:

    python aitoolkit_telora.py config_telora.yaml
"""

from __future__ import annotations

import math
import sys
from typing import Optional, TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn

if TYPE_CHECKING:
    from toolkit.lora_special import LoRASpecialNetwork


# ── Bridge initialization (self-contained, no rhombic dependency) ─────────

def _bridge_init_identity(n: int) -> np.ndarray:
    return np.eye(n, dtype=np.float64)


def _bridge_init_geometric(n: int) -> np.ndarray:
    """Geometric init from RD face-sharing coupling (requires n=6)."""
    if n != 6:
        raise ValueError("geometric bridge_mode requires n_channels=6")
    # Hard-coded coupling matrix from rhombic dodecahedron geometry.
    # coupling[i,j] = shared octahedral vertices between direction pair i and j.
    # Diagonal=4, co-planar=4, cross-planar=2.
    coupling = np.array([
        [4, 4, 2, 2, 2, 2],
        [4, 4, 2, 2, 2, 2],
        [2, 2, 4, 4, 2, 2],
        [2, 2, 4, 4, 2, 2],
        [2, 2, 2, 2, 4, 4],
        [2, 2, 2, 2, 4, 4],
    ], dtype=np.float64)
    off_diag = coupling.copy()
    np.fill_diagonal(off_diag, 0.0)
    off_diag /= off_diag.max()
    return np.eye(6, dtype=np.float64) + 0.01 * off_diag


def bridge_init(n_channels: int, mode: str = 'identity') -> np.ndarray:
    if mode == 'identity':
        return _bridge_init_identity(n_channels)
    elif mode == 'geometric':
        return _bridge_init_geometric(n_channels)
    else:
        raise ValueError(f"Unknown bridge_mode: {mode}")


# ── BridgeMid: the lora_mid module ───────────────────────────────────────

class BridgeMid(nn.Module):
    """Bridge matrix that slots into AI Toolkit's lora_mid position.

    Reshapes the rank-dimensional hidden state into (C, R) channels,
    applies a learnable C×C mixing matrix, and reshapes back.

    This is the ONLY addition TeLoRA makes to standard LoRA.
    With bridge frozen at identity, forward is a no-op reshape cycle.
    """

    def __init__(self, rank: int, n_channels: int, bridge_mode: str = 'identity'):
        super().__init__()
        if rank % n_channels != 0:
            raise ValueError(
                f"rank ({rank}) must be divisible by n_channels ({n_channels})"
            )
        self.rank = rank
        self.n_channels = n_channels
        self.channel_size = rank // n_channels

        bridge_np = bridge_init(n_channels, mode=bridge_mode)
        self.bridge = nn.Parameter(torch.from_numpy(bridge_np).float())

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (..., rank) -> (..., rank) with bridge mixing."""
        batch_shape = h.shape[:-1]
        C, R = self.n_channels, self.channel_size
        h = h.view(*batch_shape, C, R)               # (..., C, R)
        h = torch.einsum('ij,...jk->...ik', self.bridge, h)  # bridge mix
        return h.reshape(*batch_shape, self.rank)     # (..., rank)

    def extra_repr(self) -> str:
        return f"rank={self.rank}, channels={self.n_channels}"

    @torch.no_grad()
    def deviation(self) -> float:
        """Frobenius distance from identity."""
        eye = torch.eye(self.n_channels, device=self.bridge.device, dtype=self.bridge.dtype)
        return float(torch.norm(self.bridge - eye).item())


# ── TeLoRAModule: AI Toolkit LoRA with bridge ────────────────────────────

def _make_telora_module_class():
    """Dynamically create TeLoRAModule after AI Toolkit is importable."""
    from toolkit.lora_special import LoRAModule

    class TeLoRAModule(LoRAModule):
        """LoRA module with topology-enhanced bridge.

        Identical to standard LoRAModule except it inserts a BridgeMid
        between lora_down and lora_up. AI Toolkit's _call_forward already
        dispatches through self.lora_mid when it exists.
        """

        def __init__(
            self,
            lora_name,
            org_module: nn.Module,
            multiplier=1.0,
            lora_dim=4,
            alpha=1,
            dropout=None,
            rank_dropout=None,
            module_dropout=None,
            network: 'LoRASpecialNetwork' = None,
            use_bias: bool = False,
            n_channels: int = 6,
            bridge_mode: str = 'identity',
            **kwargs
        ):
            # Standard LoRA init (creates lora_down, lora_up, etc.)
            super().__init__(
                lora_name=lora_name,
                org_module=org_module,
                multiplier=multiplier,
                lora_dim=lora_dim,
                alpha=alpha,
                dropout=dropout,
                rank_dropout=rank_dropout,
                module_dropout=module_dropout,
                network=network,
                use_bias=use_bias,
                **kwargs
            )

            # Insert the bridge as lora_mid
            if not self.full_rank:
                self.lora_mid = BridgeMid(
                    rank=lora_dim,
                    n_channels=n_channels,
                    bridge_mode=bridge_mode,
                )
            # Store for diagnostics
            self._n_channels = n_channels
            self._bridge_mode = bridge_mode

        def freeze_bridge(self):
            """Freeze bridge — becomes standard LoRA."""
            if hasattr(self, 'lora_mid') and self.lora_mid is not None:
                self.lora_mid.bridge.requires_grad_(False)

        def unfreeze_bridge(self):
            """Unfreeze bridge for training."""
            if hasattr(self, 'lora_mid') and self.lora_mid is not None:
                self.lora_mid.bridge.requires_grad_(True)

    return TeLoRAModule


# ── Monkey-patch AI Toolkit ──────────────────────────────────────────────

_patched = False


def patch():
    """Patch AI Toolkit to recognize 'telora' as a network_type.

    Call this before creating any LoRASpecialNetwork instances.
    Safe to call multiple times (idempotent).
    """
    global _patched
    if _patched:
        return

    from toolkit.lora_special import LoRASpecialNetwork

    TeLoRAModule = _make_telora_module_class()

    _original_init = LoRASpecialNetwork.__init__

    def _patched_init(self, *args, **kwargs):
        _original_init(self, *args, **kwargs)

        # After standard init, check if telora was requested
        if self.network_type.lower() == 'telora':
            # Get bridge config from network_kwargs
            n_channels = 6
            bridge_mode = 'identity'
            if self.network_config is not None:
                nk = getattr(self.network_config, 'network_kwargs', {})
                n_channels = nk.get('n_channels', 6)
                bridge_mode = nk.get('bridge_mode', 'identity')

            # Replace all lora modules with TeLoRA modules
            # Re-create with bridge by injecting lora_mid into existing modules
            for lora in self.unet_loras:
                if not hasattr(lora, 'lora_mid') or lora.lora_mid is None:
                    if not lora.full_rank:
                        lora.lora_mid = BridgeMid(
                            rank=lora.lora_dim,
                            n_channels=n_channels,
                            bridge_mode=bridge_mode,
                        ).to(lora.lora_down.weight.device)
                        lora._n_channels = n_channels
                        lora._bridge_mode = bridge_mode

            for lora_list in [getattr(self, 'text_encoder_loras', [])]:
                for lora in lora_list:
                    if not hasattr(lora, 'lora_mid') or lora.lora_mid is None:
                        if not lora.full_rank:
                            lora.lora_mid = BridgeMid(
                                rank=lora.lora_dim,
                                n_channels=n_channels,
                                bridge_mode=bridge_mode,
                            ).to(lora.lora_down.weight.device)
                            lora._n_channels = n_channels
                            lora._bridge_mode = bridge_mode

            # Count bridge parameters
            bridge_params = sum(
                p.numel() for lora in self.unet_loras
                if hasattr(lora, 'lora_mid') and lora.lora_mid is not None
                for p in lora.lora_mid.parameters()
            )
            total_params = sum(p.numel() for p in self.parameters())
            print(f"[TeLoRA] Bridge injected: {n_channels} channels, "
                  f"mode={bridge_mode}")
            print(f"[TeLoRA] Bridge params: {bridge_params:,} "
                  f"({bridge_params/total_params*100:.1f}% of total)")

    LoRASpecialNetwork.__init__ = _patched_init
    _patched = True
    print("[TeLoRA] AI Toolkit patched successfully")


# ── CLI: run AI Toolkit training with TeLoRA ─────────────────────────────

if __name__ == '__main__':
    patch()

    if len(sys.argv) < 2:
        print("Usage: python aitoolkit_telora.py <config.yaml>")
        print("  Patches AI Toolkit with TeLoRA support, then runs training.")
        sys.exit(1)

    # Import and run AI Toolkit's main entry point
    from toolkit.job import get_job
    import yaml

    config_path = sys.argv[1]
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    job = get_job(config_path)
    job.run()
