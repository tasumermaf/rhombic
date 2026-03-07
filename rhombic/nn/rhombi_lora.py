"""RhombiLoRA: LoRA with FCC direction-pair bridge topology.

Standard LoRA: x -> A(r x d_in) -> h(r) -> B(d_out x r) -> dW
RhombiLoRA:    x -> A -> reshape(C, r/C) -> bridge(C x C) -> reshape(r) -> B -> dW

The ONLY addition is a learnable C x C bridge matrix between the down-
and up-projections. With bridge frozen at identity, this is exactly
standard LoRA. The 6 channels correspond to the 6 face-pair directions
of the rhombic dodecahedron.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from rhombic.nn.topology import bridge_init as _bridge_init
from rhombic.spectral import fiedler_value


class RhombiLoRALinear(nn.Module):
    """LoRA with FCC direction-pair bridge topology.

    Parameters
    ----------
    in_features : int
        Input dimension of the linear layer being adapted.
    out_features : int
        Output dimension of the linear layer being adapted.
    rank : int
        LoRA rank. Must be divisible by n_channels.
    n_channels : int
        Number of bridge channels. Default 6 (RD direction pairs).
    alpha : float
        LoRA scaling factor. Default 16.
    dropout : float
        Dropout probability on the LoRA path. Default 0.0.
    bridge_mode : str
        Bridge initialization: 'identity' or 'geometric'.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 24,
        n_channels: int = 6,
        alpha: float = 16.0,
        dropout: float = 0.0,
        bridge_mode: str = 'identity',
    ):
        super().__init__()

        if rank % n_channels != 0:
            raise ValueError(
                f"rank ({rank}) must be divisible by n_channels ({n_channels})"
            )

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.n_channels = n_channels
        self.channel_size = rank // n_channels
        self.scaling = alpha / rank

        # LoRA matrices — standard initialization
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # Bridge matrix — identity or geometric init
        bridge_np = _bridge_init(n_channels, mode=bridge_mode)
        self.bridge = nn.Parameter(
            torch.from_numpy(bridge_np).float()
        )

        # Dropout on the LoRA path
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        # Initialize lora_A with Kaiming uniform (standard PEFT convention)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute LoRA delta with bridge mixing.

        Parameters
        ----------
        x : (..., in_features) tensor

        Returns
        -------
        (..., out_features) tensor — the LoRA correction dW(x).
        """
        input_dtype = x.dtype
        x = x.to(self.lora_A.dtype)
        h = self.dropout(F.linear(x, self.lora_A))       # (..., rank)
        batch_shape = h.shape[:-1]
        C, R = self.n_channels, self.channel_size
        h = h.view(*batch_shape, C, R)                   # (..., C, R)
        h = torch.einsum('ij,...jk->...ik', self.bridge, h)
        h = h.reshape(*batch_shape, self.rank)            # (..., rank)
        return F.linear(h, self.lora_B).to(input_dtype) * self.scaling

    def freeze_bridge(self) -> None:
        """Freeze bridge weights — becomes exactly standard LoRA."""
        self.bridge.requires_grad_(False)

    def unfreeze_bridge(self) -> None:
        """Unfreeze bridge weights for training."""
        self.bridge.requires_grad_(True)

    def bridge_deviation(self) -> float:
        """Frobenius norm of (bridge - I). Monitors divergence from init."""
        with torch.no_grad():
            eye = torch.eye(
                self.n_channels, device=self.bridge.device, dtype=self.bridge.dtype
            )
            return float(torch.norm(self.bridge - eye).item())

    def bridge_fiedler(self) -> float:
        """Fiedler value of current bridge matrix (diagnostic).

        Treats |bridge[i,j]| (i != j) as edge weights of a complete graph
        on n_channels nodes, then computes algebraic connectivity.
        """
        with torch.no_grad():
            B = self.bridge.detach().cpu().numpy()
        n = B.shape[0]
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        weights = [float(abs(B[i, j])) for i, j in edges]
        return fiedler_value(n, edges, weights)

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"rank={self.rank}, channels={self.n_channels}, "
            f"scaling={self.scaling:.4f}"
        )
