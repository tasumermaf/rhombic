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


class EmanationBridge(nn.Module):
    """Hierarchical bridge: master emanates into per-layer bridges.

    Plotinus: the One produces without diminishing.
    Each layer's bridge = master + layer_offset, where layer_offset
    is a small learned perturbation.

    Parameters
    ----------
    n_channels : int
        Number of bridge channels. Default 6 (RD direction pairs).
    n_layers : int
        Number of layers that will receive emanated bridges.
    init_scale : float
        Standard deviation for offset initialization. Default 0.01.
    """

    def __init__(self, n_channels: int, n_layers: int, init_scale: float = 0.01):
        super().__init__()
        self.n_channels = n_channels
        self.n_layers = n_layers
        self.master = nn.Parameter(torch.eye(n_channels))  # The One
        self.offsets = nn.ParameterList([
            nn.Parameter(torch.randn(n_channels, n_channels) * init_scale)
            for _ in range(n_layers)
        ])

    def get_bridge(self, layer_idx: int) -> torch.Tensor:
        """Emanate the master into a layer-specific bridge."""
        return self.master + self.offsets[layer_idx]

    @property
    def coherence(self) -> float:
        """How similar are all layer bridges to the master?
        High coherence = strong emanation, low = fragmented."""
        with torch.no_grad():
            total = sum(off.norm().item() for off in self.offsets)
            return 1.0 / (1.0 + total / len(self.offsets))


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
    dynamic_bridge : bool
        If True, adds a gate projection that produces input-dependent
        bridge weights. The static bridge serves as the base; the gate
        modulates it per-token. Default False.
    gate_temperature : float
        Initial temperature for the sigmoid gate. High T (~5.0) produces
        near-uniform gating; low T (~0.1) produces hard binary gates.
        Anneal externally via set_gate_temperature(). Default 5.0.
    master_bridge : nn.Parameter, optional
        Shared master bridge parameter for emanation mode. When provided,
        no local bridge is created; instead a layer_proj parameter
        modulates the shared master bridge via sigmoid gating. The
        effective bridge becomes: master_bridge * 2 * sigmoid(layer_proj).
        layer_proj is initialized to zeros so sigmoid(0)=0.5, and 2*0.5=1.0
        gives identity-preserving initialization.
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
        dynamic_bridge: bool = False,
        gate_temperature: float = 5.0,
        master_bridge: nn.Parameter | None = None,
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
        self.dynamic_bridge = dynamic_bridge
        self.emanation = master_bridge is not None
        self._external_bridge_fn: callable | None = None

        # LoRA matrices — standard initialization
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # Bridge: either local (standard) or emanation (shared master + local projection)
        if master_bridge is not None:
            # Emanation mode: shared master bridge + per-layer sigmoid projection
            self.master_bridge = master_bridge  # shared, not owned
            self.layer_proj = nn.Parameter(
                torch.zeros(n_channels, n_channels)
            )
            # Register bridge as a read-only property for diagnostics
            # (no local bridge parameter created)
        else:
            # Standard mode: local bridge parameter
            bridge_np = _bridge_init(n_channels, mode=bridge_mode)
            self.bridge = nn.Parameter(
                torch.from_numpy(bridge_np).float()
            )

        # Dynamic bridge gate — input-dependent bridge modulation
        if dynamic_bridge:
            C = n_channels
            self.gate_proj = nn.Linear(in_features, C * C, bias=False)
            # Initialize gate_proj small so initial gating ≈ 0.5 (neutral)
            nn.init.zeros_(self.gate_proj.weight)
            self.register_buffer(
                '_gate_temperature',
                torch.tensor(gate_temperature, dtype=torch.float32),
            )
        else:
            self.gate_proj = None

        # Dropout on the LoRA path
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        # Initialize lora_A with Kaiming uniform (standard PEFT convention)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def set_gate_temperature(self, temperature: float) -> None:
        """Set gate temperature for dynamic bridge annealing."""
        if self.gate_proj is not None:
            self._gate_temperature.fill_(temperature)

    @property
    def effective_bridge(self) -> torch.Tensor:
        """The current effective bridge matrix.

        In external bridge mode (EmanationBridge), returns the result of the
        external bridge function. In emanation mode (master_bridge + sigmoid),
        returns master_bridge * 2 * sigmoid(layer_proj). In standard mode,
        returns self.bridge directly.
        """
        if self._external_bridge_fn is not None:
            return self._external_bridge_fn()
        if self.emanation:
            return self.master_bridge * (2.0 * torch.sigmoid(self.layer_proj))
        return self.bridge

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

        bridge = self.effective_bridge

        if self.gate_proj is not None:
            # Dynamic bridge: gate modulates static bridge per-token
            gate_logits = self.gate_proj(x)               # (..., C*C)
            gate = torch.sigmoid(
                gate_logits / self._gate_temperature
            )                                             # (..., C*C)
            gate = gate.view(*batch_shape, C, C)          # (..., C, C)
            # Effective bridge = bridge * gate
            effective_bridge = bridge * gate               # (..., C, C)
            h = torch.einsum('...ij,...jk->...ik', effective_bridge, h)
        else:
            h = torch.einsum('ij,...jk->...ik', bridge, h)

        h = h.reshape(*batch_shape, self.rank)            # (..., rank)
        return F.linear(h, self.lora_B).to(input_dtype) * self.scaling

    def freeze_bridge(self) -> None:
        """Freeze bridge weights — becomes exactly standard LoRA.

        In emanation mode, freezes the layer_proj (master_bridge is shared
        and should be frozen/unfrozen at the owner level).
        """
        if self.emanation:
            self.layer_proj.requires_grad_(False)
        else:
            self.bridge.requires_grad_(False)

    def unfreeze_bridge(self) -> None:
        """Unfreeze bridge weights for training.

        In emanation mode, unfreezes the layer_proj.
        """
        if self.emanation:
            self.layer_proj.requires_grad_(True)
        else:
            self.bridge.requires_grad_(True)

    def bridge_deviation(self) -> float:
        """Frobenius norm of (effective_bridge - I). Monitors divergence from init."""
        with torch.no_grad():
            b = self.effective_bridge
            eye = torch.eye(
                self.n_channels, device=b.device, dtype=b.dtype
            )
            return float(torch.norm(b - eye).item())

    def bridge_fiedler(self) -> float:
        """Fiedler value of current effective bridge matrix (diagnostic).

        Treats |bridge[i,j]| (i != j) as edge weights of a complete graph
        on n_channels nodes, then computes algebraic connectivity.
        """
        with torch.no_grad():
            B = self.effective_bridge.detach().cpu().numpy()
        n = B.shape[0]
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        weights = [float(abs(B[i, j])) for i, j in edges]
        return fiedler_value(n, edges, weights)

    def gate_temperature(self) -> float | None:
        """Current gate temperature, or None if static bridge."""
        if self.gate_proj is not None:
            return float(self._gate_temperature.item())
        return None

    def gate_sparsity(self) -> float | None:
        """Fraction of gate values below 0.1 (nearly off). Returns None if static."""
        if self.gate_proj is None:
            return None
        # Sample: compute gate on a zero input to see the default gate state
        with torch.no_grad():
            dev = self.lora_A.device
            zero = torch.zeros(1, self.in_features, device=dev)
            logits = self.gate_proj(zero)
            gate = torch.sigmoid(logits / self._gate_temperature)
            return float((gate < 0.1).float().mean().item())

    def extra_repr(self) -> str:
        s = (
            f"in={self.in_features}, out={self.out_features}, "
            f"rank={self.rank}, channels={self.n_channels}, "
            f"scaling={self.scaling:.4f}"
        )
        if self.emanation:
            s += ", emanation=True"
        if self.dynamic_bridge:
            s += f", dynamic=True, gate_T={float(self._gate_temperature):.2f}"
        return s
