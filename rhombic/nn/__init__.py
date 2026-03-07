"""rhombic.nn — Neural network modules with FCC lattice topology.

Requires PyTorch. Install with: pip install rhombic[nn]
"""

try:
    import torch  # noqa: F401
except ImportError:
    raise ImportError(
        "rhombic.nn requires PyTorch. Install with: pip install rhombic[nn]"
    )

from rhombic.nn.rhombi_lora import RhombiLoRALinear
from rhombic.nn.topology import direction_pair_coupling, bridge_init

__all__ = ["RhombiLoRALinear", "direction_pair_coupling", "bridge_init"]
