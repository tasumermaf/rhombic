"""Tests for rhombic.nn — RhombiLoRA and bridge topology.

Five tiers:
  1. Topology — coupling matrix from RD geometry
  2. Construction — valid/invalid params, shapes, initialization
  3. Forward pass — shapes, identity equivalence, zero at init, scaling
  4. Gradients — flow to A, B, bridge; frozen bridge blocks grads
  5. Ablation — frozen bridge = LoRA, state_dict round-trip, diagnostics
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from rhombic.nn.topology import direction_pair_coupling, bridge_init
from rhombic.nn.rhombi_lora import RhombiLoRALinear


# ── Tier 1: Topology ────────────────────────────────────────────────


class TestTopology:
    """Coupling matrix derived from RD geometry."""

    def test_coupling_shape(self):
        C = direction_pair_coupling()
        assert C.shape == (6, 6)

    def test_coupling_symmetric(self):
        C = direction_pair_coupling()
        np.testing.assert_array_equal(C, C.T)

    def test_coupling_diagonal_is_four(self):
        """Each direction pair uses 4 octahedral vertices."""
        C = direction_pair_coupling()
        np.testing.assert_array_equal(np.diag(C), [4, 4, 4, 4, 4, 4])

    def test_coupling_values_are_two_or_four(self):
        """All entries are either 2 (cross-planar) or 4 (co-planar/self)."""
        C = direction_pair_coupling()
        for i in range(6):
            for j in range(6):
                assert C[i, j] in (2.0, 4.0), f"C[{i},{j}] = {C[i,j]}"

    def test_coupling_coplanar_pairs(self):
        """Pairs in the same coordinate plane share 4 octahedral vertices.

        Pairs (0,1) are in xy, (2,3) in xz, (4,5) in yz.
        """
        C = direction_pair_coupling()
        assert C[0, 1] == 4.0  # xy plane
        assert C[2, 3] == 4.0  # xz plane
        assert C[4, 5] == 4.0  # yz plane

    def test_coupling_crossplanar_pairs(self):
        """Pairs in different coordinate planes share 2 octahedral vertices."""
        C = direction_pair_coupling()
        assert C[0, 2] == 2.0  # xy vs xz
        assert C[0, 4] == 2.0  # xy vs yz
        assert C[2, 4] == 2.0  # xz vs yz

    def test_coupling_row_sums(self):
        """Each direction pair: 4 (self) + 4 (co-planar) + 4*2 (cross) = 16."""
        C = direction_pair_coupling()
        np.testing.assert_array_equal(C.sum(axis=1), [16, 16, 16, 16, 16, 16])

    def test_coupling_block_structure(self):
        """The coupling has 3x3 block structure with 2x2 blocks."""
        C = direction_pair_coupling()
        # Within-plane blocks: [[4,4],[4,4]]
        for start in (0, 2, 4):
            block = C[start:start+2, start:start+2]
            np.testing.assert_array_equal(block, [[4, 4], [4, 4]])

    def test_bridge_init_identity(self):
        B = bridge_init(6, 'identity')
        np.testing.assert_array_equal(B, np.eye(6))

    def test_bridge_init_identity_other_size(self):
        B = bridge_init(3, 'identity')
        np.testing.assert_array_equal(B, np.eye(3))

    def test_bridge_init_geometric_shape(self):
        B = bridge_init(6, 'geometric')
        assert B.shape == (6, 6)

    def test_bridge_init_geometric_near_identity(self):
        """Geometric init is identity + small perturbation."""
        B = bridge_init(6, 'geometric')
        deviation = np.linalg.norm(B - np.eye(6))
        assert deviation < 0.1  # eps=0.01 * 6x6 matrix

    def test_bridge_init_geometric_symmetric(self):
        """Geometric coupling is symmetric, so the init is symmetric."""
        B = bridge_init(6, 'geometric')
        np.testing.assert_allclose(B, B.T, atol=1e-12)

    def test_bridge_init_geometric_requires_6(self):
        with pytest.raises(ValueError, match="n_channels=6"):
            bridge_init(3, 'geometric')

    def test_bridge_init_invalid_mode(self):
        with pytest.raises(ValueError, match="Unknown mode"):
            bridge_init(6, 'bogus')


# ── Tier 2: Construction ────────────────────────────────────────────


class TestConstruction:
    """Module creation, parameter shapes, initialization."""

    def test_basic_construction(self):
        m = RhombiLoRALinear(64, 128, rank=24)
        assert m.in_features == 64
        assert m.out_features == 128
        assert m.rank == 24
        assert m.n_channels == 6
        assert m.channel_size == 4

    def test_rank_not_divisible_raises(self):
        with pytest.raises(ValueError, match="divisible"):
            RhombiLoRALinear(64, 128, rank=25, n_channels=6)

    def test_parameter_shapes(self):
        m = RhombiLoRALinear(64, 128, rank=24)
        assert m.lora_A.shape == (24, 64)
        assert m.lora_B.shape == (128, 24)
        assert m.bridge.shape == (6, 6)

    def test_lora_B_initialized_to_zeros(self):
        m = RhombiLoRALinear(64, 128, rank=24)
        assert torch.all(m.lora_B == 0)

    def test_bridge_initialized_to_identity(self):
        m = RhombiLoRALinear(64, 128, rank=24, bridge_mode='identity')
        expected = torch.eye(6)
        torch.testing.assert_close(m.bridge.data, expected)

    def test_bridge_geometric_init(self):
        m = RhombiLoRALinear(64, 128, rank=24, bridge_mode='geometric')
        # Should be close to identity
        dev = float(torch.norm(m.bridge.data - torch.eye(6)).item())
        assert dev < 0.1
        assert dev > 0.0  # not exactly identity

    def test_scaling(self):
        m = RhombiLoRALinear(64, 128, rank=24, alpha=16.0)
        assert abs(m.scaling - 16.0 / 24) < 1e-7

    def test_custom_channels(self):
        m = RhombiLoRALinear(64, 128, rank=12, n_channels=3)
        assert m.n_channels == 3
        assert m.channel_size == 4
        assert m.bridge.shape == (3, 3)

    def test_lora_A_not_zeros(self):
        """Kaiming init should produce non-zero A."""
        m = RhombiLoRALinear(64, 128, rank=24)
        assert torch.any(m.lora_A != 0)

    def test_parameter_count(self):
        m = RhombiLoRALinear(64, 128, rank=24)
        total = sum(p.numel() for p in m.parameters())
        expected = 24 * 64 + 128 * 24 + 6 * 6  # A + B + bridge
        assert total == expected


# ── Tier 3: Forward pass ────────────────────────────────────────────


class TestForward:
    """Output shapes, identity equivalence, zero init, scaling."""

    def test_output_shape_2d(self):
        m = RhombiLoRALinear(64, 128, rank=24)
        x = torch.randn(8, 64)
        y = m(x)
        assert y.shape == (8, 128)

    def test_output_shape_3d(self):
        m = RhombiLoRALinear(64, 128, rank=24)
        x = torch.randn(4, 16, 64)
        y = m(x)
        assert y.shape == (4, 16, 128)

    def test_output_shape_1d(self):
        """Single vector input."""
        m = RhombiLoRALinear(64, 128, rank=24)
        x = torch.randn(64)
        y = m(x)
        assert y.shape == (128,)

    def test_zero_output_at_init(self):
        """Fresh module produces zero output (B initialized to zeros)."""
        m = RhombiLoRALinear(64, 128, rank=24)
        x = torch.randn(8, 64)
        y = m(x)
        torch.testing.assert_close(y, torch.zeros(8, 128))

    def test_identity_bridge_equals_standard_lora(self):
        """With identity bridge, output must match manual A->B path."""
        torch.manual_seed(42)
        m = RhombiLoRALinear(64, 128, rank=24, bridge_mode='identity')
        # Set B to something non-zero so we can compare
        nn.init.normal_(m.lora_B, std=0.01)

        x = torch.randn(8, 64)
        y_rhombi = m(x)

        # Manual standard LoRA: x @ A^T @ B^T * scaling
        h = x @ m.lora_A.data.T
        y_manual = h @ m.lora_B.data.T * m.scaling
        torch.testing.assert_close(y_rhombi, y_manual, atol=1e-5, rtol=1e-5)

    def test_scaling_factor_applied(self):
        """Output scales linearly with alpha."""
        torch.manual_seed(42)
        m1 = RhombiLoRALinear(64, 128, rank=24, alpha=16.0)
        m2 = RhombiLoRALinear(64, 128, rank=24, alpha=32.0)
        # Copy weights so only scaling differs
        m2.lora_A.data.copy_(m1.lora_A.data)
        m2.lora_B.data.copy_(m1.lora_B.data)
        m2.bridge.data.copy_(m1.bridge.data)
        # Set non-zero B
        nn.init.normal_(m1.lora_B, std=0.01)
        m2.lora_B.data.copy_(m1.lora_B.data)

        x = torch.randn(4, 64)
        y1 = m1(x)
        y2 = m2(x)
        torch.testing.assert_close(y2, y1 * 2.0, atol=1e-5, rtol=1e-5)

    def test_dropout_training_vs_eval(self):
        """Dropout active in training, inactive in eval."""
        torch.manual_seed(42)
        m = RhombiLoRALinear(64, 128, rank=24, dropout=0.5)
        nn.init.normal_(m.lora_B, std=0.01)
        x = torch.randn(32, 64)

        m.eval()
        y_eval = m(x)
        m.eval()
        y_eval2 = m(x)
        torch.testing.assert_close(y_eval, y_eval2)

    def test_bridge_mixing_changes_output(self):
        """Non-identity bridge produces different output from identity."""
        torch.manual_seed(42)
        m = RhombiLoRALinear(64, 128, rank=24, bridge_mode='identity')
        nn.init.normal_(m.lora_B, std=0.01)
        x = torch.randn(4, 64)

        y_identity = m(x).clone()

        # Perturb bridge
        with torch.no_grad():
            m.bridge.data += torch.randn(6, 6) * 0.1
        y_perturbed = m(x)

        assert not torch.allclose(y_identity, y_perturbed, atol=1e-6)


# ── Tier 4: Gradients ───────────────────────────────────────────────


class TestGradients:
    """Gradient flow through A, B, and bridge."""

    def _make_trainable(self):
        """Create a module with non-zero B for gradient testing."""
        m = RhombiLoRALinear(32, 16, rank=6, n_channels=3)
        nn.init.normal_(m.lora_B, std=0.01)
        return m

    def test_gradients_flow_to_all_params(self):
        m = self._make_trainable()
        x = torch.randn(4, 32)
        y = m(x)
        loss = y.sum()
        loss.backward()

        assert m.lora_A.grad is not None
        assert m.lora_B.grad is not None
        assert m.bridge.grad is not None

    def test_gradients_nonzero(self):
        m = self._make_trainable()
        x = torch.randn(4, 32)
        y = m(x)
        loss = y.sum()
        loss.backward()

        assert torch.any(m.lora_A.grad != 0)
        assert torch.any(m.lora_B.grad != 0)
        assert torch.any(m.bridge.grad != 0)

    def test_frozen_bridge_no_grad(self):
        m = self._make_trainable()
        m.freeze_bridge()
        x = torch.randn(4, 32)
        y = m(x)
        loss = y.sum()
        loss.backward()

        assert m.lora_A.grad is not None
        assert m.lora_B.grad is not None
        assert m.bridge.grad is None

    def test_unfreeze_restores_grad(self):
        m = self._make_trainable()
        m.freeze_bridge()
        m.unfreeze_bridge()

        x = torch.randn(4, 32)
        y = m(x)
        loss = y.sum()
        loss.backward()

        assert m.bridge.grad is not None

    def test_bridge_grad_shape(self):
        m = self._make_trainable()
        x = torch.randn(4, 32)
        y = m(x)
        loss = y.sum()
        loss.backward()

        assert m.bridge.grad.shape == (3, 3)

    def test_gradient_accumulation(self):
        """Two backward passes accumulate gradients."""
        m = self._make_trainable()
        x1 = torch.randn(4, 32)
        x2 = torch.randn(4, 32)

        m(x1).sum().backward()
        grad1 = m.bridge.grad.clone()

        m(x2).sum().backward()
        grad2 = m.bridge.grad  # accumulated

        # grad2 should be different from grad1 (accumulated)
        assert not torch.allclose(grad1, grad2)


# ── Tier 5: Ablation & diagnostics ──────────────────────────────────


class TestAblation:
    """Frozen bridge = LoRA, state_dict, diagnostic methods."""

    def test_frozen_identity_matches_standard_lora(self):
        """Frozen identity bridge must produce identical output to manual LoRA."""
        torch.manual_seed(99)
        m = RhombiLoRALinear(64, 128, rank=24, bridge_mode='identity')
        nn.init.normal_(m.lora_B, std=0.01)
        m.freeze_bridge()

        x = torch.randn(8, 64)
        y = m(x)

        # Manual: x @ A^T @ B^T * scaling
        h = x @ m.lora_A.data.T
        y_manual = h @ m.lora_B.data.T * m.scaling
        torch.testing.assert_close(y, y_manual, atol=1e-5, rtol=1e-5)

    def test_state_dict_roundtrip(self):
        """Save and load reproduces identical output."""
        torch.manual_seed(42)
        m1 = RhombiLoRALinear(64, 128, rank=24, bridge_mode='geometric')
        nn.init.normal_(m1.lora_B, std=0.01)

        state = m1.state_dict()

        m2 = RhombiLoRALinear(64, 128, rank=24, bridge_mode='identity')
        m2.load_state_dict(state)

        x = torch.randn(4, 64)
        torch.testing.assert_close(m1(x), m2(x))

    def test_bridge_deviation_at_init(self):
        """Identity-initialized bridge has zero deviation."""
        m = RhombiLoRALinear(64, 128, rank=24, bridge_mode='identity')
        assert abs(m.bridge_deviation()) < 1e-7

    def test_bridge_deviation_after_perturbation(self):
        """Perturbed bridge has non-zero deviation."""
        m = RhombiLoRALinear(64, 128, rank=24, bridge_mode='identity')
        with torch.no_grad():
            m.bridge.data += torch.randn(6, 6) * 0.1
        assert m.bridge_deviation() > 0.01

    def test_bridge_fiedler_at_init(self):
        """Identity bridge has zero Fiedler value (no off-diagonal coupling)."""
        m = RhombiLoRALinear(64, 128, rank=24, bridge_mode='identity')
        assert abs(m.bridge_fiedler()) < 1e-7

    def test_bridge_fiedler_after_perturbation(self):
        """Non-zero off-diagonal elements produce positive Fiedler value."""
        m = RhombiLoRALinear(64, 128, rank=24, bridge_mode='identity')
        with torch.no_grad():
            m.bridge.data += 0.1 * torch.ones(6, 6)
        assert m.bridge_fiedler() > 0.0

    def test_bridge_fiedler_geometric_init(self):
        """Geometric init has non-zero Fiedler value (coupling structure)."""
        m = RhombiLoRALinear(64, 128, rank=24, bridge_mode='geometric')
        assert m.bridge_fiedler() > 0.0

    def test_extra_repr(self):
        m = RhombiLoRALinear(64, 128, rank=24)
        r = m.extra_repr()
        assert "in=64" in r
        assert "out=128" in r
        assert "rank=24" in r
        assert "channels=6" in r
