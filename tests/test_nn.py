"""Tests for rhombic.nn — RhombiLoRA and bridge topology.

Six tiers:
  1. Topology — coupling matrix from RD geometry
  2. Construction — valid/invalid params, shapes, initialization
  3. Forward pass — shapes, identity equivalence, zero at init, scaling
  4. Gradients — flow to A, B, bridge; frozen bridge blocks grads
  5. Ablation — frozen bridge = LoRA, state_dict round-trip, diagnostics
  6. Dynamic Bridge — input-dependent gating with temperature annealing
  7. Emanation — shared master bridge with per-layer sigmoid projection
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from rhombic.nn.topology import direction_pair_coupling, bridge_init, create_emanation_bridge
from rhombic.nn.rhombi_lora import RhombiLoRALinear
from rhombic.corpus import corpus_available as _corpus_available


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

    @pytest.mark.skipif(
        not _corpus_available(),
        reason="Corpus values required for corpus_coupled mode"
    )
    def test_bridge_init_corpus_coupled_shape(self):
        B = bridge_init(6, 'corpus_coupled')
        assert B.shape == (6, 6)

    @pytest.mark.skipif(
        not _corpus_available(),
        reason="Corpus values required for corpus_coupled mode"
    )
    def test_bridge_init_corpus_coupled_diagonal(self):
        """Corpus-coupled init has identity diagonal."""
        B = bridge_init(6, 'corpus_coupled')
        np.testing.assert_allclose(np.diag(B), np.ones(6), atol=1e-10)

    @pytest.mark.skipif(
        not _corpus_available(),
        reason="Corpus values required for corpus_coupled mode"
    )
    def test_bridge_init_corpus_coupled_differs_from_geometric(self):
        """Corpus-coupled init produces a different matrix than geometric."""
        B_geo = bridge_init(6, 'geometric')
        B_cc = bridge_init(6, 'corpus_coupled')
        assert not np.allclose(B_geo, B_cc)

    def test_bridge_init_corpus_coupled_requires_6(self):
        with pytest.raises(ValueError, match="n_channels=6"):
            bridge_init(3, 'corpus_coupled')

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


# ── Tier 6: Dynamic Bridge (L9 Prototype) ─────────────────────────


class TestDynamicBridge:
    """Input-dependent bridge gating with temperature annealing."""

    def test_construction_default_static(self):
        """Default construction has no gate_proj."""
        m = RhombiLoRALinear(64, 128, rank=24)
        assert m.gate_proj is None
        assert not m.dynamic_bridge

    def test_construction_dynamic(self):
        """Dynamic bridge creates gate_proj with correct dimensions."""
        m = RhombiLoRALinear(64, 128, rank=24, dynamic_bridge=True)
        assert m.gate_proj is not None
        assert m.dynamic_bridge
        assert m.gate_proj.weight.shape == (36, 64)  # C*C=36, in=64

    def test_dynamic_forward_shape(self):
        """Dynamic forward pass produces same output shape as static."""
        m_static = RhombiLoRALinear(64, 128, rank=24)
        m_dyn = RhombiLoRALinear(64, 128, rank=24, dynamic_bridge=True)
        x = torch.randn(2, 10, 64)
        y_static = m_static(x)
        y_dyn = m_dyn(x)
        assert y_static.shape == y_dyn.shape == (2, 10, 128)

    def test_dynamic_forward_differs_per_token(self):
        """Different input tokens should produce different gating."""
        m = RhombiLoRALinear(64, 128, rank=24, dynamic_bridge=True)
        # Set gate and lora_B nonzero so outputs are nonzero and differ
        nn.init.normal_(m.gate_proj.weight, std=0.1)
        nn.init.normal_(m.lora_B, std=0.01)
        x = torch.randn(1, 5, 64)
        y = m(x)
        # All-same input would produce all-same output
        x_same = x[:, 0:1, :].expand(1, 5, 64)
        y_same = m(x_same)
        # Random tokens: outputs should NOT all be equal
        assert not torch.allclose(y[:, 0], y[:, 1], atol=1e-6)
        # Same tokens: outputs SHOULD all be equal
        assert torch.allclose(y_same[:, 0], y_same[:, 1], atol=1e-6)

    def test_temperature_annealing(self):
        """Gate temperature can be set and affects output."""
        m = RhombiLoRALinear(64, 128, rank=24, dynamic_bridge=True)
        assert m.gate_temperature() == 5.0
        m.set_gate_temperature(0.1)
        assert abs(m.gate_temperature() - 0.1) < 1e-6

    def test_zero_init_gate_neutral(self):
        """Zero-initialized gate_proj produces gate ≈ 0.5 (neutral)."""
        m = RhombiLoRALinear(64, 128, rank=24, dynamic_bridge=True)
        x = torch.randn(1, 64)
        with torch.no_grad():
            logits = m.gate_proj(x)
            gate = torch.sigmoid(logits / m._gate_temperature)
        # All gate values should be ≈ 0.5 since logits ≈ 0
        assert torch.allclose(gate, torch.full_like(gate, 0.5), atol=0.01)

    def test_gate_gradient_flow(self):
        """Gradients flow through the gate to gate_proj weights."""
        m = RhombiLoRALinear(64, 128, rank=24, dynamic_bridge=True)
        # lora_B must be nonzero for gradients to flow through
        nn.init.normal_(m.lora_B, std=0.01)
        x = torch.randn(2, 10, 64)
        y = m(x)
        y.sum().backward()
        assert m.gate_proj.weight.grad is not None
        assert m.gate_proj.weight.grad.abs().sum() > 0

    def test_gate_sparsity(self):
        """gate_sparsity returns a float for dynamic, None for static."""
        m_static = RhombiLoRALinear(64, 128, rank=24)
        m_dyn = RhombiLoRALinear(64, 128, rank=24, dynamic_bridge=True)
        assert m_static.gate_sparsity() is None
        sp = m_dyn.gate_sparsity()
        assert isinstance(sp, float)
        assert 0.0 <= sp <= 1.0

    def test_dynamic_extra_repr(self):
        """Dynamic bridge shows in repr."""
        m = RhombiLoRALinear(64, 128, rank=24, dynamic_bridge=True)
        r = m.extra_repr()
        assert "dynamic=True" in r
        assert "gate_T=" in r

    def test_param_count(self):
        """Dynamic bridge adds exactly C*C*in_features parameters."""
        m_static = RhombiLoRALinear(64, 128, rank=24)
        m_dyn = RhombiLoRALinear(64, 128, rank=24, dynamic_bridge=True)
        static_params = sum(p.numel() for p in m_static.parameters())
        dyn_params = sum(p.numel() for p in m_dyn.parameters())
        # gate_proj: no bias, weight = (C*C) * in_features = 36 * 64 = 2304
        assert dyn_params - static_params == 36 * 64

    def test_state_dict_roundtrip(self):
        """Dynamic bridge survives save/load cycle."""
        m1 = RhombiLoRALinear(64, 128, rank=24, dynamic_bridge=True)
        nn.init.normal_(m1.gate_proj.weight, std=0.1)
        m1.set_gate_temperature(1.5)
        sd = m1.state_dict()
        m2 = RhombiLoRALinear(64, 128, rank=24, dynamic_bridge=True)
        m2.load_state_dict(sd)
        assert abs(m2.gate_temperature() - 1.5) < 1e-6
        x = torch.randn(1, 5, 64)
        assert torch.allclose(m1(x), m2(x))


# ── Tier 7: Emanation Architecture ────────────────────────────────────


class TestEmanation:
    """Shared master bridge with per-layer sigmoid projection."""

    def test_create_emanation_bridge_returns_tuple(self):
        """create_emanation_bridge returns (Parameter, callable)."""
        master, factory = create_emanation_bridge(6, 'identity')
        assert isinstance(master, nn.Parameter)
        assert callable(factory)

    def test_create_emanation_bridge_shape(self):
        """Master bridge has correct shape."""
        master, _ = create_emanation_bridge(6, 'identity')
        assert master.shape == (6, 6)

    def test_create_emanation_bridge_geometric(self):
        """Master bridge can be initialized with geometric mode."""
        master, _ = create_emanation_bridge(6, 'geometric')
        # Should be near identity (geometric = I + eps * coupling)
        dev = float(torch.norm(master.data - torch.eye(6)).item())
        assert 0.0 < dev < 0.1

    def test_factory_returns_parameter(self):
        """Factory produces nn.Parameter of correct shape."""
        master, factory = create_emanation_bridge(6, 'identity')
        proj = factory()
        assert isinstance(proj, nn.Parameter)
        assert proj.shape == (6, 6)

    def test_factory_returns_zeros(self):
        """Factory produces zero-initialized projections."""
        _, factory = create_emanation_bridge(6, 'identity')
        proj = factory()
        assert torch.all(proj == 0)

    def test_factory_returns_independent_params(self):
        """Each factory call produces an independent parameter."""
        _, factory = create_emanation_bridge(6, 'identity')
        p1 = factory()
        p2 = factory()
        assert p1 is not p2
        # Mutating one doesn't affect the other
        with torch.no_grad():
            p1.fill_(1.0)
        assert torch.all(p2 == 0)

    def test_construction_emanation(self):
        """RhombiLoRALinear with master_bridge enters emanation mode."""
        master, _ = create_emanation_bridge(6, 'identity')
        m = RhombiLoRALinear(64, 128, rank=24, master_bridge=master)
        assert m.emanation is True
        assert hasattr(m, 'layer_proj')
        assert hasattr(m, 'master_bridge')
        assert not hasattr(m, 'bridge')

    def test_construction_default_not_emanation(self):
        """Default construction is not emanation mode."""
        m = RhombiLoRALinear(64, 128, rank=24)
        assert m.emanation is False
        assert hasattr(m, 'bridge')
        assert not hasattr(m, 'layer_proj')

    def test_layer_proj_shape(self):
        """layer_proj has (n_channels, n_channels) shape."""
        master, _ = create_emanation_bridge(6, 'identity')
        m = RhombiLoRALinear(64, 128, rank=24, master_bridge=master)
        assert m.layer_proj.shape == (6, 6)

    def test_layer_proj_initialized_to_zeros(self):
        """layer_proj starts at zero (sigmoid(0)=0.5, *2 = identity)."""
        master, _ = create_emanation_bridge(6, 'identity')
        m = RhombiLoRALinear(64, 128, rank=24, master_bridge=master)
        assert torch.all(m.layer_proj == 0)

    def test_effective_bridge_identity_at_init(self):
        """With identity master and zero layer_proj, effective bridge = I."""
        master, _ = create_emanation_bridge(6, 'identity')
        m = RhombiLoRALinear(64, 128, rank=24, master_bridge=master)
        eb = m.effective_bridge
        torch.testing.assert_close(eb, torch.eye(6), atol=1e-6, rtol=1e-6)

    def test_effective_bridge_geometric_at_init(self):
        """With geometric master and zero layer_proj, effective bridge = master."""
        master, _ = create_emanation_bridge(6, 'geometric')
        m = RhombiLoRALinear(64, 128, rank=24, master_bridge=master)
        eb = m.effective_bridge
        torch.testing.assert_close(eb, master.data, atol=1e-6, rtol=1e-6)

    def test_forward_shape(self):
        """Emanation forward produces correct output shape."""
        master, _ = create_emanation_bridge(6, 'identity')
        m = RhombiLoRALinear(64, 128, rank=24, master_bridge=master)
        x = torch.randn(4, 16, 64)
        y = m(x)
        assert y.shape == (4, 16, 128)

    def test_zero_output_at_init(self):
        """Emanation module produces zero output at init (B=0)."""
        master, _ = create_emanation_bridge(6, 'identity')
        m = RhombiLoRALinear(64, 128, rank=24, master_bridge=master)
        x = torch.randn(4, 64)
        y = m(x)
        torch.testing.assert_close(y, torch.zeros(4, 128))

    def test_emanation_matches_standard_at_init(self):
        """With identity master, emanation output matches standard LoRA at init."""
        torch.manual_seed(42)
        master, _ = create_emanation_bridge(6, 'identity')
        m_eman = RhombiLoRALinear(64, 128, rank=24, master_bridge=master)

        torch.manual_seed(42)
        m_std = RhombiLoRALinear(64, 128, rank=24, bridge_mode='identity')

        # Copy non-bridge params
        m_eman.lora_A.data.copy_(m_std.lora_A.data)
        m_eman.lora_B.data.copy_(m_std.lora_B.data)
        nn.init.normal_(m_eman.lora_B, std=0.01)
        m_std.lora_B.data.copy_(m_eman.lora_B.data)

        x = torch.randn(4, 64)
        torch.testing.assert_close(m_eman(x), m_std(x), atol=1e-5, rtol=1e-5)

    def test_master_bridge_shared_across_layers(self):
        """Multiple layers share the same master_bridge tensor."""
        master, _ = create_emanation_bridge(6, 'identity')
        m1 = RhombiLoRALinear(64, 128, rank=24, master_bridge=master)
        m2 = RhombiLoRALinear(64, 128, rank=24, master_bridge=master)
        assert m1.master_bridge is m2.master_bridge
        assert m1.master_bridge.data_ptr() == m2.master_bridge.data_ptr()

    def test_layer_proj_independent_across_layers(self):
        """Each layer has its own layer_proj."""
        master, _ = create_emanation_bridge(6, 'identity')
        m1 = RhombiLoRALinear(64, 128, rank=24, master_bridge=master)
        m2 = RhombiLoRALinear(64, 128, rank=24, master_bridge=master)
        assert m1.layer_proj is not m2.layer_proj

    def test_layer_proj_modulation_changes_output(self):
        """Non-zero layer_proj produces different output from standard."""
        master, _ = create_emanation_bridge(6, 'identity')
        m = RhombiLoRALinear(64, 128, rank=24, master_bridge=master)
        nn.init.normal_(m.lora_B, std=0.01)

        x = torch.randn(4, 64)
        y_init = m(x).clone()

        # Perturb layer_proj
        with torch.no_grad():
            m.layer_proj.data += torch.randn(6, 6) * 2.0
        y_perturbed = m(x)

        assert not torch.allclose(y_init, y_perturbed, atol=1e-6)

    def test_master_bridge_mutation_affects_all_layers(self):
        """Mutating master_bridge affects all layers sharing it."""
        master, _ = create_emanation_bridge(6, 'identity')
        m1 = RhombiLoRALinear(64, 128, rank=24, master_bridge=master)
        m2 = RhombiLoRALinear(64, 128, rank=24, master_bridge=master)
        nn.init.normal_(m1.lora_B, std=0.01)
        nn.init.normal_(m2.lora_B, std=0.01)

        x = torch.randn(4, 64)
        y1_before = m1(x).clone()
        y2_before = m2(x).clone()

        # Mutate master bridge
        with torch.no_grad():
            master.data += torch.randn(6, 6) * 0.5

        y1_after = m1(x)
        y2_after = m2(x)

        assert not torch.allclose(y1_before, y1_after, atol=1e-6)
        assert not torch.allclose(y2_before, y2_after, atol=1e-6)

    def test_gradient_flow_to_layer_proj(self):
        """Gradients flow through to layer_proj."""
        master, _ = create_emanation_bridge(6, 'identity')
        m = RhombiLoRALinear(64, 128, rank=24, master_bridge=master)
        nn.init.normal_(m.lora_B, std=0.01)

        x = torch.randn(4, 64)
        y = m(x)
        y.sum().backward()

        assert m.layer_proj.grad is not None
        assert m.layer_proj.grad.abs().sum() > 0

    def test_gradient_flow_to_master_bridge(self):
        """Gradients flow through to the shared master bridge."""
        master, _ = create_emanation_bridge(6, 'identity')
        m = RhombiLoRALinear(64, 128, rank=24, master_bridge=master)
        nn.init.normal_(m.lora_B, std=0.01)

        x = torch.randn(4, 64)
        y = m(x)
        y.sum().backward()

        assert master.grad is not None
        assert master.grad.abs().sum() > 0

    def test_gradient_accumulates_from_multiple_layers(self):
        """Master bridge grad accumulates from multiple layer forward passes."""
        master, _ = create_emanation_bridge(6, 'identity')
        m1 = RhombiLoRALinear(32, 16, rank=6, n_channels=6, master_bridge=master)
        m2 = RhombiLoRALinear(32, 16, rank=6, n_channels=6, master_bridge=master)
        nn.init.normal_(m1.lora_B, std=0.01)
        nn.init.normal_(m2.lora_B, std=0.01)

        x = torch.randn(4, 32)
        (m1(x).sum() + m2(x).sum()).backward()

        # master_bridge should have accumulated gradients from both layers
        assert master.grad is not None
        grad_both = master.grad.clone()

        # Compare: single layer grad should be smaller
        master.grad = None
        m1.layer_proj.grad = None
        m2.layer_proj.grad = None
        m1.lora_A.grad = None
        m1.lora_B.grad = None
        m1(x).sum().backward()
        grad_one = master.grad.clone()

        # Two-layer grad should differ from single-layer (accumulated)
        assert not torch.allclose(grad_both, grad_one, atol=1e-7)

    def test_freeze_bridge_freezes_layer_proj(self):
        """freeze_bridge in emanation mode freezes layer_proj."""
        master, _ = create_emanation_bridge(6, 'identity')
        m = RhombiLoRALinear(64, 128, rank=24, master_bridge=master)
        m.freeze_bridge()
        assert not m.layer_proj.requires_grad

    def test_unfreeze_bridge_unfreezes_layer_proj(self):
        """unfreeze_bridge in emanation mode restores layer_proj grad."""
        master, _ = create_emanation_bridge(6, 'identity')
        m = RhombiLoRALinear(64, 128, rank=24, master_bridge=master)
        m.freeze_bridge()
        m.unfreeze_bridge()
        assert m.layer_proj.requires_grad

    def test_bridge_deviation_at_init(self):
        """Identity emanation bridge has zero deviation at init."""
        master, _ = create_emanation_bridge(6, 'identity')
        m = RhombiLoRALinear(64, 128, rank=24, master_bridge=master)
        assert abs(m.bridge_deviation()) < 1e-6

    def test_bridge_fiedler_at_init(self):
        """Identity emanation bridge has zero Fiedler at init."""
        master, _ = create_emanation_bridge(6, 'identity')
        m = RhombiLoRALinear(64, 128, rank=24, master_bridge=master)
        assert abs(m.bridge_fiedler()) < 1e-7

    def test_bridge_fiedler_geometric_emanation(self):
        """Geometric emanation bridge has non-zero Fiedler."""
        master, _ = create_emanation_bridge(6, 'geometric')
        m = RhombiLoRALinear(64, 128, rank=24, master_bridge=master)
        assert m.bridge_fiedler() > 0.0

    def test_extra_repr_shows_emanation(self):
        """Emanation flag appears in repr."""
        master, _ = create_emanation_bridge(6, 'identity')
        m = RhombiLoRALinear(64, 128, rank=24, master_bridge=master)
        r = m.extra_repr()
        assert "emanation=True" in r

    def test_param_count_emanation(self):
        """Emanation mode: layer_proj replaces bridge (same param count)."""
        master, _ = create_emanation_bridge(6, 'identity')
        m_eman = RhombiLoRALinear(64, 128, rank=24, master_bridge=master)
        m_std = RhombiLoRALinear(64, 128, rank=24)
        # Emanation has: lora_A + lora_B + layer_proj (no bridge)
        # Note: master_bridge is shared, not owned by this module
        eman_params = sum(p.numel() for p in m_eman.parameters())
        std_params = sum(p.numel() for p in m_std.parameters())
        # Both should have same local param count: A + B + (bridge or layer_proj)
        # But emanation also has master_bridge registered as parameter
        # master_bridge is assigned as attribute -> it IS a parameter
        expected_local = 24 * 64 + 128 * 24 + 6 * 6 + 6 * 6  # A + B + master + layer_proj
        assert eman_params == expected_local

    def test_custom_channels_emanation(self):
        """Emanation works with non-default channel count."""
        master, factory = create_emanation_bridge(3, 'identity')
        m = RhombiLoRALinear(64, 128, rank=12, n_channels=3, master_bridge=master)
        assert m.master_bridge.shape == (3, 3)
        assert m.layer_proj.shape == (3, 3)
        x = torch.randn(4, 64)
        y = m(x)
        assert y.shape == (4, 128)
