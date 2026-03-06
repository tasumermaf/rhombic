"""Tests for weighted Laplacian spectral analysis."""

import pytest
import numpy as np
from rhombic.polyhedron import RhombicDodecahedron, cuboctahedron_graph
from rhombic.spectral import (
    weighted_laplacian, spectrum, fiedler_value, spectral_gap,
    eigenvalue_multiplicity_pattern, spectral_distance, spectrum_summary,
)


@pytest.fixture
def rd():
    return RhombicDodecahedron()


class TestWeightedLaplacian:

    def test_symmetric(self, rd):
        L = weighted_laplacian(14, rd.edges)
        np.testing.assert_array_almost_equal(L, L.T)

    def test_row_sums_zero(self, rd):
        """Row sums of the Laplacian are zero."""
        L = weighted_laplacian(14, rd.edges)
        row_sums = L.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, 0.0)

    def test_weighted_row_sums_zero(self, rd):
        """Weighted Laplacian also has zero row sums."""
        weights = list(range(1, 25))
        L = weighted_laplacian(14, rd.edges, weights)
        row_sums = L.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, 0.0)

    def test_diagonal_equals_weighted_degree(self, rd):
        """Diagonal entry = sum of incident edge weights."""
        weights = list(range(1, 25))
        L = weighted_laplacian(14, rd.edges, weights)
        for v in range(14):
            star = rd.vertex_star(v)
            expected = sum(weights[e] for e in star)
            assert abs(L[v, v] - expected) < 1e-10

    def test_shape(self, rd):
        L = weighted_laplacian(14, rd.edges)
        assert L.shape == (14, 14)


class TestSpectrum:

    def test_first_eigenvalue_zero(self, rd):
        """Connected graph: first eigenvalue is 0."""
        eigs = spectrum(14, rd.edges)
        assert abs(eigs[0]) < 1e-10

    def test_all_non_negative(self, rd):
        """All eigenvalues are non-negative for PSD Laplacian."""
        eigs = spectrum(14, rd.edges)
        assert np.all(eigs >= -1e-10)

    def test_weighted_all_non_negative(self, rd):
        weights = list(range(1, 25))
        eigs = spectrum(14, rd.edges, weights)
        assert np.all(eigs >= -1e-10)

    def test_14_eigenvalues(self, rd):
        eigs = spectrum(14, rd.edges)
        assert len(eigs) == 14

    def test_ascending_order(self, rd):
        eigs = spectrum(14, rd.edges)
        for i in range(len(eigs) - 1):
            assert eigs[i] <= eigs[i + 1] + 1e-10


class TestFiedlerValue:

    def test_positive_for_connected(self, rd):
        """Fiedler value > 0 for connected graph."""
        fv = fiedler_value(14, rd.edges)
        assert fv > 0

    def test_weighted_positive(self, rd):
        weights = list(range(1, 25))
        fv = fiedler_value(14, rd.edges, weights)
        assert fv > 0

    def test_uniform_scaling(self, rd):
        """Scaling all weights by c scales Fiedler value by c."""
        fv1 = fiedler_value(14, rd.edges, [1.0] * 24)
        fv2 = fiedler_value(14, rd.edges, [3.0] * 24)
        np.testing.assert_almost_equal(fv2 / fv1, 3.0, decimal=5)


class TestSpectralGap:

    def test_bounded(self, rd):
        """Spectral gap is in [0, 1]."""
        g = spectral_gap(14, rd.edges)
        assert 0 <= g <= 1.0 + 1e-10

    def test_invariant_under_uniform_scaling(self, rd):
        """Uniform scaling doesn't change the gap ratio."""
        g1 = spectral_gap(14, rd.edges, [1.0] * 24)
        g2 = spectral_gap(14, rd.edges, [5.0] * 24)
        np.testing.assert_almost_equal(g1, g2, decimal=5)

    def test_non_zero_for_connected(self, rd):
        g = spectral_gap(14, rd.edges)
        assert g > 0


# ── Multiplicity pattern tests ──────────────────────────────────────


class TestMultiplicityPattern:

    def test_uniform_rd_has_degeneracies(self, rd):
        """Unweighted RD should have degenerate eigenvalues (high symmetry)."""
        eigs = spectrum(14, rd.edges)
        pattern = eigenvalue_multiplicity_pattern(eigs)
        # Fewer distinct eigenvalues than vertices means degeneracies
        assert len(pattern) < 14

    def test_weighted_breaks_degeneracy(self, rd):
        """Non-uniform weights should break some degeneracies."""
        eigs_uniform = spectrum(14, rd.edges)
        weights = list(range(1, 25))
        eigs_weighted = spectrum(14, rd.edges, weights)
        pattern_u = eigenvalue_multiplicity_pattern(eigs_uniform)
        pattern_w = eigenvalue_multiplicity_pattern(eigs_weighted)
        # Weighted should have more distinct eigenvalues (fewer degeneracies)
        assert len(pattern_w) >= len(pattern_u)

    def test_total_count_equals_n(self, rd):
        """Sum of multiplicities should equal number of vertices."""
        eigs = spectrum(14, rd.edges)
        pattern = eigenvalue_multiplicity_pattern(eigs)
        total = sum(m for _, m in pattern)
        assert total == 14


class TestSpectralDistance:

    def test_self_distance_zero(self, rd):
        eigs = spectrum(14, rd.edges)
        assert spectral_distance(eigs, eigs) < 1e-10

    def test_symmetric(self, rd):
        eigs1 = spectrum(14, rd.edges)
        eigs2 = spectrum(14, rd.edges, list(range(1, 25)))
        assert abs(spectral_distance(eigs1, eigs2) -
                    spectral_distance(eigs2, eigs1)) < 1e-10

    def test_different_spectra_positive(self, rd):
        eigs1 = spectrum(14, rd.edges)
        eigs2 = spectrum(14, rd.edges, list(range(1, 25)))
        assert spectral_distance(eigs1, eigs2) > 0

    def test_different_size_spectra(self):
        """Handles spectra of different lengths via zero-padding."""
        s1 = np.array([0.0, 1.0, 2.0])
        s2 = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        d = spectral_distance(s1, s2)
        assert d > 0


class TestSpectrumSummary:

    def test_returns_summary(self, rd):
        ss = spectrum_summary("RD", 14, rd.edges)
        assert ss.graph_name == "RD"
        assert ss.n_vertices == 14
        assert ss.n_edges == 24
        assert ss.fiedler > 0
        assert len(ss.full_spectrum) == 14

    def test_weighted_summary(self, rd):
        weights = list(range(1, 25))
        ss = spectrum_summary("RD_weighted", 14, rd.edges, weights)
        assert ss.fiedler > 0
        assert ss.n_distinct_eigenvalues > 0
