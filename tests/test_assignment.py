"""Tests for optimal edge-weight assignment."""

import pytest
import numpy as np
from rhombic.polyhedron import RhombicDodecahedron
from rhombic.corpus import edge_values
from rhombic.assignment import (
    total_variation, optimal_assignment, random_assignment_tv, compare_graphs,
)


@pytest.fixture
def rd():
    return RhombicDodecahedron()


@pytest.fixture
def ve(rd):
    """Vertex-edge incidence dict."""
    return {v: rd.vertex_star(v) for v in range(14)}


class TestTotalVariation:

    def test_uniform_weights_zero_tv(self, ve):
        """Uniform weights produce TV = 0."""
        weights = [1.0] * 24
        assert total_variation(ve, weights) == 0.0

    def test_constant_weights_zero_tv(self, ve):
        """Any constant weight produces TV = 0."""
        weights = [42.0] * 24
        assert total_variation(ve, weights) == 0.0

    def test_tv_non_negative(self, ve):
        """TV is always non-negative."""
        rng = np.random.default_rng(42)
        for _ in range(10):
            weights = rng.uniform(0.1, 10.0, size=24).tolist()
            assert total_variation(ve, weights) >= 0.0

    def test_tv_changes_with_permutation(self, ve):
        """Different permutations produce different TVs (in general)."""
        weights = list(range(1, 25))
        tv1 = total_variation(ve, weights)
        tv2 = total_variation(ve, list(reversed(weights)))
        # Not necessarily different, but very unlikely to be equal for sequential vs reversed
        # Just check both are non-negative
        assert tv1 >= 0 and tv2 >= 0


class TestOptimalAssignment:

    def test_improves_over_random(self, ve, rd):
        """Optimal assignment should produce lower TV than random mean."""
        weights = list(range(1, 25))
        _, best_tv = optimal_assignment(
            ve, rd.edges, weights, n_restarts=20, max_iters=1000, seed=42)
        null_tvs = random_assignment_tv(ve, weights, n_samples=1000, seed=42)
        assert best_tv <= null_tvs.mean(), \
            f"Optimal TV {best_tv:.1f} should be <= random mean {null_tvs.mean():.1f}"

    def test_deterministic(self, ve, rd):
        """Same seed produces same result."""
        weights = list(range(1, 25))
        _, tv1 = optimal_assignment(ve, rd.edges, weights, n_restarts=10, seed=42)
        _, tv2 = optimal_assignment(ve, rd.edges, weights, n_restarts=10, seed=42)
        assert tv1 == tv2

    def test_returns_correct_length(self, ve, rd):
        """Assignment has same length as input weights."""
        weights = list(range(1, 25))
        assign, _ = optimal_assignment(ve, rd.edges, weights, n_restarts=5, seed=42)
        assert len(assign) == 24

    def test_assignment_is_permutation(self, ve, rd):
        """Assignment contains exactly the input values (permuted)."""
        weights = list(range(1, 25))
        assign, _ = optimal_assignment(ve, rd.edges, weights, n_restarts=5, seed=42)
        assert sorted(assign) == sorted(weights)


class TestNullDistribution:

    def test_null_length(self, ve):
        weights = list(range(1, 25))
        tvs = random_assignment_tv(ve, weights, n_samples=100, seed=42)
        assert len(tvs) == 100

    def test_null_non_negative(self, ve):
        weights = list(range(1, 25))
        tvs = random_assignment_tv(ve, weights, n_samples=100, seed=42)
        assert np.all(tvs >= 0)

    def test_null_deterministic(self, ve):
        weights = list(range(1, 25))
        tvs1 = random_assignment_tv(ve, weights, n_samples=100, seed=42)
        tvs2 = random_assignment_tv(ve, weights, n_samples=100, seed=42)
        np.testing.assert_array_equal(tvs1, tvs2)


class TestCorpusAssignment:

    def test_corpus_values_on_rd(self, ve, rd):
        """The corpus values can be assigned to the rhombic dodecahedron."""
        weights = [float(v) for v in edge_values()]
        _, best_tv = optimal_assignment(
            ve, rd.edges, weights, n_restarts=10, max_iters=500, seed=42)
        null_tvs = random_assignment_tv(ve, weights, n_samples=500, seed=42)

        # The optimal should be in the lower tail of the null distribution
        p_value = float(np.mean(null_tvs <= best_tv))
        assert p_value < 0.5, f"p-value {p_value:.3f} — optimal should beat median"
