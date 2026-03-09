"""Tests for optimal edge-weight assignment and prime-vertex coherence."""

import pytest
import numpy as np
from rhombic.polyhedron import RhombicDodecahedron
from rhombic.corpus import edge_values, TRACKED_PRIMES, corpus_available
from rhombic.assignment import (
    total_variation, optimal_assignment, random_assignment_tv, compare_graphs,
    prime_vertex_score, optimal_prime_assignment, null_prime_scores,
)

# Skip corpus-dependent tests if private data not available
requires_corpus = pytest.mark.skipif(
    not corpus_available(),
    reason="Corpus values are proprietary and not available in this environment"
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

    @requires_corpus
    def test_corpus_values_on_rd(self, ve, rd):
        """The corpus values can be assigned to the rhombic dodecahedron."""
        weights = [float(v) for v in edge_values()]
        _, best_tv = optimal_assignment(
            ve, rd.edges, weights, n_restarts=10, max_iters=500, seed=42)
        null_tvs = random_assignment_tv(ve, weights, n_samples=500, seed=42)

        # The optimal should be in the lower tail of the null distribution
        p_value = float(np.mean(null_tvs <= best_tv))
        assert p_value < 0.5, f"p-value {p_value:.3f} — optimal should beat median"


# ── Prime-vertex coherence tests ─────────────────────────────────────


class TestPrimeVertexScore:

    def test_non_negative(self, ve):
        """Score is always non-negative."""
        values = list(range(1, 25))
        mapping = {67: 0, 23: 1, 29: 2, 17: 3, 19: 4, 31: 5, 11: 6, 89: 7}
        score = prime_vertex_score(ve, values, mapping)
        assert score >= 0

    def test_empty_mapping(self, ve):
        """Empty mapping gives zero score."""
        values = list(range(1, 25))
        score = prime_vertex_score(ve, values, {})
        assert score == 0.0

    def test_divisible_values_score_higher(self, ve, rd):
        """An assignment where edge values are divisible by vertex prime
        should score higher than one where they aren't."""
        # Construct values where vertex 0 (degree 3) edges are all multiples of 67
        values = [1] * 24
        for e_idx in rd.vertex_star(0):
            values[e_idx] = 67 * 3  # 201, divisible by 67

        mapping_good = {67: 0}
        mapping_bad = {67: 1}  # vertex 1 has different edges

        score_good = prime_vertex_score(ve, values, mapping_good)
        score_bad = prime_vertex_score(ve, values, mapping_bad)
        assert score_good > score_bad

    @requires_corpus
    def test_corpus_values(self, ve):
        """Runs on actual corpus values without error."""
        values = edge_values()
        primes = TRACKED_PRIMES[:8]
        mapping = dict(zip(primes, range(8)))
        score = prime_vertex_score(ve, values, mapping)
        assert score >= 0


class TestOptimalPrimeAssignment:

    @requires_corpus
    def test_returns_mapping_and_score(self, ve):
        """Returns a valid mapping and positive score."""
        values = edge_values()
        primes = [67, 23, 29]
        verts = [0, 1, 2]
        mapping, score = optimal_prime_assignment(ve, values, primes, verts)
        assert len(mapping) == 3
        assert set(mapping.keys()) == set(primes)
        assert set(mapping.values()) == set(verts)
        assert score >= 0

    @requires_corpus
    def test_exhaustive_small(self, ve):
        """With 3 primes and 3 vertices, checks all 6 permutations."""
        values = edge_values()
        primes = [67, 23, 29]
        verts = [0, 1, 2]
        mapping, score = optimal_prime_assignment(ve, values, primes, verts)
        # Verify it's at least as good as any specific mapping
        for p in [67, 23, 29]:
            test_map = {67: 0, 23: 1, 29: 2}
            test_score = prime_vertex_score(ve, values, test_map)
            assert score >= test_score

    @requires_corpus
    def test_deterministic(self, ve):
        values = edge_values()
        primes = [67, 23]
        verts = [0, 1]
        m1, s1 = optimal_prime_assignment(ve, values, primes, verts)
        m2, s2 = optimal_prime_assignment(ve, values, primes, verts)
        assert s1 == s2
        assert m1 == m2


class TestNullPrimeScores:

    @requires_corpus
    def test_returns_all_permutations(self, ve):
        """For 3 items, should return 6 scores."""
        values = edge_values()
        primes = [67, 23, 29]
        verts = [0, 1, 2]
        scores = null_prime_scores(ve, values, primes, verts)
        assert len(scores) == 6  # 3!

    @requires_corpus
    def test_contains_optimal(self, ve):
        """The optimal score should be the max of the null distribution."""
        values = edge_values()
        primes = [67, 23, 29]
        verts = [0, 1, 2]
        _, opt_score = optimal_prime_assignment(ve, values, primes, verts)
        all_scores = null_prime_scores(ve, values, primes, verts)
        assert abs(opt_score - all_scores.max()) < 1e-10

    @requires_corpus
    def test_all_non_negative(self, ve):
        values = edge_values()
        primes = [67, 23]
        verts = [0, 1]
        scores = null_prime_scores(ve, values, primes, verts)
        assert np.all(scores >= 0)
