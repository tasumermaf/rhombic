"""Tests for the isopsephy corpus data module."""

import pytest
from rhombic.corpus import (
    TRUMP_VALUES, NAMES_OF_POWER, TRACKED_PRIMES,
    edge_values, prime_factors, prime_factor_set, shared_factors,
    prime_membership, weight_distributions, corpus_stats,
    CANONICAL_EDGE_ORDER,
)


class TestDataIntegrity:
    """Verify the corpus data is complete and correct."""

    def test_24_trump_values(self):
        assert len(TRUMP_VALUES) == 24

    def test_14_names_of_power(self):
        assert len(NAMES_OF_POWER) == 14

    def test_8_tracked_primes(self):
        assert len(TRACKED_PRIMES) == 8

    def test_canonical_order_length(self):
        assert len(CANONICAL_EDGE_ORDER) == 24

    def test_edge_values_length(self):
        assert len(edge_values()) == 24

    def test_all_values_positive(self):
        for v in edge_values():
            assert v > 0


class TestKnownValues:
    """Spot-check verified values from the corpus."""

    def test_geareij(self):
        """Card 21: GEAREIJ = 134 = 2 * 67."""
        assert TRUMP_VALUES[21] == 134

    def test_bral_donace_identity(self):
        """BRAL (Card 1 component) = DONACE (Card 16) = 133 = 7 * 19."""
        assert TRUMP_VALUES[16] == 133

    def test_maat(self):
        """Card 8: MA∀T standard = 342."""
        assert TRUMP_VALUES[8] == 342

    def test_dajibaa(self):
        """Card 15: DAJIBA∀ = 29 (prime)."""
        assert TRUMP_VALUES[15] == 29

    def test_belial(self):
        """Transit: BELIAL = 78 = T(12)."""
        assert TRUMP_VALUES["T"] == 78

    def test_rirajla(self):
        """Grail: RIRAJLA = 252."""
        assert TRUMP_VALUES["G"] == 252

    def test_fool_combined(self):
        """Card 0: UAT ASETEDOJ combined = 1296 = 6^4."""
        assert TRUMP_VALUES[0] == 1296

    def test_samma(self):
        assert NAMES_OF_POWER["SAMMA"] == 282

    def test_vadusfadahm(self):
        assert NAMES_OF_POWER["VADUSFADAHM"] == 671

    def test_osiris(self):
        assert NAMES_OF_POWER["OSIRIS"] == 590


class TestFactorization:
    """Verify prime factorization functions."""

    def test_134_factors(self):
        """134 = 2 * 67."""
        assert prime_factors(134) == [2, 67]

    def test_133_factors(self):
        """133 = 7 * 19."""
        assert prime_factors(133) == [7, 19]

    def test_1296_factors(self):
        """1296 = 2^4 * 3^4 = 6^4."""
        f = prime_factors(1296)
        assert f.count(2) == 4
        assert f.count(3) == 4

    def test_29_is_prime(self):
        assert prime_factors(29) == [29]

    def test_prime_factor_set(self):
        assert prime_factor_set(342) == {2, 3, 19}

    def test_factors_of_1(self):
        assert prime_factors(1) == [1]


class TestSharedFactors:

    def test_shared_134_133(self):
        """134 = 2*67, 133 = 7*19: no shared factors."""
        assert shared_factors(134, 133) == set()

    def test_shared_342_134(self):
        """342 = 2*3^2*19, 134 = 2*67: share {2}."""
        assert shared_factors(342, 134) == {2}

    def test_self_shared(self):
        """A value shares all its own factors."""
        assert shared_factors(342, 342) == {2, 3, 19}


class TestPrimeMembership:

    def test_67_divides_134(self):
        assert prime_membership(134, 67) is True

    def test_67_not_divides_133(self):
        assert prime_membership(133, 67) is False

    def test_23_divides_69(self):
        """ALEBAL = 69 = 3 * 23."""
        assert prime_membership(69, 23) is True


class TestWeightDistributions:

    def test_four_distributions(self):
        dists = weight_distributions()
        assert set(dists.keys()) == {'uniform', 'random', 'power_law', 'corpus'}

    def test_each_has_24_values(self):
        dists = weight_distributions()
        for name, values in dists.items():
            assert len(values) == 24, f"{name} has {len(values)} values"

    def test_uniform_all_ones(self):
        dists = weight_distributions()
        assert all(v == 1.0 for v in dists['uniform'])

    def test_all_non_negative(self):
        dists = weight_distributions()
        for name, values in dists.items():
            for v in values:
                assert v >= 0.0, f"{name} has negative value {v}"

    def test_deterministic(self):
        d1 = weight_distributions(seed=42)
        d2 = weight_distributions(seed=42)
        assert d1['random'] == d2['random']

    def test_different_seeds_differ(self):
        d1 = weight_distributions(seed=42)
        d2 = weight_distributions(seed=99)
        assert d1['random'] != d2['random']


class TestCorpusStats:

    def test_stats_values(self):
        s = corpus_stats()
        assert s.n_values == 24
        assert s.min_value == 18    # GEJ
        assert s.max_value == 1296  # UAT ASETEDOJ

    def test_tracked_prime_coverage(self):
        s = corpus_stats()
        # At minimum, prime 67 divides GEAREIJ(134) = 2*67
        assert s.tracked_prime_coverage[67] >= 1
        # Prime 29: DAJIBA∀ = 29 itself
        assert s.tracked_prime_coverage[29] >= 1
