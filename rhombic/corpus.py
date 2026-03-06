"""
The Falco isopsephy corpus as structured data for graph experiments.

24 trump card values map to 24 edges. 14 Names of Power map to 14 vertices.
8 tracked primes map to 8 trivalent (cubic) vertices.

The paper presents these as "a set of 24 integers with non-trivial internal
arithmetic structure, derived from an independent analytical domain."
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass


# ── Raw corpus data ───────────────────────────────────────────────────

# Standard isopsephic values, verified by tools/isopsephy.py compute
TRUMP_VALUES: dict[int | str, int] = {
    0: 1296, 1: 202, 2: 405, 3: 463, 4: 94,
    5: 64, 6: 448, 7: 771, 8: 342, 9: 153,
    10: 435, 11: 136, 12: 386, 13: 72, 14: 55,
    15: 29, "T": 78, 16: 133, 17: 18, 18: 309,
    19: 240, 20: 346, 21: 134, "G": 252,
}

TRUMP_NAMES: dict[int | str, str] = {
    0: "Fool", 1: "Magician", 2: "Priestess", 3: "Empress",
    4: "Emperor", 5: "Hierophant", 6: "Lovers", 7: "Chariot",
    8: "Justice", 9: "Hermit", 10: "Wheel", 11: "Strength",
    12: "Reversal", 13: "Renewal", 14: "Temperance", 15: "Fallen Angel",
    "T": "Transit", 16: "Tower", 17: "Star", 18: "Moon",
    19: "Sun", 20: "Judgment", 21: "World", "G": "Grail",
}

TRUMP_INSCRIPTIONS: dict[int | str, str] = {
    0: "UAT ASETEDOJ", 1: "BRAL ALEBAL", 2: "ALBAL MAT",
    3: "NIALMATAL", 4: "BANIAL", 5: "ALBAL", 6: "BARILTE",
    7: "ULITAL", 8: "MAAT", 9: "COMJL", 10: "ULE",
    11: "ADONAJ", 12: "TELAMJ", 13: "VAJNE", 14: "VELIBAA",
    15: "DAJIBAA", "T": "BELIAL", 16: "DONACE", 17: "GEJ",
    18: "ECAT", 19: "ORO", 20: "TALEJ", 21: "GEAREIJ",
    "G": "RIRAJLA",
}

NAMES_OF_POWER: dict[str, int] = {
    "SAMMA": 282, "TETRAGRAMMATHON": 1319, "VADUSFADAHM": 671,
    "EOROS": 445, "LIOTHIL": 458, "MENON": 215, "AGAFEST": 516,
    "SADAS": 406, "DESURIORIS": 1099, "SADAM": 246, "TASUMER": 1046,
    "ISIS": 420, "SET": 505, "OSIRIS": 590,
}

TRACKED_PRIMES: list[int] = [67, 23, 29, 17, 19, 31, 11, 89]

# Canonical edge ordering: deck order (0-21, T, G)
CANONICAL_EDGE_ORDER: list[int | str] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    "T", 16, 17, 18, 19, 20, 21, "G",
]


# ── Functions ─────────────────────────────────────────────────────────


def edge_values() -> list[int]:
    """The 24 trump values in canonical edge order."""
    return [TRUMP_VALUES[k] for k in CANONICAL_EDGE_ORDER]


def prime_factors(value: int) -> list[int]:
    """Complete prime factorization of value, with multiplicity."""
    if value < 2:
        return [value] if value > 0 else []
    factors = []
    n = value
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors


def prime_factor_set(value: int) -> set[int]:
    """Unique prime factors of value."""
    return set(prime_factors(value))


def shared_factors(v1: int, v2: int) -> set[int]:
    """Common prime factors between two values."""
    return prime_factor_set(v1) & prime_factor_set(v2)


def prime_membership(value: int, prime: int) -> bool:
    """Whether prime divides value."""
    return value > 0 and value % prime == 0


def weight_distributions(seed: int = 42) -> dict[str, list[float]]:
    """Four weight distributions over 24 edges.

    Returns
    -------
    dict with keys: 'uniform', 'random', 'power_law', 'corpus'
    Each value is a list of 24 floats.
    """
    rng = np.random.default_rng(seed)
    n = 24
    corpus = edge_values()

    # Normalize all distributions to [0, 1] range for comparability
    corpus_arr = np.array(corpus, dtype=np.float64)
    corpus_norm = (corpus_arr - corpus_arr.min()) / max(corpus_arr.max() - corpus_arr.min(), 1)

    uniform = [1.0] * n

    random_raw = rng.uniform(0.1, 1.0, size=n)
    random_norm = random_raw.tolist()

    # Power-law: Zipf-like distribution
    ranks = np.arange(1, n + 1, dtype=np.float64)
    power_raw = 1.0 / ranks
    power_norm = (power_raw / power_raw.max()).tolist()

    return {
        'uniform': uniform,
        'random': random_norm,
        'power_law': power_norm,
        'corpus': corpus_norm.tolist(),
    }


@dataclass
class CorpusStats:
    """Summary statistics of the corpus as a weight set."""
    n_values: int
    min_value: int
    max_value: int
    mean_value: float
    std_value: float
    n_distinct_primes: int
    tracked_prime_coverage: dict[int, int]  # prime -> count of values divisible


def corpus_stats() -> CorpusStats:
    """Compute summary statistics of the 24 trump values."""
    values = edge_values()
    arr = np.array(values)

    all_primes: set[int] = set()
    for v in values:
        all_primes.update(prime_factor_set(v))

    coverage = {}
    for p in TRACKED_PRIMES:
        coverage[p] = sum(1 for v in values if prime_membership(v, p))

    return CorpusStats(
        n_values=len(values),
        min_value=int(arr.min()),
        max_value=int(arr.max()),
        mean_value=float(arr.mean()),
        std_value=float(arr.std()),
        n_distinct_primes=len(all_primes),
        tracked_prime_coverage=coverage,
    )
