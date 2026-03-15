"""
The Falco isopsephy corpus as structured data for graph experiments.

24 trump card values map to 24 edges. 14 Names of Power map to 14 vertices.
8 tracked primes map to 8 trivalent (cubic) vertices.

The paper presents these as "a set of 24 integers with non-trivial internal
arithmetic structure, derived from an independent analytical domain."

PROPRIETARY: The specific corpus values are intellectual property of
Promptcrafted LLC. They are loaded from a local data file that is not
distributed with the public package. All non-corpus functionality
(uniform, random, power-law distributions; prime utilities; direction
weighting) works without the corpus values.
"""

from __future__ import annotations

import json
import math
import numpy as np
from dataclasses import dataclass
from pathlib import Path


# ── Corpus availability ──────────────────────────────────────────────


class CorpusUnavailable(Exception):
    """Raised when corpus values are required but not available."""
    pass


_DATA_FILE = Path(__file__).parent / "data" / "corpus_private.json"
_corpus_loaded = False
_TRUMP_VALUES: dict[int | str, int] = {}
_TRUMP_INSCRIPTIONS: dict[int | str, str] = {}
_NAMES_OF_POWER: dict[str, int] = {}


def _load_corpus() -> bool:
    """Attempt to load corpus from private data file. Returns True if successful."""
    global _corpus_loaded, _TRUMP_VALUES, _TRUMP_INSCRIPTIONS, _NAMES_OF_POWER

    if _corpus_loaded:
        return True

    if not _DATA_FILE.exists():
        return False

    try:
        with open(_DATA_FILE) as f:
            data = json.load(f)

        # Convert string keys back to int/str
        for k, v in data["trump_values"].items():
            key = int(k) if k not in ("T", "G") else k
            _TRUMP_VALUES[key] = v

        for k, v in data["trump_inscriptions"].items():
            key = int(k) if k not in ("T", "G") else k
            _TRUMP_INSCRIPTIONS[key] = v

        _NAMES_OF_POWER.update(data["names_of_power"])
        _corpus_loaded = True
        return True
    except (json.JSONDecodeError, KeyError):
        return False


def _require_corpus() -> None:
    """Raise CorpusUnavailable if corpus values are not loaded."""
    if not _load_corpus():
        raise CorpusUnavailable(
            "Corpus values are proprietary (Promptcrafted LLC). "
            "Use uniform, random, or power_law distributions instead. "
            "See: https://github.com/promptcrafted/rhombic"
        )


def corpus_available() -> bool:
    """Check whether corpus values are available without raising."""
    return _load_corpus()


# ── Public accessors (require corpus) ────────────────────────────────


def trump_values() -> dict[int | str, int]:
    """The 24 trump values keyed by card position."""
    _require_corpus()
    return dict(_TRUMP_VALUES)


def trump_inscriptions() -> dict[int | str, str]:
    """The 24 trump inscriptions keyed by card position."""
    _require_corpus()
    return dict(_TRUMP_INSCRIPTIONS)


def names_of_power() -> dict[str, int]:
    """The 14 Names of Power and their values."""
    _require_corpus()
    return dict(_NAMES_OF_POWER)


# ── Non-proprietary data ─────────────────────────────────────────────


TRUMP_NAMES: dict[int | str, str] = {
    0: "Fool", 1: "Magician", 2: "Priestess", 3: "Empress",
    4: "Emperor", 5: "Hierophant", 6: "Lovers", 7: "Chariot",
    8: "Justice", 9: "Hermit", 10: "Wheel", 11: "Strength",
    12: "Reversal", 13: "Renewal", 14: "Temperance", 15: "Fallen Angel",
    "T": "Transit", 16: "Tower", 17: "Star", 18: "Moon",
    19: "Sun", 20: "Judgment", 21: "World", "G": "Grail",
}

TRACKED_PRIMES: list[int] = [67, 23, 29, 17, 19, 31, 11, 89]

CANONICAL_EDGE_ORDER: list[int | str] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    "T", 16, 17, 18, 19, 20, 21, "G",
]


# ── Functions ────────────────────────────────────────────────────────


def edge_values() -> list[int]:
    """The 24 trump values in canonical edge order."""
    _require_corpus()
    return [_TRUMP_VALUES[k] for k in CANONICAL_EDGE_ORDER]


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


def direction_weights(values: list[float], n_directions: int) -> list[float]:
    """Map corpus values to direction buckets for direction-based weighting.

    Sorts values, splits into n_directions equal groups, returns the mean
    of each group. For FCC (n_directions=6): 4 values per bucket.
    For cubic (n_directions=3): 8 values per bucket.

    The resulting list has n_directions entries — one weight per direction pair.
    The caller maps each lattice edge to its direction pair to get per-edge weights.
    """
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    per_group = n // n_directions
    result = []
    for i in range(n_directions):
        start = i * per_group
        end = start + per_group if i < n_directions - 1 else n
        group = sorted_vals[start:end]
        result.append(float(np.mean(group)))
    return result


def weight_distributions(seed: int = 42) -> dict[str, list[float]]:
    """Four weight distributions over 24 edges.

    Returns
    -------
    dict with keys: 'uniform', 'random', 'power_law', and optionally 'corpus'
    Each value is a list of 24 floats.

    The corpus distribution requires proprietary data. If unavailable,
    only 3 distributions are returned.
    """
    rng = np.random.default_rng(seed)
    n = 24

    uniform = [1.0] * n

    random_raw = rng.uniform(0.1, 1.0, size=n)
    random_norm = random_raw.tolist()

    ranks = np.arange(1, n + 1, dtype=np.float64)
    power_raw = 1.0 / ranks
    power_norm = (power_raw / power_raw.max()).tolist()

    result = {
        'uniform': uniform,
        'random': random_norm,
        'power_law': power_norm,
    }

    if corpus_available():
        corpus = edge_values()
        corpus_arr = np.array(corpus, dtype=np.float64)
        corpus_norm = (corpus_arr - corpus_arr.min()) / max(corpus_arr.max() - corpus_arr.min(), 1)
        result['corpus'] = corpus_norm.tolist()

    return result


@dataclass
class CorpusStats:
    """Summary statistics of the corpus as a weight set."""
    n_values: int
    min_value: int
    max_value: int
    mean_value: float
    std_value: float
    n_distinct_primes: int
    tracked_prime_coverage: dict[int, int]


def corpus_stats() -> CorpusStats:
    """Compute summary statistics of the 24 trump values."""
    _require_corpus()
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


# ── Corpus-coupled bridge ────────────────────────────────────────────
#
# Stream B (proprietary): Tests corpus arithmetic as bridge programming.
# The hexagram coupling derives from the Cantong qi 6-trigram cycle;
# thread density measures shared prime divisibility across the corpus.


CHANNEL_PRIME_MAP: dict[int, int] = {
    0: 11, 1: 67, 2: 23, 3: 31, 4: 29, 5: 17,
}

CHANNEL_TRIGRAM_MAP: dict[int, tuple[int, int, int]] = {
    0: (0, 0, 0),  # Kūn ☷ — Kaos
    1: (0, 0, 1),  # Zhèn ☳ — Geometric Essence
    2: (0, 1, 1),  # Duì ☱ — Time Matrix
    3: (1, 1, 1),  # Qián ☰ — Arrow of Complexity
    4: (1, 1, 0),  # Xùn ☴ — Synchronicity
    5: (1, 0, 0),  # Gèn ☶ — Circular Causality
}


def hexagram_coupling(ch_i: int, ch_j: int) -> float:
    """Trigram pair → coupling strength in [-1, 1].

    Matching lines = resonance (positive coupling).
    Complementary lines = exchange (negative coupling).
    Self-pairing = maximum resonance (1.0).

    The coupling is computed line-by-line: each matching line contributes
    +1/3, each complementary line contributes -1/3.
    """
    tri_i = CHANNEL_TRIGRAM_MAP[ch_i]
    tri_j = CHANNEL_TRIGRAM_MAP[ch_j]
    score = 0.0
    for a, b in zip(tri_i, tri_j):
        if a == b:
            score += 1.0 / 3.0  # resonance
        else:
            score -= 1.0 / 3.0  # exchange
    return score


def thread_density(ch_i: int, ch_j: int, values: list[int]) -> float:
    """Geometric mean of individual channel prime densities.

    For channels i and j, computes sqrt(d_i * d_j) where d_k is the
    fraction of corpus values divisible by channel k's tracked prime.

    Joint divisibility (values divisible by BOTH primes) is too strict:
    the tracked primes are large (11-89) and most products exceed corpus
    value range. The geometric mean captures "both channels are
    thread-active" without requiring the same values.

    Returns a float in [0, 1].
    """
    if not values:
        return 0.0
    p_i = CHANNEL_PRIME_MAP[ch_i]
    p_j = CHANNEL_PRIME_MAP[ch_j]
    n = len(values)
    d_i = sum(1 for v in values if v % p_i == 0) / n
    d_j = sum(1 for v in values if v % p_j == 0) / n
    return float(np.sqrt(d_i * d_j))


def corpus_coupled_matrix(values: list[int]) -> np.ndarray:
    """Full 6×6 corpus-coupled bridge initialization matrix.

    Off-diagonal: hexagram_coupling × geometric_coupling × (1 + thread_density).
    Diagonal: 1.0 (identity baseline).
    Off-diagonal entries normalized so max |off-diag| = 0.01.

    The hexagram × geometric product is the primary coupling signal.
    Thread density amplifies pairs whose primes are both thread-active
    in the corpus. The (1 + td) form ensures coupling is nonzero even
    when thread density is sparse (which it is — most tracked primes
    divide few of the 24 primary values).

    This puts corpus-derived coupling on the OFF-DIAGONAL (coupling between
    channels), not the diagonal (channel scaling). L-026 showed that diagonal
    weighting was the wrong approach — this corrects it.
    """
    from rhombic.nn.topology import direction_pair_coupling

    n = 6
    geo = direction_pair_coupling()
    # Normalize geometric coupling to [0, 1]
    geo_off = geo.copy()
    np.fill_diagonal(geo_off, 0.0)
    geo_max = geo_off.max()
    if geo_max > 0:
        geo_off /= geo_max

    M = np.eye(n, dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            hx = hexagram_coupling(i, j)
            td = thread_density(i, j, values)
            gx = geo_off[i, j]
            M[i, j] = hx * gx * (1.0 + td)

    # Normalize off-diagonal to max absolute value = 0.01
    off_diag = M.copy()
    np.fill_diagonal(off_diag, 0.0)
    off_max = np.abs(off_diag).max()
    if off_max > 0:
        off_diag = off_diag / off_max * 0.01
        np.fill_diagonal(M, 0.0)
        M = off_diag
        np.fill_diagonal(M, 1.0)

    return M
