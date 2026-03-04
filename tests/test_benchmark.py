"""Smoke tests for the benchmark suite."""

import pytest
from rhombic.benchmark import run_benchmark


@pytest.fixture(scope="module")
def small_result():
    """Run benchmark once at small scale for all tests."""
    return run_benchmark(target_nodes=125, compute_paths=True)


def test_benchmark_runs(small_result):
    """Benchmark completes without error at small scale."""
    assert small_result.cubic_nodes > 0
    assert small_result.fcc_nodes > 0
    assert small_result.elapsed_seconds > 0


def test_fcc_shorter_paths(small_result):
    """FCC lattice has shorter average paths than cubic."""
    assert small_result.fcc_avg_path is not None
    assert small_result.cubic_avg_path is not None
    assert small_result.fcc_avg_path < small_result.cubic_avg_path


def test_fcc_smaller_diameter(small_result):
    """FCC lattice has smaller diameter than cubic."""
    assert small_result.fcc_diameter is not None
    assert small_result.cubic_diameter is not None
    assert small_result.fcc_diameter < small_result.cubic_diameter


def test_fcc_higher_fiedler(small_result):
    """FCC lattice has higher algebraic connectivity."""
    assert small_result.fcc_fiedler is not None
    assert small_result.cubic_fiedler is not None
    assert small_result.fcc_fiedler > small_result.cubic_fiedler


def test_fcc_more_edges(small_result):
    """FCC lattice uses more edges (the documented cost)."""
    assert small_result.fcc_edges > small_result.cubic_edges


def test_fault_curves_populated(small_result):
    """Fault tolerance curves have entries."""
    assert len(small_result.cubic_fault_curve) > 0
    assert len(small_result.fcc_fault_curve) > 0
    # First entry (no removal) should be ~1.0
    assert small_result.cubic_fault_curve[0] > 0.9
    assert small_result.fcc_fault_curve[0] > 0.9
