# Rung 4: Context Architecture — Raw Results

Three benchmarks at three scales. Clustered embeddings (128D, 5 clusters)
projected to 3D via random orthogonal projection, assigned to lattice
nodes. All results reproducible via `python -m rhombic.context`.

## Benchmark 1: Neighborhood Recall

Fraction of an embedding's true 10 nearest neighbors (in 128D space)
captured by its lattice node's k-hop neighborhood.

### Scale ~125 (Cubic=125, FCC=108, 125 embeddings)

| Hops | Cubic Recall | FCC Recall | Δ | Cubic Size | FCC Size |
|------|-------------|-----------|------|-----------|---------|
| 1 | 0.641 | 0.903 | +0.262 | 6.9 | 12.6 |
| 2 | 0.932 | 0.998 | +0.065 | 22.4 | 41.3 |
| 3 | 0.990 | 1.000 | +0.010 | 47.3 | 77.2 |

### Scale ~500 (Cubic=512, FCC=500, 500 embeddings)

| Hops | Cubic Recall | FCC Recall | Δ | Cubic Size | FCC Size |
|------|-------------|-----------|------|-----------|---------|
| 1 | 0.547 | 0.763 | +0.217 | 7.0 | 13.0 |
| 2 | 0.840 | 0.971 | +0.131 | 24.6 | 53.0 |
| 3 | 0.967 | 0.999 | +0.032 | 59.3 | 129.7 |

### Scale ~1000 (Cubic=1000, FCC=864, 500 embeddings)

| Hops | Cubic Recall | FCC Recall | Δ | Cubic Size | FCC Size |
|------|-------------|-----------|------|-----------|---------|
| 1 | 0.508 | 0.658 | +0.150 | 7.0 | 13.0 |
| 2 | 0.756 | 0.915 | +0.160 | 24.8 | 54.2 |
| 3 | 0.910 | 0.988 | +0.078 | 61.3 | 137.0 |

## Benchmark 2: Information Diffusion

Signal pulse from a central node. Propagation via max(neighbor * 0.9).
Rounds to reach fraction of total lattice nodes.

| Scale | 50% (Cubic) | 50% (FCC) | Speedup | 80% (Cubic) | 80% (FCC) | Speedup |
|-------|------------|----------|---------|------------|----------|---------|
| ~125 | 4 | 2 | 2.00x | 5 | 3 | 1.67x |
| ~500 | 6 | 4 | 1.50x | 8 | 5 | 1.60x |
| ~1000 | 7 | 5 | 1.40x | 10 | 6 | 1.67x |

## Benchmark 3: Consensus Speed

Laplacian averaging to convergence (ε=0.05). Initial states: zero mean,
unit variance.

| Scale | Cubic (rounds) | FCC (rounds) | Speedup |
|-------|---------------|-------------|---------|
| ~125 | 23 | 23 | 1.00x |
| ~500 | 52 | 33 | 1.58x |
| ~1000 | 56 | 60 | 0.93x |
