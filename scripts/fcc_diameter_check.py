# -*- coding: utf-8 -*-
"""FCC vs cubic diameter comparison at matched N — the LAT-001 motivating fact.

Lands the Director's 2026-08-04 State-of-Play computation as a reproducible
in-tree artifact (his was computed off-tree). Diameter is the term in the
Reasoning-by-Superposition step bound (arXiv:2505.12514): D continuous-thought
steps solve reachability, so the diameter advantage is the predicted step-count
advantage.

FIX 2026-08-04 (supersedes the defective first landing, e238685): the first
version hard-coded (cubic_m, fcc_m) constructor pairs assuming FCCLattice(m)
yields ~m^3 nodes. It yields exactly 4m^3 (4-atom basis over m^3 cells), so
the script compared FCC at up to ~10x the cubic N and inverted the result.

The Director's matched-N table (216/256, 512/500, 1000/1099, 1728/1688) is
the even-parity FCC site set of a doubled-coordinate box {0..b}^3 at
b = 7, 9, 12, 14: N = ceil((b+1)^3 / 2). FCCLattice(n) realizes exactly the
odd-b members of that family (b = 2n-1, N = 4n^3) — his first two pairs ARE
FCCLattice(4) and FCCLattice(5) — but no constructor argument reaches the
even-b boxes (1099, 1688). This script therefore builds the parity-box FCC
graph directly (same 12 neighbor offsets as rhombic.lattice, in doubled
coordinates) and CROSS-CHECKS it against the library class at both odd-b
sizes: identical node count, edge count, and diameter, asserted before any
comparison row is printed.
"""
import networkx as nx

from rhombic.lattice import CubicLattice, FCCLattice

# Matched-N pairs per the Director's table: (cubic side n, fcc box side b).
# Realized counts asserted below: 216/256, 512/500, 1000/1099, 1728/1688.
PAIRS = [(6, 7), (8, 9), (10, 12), (12, 14)]
EXPECTED_N = {7: 256, 9: 500, 12: 1099, 14: 1688}

# The 12 FCC neighbor offsets of rhombic.lattice.FCC_NEIGHBOR_OFFSETS
# ((+-0.5, +-0.5, 0) family), expressed in doubled coordinates.
FCC_OFFSETS = [(dx, dy, dz)
               for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
               if dx * dx + dy * dy + dz * dz == 2]


def fcc_box(b: int) -> nx.Graph:
    """FCC graph on the even-parity integer sites of {0..b}^3 (doubled
    coordinates). For odd b this is graph-identical to FCCLattice((b+1)//2)."""
    nodes = [(x, y, z)
             for x in range(b + 1) for y in range(b + 1) for z in range(b + 1)
             if (x + y + z) % 2 == 0]
    node_set = set(nodes)
    g = nx.Graph()
    g.add_nodes_from(nodes)
    for (x, y, z) in nodes:
        for dx, dy, dz in FCC_OFFSETS:
            nb = (x + dx, y + dy, z + dz)
            if nb in node_set:
                g.add_edge((x, y, z), nb)
    return g


# ── Cross-check: parity-box construction == library class at odd b ──
for n_cells in (4, 5):
    lib = FCCLattice(n_cells).to_networkx()
    box = fcc_box(2 * n_cells - 1)
    assert lib.number_of_nodes() == box.number_of_nodes(), \
        (n_cells, lib.number_of_nodes(), box.number_of_nodes())
    assert lib.number_of_edges() == box.number_of_edges(), \
        (n_cells, lib.number_of_edges(), box.number_of_edges())
    assert nx.diameter(lib) == nx.diameter(box), n_cells
print("cross-check: fcc_box(2n-1) == FCCLattice(n) for n=4,5 "
      "(nodes, edges, diameter) — OK")

rows = []
for cn, b in PAIRS:
    gc = CubicLattice(cn).to_networkx()
    gf = fcc_box(b)
    nc, nf = gc.number_of_nodes(), gf.number_of_nodes()
    assert nc == cn ** 3, (cn, nc)
    assert nf == EXPECTED_N[b], (b, nf, EXPECTED_N[b])
    dc, df = nx.diameter(gc), nx.diameter(gf)
    ac = nx.average_shortest_path_length(gc)
    af = nx.average_shortest_path_length(gf)
    rows.append((nc, ac, dc, nf, af, df,
                 100 * (nf - nc) / nc,
                 100 * (1 - af / ac), 100 * (1 - df / dc)))

print(f"\n{'cubic N':>8} {'ASP':>7} {'D':>3} | {'fcc N':>6} {'ASP':>7} {'D':>3} |"
      f" {'dN%':>6} {'ASP gain%':>9} {'D gain%':>8}")
for r in rows:
    print(f"{r[0]:>8} {r[1]:>7.3f} {r[2]:>3} | {r[3]:>6} {r[4]:>7.3f} {r[5]:>3} |"
          f" {r[6]:>+6.1f} {r[7]:>9.1f} {r[8]:>8.1f}")
asp_mean = sum(r[7] for r in rows) / len(rows)
d_mean = sum(r[8] for r in rows) / len(rows)
print(f"\nMEAN_ASP_REDUCTION      = {asp_mean:.1f}%")
print(f"MEAN_DIAMETER_REDUCTION = {d_mean:.1f}%   <- the superposition step-bound term")
