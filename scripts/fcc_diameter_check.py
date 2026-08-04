# -*- coding: utf-8 -*-
"""FCC vs cubic diameter comparison at matched N — the LAT-001 motivating fact.

Lands the Director's 2026-08-04 State-of-Play computation as a reproducible
in-tree artifact (his was computed off-tree). Diameter is the term in the
Reasoning-by-Superposition step bound (arXiv:2505.12514): D continuous-thought
steps solve reachability, so the diameter advantage is the predicted step-count
advantage. Uses the library's own lattice classes; BFS via networkx.
"""
import networkx as nx

from rhombic.lattice import CubicLattice, FCCLattice

# Matched-N pairs per the Director's table (cubic M, fcc M).
PAIRS = [(6, 8), (8, 10), (10, 13), (12, 15)]

rows = []
for cm, fm in PAIRS:
    gc = CubicLattice(cm).to_networkx()
    gf = FCCLattice(fm).to_networkx()
    dc, df = nx.diameter(gc), nx.diameter(gf)
    ac = nx.average_shortest_path_length(gc)
    af = nx.average_shortest_path_length(gf)
    rows.append((gc.number_of_nodes(), ac, dc,
                 gf.number_of_nodes(), af, df,
                 100 * (1 - af / ac), 100 * (1 - df / dc)))

print(f"{'cubic N':>8} {'ASP':>7} {'D':>3} | {'fcc N':>6} {'ASP':>7} {'D':>3} |"
      f" {'ASP gain%':>9} {'D gain%':>8}")
for r in rows:
    print(f"{r[0]:>8} {r[1]:>7.3f} {r[2]:>3} | {r[3]:>6} {r[4]:>7.3f} {r[5]:>3} |"
          f" {r[6]:>9.1f} {r[7]:>8.1f}")
asp_mean = sum(r[6] for r in rows) / len(rows)
d_mean = sum(r[7] for r in rows) / len(rows)
print(f"\nMEAN_ASP_REDUCTION      = {asp_mean:.1f}%")
print(f"MEAN_DIAMETER_REDUCTION = {d_mean:.1f}%   <- the superposition step-bound term")
