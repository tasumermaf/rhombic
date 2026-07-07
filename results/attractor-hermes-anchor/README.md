# Spectral-attractor Hermes anchor (H-ch4, H-ch8)

The `tab:attractor` / bifurcation numbers in Paper 4 for the two spectral-only
runs **H-ch4** (n=4) and **H-ch8** (n=8) were not cleanly backed on local disk:

- The local `results/channel-ablation/H-ch4/` artifact is a **1,100-step partial**
  (Fiedler 0.0836) — the value the draft table originally carried, mislabelled as
  the 10,000-step result.
- `H-ch8` was **not on local disk at all**.

Both complete 10,000-step runs are on Hermes; their `results.json` + `config.json`
were fetched here (2026-07-06) to anchor the paper's numbers locally:

| Run | Steps | Fiedler (final) | Paper value | Status |
|-----|-------|-----------------|-------------|--------|
| H-ch4 | 10,000 | **0.0918** | 0.0918 (bifurcation table) | anchored — corrects `tab:attractor` (was 0.0836, the 1,100-step partial) |
| H-ch8 | 10,000 | **0.0944** | 0.0944 | anchored |

H-ch3 (0.0951) and H-ch12 (0.1019) are complete on local disk already.

**Interpretation (BM-000):** the spectral-only Fiedler band (0.0918–0.1019, CoV
3.9%) sits *inside* the identity-plus-noise (ε=0.05) null ensemble — Fiedler ≈0.09
at ≈17th percentile (BM-000 M2). The "spectral attractor" is therefore a
near-initialization resting state, not a distinct structured equilibrium; Paper 4
§5 and the intro/discussion were reframed accordingly. The contrastive Fiedler
values (0.00009 / 0.000191) fall two orders of magnitude below every null and are
the genuine anomaly — the bifurcation is the null-validated result.

Source on Hermes: `~/rhombic/results/channel-ablation/{H-ch4,H-ch8}/`.
