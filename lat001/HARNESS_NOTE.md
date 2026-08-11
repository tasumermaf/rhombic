# LAT-001 Harness — Build Note

**2026-08-11 · Track B (harness + CPU smoke only, card §5). No GPU used at any
point. Real runs remain gated on the Director's grade of
`docs/CARD_LAT001_DRAFT_2026-08-04.md` — nothing here is registered.**

## Provenance

Built 2026-08-05 by the `lat001-harness-build` workflow (spec subagent → three
parallel implementers per SPEC §1 ownership), stopped mid-build by the
research pause before its review stage ran. Continuation hand-authored
2026-08-11: test suite run, CPU smoke run to completion, adversarial review by
LAORA (`C:/falco/docs/lora_lat001_harness_review_2026-08-11.md`, falco
056563d — verdict **COMMIT AFTER FIXES**, 3 blockers / 8 fixes / 12 notes),
mechanical fixes applied by a fresh implementer agent and hub-verified at
source. The novelty review committed separately (b0a145f, UNCLAIMED with
boundaries mapped).

## Smoke result — NOT a pass, and honestly so

`python -m lat001.evaluate --smoke`, 2026-08-11, CPU, 976.5 s total:

```
steps              = 300           cells = cubic6/0, fcc12/0
SA1_pass           = True          ratio 0.5215 (cubic6) / 0.5173 (fcc12)
SA2_pass           = True
SA3_pass           = True          per-cell p 0.2266 / 0.0923 · pooled p 0.83882 (n_pairs 332)
SA4_pass           = False         z = 20.326 (cubic6) / 0.362 (fcc12)
SA5_pass           = True          (bitwise determinism)
all_pass           = False         exit 1
```

**Diagnosis (review F-1, N-3, N-4, all measured):** the 300-step SMOKE_MODEL
collapses to a constant predictor (node 18 on 140/140 cubic6 eval items at
every c; final loss 3.1301 vs best-constant 3.1268). SA-1's "pass" is the
vocabulary-support collapse — ln(27)/ln(550) = 0.5223 predicts the measured
0.5215 — not learning. SA-4's cubic6 "fail" is the anti-conservative pooled
statistic reacting to that constant predictor (a frequent-valid-hop constant
beats the permuted-successor guess baseline without reading the graph); the
review shows a fail can be a false alarm while a pass is meaningful. The
plumbing the smoke exists to validate — end-to-end build/train/eval/null/
invariance/determinism — all exercised and working.

**Consequence, per SPEC §1 ("card wins; flag, do not silently resolve"):** the
§8 criteria need a dated amendment before the smoke can be *meaningful*:
SA-1 gated on the task-computed constant-predictor baseline (F-1), SA-2's
d-coverage condition resolved explicitly (F-2), and a mechanism-liveness SA-6
(N-2). Those go to the Director alongside the card — they are not edited here.
Until then the smoke record stands as: **harness validated, criteria not yet
informative, all_pass = False.**

## Mechanical fixes applied this pass (review IDs)

- **B-1** — `test_t4_model_forward_and_grad` added (Section B was an empty
  banner): forward shape, c+1 extension via ln_f hook, gradient reaches
  tok_emb through c=3, forward-value liveness across c.
- **F-2** — wrong causal comment corrected (d-coverage gap = plain seeded 20%
  split, not `_stratified_cap`). Pass condition untouched.
- **F-3** — `_stratified_cap` now has a forcing test (max_eval_pairs=50).
- **F-4** — `evaluate()` no longer silently relocates the model; `device=None`
  → `_model_device(model)`, matching its siblings.
- **F-6** — SA-5 re-train writes to `checkpoint_dir + "_sa5"`; state_dict
  tensors compared, not only final_loss.
- **F-8** — relative `checkpoint_dir` resolved against the repo root
  (common.py stays frozen).
- **N-9** — documented `c_values=None` default now reachable.
- **B-2 (adapted)** — `lat001/results/` gitignored; the smoke completed after
  the review's mid-run snapshot (its "never completed" evidence was a live
  read at 11:08–11:10), so results stay on disk as the run record and this
  note carries the numbers.

Suite after fixes: **11 passed** (`tests/test_lat001.py`).

## Open before any GPU (review, unchanged)

**B-3** materialized attention makes size indices 1–2 unrunnable (SDPA +
cached causal mask + KV-cache, then re-verify SA-5 with a pinned backend) ·
**F-5** `CUBLAS_WORKSPACE_CONFIG` for the determinism flag on CUDA · **F-7**
`EXCLUDED_FROM_BANK` marker + `gpu_guard` in the runner (card §5) · **N-8**
record networkx/torch/numpy/python versions in checkpoints (fingerprint is
networkx-version-coupled) · **N-5** timing pilots must run at size index 2 or
the card's ≤4.375 GPU-day ceiling does not bind. Registration-phase design
gaps to hand back with the card: inference unit = (graph, seed) cell; power
preconditions for SA-3/SA-4; the N-6 init-scale ablation (LR tuned per arm)
resolved before the per-topology LR grid.
