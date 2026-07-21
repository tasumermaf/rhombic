# asset1_d3_labels.py — Adversarial Verifier Note

**Context.** The Asset-1 experiment card scoped D3 Step 6 (post-merge GPU label
generation) as "external to this pipeline" — no runner shipped. `asset1_d3_labels.py`
was written on delivery day to produce the labels the D3 AUC rests on. Because the
same session wrote it, it was put through a **fresh-context adversarial verifier
BEFORE the ~2-hour GPU run**, charged with finding any defect that would produce
wrong labels or crash. Verdict: **FIX-FIRST** — two blocking defects, both fixed and
dry-run-verified before launch.

## Defect 1 (BLOCKING, reproduced) — flat-vs-nested merge state

The runner read each merged adapter `.pt` with a bare `torch.load(...)` and treated it
as a nested `{module: {field: tensor}}` dict. But `asset1_d3_merge.merge_adapters`
saves a **flat** dict keyed `"{safe_module}.{field}"` (384 top-level keys for llama's
64 modules). The verifier loaded an actual merge file and reproduced the exact crash:
`next(iter(first.values()))["rank"]` → `IndexError` on the first family, before the
model even loaded; and `_install_state`'s `set(safe_to_dotted) == set(state)` check
would have raised `ValueError` on the flat keys.

**Fix:** both `torch.load` calls replaced with `asset1_canonicalize.load_adapter_modules(path)`
— the production loader that parses the flat keys into the nested form
`_install_state` expects (the D3 merge docstring specifies this loader). Dry-run
confirmed: 64 nested modules, each with lora_A/lora_B/bridge, safe module-name keys.

## Defect 2 (latent) — machine-absolute manifest path

`native_final_val` read `metrics.json` via the manifest's stored `run_dir`, which is a
machine-absolute path from the writing machine. `asset1_analysis_io`'s foundation rule
explicitly distrusts these and re-derives run dirs from `bank_root`. Correct on this
box, a `FileNotFoundError` waiting for any move/copy.

**Fix:** `_run_dir(bank_root, manifest, run_index)` re-derives the path as
`bank_root / family_short / task / run_<idx:03d>`.

## Native-loss shortcut, verified equivalent

The runner takes native val-losses from each run's trainer-logged `metrics.json`
final instead of re-evaluating them on GPU (halving the label GPU cost). Before
adopting it, all **36** of D2 Stage B's *fresh* native evaluations were compared to
the trainer finals: **0.00000% relative divergence** (same `evaluate_val_loss`, same
locked val_seed=777 split, deterministic). The shortcut is exact.

## Verified-correct (no action)

Interlock returns the manifest and gates on 480/480; degradation formula
`max((m−n)/n)` over endpoints with m,n = exp(loss) equals the DECIDED primary rule
(conflict iff either endpoint ≥ 5% relative); same-task shortcut valid (same split,
same installed state, deterministic); instrument parity with D2 Stage B
(`build_dataset` val split, `EVAL_BATCH_SIZE`, `MAX_LEN`, `evaluate_val_loss`); output
schema matches `load_labels`' required + optional-metric fields; no writes into the
bank tree.

*Both defects fixed and re-verified before the GPU run produced a single label.*
