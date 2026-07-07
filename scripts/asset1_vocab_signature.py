"""Asset 1 — OUTPUT-REFERENCED (vocab-signature) adapter representation.

Proposed as pre-registered D1 representation arm #3 ("vocab_signature"),
alongside 'raw' (flattened factors) and 'canonical' (W2T QR->SVD,
asset1_canonicalize.py). Motivated by the Anthropic global-workspace
J-lens (GLOBAL_WORKSPACE_MAPPING_2026-07-07.md, §2.2 / shortlist item 1):
read an adapter's internal directions through their effect on the OUTPUT
vocabulary distribution, not through their weight-space factorization.

Construction (exact, CPU, cheap)
--------------------------------
Per module, the TRUE effective update is Delta = B' @ A' with
(B', A') = asset1_canonicalize.effective_factors (bridge + alpha/rank
scaling absorbed — the same absorption used by every other Asset-1
representation, so all three D1 arms describe the identical object).
Delta is NEVER materialized: probe responses are computed factored,
Y = B' @ (A' @ X).

    1. PROBE INPUTS.  X in R^{d_in x n_probes}, iid N(0, 1/d_in)
       (E||column||^2 = 1), drawn from numpy default_rng seeded with
       [seed, 71, d_in, n_probes] — shared by every module and every
       adapter with the same d_in (within a family, all of them).
    2. MODULE RESPONSE.  Y = Delta @ X in R^{d_out x n_probes}, float64.
    3. RESIDUAL PLACEMENT.  Modules with d_out == d_model (q_proj /
       o_proj on both bank families) pass through unchanged. Modules
       with d_out < d_model (k_proj / v_proj under GQA) are handled per
       ``kv_mode``: 'zero_pad' (default — rows 0..d_out-1 of the
       residual coordinate frame, zeros elsewhere) or 'exclude'.
       ZERO-PAD IS AN APPROXIMATION (k/v outputs live in head space and
       reach the logits only through attention, which this cheap level
       ignores; the true propagation is Level B, asset1_jacobian_lens
       .py). DIRECTOR-APPROVED 2026-07-07 WITH CONDITION
       (docs/DIRECTOR_RULING_PREREG_A3A5_2026-07-07.md): the
       approximation is surfaced in every output artifact — the layout
       block names the zero-padded modules — and the D1 arm reports
       BOTH kv_mode variants (zero_pad primary, exclude secondary) so
       an outcome-(c) deficit cannot be silently attributed to output-
       null structure when it might be the approximation. Level B (the
       true J-lens) is the arbiter; any outcome-(c) reading is
       PROVISIONAL pending it.
    4. NORM HANDLING.  The frozen model's final RMSNorm is linearized as
       its elementwise gain only: y -> g * y with g = model.norm.weight.
       The input-dependent 1/rms(h) scalar is DROPPED (linearization at
       unit RMS). This keeps the whole map LINEAR in Delta — which is
       what makes the signature gauge-invariant by construction.
    5. VOCAB PROJECTION.  Logit response Lam = W_eff @ y_res with
       W_eff = W_U * g (row-wise gain), W_U the frozen model's
       unembedding (lm_head.weight, or model.embed_tokens.weight under
       tied embeddings — both bank families tie).
    6. FIXED-LENGTH SIGNATURE per module (top-k vs dense decision —
       dense V x n_probes is ~150k x 16 per module and is not a usable
       feature; both retained blocks are output-referenced because W_U
       enters both):
         'sketch' block — S^T @ Lam in R^{sketch_dim x n_probes},
             flattened sketch-dim-major. S in R^{V x sketch_dim}, iid
             N(0, 1/sketch_dim) (JL convention, norm-preserving in
             expectation), seeded [seed, 72, V, sketch_dim] — shared
             within a family (same V). Computed as T @ y with
             T = S^T @ W_eff precomputed ONCE per family (sketch_dim x
             d_model), so the per-adapter cost never touches V.
         'topk' block — the k signed logit responses of largest
             magnitude, computed on the module's MEAN probe response
             m = mean_p(y_res): l = W_eff @ m, take top-k by |l|,
             ties broken by ascending vocab index (stable, documented,
             chunk-invariant). SIGNED VALUES ONLY are kept (token ids
             are family-specific categoricals and are excluded from the
             feature — PINNED DEFAULT). Mean-response (not per-probe)
             top-k keeps the cost to ONE (V x d_model) @ (d_model x
             n_modules) GEMM per adapter.
    7. ADAPTER SIGNATURE.  Concatenation over sorted module names of
       [sketch block; topk block] — float32, fixed length
       n_modules_kept * (sketch_dim * n_probes + topk).

Gauge invariance
----------------
Every block is a deterministic function of Delta alone (steps 2-6 are a
fixed linear map followed by a Delta-only top-k selection), so any
GL(r) reparameterization (B -> B G, A -> G^{-1} A), which leaves Delta
unchanged up to float64 round-off, leaves the signature unchanged to
~1e-10 — property-tested in tests/test_asset1_vocab_signature.py with
random GL(r) transforms (the test pattern of tests/test_canonicalize.py).

W_U loading (lazy, partial, CPU-only)
-------------------------------------
``load_unembedding`` resolves the HF hub cache exactly like asset1_bank
(env HF_HUB_CACHE, else HF_HOME/hub, else huggingface_hub.constants
lazily, else ~/.cache/huggingface/hub; on this workstation HF_HOME
points at C:\\falco\\hf-cache), locates the model snapshot WITHOUT
importing transformers, and reads ONLY the unembedding + final-norm
tensors via safetensors partial load (safe_open / get_tensor per key —
never a full state dict, never a model instantiation, never CUDA).
Tied-embedding fallback: if 'lm_head.weight' is absent (both bank
families tie), 'model.embed_tokens.weight' is used and the fallback is
recorded. The returned dict carries loaded_keys and files_opened so
runs self-document the partial load; the property is proven against a
stub snapshot in the tests. NOT to be called during the campaign's GPU
work — it is pure CPU file IO, but the policy is post-bank/Hermes.

RAM / chunking
--------------
The only large object is W_U (float32: Qwen ~0.93 GB, Llama ~1.05 GB),
held once per family. All V-sized products are row-chunked
(``vocab_chunk``, default 32768; chunks upcast float32->float64 exactly;
the top-k SELECTION RULE is chunk-invariant and values agree across
chunk sizes to float64 round-off — BLAS blocking may differ by shape;
peak extra RAM is ~vocab_chunk * d_model * 8 bytes ~ 0.4 GB). Per-run
signatures are ~n_modules * (sketch_dim*n_probes + topk) floats
(Qwen: 112 * 136 = 15,232; Llama-3.2-1B: 64 * 136 = 8,704 at the
pinned defaults) — a 480-run pass is ~23 MB
of features, streamed one adapter at a time. A full-bank pass fits RAM
with a wide margin.

Cross-family caveat (DESIGN ONLY — Director decision required)
--------------------------------------------------------------
Qwen2.5 (V=151,936) and Llama-3.2 (V=128,256) vocabularies differ, so
vocab signatures are WITHIN-FAMILY representations (the D1 H1 setting).
For any cross-family use, the designed (NOT built) mapping is:
  (1) load both tokenizers; compute the intersection of literal token
      strings (byte-level normalization applied identically to both
      vocab dumps);
  (2) sort the intersection lexicographically (deterministic order);
  (3) restrict each family's W_eff to its rows for the shared tokens in
      that shared order -> W_shared^F in R^{|T_shared| x d_F};
  (4) draw the vocab sketch on the SHARED axis (seeded
      [seed, 72, |T_shared|, sketch_dim]) so sketch and top-k blocks
      live in one output-referenced coordinate system;
  (5) residual family scale signatures (depth/width) remain — the D1 H2
      shift control (familywise standardization) still applies.
Open Director decisions: approve the token-string intersection as the
alignment; decide whether top-k runs over shared-token ids only; decide
the treatment of tokens whose strings match but whose tokenizer merge
contexts differ. Recorded in every output JSON.

Pre-registration hygiene
------------------------
Real-bank invocations pass through asset1_analysis_io
.require_complete_bank (the interlock); --out-dir inside the bank tree
is refused; the acceptance path is --synthetic-selftest on
asset1_synth.make_synthetic_bank fixtures with a stub readout. Nothing
here reads results/asset1-bank while the campaign is live.

D1 integration (LANDED 2026-07-07 — authorized by the Director's A3
ruling, docs/DIRECTOR_RULING_PREREG_A3A5_2026-07-07.md)
-----------------------------------------------------------------------
asset1_d1_identifiability.py carries 'vocab_signature' in its
--representation choices; its _feat dispatch calls
``vocab_signature_feature(run_dir, readout, ...)`` with a per-family
readout built once in analyze_bank. Analysis machinery (Gram, LOO,
permutation null, H1 lock) is IDENTICAL to the raw/canonical arms.
Per the ruling's condition, the D1 arm runs the signature under BOTH
kv_mode variants: 'vocab_signature' (zero_pad, the pinned primary) and
'vocab_signature_kv_exclude' (secondary, disambiguation-only — not a
separate H1 headline claim).

Usage
-----
    python scripts/asset1_vocab_signature.py --synthetic-selftest \
        --out-dir results/asset1-vocabsig-selftest
    # post-bank, per the amendment:
    python scripts/asset1_vocab_signature.py \
        --bank-root results/asset1-bank --out-dir results/asset1-vocabsig
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for _p in (str(SCRIPTS_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import asset1_analysis_io as aio  # noqa: E402  (torch-only; no HF import)
from asset1_canonicalize import (  # noqa: E402
    effective_factors, load_adapter_modules)

# ── Pinned defaults (every one recorded in output JSON; Director
#    sign-off items are marked in PINNED_DEFAULTS below) ─────────────

N_PROBES = 16          # seeded Gaussian input probes per module
SKETCH_DIM = 8         # vocab-sketch dimension (dense-sketch block)
TOPK = 8               # top-|logit| signed values on the mean response
KV_MODE = "zero_pad"   # d_out < d_model placement: zero_pad | exclude
VOCAB_CHUNK = 32768    # W_U row-chunk for all V-sized products
DEFAULT_SEED = 0       # probe + sketch seed
STUB_VOCAB_SIZE = 64   # stub readout vocab (synthetic fixtures only)

_STREAM_PROBE = 71     # rng stream tags (asset1 house convention:
_STREAM_SKETCH = 72    # integer-list default_rng seeds, fixed tags)
_STREAM_STUB = 73

PINNED_DEFAULTS = {
    "n_probes": N_PROBES,
    "sketch_dim": SKETCH_DIM,
    "topk": TOPK,
    "kv_mode": (
        f"{KV_MODE} — k/v (d_out < d_model) responses zero-padded into "
        "residual coordinates before the norm-gain + W_U readout. "
        "APPROXIMATION (true k/v->logit propagation goes through "
        "attention; that is Level B, asset1_jacobian_lens.py). "
        "DIRECTOR APPROVED 2026-07-07 WITH CONDITION "
        "(docs/DIRECTOR_RULING_PREREG_A3A5_2026-07-07.md): surfaced in "
        "every output (layout.modules_zero_padded); the D1 arm reports "
        "BOTH variants — zero_pad (primary) and exclude (secondary, "
        "disambiguation-only); Level B is the arbiter and an "
        "outcome-(c) reading is PROVISIONAL pending it."),
    "norm_handling": (
        "final RMSNorm linearized as elementwise gain g only; the "
        "input-dependent 1/rms scalar is dropped (keeps the map linear "
        "in Delta => gauge invariance by construction)."),
    "probe_distribution": (
        "X ~ N(0, 1/d_in), default_rng([seed, 71, d_in, n_probes]); "
        "shared across modules and adapters of equal d_in."),
    "sketch_distribution": (
        "S ~ N(0, 1/sketch_dim), default_rng([seed, 72, V, sketch_dim]); "
        "shared within a family (same V); JL norm-preserving scaling."),
    "topk_construction": (
        "computed on the MEAN probe response per module (one GEMM per "
        "adapter); SIGNED values by descending |logit|, ties by "
        "ascending vocab index (stable, chunk-invariant); token ids "
        "EXCLUDED from the feature (family-specific categoricals)."),
    "aggregation": (
        "concat over sorted module names of [sketch(sketch_dim x "
        "n_probes, sketch-major); topk(k)] -> float32."),
    "unembedding_source": (
        "safetensors partial load of lm_head.weight (fallback "
        "model.embed_tokens.weight under tied embeddings) + "
        "model.norm.weight from the HF hub cache; CPU only; no "
        "transformers import."),
    "compute_dtype": "float64 (float32 W_U storage, exact chunk upcast)",
    "vocab_chunk": VOCAB_CHUNK,
    "seed": DEFAULT_SEED,
    "scope": (
        "WITHIN-FAMILY representation. Cross-family shared-token-string "
        "mapping is DESIGN ONLY — see cross_family_design; Director "
        "decision required before any cross-family use."),
}

CROSS_FAMILY_DESIGN = {
    "status": "DESIGN ONLY — NOT BUILT — Director decision required",
    "mapping": [
        "1. load both tokenizers; intersect literal token STRINGS "
        "(identical byte-level normalization on both vocab dumps)",
        "2. sort the intersection lexicographically (deterministic)",
        "3. restrict each family's W_eff to shared-token rows in that "
        "order -> W_shared^F in R^{|T_shared| x d_F}",
        "4. draw the vocab sketch on the SHARED axis "
        "([seed, 72, |T_shared|, sketch_dim]) so sketch/topk blocks "
        "share one output-referenced coordinate system",
        "5. family scale signatures remain -> D1 H2 familywise "
        "standardization still applies",
    ],
    "open_director_decisions": [
        "approve token-string intersection as the alignment",
        "top-k over shared-token ids only, or family-native ids",
        "treatment of string-equal tokens with different merge contexts",
    ],
}


# ── Seeded probe / sketch generators ────────────────────────────────


def _seeded_gaussian(rows: int, cols: int, seed: int, stream: int,
                     scale: float) -> np.ndarray:
    """Deterministic N(0, scale^2) matrix keyed on every argument.

    default_rng integer-list seeding (house convention, asset1_synth) —
    identical (rows, cols, seed, stream) always yields the identical
    matrix, across modules, adapters, and processes.
    """
    rng = np.random.default_rng([seed, stream, rows, cols])
    return rng.standard_normal((rows, cols)) * scale


def probe_inputs(d_in: int, n_probes: int, seed: int) -> np.ndarray:
    """Input probes X (d_in x n_probes), N(0, 1/d_in): E||col||^2 = 1."""
    return _seeded_gaussian(d_in, n_probes, seed, _STREAM_PROBE,
                            d_in ** -0.5)


def vocab_sketch(vocab_size: int, sketch_dim: int, seed: int) -> np.ndarray:
    """Vocab sketch S (V x sketch_dim), N(0, 1/sketch_dim): JL scaling,
    E||S^T lam||^2 = ||lam||^2. Shared by asset1_jacobian_lens (the two
    output-referenced levels use the IDENTICAL sketch)."""
    return _seeded_gaussian(vocab_size, sketch_dim, seed, _STREAM_SKETCH,
                            sketch_dim ** -0.5)


# ── W_U loading (lazy safetensors partial load; no transformers) ────


def resolve_hf_hub_cache() -> str:
    """HF hub cache directory, resolved like asset1_bank: env first,
    huggingface_hub.constants lazily, else the default location."""
    env = os.environ.get("HF_HUB_CACHE")
    if env:
        return env
    home = os.environ.get("HF_HOME")
    if home:
        return os.path.join(home, "hub")
    try:
        from huggingface_hub.constants import HF_HUB_CACHE  # lazy
        return HF_HUB_CACHE
    except Exception:
        return os.path.join(os.path.expanduser("~"), ".cache",
                            "huggingface", "hub")


def _resolve_snapshot(model_id: str, cache_dir: str | Path) -> Path:
    """Locate the model's snapshot dir in the hub cache (no network)."""
    repo_dir = Path(cache_dir) / ("models--" + model_id.replace("/", "--"))
    snaps = repo_dir / "snapshots"
    ref_main = repo_dir / "refs" / "main"
    if ref_main.exists():
        rev = ref_main.read_text(encoding="utf-8").strip()
        cand = snaps / rev
        if cand.exists():
            return cand
    if snaps.exists():
        dirs = sorted(d for d in snaps.iterdir() if d.is_dir())
        if len(dirs) == 1:
            return dirs[0]
        if len(dirs) > 1:
            raise FileNotFoundError(
                f"ambiguous snapshots for {model_id} under {snaps}: "
                f"{[d.name for d in dirs]} and no refs/main")
    raise FileNotFoundError(
        f"no cached snapshot for {model_id} under {cache_dir} "
        f"(looked in {repo_dir})")


def _key_to_file_map(snapshot: Path) -> dict[str, Path]:
    """{tensor_key: weight_file} for a snapshot — from the safetensors
    index if sharded, else by enumerating the single file's keys."""
    from safetensors import safe_open  # lazy — no transformers

    index = snapshot / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text(encoding="utf-8"))[
            "weight_map"]
        return {k: snapshot / fname for k, fname in weight_map.items()}
    single = snapshot / "model.safetensors"
    if not single.exists():
        raise FileNotFoundError(
            f"no model.safetensors[.index.json] under {snapshot}")
    with safe_open(str(single), framework="pt", device="cpu") as f:
        return {k: single for k in f.keys()}


def load_unembedding(model_id: str, cache_dir: str | Path | None = None,
                     *, unembed_keys: tuple[str, ...] = (
                         "lm_head.weight", "model.embed_tokens.weight"),
                     norm_key: str = "model.norm.weight") -> dict:
    """PARTIAL load of the frozen model's output readout — CPU only.

    Reads ONLY the unembedding tensor (first present of ``unembed_keys``
    — lm_head, else the tied embedding) and the final-norm gain via
    safetensors safe_open/get_tensor. Never instantiates a model, never
    imports transformers, never touches CUDA. Opens only the shard
    file(s) that hold the wanted keys (proven against a stub snapshot in
    the tests via loaded_keys / files_opened).

    Returns dict: W_U (V, d_model) float32 cpu tensor, norm_g (d_model,)
    float32 cpu tensor, vocab_size, d_model, loaded_keys, files_opened,
    tied_embeddings_fallback, snapshot.
    """
    from safetensors import safe_open  # lazy — no transformers

    cache_dir = cache_dir if cache_dir is not None else resolve_hf_hub_cache()
    snapshot = _resolve_snapshot(model_id, cache_dir)
    key_map = _key_to_file_map(snapshot)

    unembed_key = next((k for k in unembed_keys if k in key_map), None)
    if unembed_key is None:
        raise KeyError(
            f"none of {unembed_keys} present in {snapshot} "
            f"({len(key_map)} keys)")
    if norm_key not in key_map:
        raise KeyError(f"{norm_key!r} not present in {snapshot}")

    wanted = {unembed_key, norm_key}
    by_file: dict[Path, list[str]] = {}
    for k in sorted(wanted):
        by_file.setdefault(key_map[k], []).append(k)

    tensors: dict[str, torch.Tensor] = {}
    files_opened: list[str] = []
    for fpath in sorted(by_file):
        files_opened.append(fpath.name)
        with safe_open(str(fpath), framework="pt", device="cpu") as f:
            for k in by_file[fpath]:
                tensors[k] = f.get_tensor(k).to(torch.float32)

    W_U = tensors[unembed_key]
    g = tensors[norm_key]
    if W_U.ndim != 2 or g.ndim != 1 or W_U.shape[1] != g.shape[0]:
        raise ValueError(
            f"shape mismatch: {unembed_key} {tuple(W_U.shape)} vs "
            f"{norm_key} {tuple(g.shape)}")
    return {
        "W_U": W_U,
        "norm_g": g,
        "vocab_size": int(W_U.shape[0]),
        "d_model": int(W_U.shape[1]),
        "loaded_keys": sorted(wanted),
        "files_opened": files_opened,
        "tied_embeddings_fallback": unembed_key != unembed_keys[0],
        "snapshot": str(snapshot),
    }


def make_stub_readout(d_model: int, vocab_size: int = STUB_VOCAB_SIZE,
                      seed: int = DEFAULT_SEED) -> "VocabReadout":
    """Seeded stand-in readout for synthetic fixtures and tests.

    W_U ~ N(0, 1/sqrt(d_model)) and a positive gain g ~ U[0.5, 1.5],
    keyed [seed, 73, vocab_size, d_model] — deterministic, no HF cache.
    """
    rng = np.random.default_rng([seed, _STREAM_STUB, vocab_size, d_model])
    W = rng.standard_normal((vocab_size, d_model)) / (d_model ** 0.5)
    g = rng.uniform(0.5, 1.5, size=d_model)
    return VocabReadout(torch.from_numpy(W.astype(np.float32)),
                        torch.from_numpy(g.astype(np.float32)),
                        sketch_seed=seed)


# ── The readout object (W_eff, precomputed sketch map, chunked topk) ─


class VocabReadout:
    """Frozen-model output readout: W_eff = W_U * g, its precomputed
    sketch map T = S^T @ W_eff, and a chunk-invariant signed top-k.

    W_U is stored float32 (as loaded); every V-sized product upcasts
    row-chunks to float64 exactly, so chunking never changes results.
    """

    def __init__(self, W_U: torch.Tensor, norm_g: torch.Tensor, *,
                 sketch_dim: int = SKETCH_DIM, sketch_seed: int = DEFAULT_SEED,
                 vocab_chunk: int = VOCAB_CHUNK):
        W = np.asarray(W_U.detach().to(torch.float32).cpu().numpy())
        g = np.asarray(norm_g.detach().to(torch.float64).cpu().numpy())
        if W.ndim != 2 or g.shape != (W.shape[1],):
            raise ValueError(
                f"W_U (V, d) and norm_g (d,) required; got "
                f"{W.shape} / {g.shape}")
        if vocab_chunk < 1:
            raise ValueError("vocab_chunk must be >= 1")
        self.W = W                              # (V, d) float32
        self.g = g                              # (d,)  float64
        self.vocab_size, self.d_model = W.shape
        self.sketch_dim = int(sketch_dim)
        self.sketch_seed = int(sketch_seed)
        self.vocab_chunk = int(vocab_chunk)
        self.S = vocab_sketch(self.vocab_size, self.sketch_dim,
                              self.sketch_seed)  # (V, s) float64
        # T = S^T @ (W * g), accumulated in row chunks (float64 exact).
        T = np.zeros((self.sketch_dim, self.d_model), dtype=np.float64)
        for lo in range(0, self.vocab_size, self.vocab_chunk):
            hi = min(lo + self.vocab_chunk, self.vocab_size)
            Wc = self.W[lo:hi].astype(np.float64) * self.g[None, :]
            T += self.S[lo:hi].T @ Wc
        self.T = T                              # (s, d) float64

    def topk_signed_logits(self, M: np.ndarray, k: int) -> np.ndarray:
        """Signed top-k logit responses per column of M (d_model x n).

        Selection rule (deterministic, chunk-invariant): descending
        |value|, ties broken by ascending vocab index. Returns (k, n)
        float64 SIGNED values in that order.
        """
        M = np.asarray(M, dtype=np.float64)
        if M.ndim != 2 or M.shape[0] != self.d_model:
            raise ValueError(f"M must be (d_model, n); got {M.shape}")
        if not 1 <= k <= self.vocab_size:
            raise ValueError(f"topk={k} outside [1, V={self.vocab_size}]")
        n = M.shape[1]
        run_vals = np.zeros((0, n), dtype=np.float64)
        run_idx = np.zeros((0, n), dtype=np.int64)
        for lo in range(0, self.vocab_size, self.vocab_chunk):
            hi = min(lo + self.vocab_chunk, self.vocab_size)
            Wc = self.W[lo:hi].astype(np.float64) * self.g[None, :]
            L = Wc @ M                                        # (c, n)
            take = min(k, L.shape[0])
            order = np.argsort(-np.abs(L), axis=0, kind="stable")[:take]
            run_vals = np.vstack(
                [run_vals, np.take_along_axis(L, order, axis=0)])
            run_idx = np.vstack([run_idx, order + lo])
            if run_vals.shape[0] > k:            # prune to running top-k
                keep_v = np.empty((k, n), dtype=np.float64)
                keep_i = np.empty((k, n), dtype=np.int64)
                for j in range(n):
                    sel = np.lexsort(
                        (run_idx[:, j], -np.abs(run_vals[:, j])))[:k]
                    keep_v[:, j] = run_vals[sel, j]
                    keep_i[:, j] = run_idx[sel, j]
                run_vals, run_idx = keep_v, keep_i
        out = np.empty((k, n), dtype=np.float64)
        for j in range(n):
            sel = np.lexsort((run_idx[:, j], -np.abs(run_vals[:, j])))[:k]
            out[:, j] = run_vals[sel, j]
        return out


# ── Signature computation ───────────────────────────────────────────


def signature_for_modules(modules: dict[str, dict[str, torch.Tensor]],
                          readout: VocabReadout, *,
                          n_probes: int = N_PROBES, topk: int = TOPK,
                          seed: int = DEFAULT_SEED,
                          kv_mode: str = KV_MODE
                          ) -> tuple[np.ndarray, dict]:
    """Vocab signature of one adapter given as {module: fields}.

    Returns (signature float32 1-D, layout dict). Module order: sorted
    names (deterministic, identical across adapters of a family). See
    the module docstring for the exact construction; every step is a
    function of Delta only => gauge-invariant.
    """
    if kv_mode not in ("zero_pad", "exclude"):
        raise ValueError(f"kv_mode must be zero_pad|exclude, got {kv_mode!r}")
    d_model = readout.d_model
    kept: list[str] = []
    excluded: list[str] = []
    zero_padded: list[str] = []
    sketch_blocks: list[np.ndarray] = []
    means: list[np.ndarray] = []
    for name in sorted(modules):
        B_eff, A_eff = effective_factors(modules[name])   # float64 torch
        Bn = B_eff.numpy()
        An = A_eff.numpy()
        d_out, d_in = Bn.shape[0], An.shape[1]
        X = probe_inputs(d_in, n_probes, seed)
        Y = Bn @ (An @ X)                                  # (d_out, p)
        if d_out == d_model:
            Yr = Y
        elif d_out < d_model:
            if kv_mode == "exclude":
                excluded.append(name)
                continue
            Yr = np.zeros((d_model, n_probes), dtype=np.float64)
            Yr[:d_out] = Y
            zero_padded.append(name)
        else:
            raise ValueError(
                f"module {name!r}: d_out {d_out} > d_model {d_model} — "
                f"not a residual-writable module under this readout")
        kept.append(name)
        sketch_blocks.append((readout.T @ Yr).ravel())     # (s*p,)
        means.append(Yr.mean(axis=1))
    if not kept:
        raise ValueError("no modules kept — nothing to sign")

    if topk > 0:
        M = np.stack(means, axis=1)                        # (d, n_kept)
        tk = readout.topk_signed_logits(M, topk)           # (k, n_kept)
        parts = [np.concatenate([sketch_blocks[i], tk[:, i]])
                 for i in range(len(kept))]
    else:
        parts = sketch_blocks
    sig = np.concatenate(parts).astype(np.float32)
    layout = {
        "n_modules_kept": len(kept),
        "modules_excluded": excluded,
        "modules_zero_padded": zero_padded,
        "kv_handling_note": (
            "zero_pad is a pinned APPROXIMATION (k/v head-space responses "
            "reach the logits only through attention, ignored at this "
            "level); Director condition 2026-07-07: surfaced here, both "
            "kv_mode variants reported in the D1 arm, Level B "
            "(asset1_jacobian_lens.py) is the arbiter."),
        "per_module_dim": readout.sketch_dim * n_probes + max(topk, 0),
        "sketch_block_dim": readout.sketch_dim * n_probes,
        "topk_block_dim": max(topk, 0),
        "dim": int(sig.size),
        "n_probes": n_probes,
        "sketch_dim": readout.sketch_dim,
        "topk": topk,
        "kv_mode": kv_mode,
        "seed": seed,
    }
    return sig, layout


def signature_for_adapter(adapter_state_path: str | Path,
                          readout: VocabReadout, **cfg
                          ) -> tuple[np.ndarray, dict]:
    """Vocab signature of a bank adapter_state.pt (bridge + scaling
    absorbed via effective_factors — the same effective DW every other
    Asset-1 representation describes)."""
    return signature_for_modules(load_adapter_modules(adapter_state_path),
                                 readout, **cfg)


def vocab_signature_feature(run_dir: str | Path, readout: VocabReadout,
                            **cfg) -> np.ndarray:
    """D1 feature hook — mirrors asset1_d1_identifiability._raw_feature /
    _canonical_feature; arm #3's dispatch in build_feature_memmap calls
    this (LANDED 2026-07-07 per the Director's A3 ruling — see module
    docstring, 'D1 integration')."""
    sig, _ = signature_for_adapter(Path(run_dir) / "adapter_state.pt",
                                   readout, **cfg)
    return sig


# ── Bank pass ───────────────────────────────────────────────────────


def _guard_out_dir(out_dir: Path) -> None:
    if "asset1-bank" in out_dir.resolve().parts:
        raise SystemExit(
            f"[vocabsig] REFUSING --out-dir {out_dir}: inside the live "
            f"campaign tree (results/asset1-bank is write-protected).")


def _infer_d_model(run_dir: Path) -> int:
    """Family d_model = max module d_out (q/o write the residual stream
    at d_model on both bank families and on the synthetic fixtures)."""
    modules = load_adapter_modules(Path(run_dir) / "adapter_state.pt")
    return max(int(m["lora_B"].shape[0]) for m in modules.values())


def compute_bank_signatures(bank_root: Path, out_dir: Path, *,
                            readout_mode: str = "hf",
                            stub_vocab_size: int = STUB_VOCAB_SIZE,
                            n_probes: int = N_PROBES,
                            sketch_dim: int = SKETCH_DIM, topk: int = TOPK,
                            kv_mode: str = KV_MODE,
                            vocab_chunk: int = VOCAB_CHUNK,
                            seed: int = DEFAULT_SEED,
                            expected_total: int = aio.EXPECTED_TOTAL_RUNS,
                            allow_partial: bool = False) -> dict:
    """Compute vocab signatures for every COMPLETE run of a bank.

    Writes one npz per family (signatures, run_index, task, replicate)
    plus vocab_signatures.json recording every pinned default. Gated by
    require_complete_bank; ``expected_total`` is overridden only by
    synthetic fixtures (never a CLI knob on the real bank).
    ``readout_mode``: 'hf' (partial-load the real W_U from the hub
    cache) or 'stub' (seeded stand-in — synthetic fixtures ONLY).
    """
    bank_root = Path(bank_root)
    out_dir = Path(out_dir)
    _guard_out_dir(out_dir)
    manifest = aio.require_complete_bank(bank_root,
                                         allow_partial=allow_partial,
                                         expected_total=expected_total)
    out_dir.mkdir(parents=True, exist_ok=True)

    fam_entries = manifest["campaign"]["families"]
    records = list(aio.iter_runs(bank_root))
    results: dict = {
        "analysis": ("vocab-signature (output-referenced) representation "
                     "— proposed D1 arm #3"),
        "generated_by": "scripts/asset1_vocab_signature.py",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "bank_root": str(bank_root),
        "allow_partial_bank": bool(allow_partial),
        "exploratory_only": bool(allow_partial),
        "parameters": {
            "readout_mode": readout_mode,
            "stub_vocab_size": (stub_vocab_size if readout_mode == "stub"
                                else None),
            "n_probes": n_probes, "sketch_dim": sketch_dim, "topk": topk,
            "kv_mode": kv_mode, "vocab_chunk": vocab_chunk, "seed": seed,
        },
        "pinned_defaults": PINNED_DEFAULTS,
        "cross_family_design": CROSS_FAMILY_DESIGN,
        "families": {},
        "flags": [
            "kv_mode zero_pad is a pinned approximation — DIRECTOR "
            "APPROVED 2026-07-07 WITH CONDITION "
            "(docs/DIRECTOR_RULING_PREREG_A3A5_2026-07-07.md): surfaced "
            "in output (layout.modules_zero_padded); D1 reports BOTH "
            "kv_mode variants; Level B (asset1_jacobian_lens.py) is the "
            "arbiter — outcome-(c) readings are PROVISIONAL pending it.",
            "top-k block: signed values only, mean-probe response, ties "
            "by ascending vocab index — pinned defaults.",
            "within-family scope; cross-family mapping is DESIGN ONLY "
            "(see cross_family_design) — Director decision required.",
            "stub readout is for synthetic fixtures only; real-bank runs "
            "must use readout_mode='hf'." if readout_mode == "stub"
            else "readout_mode='hf': W_U + final-norm partial-loaded "
                 "from the HF hub cache (loaded_keys recorded per "
                 "family).",
        ],
    }

    for fam in fam_entries:
        short = fam["short"]
        fam_records = [r for r in records if r["family_short"] == short]
        if not fam_records:
            continue
        if readout_mode == "hf":
            info = load_unembedding(fam["model"])
            readout = VocabReadout(info["W_U"], info["norm_g"],
                                   sketch_dim=sketch_dim, sketch_seed=seed,
                                   vocab_chunk=vocab_chunk)
            readout_meta = {k: info[k] for k in
                            ("vocab_size", "d_model", "loaded_keys",
                             "files_opened", "tied_embeddings_fallback",
                             "snapshot")}
        elif readout_mode == "stub":
            d_model = _infer_d_model(fam_records[0]["run_dir"])
            readout = make_stub_readout(d_model, stub_vocab_size, seed)
            readout_meta = {"vocab_size": readout.vocab_size,
                            "d_model": readout.d_model,
                            "stub": True}
        else:
            raise ValueError(f"readout_mode must be hf|stub, got "
                             f"{readout_mode!r}")

        rows, layout = [], None
        for rec in fam_records:
            sig, layout = signature_for_adapter(
                Path(rec["run_dir"]) / "adapter_state.pt", readout,
                n_probes=n_probes, topk=topk, seed=seed, kv_mode=kv_mode)
            rows.append(sig)
        X = np.stack(rows)
        npz_path = out_dir / f"vocab_signatures_{short}.npz"
        np.savez_compressed(
            npz_path, signatures=X,
            run_index=np.array([r["run_index"] for r in fam_records]),
            task=np.array([r["task"] for r in fam_records]),
            replicate=np.array([r["replicate"] for r in fam_records]))
        results["families"][short] = {
            "model": fam["model"],
            "n_runs": len(fam_records),
            "signature_dim": int(X.shape[1]),
            "layout": layout,
            "readout": readout_meta,
            "npz": str(npz_path),
        }
        print(f"[vocabsig] {short}: {X.shape[0]} runs x {X.shape[1]} dims "
              f"-> {npz_path}", flush=True)

    json_path = out_dir / "vocab_signatures.json"
    json_path.write_text(json.dumps(results, indent=2, default=str),
                         encoding="utf-8")
    print(f"[vocabsig] manifest written: {json_path}", flush=True)
    return results


# ── Synthetic self-test (never touches the real bank) ───────────────


def loo_nearest_centroid_accuracy(X: np.ndarray, labels: np.ndarray,
                                  normalize: bool = True) -> float:
    """Leave-one-out nearest-centroid accuracy — the selftest acceptance
    metric for planted-difference sensitivity.

    Signatures are L2-normalized first (default): the synthetic
    generator's dev_mag varies 0.5-1.5x per run ALONG the fixed task
    direction (asset1_synth docstring), so within-task spread is
    radial by construction; direction, not magnitude, carries the
    planted task identity. Deterministic; no fitted model.
    """
    X = np.asarray(X, dtype=np.float64)
    if normalize:
        X = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True),
                        1e-30, None)
    labels = np.asarray(labels)
    n = X.shape[0]
    correct = 0
    for i in range(n):
        best_t, best_d = None, np.inf
        for t in sorted(set(labels.tolist())):
            idx = [j for j in range(n) if j != i and labels[j] == t]
            if not idx:
                continue
            d = float(np.linalg.norm(X[i] - X[idx].mean(axis=0)))
            if d < best_d:
                best_t, best_d = t, d
        correct += int(best_t == labels[i])
    return correct / n


def run_synthetic_selftest(out_dir: Path, seed: int = DEFAULT_SEED) -> dict:
    """Acceptance test on asset1_synth fixtures with a stub readout.

    Asserts: (1) determinism (recompute == bitwise equal); (2) planted-
    difference sensitivity — on the task_effect=1.0 bank, LOO nearest-
    centroid accuracy on L2-normalized signatures is 1.0 per family;
    (3) the task_effect=0.0 bank stays near chance (<= 0.7 at chance
    1/3). Raises AssertionError on failure; writes selftest JSON.
    """
    import asset1_synth as synth

    out_dir = Path(out_dir)
    _guard_out_dir(out_dir)
    base = out_dir / "selftest"
    synth_kw = dict(n_families=2, n_tasks=3, n_reps=4, n_layers=2,
                    d_model=16, rank=4, n_channels=2, seed=8)
    eff = synth.make_synthetic_bank(base / "bank_effect",
                                    task_effect=1.0, **synth_kw)
    nul = synth.make_synthetic_bank(base / "bank_null",
                                    task_effect=0.0, **synth_kw)

    report: dict = {"selftest": True, "families": {}}
    for tag, bank, info in (("effect", base / "bank_effect", eff),
                            ("null", base / "bank_null", nul)):
        res = compute_bank_signatures(
            bank, base / f"out_{tag}", readout_mode="stub",
            expected_total=info["n_runs"], seed=seed)
        for short, fam_out in res["families"].items():
            data = np.load(fam_out["npz"], allow_pickle=False)
            X = data["signatures"].astype(np.float64)
            tasks = data["task"]
            acc = loo_nearest_centroid_accuracy(X, tasks)
            report["families"].setdefault(short, {})[tag] = {
                "loo_nearest_centroid_accuracy": acc,
                "chance": 1.0 / len(set(tasks.tolist())),
            }
            if tag == "effect":
                assert acc == 1.0, (
                    f"{short}/effect: planted task structure NOT fully "
                    f"recovered (LOO nearest-centroid acc {acc:.4f} < 1.0)")
            else:
                assert acc <= 0.7, (
                    f"{short}/null: spurious separation on pure noise "
                    f"(LOO nearest-centroid acc {acc:.4f} > 0.7, "
                    f"chance {1.0 / len(set(tasks.tolist())):.3f})")

    # determinism spot check: recompute one signature twice
    first = list(aio.iter_runs(base / "bank_effect"))[0]
    d_model = _infer_d_model(first["run_dir"])
    ro = make_stub_readout(d_model, seed=seed)
    s1, _ = signature_for_adapter(
        Path(first["run_dir"]) / "adapter_state.pt", ro, seed=seed)
    s2, _ = signature_for_adapter(
        Path(first["run_dir"]) / "adapter_state.pt", ro, seed=seed)
    assert np.array_equal(s1, s2), "signature not bitwise deterministic"
    report["determinism_bitwise"] = True

    (out_dir / "selftest_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print("[vocabsig] SYNTHETIC SELF-TEST PASSED: planted separation "
          "detected, null bank unseparated, bitwise determinism.",
          flush=True)
    return report


# ── CLI ─────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Output-referenced (vocab-signature) adapter "
                    "representation — proposed D1 arm #3. Refuses a "
                    "partial bank unless --allow-partial-bank.")
    parser.add_argument("--bank-root", type=Path, default=aio.REAL_BANK_ROOT)
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="Output dir (never inside the bank tree)")
    parser.add_argument("--readout", choices=("hf", "stub"), default="hf",
                        help="'hf' = partial-load real W_U from the hub "
                             "cache; 'stub' = seeded stand-in (synthetic "
                             "fixtures ONLY)")
    parser.add_argument("--stub-vocab-size", type=int,
                        default=STUB_VOCAB_SIZE)
    parser.add_argument("--n-probes", type=int, default=N_PROBES)
    parser.add_argument("--sketch-dim", type=int, default=SKETCH_DIM)
    parser.add_argument("--topk", type=int, default=TOPK)
    parser.add_argument("--kv-mode", choices=("zero_pad", "exclude"),
                        default=KV_MODE)
    parser.add_argument("--vocab-chunk", type=int, default=VOCAB_CHUNK)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--allow-partial-bank", action="store_true",
                        help="Override the completeness interlock (loud "
                             "PRE-REGISTRATION WARNING; exploratory only)")
    parser.add_argument("--synthetic-selftest", action="store_true",
                        help="Build synthetic fixture banks under "
                             "--out-dir and run the acceptance test; "
                             "never touches the real bank")
    args = parser.parse_args(argv)

    if args.synthetic_selftest:
        run_synthetic_selftest(args.out_dir, seed=args.seed)
        return

    compute_bank_signatures(
        args.bank_root, args.out_dir, readout_mode=args.readout,
        stub_vocab_size=args.stub_vocab_size, n_probes=args.n_probes,
        sketch_dim=args.sketch_dim, topk=args.topk, kv_mode=args.kv_mode,
        vocab_chunk=args.vocab_chunk, seed=args.seed,
        allow_partial=args.allow_partial_bank)


if __name__ == "__main__":
    main()
