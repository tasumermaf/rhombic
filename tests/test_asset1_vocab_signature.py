"""Tests for the output-referenced adapter representations (Lane E-1).

Covers the pre-registered properties of asset1_vocab_signature (Level A,
cheap/exact/CPU) and the CPU-safe surfaces of asset1_jacobian_lens
(Level B, GPU-gated — Stage B itself is NOT exercised here):

  1. GAUGE INVARIANCE — signatures of (B G, G^{-1} A) match signatures
     of (B, A) to ~1e-10 for random GL(r), r in {6, 24} (the
     test_canonicalize.py pattern; the signature depends only on Delta).
  2. DETERMINISM — bitwise-identical recomputation, including across a
     freshly rebuilt readout object.
  3. PLANTED-DIFFERENCE SENSITIVITY — on asset1_synth fixtures with
     task_effect=1.0, different-task adapters are distinguishable
     (between-task centroid distance > max within-task distance) and
     identical adapters give identical signatures; a task_effect=0.0
     bank shows no such separation.
  4. STUB-W_U PARTIAL LOAD — load_unembedding on a stub HF snapshot
     reads ONLY the unembedding + final-norm tensors (loaded_keys), and
     under a sharded index it opens ONLY the shard files that hold them
     (files_opened); tied-embedding fallback covered.
  5. IMPORT HYGIENE — importing either module (in a fresh interpreter)
     pulls in neither transformers nor CUDA; signature computation
     leaves CUDA uninitialized.
  6. GPU GATE — asset1_jacobian_lens --estimate refuses without the
     explicit acknowledgment flag, before any lazy transformers import.
"""

from __future__ import annotations

import json
import subprocess
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
for _p in (str(SCRIPTS_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

torch = pytest.importorskip("torch")

import asset1_jacobian_lens as jlens  # noqa: E402
import asset1_synth as synth  # noqa: E402
import asset1_vocab_signature as vs  # noqa: E402

GAUGE_ATOL = 1e-10   # pre-registered invariance tolerance (Delta-exact)


# ── Helpers ─────────────────────────────────────────────────────────


def _random_factors(d_out: int, r: int, d_in: int, seed: int):
    gen = torch.Generator().manual_seed(seed)
    B = torch.randn(d_out, r, generator=gen, dtype=torch.float64)
    A = torch.randn(r, d_in, generator=gen, dtype=torch.float64)
    return B, A


def _random_invertible(r: int, seed: int) -> torch.Tensor:
    """Well-conditioned G in GL(r) (test_canonicalize.py pattern) so the
    test measures the signature, not round-off through a singular G."""
    gen = torch.Generator().manual_seed(seed)
    for _ in range(64):
        G = torch.randn(r, r, generator=gen, dtype=torch.float64)
        if torch.linalg.cond(G).item() < 1e3:
            return G
    raise RuntimeError("could not draw a well-conditioned G")


def _modules_from_factors(B, A, name="model_layers_0_self_attn_q_proj"):
    """A plain-LoRA module dict (no bridge -> effective_factors uses
    E = I, so the GL gauge acts directly on the effective update)."""
    return {name: {"lora_A": A, "lora_B": B}}


def _stub_readout(d_model: int, vocab: int = 64, seed: int = 0):
    return vs.make_stub_readout(d_model, vocab, seed)


# ── 1. Gauge invariance ─────────────────────────────────────────────


@pytest.mark.parametrize("r,d_out,d_in", [(6, 40, 32), (24, 96, 64)])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_gauge_invariance(r, d_out, d_in, seed):
    B, A = _random_factors(d_out, r, d_in, seed)
    G = _random_invertible(r, seed + 100)
    readout = _stub_readout(d_out, vocab=96, seed=0)
    s1, _ = vs.signature_for_modules(_modules_from_factors(B, A), readout)
    s2, _ = vs.signature_for_modules(
        _modules_from_factors(B @ G, torch.linalg.inv(G) @ A), readout)
    diff = np.abs(s1.astype(np.float64) - s2.astype(np.float64)).max()
    assert diff < GAUGE_ATOL, f"vocab signature not GL({r})-invariant: {diff:.3e}"


def test_gauge_invariance_multi_module_zero_pad():
    """Mixed d_out (q at d_model, k/v at d_model//2, zero-padded) —
    invariance must hold through the residual-placement step too."""
    d_model = 32
    modules = {}
    factors = {}
    for name, d_out in (("model_layers_0_self_attn_q_proj", d_model),
                        ("model_layers_0_self_attn_k_proj", d_model // 2),
                        ("model_layers_0_self_attn_o_proj", d_model)):
        B, A = _random_factors(d_out, 6, d_model,
                               seed=zlib.crc32(name.encode()) % 1000)
        modules[name] = {"lora_A": A, "lora_B": B}
        factors[name] = (B, A)
    readout = _stub_readout(d_model, vocab=80, seed=1)
    s1, lay1 = vs.signature_for_modules(modules, readout)
    G = _random_invertible(6, seed=42)
    transformed = {
        name: {"lora_A": torch.linalg.inv(G) @ A, "lora_B": B @ G}
        for name, (B, A) in factors.items()}
    s2, _ = vs.signature_for_modules(transformed, readout)
    assert lay1["n_modules_kept"] == 3
    diff = np.abs(s1.astype(np.float64) - s2.astype(np.float64)).max()
    assert diff < GAUGE_ATOL


def test_jlens_signature_gauge_invariant_and_deterministic():
    """Stage-C jlens signature (stub lenses) is a fixed linear map of
    Delta: gauge-invariant and bitwise deterministic."""
    B, A = _random_factors(24, 6, 24, seed=3)
    rng = np.random.default_rng(5)
    lenses = {"model.layers.0.self_attn.q_proj":
              rng.standard_normal((4, 24))}
    mods = _modules_from_factors(B, A)
    s1, lay = jlens.jlens_signature_for_modules(mods, lenses)
    s1b, _ = jlens.jlens_signature_for_modules(mods, lenses)
    assert np.array_equal(s1, s1b)
    assert lay["dim"] == 4 * vs.N_PROBES
    G = _random_invertible(6, seed=7)
    s2, _ = jlens.jlens_signature_for_modules(
        _modules_from_factors(B @ G, torch.linalg.inv(G) @ A), lenses)
    assert np.abs(s1.astype(np.float64)
                  - s2.astype(np.float64)).max() < GAUGE_ATOL


# ── 2. Determinism ──────────────────────────────────────────────────


def test_signature_bitwise_deterministic_across_readout_rebuild():
    B, A = _random_factors(40, 6, 32, seed=11)
    mods = _modules_from_factors(B, A)
    s1, _ = vs.signature_for_modules(mods, _stub_readout(40, 64, seed=2))
    s2, _ = vs.signature_for_modules(mods, _stub_readout(40, 64, seed=2))
    assert np.array_equal(s1, s2), "not bitwise deterministic"
    s3, _ = vs.signature_for_modules(mods, _stub_readout(40, 64, seed=3))
    assert not np.array_equal(s1, s3), "seed does not enter the sketch"


def test_probe_and_sketch_generators_deterministic_and_shared():
    assert np.array_equal(vs.probe_inputs(32, 16, 0),
                          vs.probe_inputs(32, 16, 0))
    assert np.array_equal(vs.vocab_sketch(64, 8, 0),
                          vs.vocab_sketch(64, 8, 0))
    # Level B imports the SAME generators (identical objects) — the two
    # output-referenced levels differ only in the readout map.
    assert jlens.probe_inputs is vs.probe_inputs
    assert jlens.vocab_sketch is vs.vocab_sketch


def test_topk_chunk_invariance():
    """Chunked top-k must equal unchunked exactly (the documented
    chunk-invariant selection rule)."""
    W = torch.randn(97, 12, generator=torch.Generator().manual_seed(1))
    g = torch.rand(12, generator=torch.Generator().manual_seed(2)) + 0.5
    M = np.random.default_rng(3).standard_normal((12, 5))
    r_full = vs.VocabReadout(W, g, vocab_chunk=1000)
    r_chunk = vs.VocabReadout(W, g, vocab_chunk=7)
    tk_full = r_full.topk_signed_logits(M, 8)
    tk_chunk = r_chunk.topk_signed_logits(M, 8)
    # Selection rule is chunk-invariant; VALUES agree to float64
    # round-off (BLAS blocking differs across chunk shapes).
    assert np.allclose(tk_full, tk_chunk, rtol=1e-12, atol=1e-12)
    assert np.allclose(r_full.T, r_chunk.T, rtol=1e-12, atol=1e-12)
    # descending |value| order per column
    assert np.all(np.diff(np.abs(tk_full), axis=0) <= 0)


def test_kv_mode_exclude_and_layout():
    d_model = 24
    modules = {}
    for name, d_out in (("model_layers_0_self_attn_q_proj", d_model),
                        ("model_layers_0_self_attn_k_proj", d_model // 2)):
        B, A = _random_factors(d_out, 6, d_model, seed=99)
        modules[name] = {"lora_A": A, "lora_B": B}
    readout = _stub_readout(d_model, 64, seed=0)
    _, lay_pad = vs.signature_for_modules(modules, readout,
                                          kv_mode="zero_pad")
    sig_ex, lay_ex = vs.signature_for_modules(modules, readout,
                                              kv_mode="exclude")
    assert lay_pad["n_modules_kept"] == 2 and not lay_pad["modules_excluded"]
    assert lay_ex["n_modules_kept"] == 1
    assert lay_ex["modules_excluded"] == ["model_layers_0_self_attn_k_proj"]
    per = readout.sketch_dim * vs.N_PROBES + vs.TOPK
    assert lay_pad["dim"] == 2 * per and sig_ex.size == per
    with pytest.raises(ValueError):
        vs.signature_for_modules(modules, readout, kv_mode="bogus")


# ── 3. Planted-difference sensitivity (synthetic fixtures) ──────────


@pytest.fixture(scope="module")
def synth_banks(tmp_path_factory):
    root = tmp_path_factory.mktemp("vocabsig-fixture")
    kw = dict(n_families=1, n_tasks=2, n_reps=3, n_layers=2,
              d_model=16, rank=4, n_channels=2, seed=8)
    eff = synth.make_synthetic_bank(root / "effect", task_effect=1.0, **kw)
    nul = synth.make_synthetic_bank(root / "null", task_effect=0.0, **kw)
    return {"effect": (root / "effect", eff), "null": (root / "null", nul)}


def _bank_signatures(bank_root, n_runs, seed=0):
    import asset1_analysis_io as aio
    readout = None
    rows, tasks = [], []
    for rec in aio.iter_runs(bank_root):
        if readout is None:
            readout = _stub_readout(
                vs._infer_d_model(rec["run_dir"]), seed=seed)
        sig, _ = vs.signature_for_adapter(
            Path(rec["run_dir"]) / "adapter_state.pt", readout, seed=seed)
        rows.append(sig.astype(np.float64))
        tasks.append(rec["task"])
    assert len(rows) == n_runs
    return np.stack(rows), np.array(tasks)


def test_planted_difference_detected(synth_banks):
    """Effect bank: LOO nearest-centroid on L2-normalized signatures
    recovers the planted tasks perfectly (the selftest acceptance
    metric — normalization removes the generator's radial dev_mag
    variance; see loo_nearest_centroid_accuracy docstring)."""
    bank, info = synth_banks["effect"]
    X, tasks = _bank_signatures(bank, info["n_runs"])
    acc = vs.loo_nearest_centroid_accuracy(X, tasks)
    assert acc == 1.0, f"planted task structure not recovered: acc {acc}"


def test_null_bank_not_separated(synth_banks):
    bank, info = synth_banks["null"]
    X, tasks = _bank_signatures(bank, info["n_runs"])
    acc = vs.loo_nearest_centroid_accuracy(X, tasks)
    assert acc < 0.9, (
        f"spurious separation on pure noise: LOO nearest-centroid acc "
        f"{acc} (chance 0.5, n=6)")


def test_identical_adapters_identical_signatures(synth_banks):
    import asset1_analysis_io as aio
    bank, _ = synth_banks["effect"]
    rec = next(iter(aio.iter_runs(bank)))
    readout = _stub_readout(vs._infer_d_model(rec["run_dir"]), seed=0)
    path = Path(rec["run_dir"]) / "adapter_state.pt"
    s1, _ = vs.signature_for_adapter(path, readout)
    s2, _ = vs.signature_for_adapter(path, readout)
    assert np.array_equal(s1, s2)


# ── 4. Stub-W_U partial load ────────────────────────────────────────


def _write_stub_snapshot(cache_dir: Path, model_id: str,
                         tensors_by_file: dict[str, dict],
                         index: bool) -> Path:
    """Fake HF hub cache entry: refs/main + snapshots/<rev>/ with one or
    more safetensors files (and an index when sharded)."""
    from safetensors.torch import save_file

    repo = cache_dir / ("models--" + model_id.replace("/", "--"))
    rev = "abc123def"
    (repo / "refs").mkdir(parents=True)
    (repo / "refs" / "main").write_text(rev, encoding="utf-8")
    snap = repo / "snapshots" / rev
    snap.mkdir(parents=True)
    weight_map = {}
    for fname, tensors in tensors_by_file.items():
        save_file(tensors, str(snap / fname))
        for k in tensors:
            weight_map[k] = fname
    if index:
        (snap / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": weight_map}), encoding="utf-8")
    return snap


def test_partial_load_single_file_with_decoys(tmp_path):
    V, d = 32, 8
    gen = torch.Generator().manual_seed(0)
    tensors = {
        "lm_head.weight": torch.randn(V, d, generator=gen),
        "model.norm.weight": torch.rand(d, generator=gen) + 0.5,
        "model.embed_tokens.weight": torch.randn(V, d, generator=gen),
        "model.layers.0.self_attn.q_proj.weight":
            torch.randn(d, d, generator=gen),
    }
    _write_stub_snapshot(tmp_path, "stub/tiny",
                         {"model.safetensors": tensors}, index=False)
    info = vs.load_unembedding("stub/tiny", cache_dir=tmp_path)
    assert info["vocab_size"] == V and info["d_model"] == d
    assert info["loaded_keys"] == ["lm_head.weight", "model.norm.weight"]
    assert info["tied_embeddings_fallback"] is False
    assert torch.allclose(info["W_U"], tensors["lm_head.weight"])
    assert torch.allclose(info["norm_g"], tensors["model.norm.weight"])
    # decoy tensors were never requested
    assert "model.layers.0.self_attn.q_proj.weight" not in info["loaded_keys"]


def test_partial_load_sharded_opens_only_needed_files(tmp_path):
    """Sharded snapshot: the decoy-only shard must never be opened —
    the on-disk proof of the partial load."""
    V, d = 16, 4
    gen = torch.Generator().manual_seed(1)
    shard_a = {"lm_head.weight": torch.randn(V, d, generator=gen),
               "model.norm.weight": torch.rand(d, generator=gen) + 0.5}
    shard_b = {"model.layers.0.mlp.up_proj.weight":
               torch.randn(d, d, generator=gen)}
    _write_stub_snapshot(
        tmp_path, "stub/sharded",
        {"model-00001-of-00002.safetensors": shard_a,
         "model-00002-of-00002.safetensors": shard_b}, index=True)
    info = vs.load_unembedding("stub/sharded", cache_dir=tmp_path)
    assert info["files_opened"] == ["model-00001-of-00002.safetensors"]
    assert info["loaded_keys"] == ["lm_head.weight", "model.norm.weight"]


def test_partial_load_tied_embedding_fallback(tmp_path):
    V, d = 16, 4
    gen = torch.Generator().manual_seed(2)
    tensors = {"model.embed_tokens.weight": torch.randn(V, d, generator=gen),
               "model.norm.weight": torch.rand(d, generator=gen) + 0.5}
    _write_stub_snapshot(tmp_path, "stub/tied",
                         {"model.safetensors": tensors}, index=False)
    info = vs.load_unembedding("stub/tied", cache_dir=tmp_path)
    assert info["tied_embeddings_fallback"] is True
    assert torch.allclose(info["W_U"], tensors["model.embed_tokens.weight"])


def test_partial_load_missing_model_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        vs.load_unembedding("stub/absent", cache_dir=tmp_path)


# ── 5. Import hygiene / no CUDA ─────────────────────────────────────


def test_import_hygiene_fresh_interpreter():
    """Fresh interpreter: importing both lane modules must pull in
    neither transformers nor initialize CUDA (guardrail 1)."""
    code = (
        "import sys; sys.path.insert(0, r'{s}'); sys.path.insert(0, r'{r}')\n"
        "import asset1_vocab_signature, asset1_jacobian_lens\n"
        "assert 'transformers' not in sys.modules, 'transformers imported'\n"
        "assert 'datasets' not in sys.modules, 'datasets imported'\n"
        "import torch\n"
        "assert not torch.cuda.is_initialized(), 'CUDA initialized'\n"
        "print('OK')\n"
    ).format(s=str(SCRIPTS_DIR), r=str(REPO_ROOT))
    proc = subprocess.run([sys.executable, "-c", code],
                          capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout


def test_no_cuda_after_signature_computation():
    B, A = _random_factors(24, 6, 24, seed=5)
    vs.signature_for_modules(_modules_from_factors(B, A),
                             _stub_readout(24, 48, seed=0))
    assert not torch.cuda.is_initialized()


# ── 6. GPU gate (Stage B refusal) ───────────────────────────────────


def test_jlens_estimate_refused_without_gate_flag(tmp_path):
    with pytest.raises(SystemExit, match="i-have-gpu-and-bank-is-complete"):
        jlens.main(["--estimate", "--family", "qwen2.5-1.5b",
                    "--out-dir", str(tmp_path)])


def test_jlens_plan_only_is_cpu_and_deterministic(tmp_path):
    jlens.main(["--plan-only", "--out-dir", str(tmp_path)])
    plan = json.loads((tmp_path / "jlens_plan.json").read_text(
        encoding="utf-8"))
    assert plan["n_contexts"] == 32
    assert plan["contexts_sha256"] == jlens._contexts_sha256(
        jlens.DEFAULT_CONTEXTS)
    assert plan["sketch_dim"] == vs.SKETCH_DIM
    assert "pinned_defaults" in plan
    assert "transformers" not in jlens.__dict__
    assert not torch.cuda.is_initialized()


def test_out_dir_guard_refuses_bank_tree():
    with pytest.raises(SystemExit, match="asset1-bank"):
        vs._guard_out_dir(REPO_ROOT / "results" / "asset1-bank" / "x")
