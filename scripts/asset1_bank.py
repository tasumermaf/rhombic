"""Asset 1 — 480-adapter bank campaign runner (pre-registered, locked card).

THE LOCKED CARD:
    2 families x 6 tasks x N=40 independent runs = 480 adapters, SEQUENTIAL
    on one GPU. Per run: 2,000 steps, LoRA rank 24, n_channels 6,
    batch 2 x grad-accum 8, bf16 base / fp32 adapters, target modules
    q/k/v/o_proj, RhombiLoRALinear(bridge_mode="identity"), LM loss ONLY —
    no Steersman, no contrastive, no spectral losses. This replicates the
    pilot regime of scripts/train_task_fingerprint.py (model wrapping,
    training loop, lr/schedule, max_len conventions reused from there).

SCALE-CHANGE NOTE (recorded deviation from the pilot bank): the pilot bank
    (train_task_fingerprint.py / train_exp2_scale.py) was trained on
    Qwen/Qwen2.5-7B-Instruct. THIS bank trains Qwen/Qwen2.5-1.5B-Instruct
    and meta-llama/Llama-3.2-1B-Instruct per the locked experiment card.
    Every run's config.json records the model actually used.

Per-run independence (locked): seed = 10000 + run_index,
    data_seed = 20000 + run_index, with run_index the GLOBAL index 0..479.
    Manifest ordering is ROUND-ROBIN across the 12 (family, task) cells,
    replicate-major — an interrupted campaign leaves a balanced partial bank.

Data design (locked, see asset1_datasets.py): per task a FIXED validation
    split (first 500 examples under the canonical val_seed=777 shuffle,
    identical across all runs of the task) and a training pool whose ORDER
    is shuffled per run by data_seed. Train AND val loss are recorded every
    100 steps (D-aux needs the generalization-gap trajectory).

License-blocked family handling: meta-llama is currently gated for this
    account. The runner probes each family once (config.json download); a
    blocked family's runs are skipped with status BLOCKED (no marker file is
    written, so a later invocation — after access lands — picks them up).

Process model: each run executes in a FRESH subprocess (true process-level
    determinism; leaks/fragmentation cannot accumulate across the ~2-week
    campaign; a CUDA crash in one run cannot kill the campaign). The child
    seeds python/numpy/torch from the run's seed and PYTHONHASHSEED is set
    in its environment.

Filesystem protocol per run (results/asset1-bank/{family_short}/{task}/run_{idx:03d}/):
    adapter_state.pt            all lora_A / lora_B / bridge tensors, fp32
    bridge_final_{module}.npy   per wrapped module (pilot naming — analysis reuse)
    bridge_step0_{module}.npy   initial bridges (pilot parity)
    config.json                 every parameter + git hash + versions + timestamps
    metrics.json                train/val loss every 100 steps + finals + wall time
    COMPLETE                    marker written LAST (atomic completion signal)
    FAILED + error.log          on failure (retried once at campaign end)

Campaign controls (in the bank root):
    PAUSE  — between runs, the campaign logs and polls every 60 s until removed
    STOP   — graceful shutdown after the current run

Usage:
    python scripts/asset1_bank.py --dry-run
    python scripts/asset1_bank.py --smoke
    python scripts/asset1_bank.py                     # the full campaign
    python scripts/asset1_bank.py --family qwen --max-runs 10
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for _p in (str(SCRIPTS_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from asset1_datasets import (  # noqa: E402
    POOL_CAP,
    VAL_SEED,
    VAL_SIZE,
    TASK_LONG_NAMES,
    TASK_REGISTRY,
    build_dataset,
    ids_sha256,
)

# ── Locked campaign constants ───────────────────────────────────────

FAMILIES: list[dict] = [
    {"model": "Qwen/Qwen2.5-1.5B-Instruct", "short": "qwen2.5-1.5b"},
    {"model": "meta-llama/Llama-3.2-1B-Instruct", "short": "llama3.2-1b"},
]
TASKS: list[str] = ["alpaca", "code", "math", "xsum", "squad", "agnews"]
N_REPLICATES = 40

SEED_BASE = 10_000
DATA_SEED_BASE = 20_000

MAX_STEPS = 2_000
RANK = 24
N_CHANNELS = 6
LORA_ALPHA = 16.0          # RhombiLoRALinear default — recorded explicitly
BRIDGE_MODE = "identity"
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
BATCH_SIZE = 4          # A1 (Director-adopted 2026-07-06): 2->4, effective batch
GRAD_ACCUM = 4          # unchanged at 16. Bit-equivalent to 2x8 (equal-token padding
                        # + accum-boundary clip + optimizer-step-counted schedule).
LR = 2e-4
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
WARMUP_STEPS = 100         # pilot (train_task_fingerprint.py) convention
MAX_LEN = 512
EVAL_INTERVAL = 100        # record train AND val loss every 100 steps
EVAL_BATCH_SIZE = 4        # 500 val examples -> 125 equal batches (exact mean)

BLOCKED_EXIT_CODE = 78     # child exit code signalling a license-gated family

DEFAULT_BANK_ROOT = REPO_ROOT / "results" / "asset1-bank"
SMOKE_BANK_ROOT = REPO_ROOT / "results" / "asset1-smoke"
SMOKE_TASKS = ["alpaca", "squad"]
SMOKE_MAX_STEPS = 50
SMOKE_POOL_CAP = 2_000     # smoke bounds eager tokenization; same code path


# ── Campaign / run specs ────────────────────────────────────────────


@dataclass
class CampaignSpec:
    tag: str
    bank_root: Path
    families: list[dict]
    tasks: list[str]
    n_replicates: int
    max_steps: int
    pool_cap: int

    @property
    def n_cells(self) -> int:
        return len(self.families) * len(self.tasks)


@dataclass
class RunSpec:
    run_index: int
    family: str
    family_short: str
    task: str
    replicate: int
    seed: int
    data_seed: int
    run_dir: Path

    def to_manifest_entry(self, status: str) -> dict:
        d = asdict(self)
        d["run_dir"] = str(self.run_dir)
        d["status"] = status
        return d


def make_campaign_spec(smoke: bool = False) -> CampaignSpec:
    if smoke:
        return CampaignSpec(
            tag="smoke", bank_root=SMOKE_BANK_ROOT, families=FAMILIES,
            tasks=SMOKE_TASKS, n_replicates=1,
            max_steps=SMOKE_MAX_STEPS, pool_cap=SMOKE_POOL_CAP,
        )
    return CampaignSpec(
        tag="bank", bank_root=DEFAULT_BANK_ROOT, families=FAMILIES,
        tasks=TASKS, n_replicates=N_REPLICATES,
        max_steps=MAX_STEPS, pool_cap=POOL_CAP,
    )


def generate_manifest(spec: CampaignSpec) -> list[RunSpec]:
    """Round-robin across the (family, task) cells, replicate-major.

    run_index is GLOBAL (0..N-1): seeds derived from it are distinct across
    the entire campaign. Directory names use the global index, so a cell's
    runs are e.g. run_000, run_012, run_024, ... for a 12-cell campaign.
    """
    cells = [(fam, task) for fam in spec.families for task in spec.tasks]
    runs: list[RunSpec] = []
    idx = 0
    for rep in range(spec.n_replicates):
        for fam, task in cells:
            runs.append(RunSpec(
                run_index=idx,
                family=fam["model"],
                family_short=fam["short"],
                task=task,
                replicate=rep,
                seed=SEED_BASE + idx,
                data_seed=DATA_SEED_BASE + idx,
                run_dir=spec.bank_root / fam["short"] / task / f"run_{idx:03d}",
            ))
            idx += 1
    return runs


# ── Small utilities ─────────────────────────────────────────────────


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_write_json(path: Path, obj) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2))


def campaign_log(bank_root: Path, msg: str) -> None:
    """Append-only, one line per event."""
    bank_root.mkdir(parents=True, exist_ok=True)
    line = f"{utc_now()} | {msg}\n"
    with open(bank_root / "campaign.log", "a", encoding="utf-8") as f:
        f.write(line)
    print(f"[campaign] {msg}", flush=True)


def git_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
        ).strip()
    except Exception:
        return "unknown"


def library_versions() -> dict:
    import torch
    versions = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
    }
    try:
        import transformers
        versions["transformers"] = transformers.__version__
    except Exception:
        versions["transformers"] = "unavailable"
    try:
        import datasets
        versions["datasets"] = datasets.__version__
    except Exception:
        versions["datasets"] = "unavailable"
    if torch.cuda.is_available():
        versions["gpu"] = torch.cuda.get_device_name(0)
    return versions


def is_blocked_error(exc: BaseException) -> bool:
    """Classify a model-load failure as license/gating (vs infrastructure)."""
    name = type(exc).__name__
    text = f"{name}: {exc}".lower()
    return (
        "gatedrepo" in name.lower()
        or "gated repo" in text
        or "gated" in text and ("403" in text or "401" in text or "access" in text)
        or "403" in text
        or "401" in text
        or "awaiting a review" in text
    )


def probe_family_access(model_name: str) -> str:
    """Return 'OK', 'BLOCKED', or 'UNKNOWN' (probe inconclusive -> attempt).

    hf_hub_download of config.json is the reliable probe: model_info()
    succeeds on gated repos (metadata is public) and cannot be trusted.
    """
    try:
        from huggingface_hub import hf_hub_download
        hf_hub_download(model_name, "config.json")
        return "OK"
    except Exception as e:
        if is_blocked_error(e):
            return "BLOCKED"
        return "UNKNOWN"


def disk_preflight(spec: CampaignSpec) -> list[str]:
    """Report free space at the RESOLVED HF cache and the bank root.

    Motivation (2026-07-03 dry run): the default HF cache
    (~/.cache/huggingface) on this machine is a junction onto a nearly-full
    drive; XSum/SQuAD downloads failed with os error 112. The resolved
    location is what matters — os.path.realpath follows junctions.
    Redirect per-invocation with HF_HUB_CACHE / HF_DATASETS_CACHE if needed
    (never persisted by this runner).
    """
    import shutil
    warnings: list[str] = []
    targets = {}
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
        targets["hf_hub_cache"] = HF_HUB_CACHE
    except Exception:
        targets["hf_hub_cache"] = os.path.join(
            os.path.expanduser("~"), ".cache", "huggingface", "hub")
    targets["hf_datasets_cache"] = os.environ.get(
        "HF_DATASETS_CACHE",
        os.path.join(os.path.expanduser("~"), ".cache", "huggingface",
                     "datasets"))
    targets["bank_root"] = str(spec.bank_root)

    thresholds_gb = {"hf_hub_cache": 5.0, "hf_datasets_cache": 5.0,
                     "bank_root": 20.0}
    print("Disk preflight (resolved paths):")
    for label, path in targets.items():
        probe = Path(path)
        while not probe.exists() and probe.parent != probe:
            probe = probe.parent
        resolved = os.path.realpath(probe)
        free_gb = shutil.disk_usage(resolved).free / 1e9
        print(f"  {label:<18} {path}")
        print(f"  {'':<18} -> {resolved} | free: {free_gb:.1f} GB")
        if free_gb < thresholds_gb[label]:
            warnings.append(
                f"LOW DISK: {label} resolves to {resolved} with only "
                f"{free_gb:.1f} GB free (< {thresholds_gb[label]} GB). "
                f"Set HF_HUB_CACHE/HF_DATASETS_CACHE to a roomier drive "
                f"or free space before the campaign.")
    for w in warnings:
        print(f"  *** {w}")
    return warnings


def scan_run_status(run: RunSpec) -> str:
    if (run.run_dir / "COMPLETE").exists():
        return "COMPLETE"
    if (run.run_dir / "FAILED").exists():
        return "FAILED"
    return "PENDING"


def write_manifest_file(spec: CampaignSpec, runs: list[RunSpec],
                        statuses: dict[int, str]) -> None:
    obj = {
        "updated_at": utc_now(),
        "campaign": {
            "tag": spec.tag,
            "bank_root": str(spec.bank_root),
            "families": spec.families,
            "tasks": spec.tasks,
            "n_replicates": spec.n_replicates,
            "n_runs": len(runs),
            "max_steps": spec.max_steps,
            "pool_cap": spec.pool_cap,
            "seed_base": SEED_BASE,
            "data_seed_base": DATA_SEED_BASE,
            "val_seed": VAL_SEED,
            "val_size": VAL_SIZE,
        },
        "status_counts": _status_counts(statuses),
        "runs": [r.to_manifest_entry(statuses[r.run_index]) for r in runs],
    }
    atomic_write_json(spec.bank_root / "bank_manifest.json", obj)


def _status_counts(statuses: dict[int, str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for s in statuses.values():
        counts[s] = counts.get(s, 0) + 1
    return dict(sorted(counts.items()))


# ── Single-run execution (child process) ────────────────────────────


def set_run_determinism(seed: int) -> None:
    """Fresh process-level determinism: python / numpy / torch (+CUDA).

    PYTHONHASHSEED is set by the parent in the child's environment (it must
    exist before interpreter start to take effect).
    """
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def inject_rhombi_lora(model, rank: int, n_channels: int,
                       target_modules: list[str]):
    """Wrap q/k/v/o projections with RhombiLoRALinear.

    Reuses the pilot's model-wrapping via train_task_fingerprint.inject_lora
    (identity bridge, fp32 adapters, frozen base inside the wrapper).
    """
    from train_task_fingerprint import TaskConfig, inject_lora

    cfg = TaskConfig(
        task_name="asset1", dataset_name="asset1",
        rank=rank, n_channels=n_channels,
    )
    cfg.target_modules = list(target_modules)
    return inject_lora(model, cfg)


def evaluate_val_loss(model, dataloader, device) -> float:
    """Mean val loss over the FULL fixed validation split.

    All sequences are padded to max_len with loss over every position
    (pilot label convention), so every batch carries an identical token
    count and the mean of per-batch means is the exact global mean.
    """
    import torch
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            out = model(input_ids=input_ids, attention_mask=attention_mask,
                        labels=labels)
            total += out.loss.item()
            n += 1
    model.train()
    return total / max(n, 1)


def verify_gradient_checkpointing(model, injected) -> None:
    """After the first backward: prove grads flow through the RhombiLoRA
    wrapping under gradient checkpointing, and that the base stays frozen.

    Note: lora_B is zero-initialized, so lora_A/bridge grads are exactly
    zero at step 0 — the meaningful signals are (a) backward succeeded,
    (b) lora_B has a non-None, nonzero gradient, (c) grads exist on all
    adapter params, (d) no base parameter accumulated a gradient.
    """
    import torch
    b_norm_total = 0.0
    for name, lora in injected.items():
        for pname in ("lora_A", "lora_B", "bridge"):
            p = getattr(lora, pname)
            if p.grad is None:
                raise RuntimeError(
                    f"gradient checkpointing verification FAILED: "
                    f"{name}.{pname}.grad is None"
                )
            if not torch.isfinite(p.grad).all():
                raise RuntimeError(
                    f"gradient checkpointing verification FAILED: "
                    f"non-finite grad on {name}.{pname}"
                )
        b_norm_total += lora.lora_B.grad.float().norm().item()
    if b_norm_total <= 0.0:
        raise RuntimeError(
            "gradient checkpointing verification FAILED: all lora_B grads "
            "are zero after the first backward"
        )
    for pname, p in model.named_parameters():
        if p.requires_grad:
            continue
        if p.grad is not None and p.grad.abs().sum().item() > 0:
            raise RuntimeError(
                f"base parameter {pname} accumulated gradient — base not frozen"
            )
    print(f"  Gradient checkpointing VERIFIED "
          f"(sum lora_B grad norm = {b_norm_total:.3e})", flush=True)


def run_single(spec: CampaignSpec, run: RunSpec) -> int:
    """Execute one run end-to-end. Returns process exit code.

    0 = COMPLETE, BLOCKED_EXIT_CODE = family license-blocked, 1 = FAILED.
    """
    out_dir = run.run_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if (out_dir / "COMPLETE").exists():
        print(f"Run {run.run_index}: COMPLETE marker present — skipping.")
        return 0
    # Clear a stale FAILED marker (this attempt supersedes it)
    for stale in ("FAILED", "error.log"):
        p = out_dir / stale
        if p.exists():
            p.unlink()

    started_at = utc_now()
    t_start = time.time()

    import torch
    from torch.utils.data import DataLoader

    set_run_determinism(run.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'=' * 70}")
    print(f"Asset 1 run {run.run_index:03d} | {run.family_short} | "
          f"{run.task} ({TASK_LONG_NAMES[run.task]}) | replicate {run.replicate}")
    print(f"seed={run.seed} data_seed={run.data_seed} steps={spec.max_steps}")
    print(f"{'=' * 70}\n", flush=True)

    # ── Model + tokenizer (BLOCKED detection lives here) ──
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(run.family)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            run.family, dtype=torch.bfloat16, device_map=device,
        )
    except Exception as e:
        if is_blocked_error(e):
            print(f"Run {run.run_index}: family {run.family} is license-blocked "
                  f"({type(e).__name__}). Exiting with BLOCKED status.")
            return BLOCKED_EXIT_CODE
        err = traceback.format_exc()
        print(err, file=sys.stderr, flush=True)
        try:
            (out_dir / "error.log").write_text(err, encoding="utf-8")
            atomic_write_text(out_dir / "FAILED", utc_now() + "\n")
        except Exception:
            pass
        return 1

    try:
        # ── Gradient checkpointing (pilot conventions + explicit use_cache) ──
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(
                make_inputs_require_grad)
        if hasattr(model, "config"):
            model.config.use_cache = False

        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        injected = inject_rhombi_lora(model, RANK, N_CHANNELS, TARGET_MODULES)
        print(f"Injected {len(injected)} RhombiLoRA adapters "
              f"(rank={RANK}, n_channels={N_CHANNELS}, bridge={BRIDGE_MODE})")
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {trainable:,}", flush=True)

        # ── Datasets: fixed val split + per-run-shuffled train pool ──
        print("Building datasets...", flush=True)
        train_ds = build_dataset(run.task, tokenizer, "train", run.data_seed,
                                 MAX_LEN, pool_cap=spec.pool_cap)
        val_ds = build_dataset(run.task, tokenizer, "val", run.data_seed,
                               MAX_LEN, pool_cap=spec.pool_cap)
        val_hash = ids_sha256(val_ds.example_ids)
        print(f"Dataset source: {train_ds.dataset_source} | "
              f"raw={train_ds.n_total_raw} val={len(val_ds)} "
              f"pool={len(train_ds)} | val_ids sha256={val_hash[:16]}...",
              flush=True)

        # Per-run epoch reshuffling, deterministic in data_seed (the pool
        # composition + initial order already derive from data_seed inside
        # the dataset; the generator makes multi-epoch reshuffles — GSM8K
        # cycles ~4.6 epochs — deterministic too, matching the pilot's
        # shuffle=True regime without its global-RNG nondeterminism).
        g = torch.Generator()
        g.manual_seed(run.data_seed)
        train_loader = DataLoader(
            train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=g,
            num_workers=0, pin_memory=True, drop_last=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False,
            num_workers=0, pin_memory=True,
        )

        # ── Optimizer + pilot LR schedule ──
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=LR, weight_decay=WEIGHT_DECAY,
        )

        def lr_schedule(step):
            if step < WARMUP_STEPS:
                return step / max(WARMUP_STEPS, 1)
            progress = (step - WARMUP_STEPS) / max(
                spec.max_steps - WARMUP_STEPS, 1)
            return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

        # ── config.json (written at start; finalized at end) ──
        config = {
            "asset": "asset1-bank",
            "campaign_tag": spec.tag,
            "run_index": run.run_index,
            "replicate": run.replicate,
            "family": run.family,
            "family_short": run.family_short,
            "task": run.task,
            "task_name": TASK_LONG_NAMES[run.task],
            "model": run.family,
            "dataset": {
                "source": train_ds.dataset_source,
                "config_name": TASK_REGISTRY[run.task].dataset_config_name,
                "hf_split": "train",
                "split_design": (
                    "canonical shuffle = numpy RandomState(val_seed) "
                    "permutation of the raw train split; val = first "
                    "val_size canonical ids (identical across runs); "
                    "train pool = next pool_cap canonical ids (fixed "
                    "composition), order shuffled per-run by "
                    "RandomState(data_seed); DataLoader epoch reshuffle "
                    "uses torch.Generator(data_seed)"
                ),
                "val_seed": VAL_SEED,
                "val_size": VAL_SIZE,
                "pool_cap": spec.pool_cap,
                "n_total_raw": train_ds.n_total_raw,
                "n_val": len(val_ds),
                "n_pool": len(train_ds),
                "val_ids_sha256": val_hash,
            },
            "seed": run.seed,
            "data_seed": run.data_seed,
            "steps": spec.max_steps,
            "lr": LR,
            "lr_schedule": (
                f"linear warmup {WARMUP_STEPS} steps + cosine to floor 0.1 "
                "(pilot train_task_fingerprint.py convention)"
            ),
            "warmup_steps": WARMUP_STEPS,
            "weight_decay": WEIGHT_DECAY,
            "grad_clip": GRAD_CLIP,
            "rank": RANK,
            "n_channels": N_CHANNELS,
            "lora_alpha": LORA_ALPHA,
            "bridge_mode": BRIDGE_MODE,
            "target_modules": TARGET_MODULES,
            "n_injected_modules": len(injected),
            "batch_size": BATCH_SIZE,
            "gradient_accumulation": GRAD_ACCUM,
            "effective_batch": BATCH_SIZE * GRAD_ACCUM,
            "batch_geometry": f"bs{BATCH_SIZE}xga{GRAD_ACCUM}",  # A1 cohort tag
            "batch_geometry_note": (
                "A1 amendment adopted 2026-07-06: bs4xga4 replaces bs2xga8 "
                "(effective batch 16 unchanged; bit-equivalent up to float "
                "summation order under the equal-token padding convention). "
                "The bs2xga8 cohort (runs 0-54) is archived at "
                "results/asset1-bank-bs2x8-archive/ and re-run here for "
                "provenance uniformity. Never mix cohorts by this tag."
            ),
            "max_len": MAX_LEN,
            "eval_interval": EVAL_INTERVAL,
            "eval_batch_size": EVAL_BATCH_SIZE,
            "dtype": "bfloat16 base / float32 adapters",
            "gradient_checkpointing": True,
            "loss": "LM loss ONLY (no Steersman / contrastive / spectral)",
            "label_convention": (
                "labels = input_ids.clone() over the full padded sequence "
                "(pilot convention, kept for regime comparability)"
            ),
            "scale_note": (
                "pilot bank was Qwen2.5-7B; this bank is 1.5B/1B per the "
                "locked experiment card — recorded scale change"
            ),
            "git_commit": git_commit_hash(),
            "library_versions": library_versions(),
            "started_at": started_at,
            "finished_at": None,
            "wall_time_seconds": None,
        }
        atomic_write_json(out_dir / "config.json", config)

        # ── Step-0 bridges (pilot parity) ──
        for name, lora in injected.items():
            safe = name.replace(".", "_")
            np.save(out_dir / f"bridge_step0_{safe}.npy",
                    lora.bridge.detach().cpu().numpy())

        # ── Baseline val loss (step 0 of the generalization-gap trajectory) ──
        print("Evaluating baseline val loss (step 0)...", flush=True)
        records: list[dict] = []
        val0 = evaluate_val_loss(model, val_loader, device)
        records.append({
            "step": 0, "train_loss": None, "val_loss": val0,
            "lr": 0.0, "wall_time_s": round(time.time() - t_start, 1),
        })
        print(f"  Step 0 | val: {val0:.4f}", flush=True)

        def flush_metrics(final: dict | None = None):
            atomic_write_json(out_dir / "metrics.json", {
                "records": records,
                "final": final,
                "wall_time_seconds": round(time.time() - t_start, 1),
            })

        flush_metrics()

        # ── Training loop (pilot structure; global micro-step accumulation
        #    counter so windows are always exactly GRAD_ACCUM batches) ──
        model.train()
        global_step = 0
        micro_step = 0
        acc_loss = 0.0
        window_steps = 0
        verified_gc = False
        done = False

        while not done:
            for batch in train_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / GRAD_ACCUM
                loss.backward()
                acc_loss += loss.item()
                micro_step += 1

                if not verified_gc:
                    verify_gradient_checkpointing(model, injected)
                    verified_gc = True

                if micro_step % GRAD_ACCUM == 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        GRAD_CLIP,
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                    window_steps += 1

                    if (global_step % EVAL_INTERVAL == 0
                            or global_step >= spec.max_steps):
                        train_avg = acc_loss / max(window_steps, 1)
                        val_loss = evaluate_val_loss(model, val_loader, device)
                        elapsed = time.time() - t_start
                        sps = global_step / max(elapsed, 0.01)
                        eta = (spec.max_steps - global_step) / max(sps, 1e-9)
                        records.append({
                            "step": global_step,
                            "train_loss": train_avg,
                            "val_loss": val_loss,
                            "lr": scheduler.get_last_lr()[0],
                            "wall_time_s": round(elapsed, 1),
                        })
                        print(f"  Step {global_step:>5}/{spec.max_steps} | "
                              f"train: {train_avg:.4f} | val: {val_loss:.4f} | "
                              f"gap: {val_loss - train_avg:+.4f} | "
                              f"{sps:.3f} step/s | ETA: {eta / 3600:.2f}h",
                              flush=True)
                        acc_loss = 0.0
                        window_steps = 0
                        flush_metrics()

                    if global_step >= spec.max_steps:
                        done = True
                        break

        # ── Save artifacts (COMPLETE marker written LAST) ──
        print("\nSaving artifacts...", flush=True)
        adapter_state = {}
        for name, lora in injected.items():
            safe = name.replace(".", "_")
            adapter_state[f"{safe}.lora_A"] = lora.lora_A.detach().cpu().float()
            adapter_state[f"{safe}.lora_B"] = lora.lora_B.detach().cpu().float()
            adapter_state[f"{safe}.bridge"] = (
                lora.effective_bridge.detach().cpu().float())
            adapter_state[f"{safe}.scaling"] = torch.tensor(lora.scaling)
            adapter_state[f"{safe}.n_channels"] = torch.tensor(lora.n_channels)
            adapter_state[f"{safe}.rank"] = torch.tensor(lora.rank)
            np.save(out_dir / f"bridge_final_{safe}.npy",
                    lora.effective_bridge.detach().cpu().numpy())
        torch.save(adapter_state, out_dir / "adapter_state.pt")

        wall = round(time.time() - t_start, 1)
        final = {
            "step": global_step,
            "train_loss": records[-1]["train_loss"],
            "val_loss": records[-1]["val_loss"],
        }
        flush_metrics(final=final)

        config["finished_at"] = utc_now()
        config["wall_time_seconds"] = wall
        atomic_write_json(out_dir / "config.json", config)

        # Atomic completion signal — written LAST
        atomic_write_text(out_dir / "COMPLETE", utc_now() + "\n")
        print(f"Run {run.run_index:03d} COMPLETE in {wall / 3600:.2f}h "
              f"(final train {final['train_loss']:.4f} / "
              f"val {final['val_loss']:.4f})", flush=True)
        return 0

    except Exception:
        err = traceback.format_exc()
        print(err, file=sys.stderr, flush=True)
        try:
            (out_dir / "error.log").write_text(err, encoding="utf-8")
            atomic_write_text(out_dir / "FAILED", utc_now() + "\n")
        except Exception:
            pass
        return 1


# ── Campaign orchestration (parent process) ─────────────────────────


def family_matches(run: RunSpec, family_filter: str | None) -> bool:
    if not family_filter:
        return True
    f = family_filter.lower()
    return f in run.family.lower() or f in run.family_short.lower()


def wait_for_pause(spec: CampaignSpec) -> None:
    pause = spec.bank_root / "PAUSE"
    while pause.exists():
        campaign_log(spec.bank_root,
                     "PAUSE file present — sleeping 60s (remove to resume)")
        time.sleep(60)
        if (spec.bank_root / "STOP").exists():
            return


def stop_requested(spec: CampaignSpec) -> bool:
    return (spec.bank_root / "STOP").exists()


def launch_run(spec: CampaignSpec, run: RunSpec) -> int:
    """Spawn the run in a fresh child process; return its exit code."""
    cmd = [sys.executable, str(Path(__file__).resolve()),
           "--run-single", str(run.run_index)]
    if spec.tag == "smoke":
        cmd.append("--smoke")
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(run.seed)
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env)
    return proc.returncode


def execute_run(spec: CampaignSpec, run: RunSpec,
                statuses: dict[int, str],
                family_access: dict[str, str]) -> str:
    """Run one manifest entry via subprocess; classify + persist status."""
    campaign_log(spec.bank_root,
                 f"RUN_START idx={run.run_index} family={run.family_short} "
                 f"task={run.task} rep={run.replicate} seed={run.seed}")
    t0 = time.time()
    rc = launch_run(spec, run)
    dur = time.time() - t0

    if (run.run_dir / "COMPLETE").exists():
        status = "COMPLETE"
    elif rc == BLOCKED_EXIT_CODE:
        status = "BLOCKED"
        family_access[run.family] = "BLOCKED"
    else:
        status = "FAILED"
        # Hard crash (no in-run handler) — parent writes the markers.
        if not (run.run_dir / "FAILED").exists():
            run.run_dir.mkdir(parents=True, exist_ok=True)
            (run.run_dir / "error.log").write_text(
                f"child process exited with code {rc} without writing "
                f"FAILED marker (hard crash / OOM / signal)\n",
                encoding="utf-8")
            atomic_write_text(run.run_dir / "FAILED", utc_now() + "\n")

    statuses[run.run_index] = status
    campaign_log(spec.bank_root,
                 f"RUN_END idx={run.run_index} status={status} rc={rc} "
                 f"dur={dur:.0f}s")
    return status


def run_campaign(spec: CampaignSpec, family_filter: str | None,
                 max_runs: int) -> None:
    spec.bank_root.mkdir(parents=True, exist_ok=True)
    runs = generate_manifest(spec)
    statuses = {r.run_index: scan_run_status(r) for r in runs}
    write_manifest_file(spec, runs, statuses)

    campaign_log(spec.bank_root,
                 f"CAMPAIGN_START tag={spec.tag} runs={len(runs)} "
                 f"filter={family_filter or 'none'} max_runs={max_runs or 'all'} "
                 f"git={git_commit_hash()[:12]}")

    for warning in disk_preflight(spec):
        campaign_log(spec.bank_root, f"WARNING {warning}")

    # One access probe per family; BLOCKED families are skipped (no marker,
    # so a later campaign invocation retries them once access lands).
    family_access: dict[str, str] = {}
    for fam in spec.families:
        family_access[fam["model"]] = probe_family_access(fam["model"])
        campaign_log(spec.bank_root,
                     f"FAMILY_ACCESS {fam['model']} = {family_access[fam['model']]}")

    executed = 0
    stopped = False

    def budget_left() -> bool:
        return not max_runs or executed < max_runs

    # ── Main pass ──
    for run in runs:
        if not family_matches(run, family_filter):
            continue
        if statuses[run.run_index] == "COMPLETE":
            continue
        if stop_requested(spec):
            stopped = True
            break
        wait_for_pause(spec)
        if stop_requested(spec):
            stopped = True
            break
        if family_access.get(run.family) == "BLOCKED":
            statuses[run.run_index] = "BLOCKED"
            campaign_log(spec.bank_root,
                         f"RUN_BLOCKED idx={run.run_index} "
                         f"family={run.family_short} (license-gated; will be "
                         f"picked up on a future invocation)")
            write_manifest_file(spec, runs, statuses)
            continue
        if not budget_left():
            campaign_log(spec.bank_root, f"MAX_RUNS reached ({max_runs})")
            break
        executed += 1
        execute_run(spec, run, statuses, family_access)
        write_manifest_file(spec, runs, statuses)

    # ── Retry pass: FAILED runs retried once ──
    if not stopped:
        failed = [r for r in runs
                  if statuses[r.run_index] == "FAILED"
                  and family_matches(r, family_filter)]
        if failed:
            campaign_log(spec.bank_root,
                         f"RETRY_PASS {len(failed)} failed run(s)")
        for run in failed:
            if stop_requested(spec):
                stopped = True
                break
            wait_for_pause(spec)
            if stop_requested(spec):
                stopped = True
                break
            if family_access.get(run.family) == "BLOCKED":
                continue
            campaign_log(spec.bank_root, f"RETRY idx={run.run_index}")
            execute_run(spec, run, statuses, family_access)
            write_manifest_file(spec, runs, statuses)

    # ── Final report ──
    counts = _status_counts(statuses)
    campaign_log(spec.bank_root,
                 f"CAMPAIGN_END stopped={stopped} executed={executed} "
                 f"counts={counts}")
    remaining = [r.run_index for r in runs
                 if statuses[r.run_index] in ("FAILED", "PENDING", "BLOCKED")]
    print(f"\n{'=' * 70}")
    print(f"Campaign summary ({spec.tag}): {counts}")
    if remaining:
        print(f"Remaining (non-COMPLETE) run indices: "
              f"{remaining[:40]}{' ...' if len(remaining) > 40 else ''}")
        print("Re-running this command resumes exactly where it left off "
              "(COMPLETE runs are skipped; BLOCKED re-attempted).")
    else:
        print("All runs COMPLETE.")
    print(f"{'=' * 70}")


# ── Dry run ─────────────────────────────────────────────────────────


def dry_run(spec: CampaignSpec) -> None:
    runs = generate_manifest(spec)
    statuses = {r.run_index: scan_run_status(r) for r in runs}

    print(f"\n{'=' * 70}")
    print(f"DRY RUN — Asset 1 campaign manifest ({spec.tag})")
    print(f"{'=' * 70}")
    print(f"Total runs: {len(runs)} "
          f"({len(spec.families)} families x {len(spec.tasks)} tasks x "
          f"{spec.n_replicates} replicates)")
    cells: dict[tuple, int] = {}
    for r in runs:
        cells[(r.family_short, r.task)] = cells.get((r.family_short, r.task), 0) + 1
    for (fam, task), n in cells.items():
        print(f"  {fam:>14} / {task:<8} : {n} runs")
    print(f"Ordering: round-robin across {len(cells)} cells, replicate-major")
    print("First cycle:", [(r.family_short, r.task) for r in runs[:len(cells)]])
    print(f"Seeds: seed = {SEED_BASE}+idx, data_seed = {DATA_SEED_BASE}+idx "
          f"(global run index)")
    print(f"Existing status counts: {_status_counts(statuses)}")

    print(f"\n{'-' * 70}")
    disk_preflight(spec)

    # ── Family access ──
    print(f"\n{'-' * 70}\nFamily access probe:")
    tokenizer_family = None
    for fam in spec.families:
        access = probe_family_access(fam["model"])
        print(f"  {fam['model']:<40} {access}")
        if access == "OK" and tokenizer_family is None:
            tokenizer_family = fam["model"]
    if tokenizer_family is None:
        print("  No family accessible — cannot run the dataset check.")
        return

    # ── Dataset availability + one formatted example per task ──
    print(f"\n{'-' * 70}")
    print(f"Dataset availability check (tokenizer: {tokenizer_family})")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_family)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_tasks = TASKS if spec.tag == "bank" else spec.tasks
    for task in all_tasks:
        print(f"\n### Task: {task} ({TASK_LONG_NAMES[task]})")
        try:
            ds = build_dataset(task, tokenizer, "val", data_seed=0,
                               max_len=MAX_LEN, pool_cap=spec.pool_cap,
                               keep_text=True)
            print(f"  source: {ds.dataset_source}")
            print(f"  raw train rows: {ds.n_total_raw} | fixed val: {len(ds)} "
                  f"| train pool: {ds.pool_size}")
            print(f"  val_ids sha256: {ids_sha256(ds.example_ids)[:16]}...")
            example = ds.formatted_texts[0]
            snippet = example if len(example) <= 700 else example[:700] + " [...]"
            print("  --- formatted example [val idx 0] ---")
            for line in snippet.splitlines():
                print(f"  | {line}")
            print("  --- end example ---")
            print(f"  STATUS: OK")
        except Exception as e:
            print(f"  STATUS: FAILED — {type(e).__name__}: {e}")

    print(f"\n{'=' * 70}\nDry run complete. No training was performed.\n")


# ── CLI ─────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Asset 1 — 480-adapter bank campaign runner")
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke campaign: 2 runs per family (alpaca + "
                             "squad) at 50 steps, output results/asset1-smoke/")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print manifest + dataset availability check "
                             "only (no training)")
    parser.add_argument("--max-runs", type=int, default=0,
                        help="Execute at most N runs this session "
                             "(0 = unlimited)")
    parser.add_argument("--family", type=str, default=None,
                        help="Only execute runs whose family matches this "
                             "substring (e.g. 'qwen')")
    parser.add_argument("--run-single", type=int, default=None,
                        help=argparse.SUPPRESS)  # internal: child-process mode
    args = parser.parse_args()

    spec = make_campaign_spec(smoke=args.smoke)

    if args.run_single is not None:
        runs = generate_manifest(spec)
        if not (0 <= args.run_single < len(runs)):
            parser.error(f"--run-single {args.run_single} out of range "
                         f"(0..{len(runs) - 1})")
        run = runs[args.run_single]
        assert run.run_index == args.run_single
        sys.exit(run_single(spec, run))

    if args.dry_run:
        dry_run(spec)
        return

    run_campaign(spec, args.family, args.max_runs)


if __name__ == "__main__":
    main()
