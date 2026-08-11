# -*- coding: utf-8 -*-
"""LAT-001 training loop (SPEC.md section 5).

Per optimizer step: sample ``batch_size`` examples with replacement from
the train pool (seeded), sample ONE ``c ~ Uniform{c_min..c_max}`` for the
whole batch (seeded), encode lazily via tasks.encode_batch, forward with
that ``c``, and take soft-target cross-entropy against the uniform
next-hop distribution from tasks.make_targets. AdamW (0.9, 0.95), linear
warmup then constant LR, grad-clip. Checkpoints carry the task
fingerprint so evaluate.py can refuse a checkpoint trained on a
different graph (the interlock).

Determinism (SPEC section 5): all RNG seeded at train() entry from
derive_seed(cfg.seed, tag); batch and c sampling use dedicated
random.Random streams so their draws cannot interleave with anything
else. CPU-only in this workflow -- ``device`` exists for the later
registered GPU phase and defaults to "cpu".

Owner: Implementer C (train). Imports: common, tasks, model, torch
(SPEC section 1).
"""
from __future__ import annotations

import dataclasses
import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from lat001.common import ModelConfig, TaskData, TrainConfig, derive_seed
from lat001.model import ContinuousThoughtTransformer

_REPO_ROOT = Path(__file__).resolve().parents[1]


# ── Result type (SPEC section 7.3) ───────────────────────────────────
@dataclass
class TrainResult:
    losses: list[float]           # per-step loss, all steps
    final_loss: float
    checkpoint_path: str          # last checkpoint written
    steps: int


# ── Loss (SPEC section 5, pinned formula) ────────────────────────────
def soft_target_cross_entropy(logits: torch.Tensor,
                              targets: torch.Tensor) -> torch.Tensor:
    """-(target * log_softmax(logits)).sum(-1).mean() -- targets are the
    uniform soft distribution over next_hops node tokens (zeros elsewhere)."""
    return -(targets * F.log_softmax(logits, dim=-1)).sum(-1).mean()


# ── Checkpointing (SPEC section 5) ───────────────────────────────────
def _save_checkpoint(path: str, model: ContinuousThoughtTransformer,
                     model_config: ModelConfig, cfg: TrainConfig,
                     task_fingerprint: str, step: int,
                     losses: list[float]) -> None:
    # Pinned key set. Configs are stored as plain dicts so the payload
    # stays torch.load(weights_only=True)-safe across torch versions.
    torch.save({
        "model_state": model.state_dict(),
        "model_config": dataclasses.asdict(model_config),
        "train_config": dataclasses.asdict(cfg),
        "task_fingerprint": task_fingerprint,
        "step": step,
        "loss_history": list(losses),
    }, path)


def load_checkpoint(path: str, expect_fingerprint: str | None = None
                    ) -> tuple[ContinuousThoughtTransformer, dict]:
    """Rebuild the model from a checkpoint; returns (model, payload).

    If ``expect_fingerprint`` is given and does not match the stored
    task_fingerprint, raises ValueError -- evaluate.py uses this to
    refuse a checkpoint trained on a different task (SPEC section 5).
    Model is returned on CPU in eval mode; callers move devices.
    """
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if expect_fingerprint is not None \
            and payload["task_fingerprint"] != expect_fingerprint:
        raise ValueError(
            f"task fingerprint mismatch: checkpoint "
            f"{payload['task_fingerprint'][:12]}... != expected "
            f"{expect_fingerprint[:12]}... ({path})")
    model = ContinuousThoughtTransformer(ModelConfig(**payload["model_config"]))
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, payload


# ── Training (SPEC sections 5, 7.3) ──────────────────────────────────
def train(task: TaskData, model_config: ModelConfig,
          cfg: TrainConfig) -> TrainResult:
    # Deferred import: tasks.py (Implementer A) lands in parallel; the
    # dependency direction (train <- tasks, SPEC section 1) is unchanged --
    # deferring only lets model.py/train.py import clean before it exists.
    from lat001 import tasks

    # Determinism setup (SPEC section 5) -- all sub-seeds via derive_seed.
    torch.manual_seed(derive_seed(cfg.seed, "torch"))
    random.seed(derive_seed(cfg.seed, "random"))
    np.random.seed(derive_seed(cfg.seed, "numpy"))
    torch.use_deterministic_algorithms(True)
    rng_batch = random.Random(derive_seed(cfg.seed, "batch"))
    rng_c = random.Random(derive_seed(cfg.seed, "c"))

    if not task.train:
        raise ValueError("task has an empty train pool")
    # Worst-case sequence must fit: prompt + c_max latents + <A>.
    worst = task.seq_len + cfg.c_max + 1
    if worst > model_config.max_seq_len:
        raise ValueError(
            f"seq_len {task.seq_len} + c_max {cfg.c_max} + 1 = {worst} "
            f"exceeds max_seq_len {model_config.max_seq_len}")

    device = torch.device(cfg.device)   # "cpu" in this workflow (hard rule)
    model = ContinuousThoughtTransformer(model_config).to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            betas=(0.9, 0.95),
                            weight_decay=cfg.weight_decay)

    # A RELATIVE checkpoint_dir resolves against the repo root, not the cwd.
    # common.py is frozen, so its default ("lat001/results/ckpt") stays
    # relative; resolving it here keeps SPEC §0's "results write only under
    # lat001/results/" true no matter where train() is called from (harness
    # review 2026-08-11, F-8). Absolute paths — what run_smoke and the tests
    # pass — are used verbatim.
    ckpt_dir = Path(cfg.checkpoint_dir)
    if not ckpt_dir.is_absolute():
        ckpt_dir = _REPO_ROOT / ckpt_dir
    ckpt_dir = str(ckpt_dir)

    os.makedirs(ckpt_dir, exist_ok=True)
    losses: list[float] = []
    checkpoint_path = ""
    for step in range(1, cfg.steps + 1):
        # Linear warmup to cfg.lr over warmup_steps, then constant.
        lr = cfg.lr * min(1.0, step / max(1, cfg.warmup_steps))
        for group in opt.param_groups:
            group["lr"] = lr

        batch = rng_batch.choices(task.train, k=cfg.batch_size)  # with replacement
        c = rng_c.randint(cfg.c_min, cfg.c_max)                  # one c per batch, inclusive
        tokens = tasks.encode_batch(task, batch).to(device)
        targets = tasks.make_targets(task, batch).to(device)

        logits = model(tokens, c)
        loss = soft_target_cross_entropy(logits, targets)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip and cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        losses.append(float(loss.item()))

        if cfg.log_every and step % cfg.log_every == 0:
            print(f"step={step} loss={losses[-1]:.6f} lr={lr:.3e} c={c}")
        if step % cfg.checkpoint_every == 0 or step == cfg.steps:
            checkpoint_path = os.path.join(ckpt_dir,
                                           f"ckpt_step{step:06d}.pt")
            _save_checkpoint(checkpoint_path, model, model_config, cfg,
                             task.fingerprint, step, losses)

    return TrainResult(losses=losses, final_loss=losses[-1],
                       checkpoint_path=checkpoint_path, steps=cfg.steps)


# ── Self-checks (CPU; checkpoints go to a tempdir, not the repo) ─────
if __name__ == "__main__":
    import tempfile

    from lat001.common import SMOKE_MODEL

    # Checkpoint round-trip + fingerprint interlock (no tasks.py needed).
    torch.manual_seed(0)
    model = ContinuousThoughtTransformer(SMOKE_MODEL)
    ref_cfg = TrainConfig(steps=1, batch_size=2, lr=1e-3, c_min=0, c_max=2,
                          seed=42)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ckpt_step000001.pt")
        _save_checkpoint(path, model, SMOKE_MODEL, ref_cfg, "f" * 64, 1, [1.0])
        loaded, payload = load_checkpoint(path, expect_fingerprint="f" * 64)
        assert payload["step"] == 1 and payload["loss_history"] == [1.0]
        for k, v in model.state_dict().items():
            assert torch.equal(v, loaded.state_dict()[k]), k
        try:
            load_checkpoint(path, expect_fingerprint="0" * 64)
            raise AssertionError("fingerprint interlock did not fire")
        except ValueError:
            pass
    print("checkpoint_roundtrip = PASS")
    print("fingerprint_interlock = PASS")

    # End-to-end mini-train iff tasks.py has landed (parallel implementer).
    try:
        from lat001 import tasks  # noqa: F401
        have_tasks = True
    except ImportError:
        have_tasks = False
    if not have_tasks:
        print("end_to_end = SKIP (lat001/tasks.py not present yet)")
    else:
        from lat001.common import TaskConfig
        task = tasks.build_task(TaskConfig(topology="cubic6", size_index=0,
                                           seed=42))
        with tempfile.TemporaryDirectory() as tmp:
            cfg = TrainConfig(steps=20, batch_size=8, lr=1e-3, c_min=0,
                              c_max=task.d_max + 2, seed=42,
                              checkpoint_dir=tmp, checkpoint_every=20,
                              log_every=10)
            r1 = train(task, SMOKE_MODEL, cfg)
            assert all(np.isfinite(r1.losses)) and len(r1.losses) == 20
            m, _ = load_checkpoint(r1.checkpoint_path,
                                   expect_fingerprint=task.fingerprint)
            r2 = train(task, SMOKE_MODEL, cfg)   # SA-5 mini: bitwise repeat
            assert r1.final_loss == r2.final_loss, (r1.final_loss,
                                                    r2.final_loss)
            assert r1.losses == r2.losses
        print(f"end_to_end = PASS (20 steps, final_loss={r1.final_loss:.6f}, "
              f"determinism bitwise)")
    print("ALL TRAIN SELF-CHECKS PASS")
