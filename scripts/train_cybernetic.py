"""Experiment 3: Cybernetic Bridge Training.

The lattice IS the steersman.

Previous experiments:
  Exp 2:   Open loop — train, checkpoint, analyze post-hoc
  Exp 2.5: Geometric data — still open loop, fixed schedule
  Exp 2.6: Contrastive loss — directional pressure but static scheduling

This experiment closes the loop. Every N steps:
  1. SENSOR: Extract bridge matrices from all adapters
  2. OSCILLOSCOPE: Compute spectral diagnostics (Fiedler, co/cross ratio,
     spectral gap, deviation, eigenvalue distribution)
  3. STEERSMAN: Decide parameter adjustments based on measurement history
     and derivative trends (PID-like control)
  4. ACTUATOR: Adjust learning rate, contrastive weight, spectral
     regularization weight, bridge learning rate
  5. SYSTEM: Training continues with adjusted parameters → back to SENSOR

Two new loss components:
  - Spectral regularization: differentiable Fiedler proxy drives algebraic
    connectivity directly as a training objective (not post-hoc measurement)
  - Adaptive contrastive: same as Exp 2.6 but weight determined by the
    Steersman based on co/cross ratio trajectory

The Steersman (kybernetes) implements three control laws:
  1. CONNECTIVITY: If Fiedler declining → increase spectral reg
  2. DIRECTIONALITY: If co/cross stagnant → increase contrastive weight
  3. STABILITY: If deviation growing too fast → dampen bridge LR

Deployment:
  # Quick local test (6 minutes)
  python scripts/train_cybernetic.py --max-steps 500 --feedback-interval 50

  # Full run on RunPod A100
  python scripts/train_cybernetic.py --max-steps 20000 --output results/exp3

  # Extended convergence study
  python scripts/train_cybernetic.py --max-steps 50000 --feedback-interval 200

Usage:
  TASUMER MAF = 1093 = kybernetes (exact). The lattice steers itself.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from rhombic.nn.rhombi_lora import RhombiLoRALinear, EmanationBridge
from rhombic.nn.topology import direction_pair_coupling
from rhombic.spectral import (
    weighted_laplacian,
    fiedler_value,
    spectral_gap,
    spectrum,
    eigenvalue_multiplicity_pattern,
)

# Reuse from Exp 2
from train_exp2_scale import (
    ExperimentConfig,
    AlpacaDataset,
    inject_lora,
    collect_metrics,
    gradient_effective_rank,
    coplanar_crossplanar_ratio,
    evaluate,
)

# Co-planar pairs from Exp 2.6
from train_contrastive_bridge import (
    _compute_pair_indices,
    _compute_wrong_label_pairs,
    _compute_resonance_pairs,
    contrastive_bridge_loss,
    circular_resonance_loss,
)


# ── Differentiable spectral diagnostics ──────────────────────────────


def differentiable_fiedler(bridge: torch.Tensor) -> torch.Tensor:
    """Compute Fiedler value of bridge matrix as a differentiable operation.

    Constructs the weighted Laplacian from |B[i,j]| off-diagonal elements,
    then returns the second-smallest eigenvalue via torch.linalg.eigvalsh
    (which IS differentiable through autograd).

    Parameters
    ----------
    bridge : (C, C) tensor — the bridge matrix

    Returns
    -------
    Scalar tensor — the algebraic connectivity (Fiedler value)
    """
    C = bridge.shape[0]
    # Absolute off-diagonal weights
    W = bridge.abs()
    W = W - torch.diag(torch.diag(W))  # zero diagonal

    # Weighted Laplacian: L[i,i] = sum of row, L[i,j] = -W[i,j]
    L = -W
    L = L + torch.diag(W.sum(dim=1))

    # Eigenvalues (ascending) — differentiable
    eigenvalues = torch.linalg.eigvalsh(L)
    return eigenvalues[1]  # Second smallest = Fiedler


def spectral_reg_loss(
    injected: dict[str, RhombiLoRALinear],
    target_fiedler: float = 0.5,
) -> torch.Tensor:
    """Spectral regularization: drive Fiedler value toward target.

    L_spectral = mean((fiedler_i - target)^2) for all adapters.

    The loss is minimized when every bridge's algebraic connectivity
    equals the target. The Steersman adjusts the target dynamically.
    """
    device = next(iter(injected.values())).effective_bridge.device
    losses = []

    for lora in injected.values():
        B = lora.effective_bridge
        if B.shape[0] < 2:
            continue
        fv = differentiable_fiedler(B)
        losses.append((fv - target_fiedler) ** 2)

    if not losses:
        return torch.tensor(0.0, device=device)
    return torch.stack(losses).mean()


# ── The Steersman (kybernetes) ───────────────────────────────────────


@dataclass
class SteersmanState:
    """Snapshot of what the Steersman observes and decides."""
    step: int
    # Measurements (sensor → oscilloscope)
    fiedler_mean: float
    fiedler_std: float
    co_cross_ratio: Optional[float]
    spectral_gap_mean: float
    deviation_mean: float
    deviation_std: float
    grad_eff_rank: float
    eigenvalue_pattern: list  # multiplicity pattern from first adapter
    # Decisions (steersman → actuator)
    lr_scale: float          # multiplier on base LR
    contrastive_weight: float
    spectral_weight: float
    spectral_target: float   # target Fiedler value
    bridge_lr_scale: float   # multiplier on bridge param LR
    # Derivative estimates
    fiedler_trend: float     # slope of Fiedler over last K measurements
    co_cross_trend: float    # slope of co/cross over last K measurements
    deviation_trend: float   # slope of deviation over last K measurements
    # Diagnostic
    control_signals: dict = field(default_factory=dict)


class Steersman:
    """Cybernetic controller for RhombiLoRA bridge training.

    Implements three control laws based on spectral measurement history:
    1. CONNECTIVITY: Fiedler value declining → increase spectral reg
    2. DIRECTIONALITY: co/cross ratio stagnant → increase contrastive weight
    3. STABILITY: deviation growing too fast → dampen bridge LR

    The Steersman maintains a sliding window of measurements and computes
    linear trends (derivatives) over that window. Control outputs are
    bounded to prevent oscillation.
    """

    def __init__(
        self,
        # Control law parameters
        base_contrastive_weight: float = 0.1,
        base_spectral_weight: float = 0.05,
        initial_spectral_target: float = 0.1,
        # Bounds
        max_contrastive_weight: float = 0.5,
        max_spectral_weight: float = 0.2,
        max_bridge_lr_scale: float = 3.0,
        min_bridge_lr_scale: float = 0.1,
        # Sensitivity
        fiedler_decline_threshold: float = -0.001,
        co_cross_stagnation_band: float = 0.02,
        deviation_rate_threshold: float = 0.05,
        # Spectral target adaptation
        spectral_target_tracking_rate: float = 0.1,
        # Window size for trend estimation
        window_size: int = 5,
    ):
        self.base_contrastive = base_contrastive_weight
        self.base_spectral = base_spectral_weight
        self.spectral_target = initial_spectral_target

        self.max_contrastive = max_contrastive_weight
        self.max_spectral = max_spectral_weight
        self.max_bridge_lr = max_bridge_lr_scale
        self.min_bridge_lr = min_bridge_lr_scale

        self.fiedler_decline_thresh = fiedler_decline_threshold
        self.co_cross_stagnation = co_cross_stagnation_band
        self.deviation_rate_thresh = deviation_rate_threshold
        self.target_tracking_rate = spectral_target_tracking_rate

        self.window_size = window_size
        self.history: list[SteersmanState] = []

        # Current actuator outputs
        self._lr_scale = 1.0
        self._contrastive_weight = base_contrastive_weight
        self._spectral_weight = base_spectral_weight
        self._bridge_lr_scale = 1.0

    def _trend(self, values: list[float]) -> float:
        """Linear regression slope over the last window_size values."""
        # Filter out None, NaN, and inf
        clean = [v for v in values if v is not None and np.isfinite(v)]
        if len(clean) < 2:
            return 0.0
        v = clean[-self.window_size:]
        n = len(v)
        x = np.arange(n, dtype=np.float64)
        y = np.array(v, dtype=np.float64)
        # OLS slope
        x_mean = x.mean()
        y_mean = y.mean()
        denom = ((x - x_mean) ** 2).sum()
        if denom < 1e-12:
            return 0.0
        slope = ((x - x_mean) * (y - y_mean)).sum() / denom
        return float(slope) if np.isfinite(slope) else 0.0

    def observe_and_decide(
        self,
        step: int,
        injected: dict[str, RhombiLoRALinear],
    ) -> SteersmanState:
        """The full cybernetic loop: sense → analyze → decide.

        Parameters
        ----------
        step : current training step
        injected : dict of adapter name → RhombiLoRALinear module

        Returns
        -------
        SteersmanState with measurements and decisions
        """
        # ── SENSOR: extract bridge matrices ──
        bridges = {}
        for name, lora in injected.items():
            bridges[name] = lora.effective_bridge.detach().cpu().numpy()

        # ── OSCILLOSCOPE: spectral diagnostics ──
        fiedlers = []
        gaps = []
        deviations = []
        patterns = []
        for name, B in bridges.items():
            n = B.shape[0]
            edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
            weights = [float(abs(B[i, j])) for i, j in edges]
            fiedlers.append(fiedler_value(n, edges, weights))
            gaps.append(spectral_gap(n, edges, weights))
            # Deviation from identity
            dev = float(np.linalg.norm(B - np.eye(n)))
            deviations.append(dev)
            # Eigenvalue pattern (first adapter only for logging)
            if not patterns:
                eigs = spectrum(n, edges, weights)
                patterns.append(eigenvalue_multiplicity_pattern(eigs))

        f_mean = float(np.mean(fiedlers))
        f_std = float(np.std(fiedlers))
        gap_mean = float(np.mean(gaps))
        d_mean = float(np.mean(deviations))
        d_std = float(np.std(deviations))

        # Co/cross ratio — aggregate across ALL bridges
        co_cross_values = []
        for B in bridges.values():
            r = coplanar_crossplanar_ratio(B)
            if r is not None:
                ratio = r["ratio"]
                if np.isfinite(ratio):
                    co_cross_values.append(ratio)
        co_cross = float(np.mean(co_cross_values)) if co_cross_values else None

        # Gradient effective rank
        eff_rank = gradient_effective_rank(injected)

        # ── STEERSMAN: compute trends and decide ──
        fiedler_history = [s.fiedler_mean for s in self.history]
        co_cross_history = [
            s.co_cross_ratio for s in self.history
            if s.co_cross_ratio is not None
        ]
        deviation_history = [s.deviation_mean for s in self.history]

        fiedler_trend = self._trend(fiedler_history + [f_mean])
        co_cross_trend = self._trend(
            co_cross_history + ([co_cross] if co_cross is not None else [])
        )
        deviation_trend = self._trend(deviation_history + [d_mean])

        signals = {}

        # Control Law 1: CONNECTIVITY
        # If Fiedler is declining, boost spectral regularization
        if fiedler_trend < self.fiedler_decline_thresh:
            boost = min(
                abs(fiedler_trend) * 10.0,  # proportional gain
                self.max_spectral - self._spectral_weight,
            )
            self._spectral_weight = min(
                self._spectral_weight + boost,
                self.max_spectral,
            )
            signals["connectivity"] = f"DECLINING (trend={fiedler_trend:.5f}), spec_reg +{boost:.4f}"
        elif fiedler_trend > 0.001:
            # Fiedler growing — ease off spectral pressure to let it find its own way
            decay = min(0.01, self._spectral_weight * 0.1)
            self._spectral_weight = max(
                self._spectral_weight - decay,
                self.base_spectral * 0.5,
            )
            signals["connectivity"] = f"IMPROVING (trend={fiedler_trend:.5f}), spec_reg -{decay:.4f}"
        else:
            signals["connectivity"] = f"STABLE (trend={fiedler_trend:.5f})"

        # Adapt spectral target: track upward but slowly
        if f_mean > self.spectral_target:
            self.spectral_target += self.target_tracking_rate * (
                f_mean - self.spectral_target
            )
            signals["target_adaptation"] = f"target → {self.spectral_target:.4f}"

        # Control Law 2: DIRECTIONALITY
        # If co/cross ratio is stagnant near 1.0, increase contrastive pressure
        if co_cross is not None and len(co_cross_history) >= 2:
            if abs(co_cross_trend) < self.co_cross_stagnation and co_cross < 1.1:
                boost = min(0.02, self.max_contrastive - self._contrastive_weight)
                self._contrastive_weight = min(
                    self._contrastive_weight + boost,
                    self.max_contrastive,
                )
                signals["directionality"] = (
                    f"STAGNANT (ratio={co_cross:.3f}, trend={co_cross_trend:.5f}), "
                    f"contrastive +{boost:.4f}"
                )
            elif co_cross > 1.2:
                # Strong directional preference achieved — ease off
                decay = min(0.01, self._contrastive_weight * 0.1)
                self._contrastive_weight = max(
                    self._contrastive_weight - decay,
                    self.base_contrastive * 0.25,
                )
                signals["directionality"] = (
                    f"STRONG (ratio={co_cross:.3f}), contrastive -{decay:.4f}"
                )
            else:
                signals["directionality"] = (
                    f"MOVING (ratio={co_cross:.3f}, trend={co_cross_trend:.5f})"
                )

        # Control Law 3: STABILITY
        # If deviation growing too fast, dampen bridge learning
        if deviation_trend > self.deviation_rate_thresh:
            dampen = max(0.8, 1.0 - deviation_trend)
            self._bridge_lr_scale = max(
                self._bridge_lr_scale * dampen,
                self.min_bridge_lr,
            )
            signals["stability"] = (
                f"FAST GROWTH (trend={deviation_trend:.5f}), "
                f"bridge_lr ×{dampen:.3f} → {self._bridge_lr_scale:.3f}"
            )
        elif deviation_trend < -0.01:
            # Bridge converging back — can restore LR
            recover = min(1.1, 1.0 + abs(deviation_trend))
            self._bridge_lr_scale = min(
                self._bridge_lr_scale * recover,
                self.max_bridge_lr,
            )
            signals["stability"] = (
                f"CONVERGING (trend={deviation_trend:.5f}), "
                f"bridge_lr ×{recover:.3f} → {self._bridge_lr_scale:.3f}"
            )
        else:
            signals["stability"] = f"STABLE (trend={deviation_trend:.5f})"

        # Build state snapshot
        state = SteersmanState(
            step=step,
            fiedler_mean=f_mean,
            fiedler_std=f_std,
            co_cross_ratio=co_cross,
            spectral_gap_mean=gap_mean,
            deviation_mean=d_mean,
            deviation_std=d_std,
            grad_eff_rank=eff_rank,
            eigenvalue_pattern=patterns[0] if patterns else [],
            lr_scale=self._lr_scale,
            contrastive_weight=self._contrastive_weight,
            spectral_weight=self._spectral_weight,
            spectral_target=self.spectral_target,
            bridge_lr_scale=self._bridge_lr_scale,
            fiedler_trend=fiedler_trend,
            co_cross_trend=co_cross_trend,
            deviation_trend=deviation_trend,
            control_signals=signals,
        )

        self.history.append(state)
        return state


# ── Cybernetic Training Loop ─────────────────────────────────────────


def train_cybernetic(
    config: ExperimentConfig,
    output_dir: Path,
    # Feedback parameters
    feedback_interval: int = 100,
    # Contrastive topology override
    contrastive_topology: str = "auto",
    # Steersman initial values
    initial_contrastive: float = 0.1,
    initial_spectral: float = 0.05,
    initial_spectral_target: float = 0.1,
    # Optional Steersman overrides (corpus preset passes these)
    fiedler_decline_threshold: Optional[float] = None,
    co_cross_stagnation_band: Optional[float] = None,
    deviation_rate_threshold: Optional[float] = None,
    # Emanation architecture
    emanation: bool = False,
    # Model export
    save_merged: bool = False,
):
    """Experiment 3: Cybernetic bridge training with closed-loop feedback.

    The lattice diagnoses itself and adjusts training parameters in real time.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config_dict = asdict(config)
    config_dict["experiment"] = "3.0"
    config_dict["feedback_interval"] = feedback_interval
    config_dict["initial_contrastive"] = initial_contrastive
    config_dict["initial_spectral"] = initial_spectral
    config_dict["initial_spectral_target"] = initial_spectral_target
    config_dict["emanation"] = emanation
    with open(output_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    if contrastive_topology == "wrong-labels":
        co_pairs, cross_pairs = _compute_wrong_label_pairs()
        geom_label = "WRONG-LABELS (random partition of 6 channels)"
    elif contrastive_topology == "resonance":
        co_pairs, cross_pairs = _compute_resonance_pairs(config.n_channels)
        geom_label = "RESONANCE (prime threading topology)"
    else:
        co_pairs, cross_pairs = _compute_pair_indices(config.n_channels)
        geom_label = {6: "RD (rhombic dodecahedron)", 8: "Tesseract (4D hypercube)"}.get(
            config.n_channels, f"n={config.n_channels} (no geometric prior)"
        )
    contrastive_active = bool(co_pairs and cross_pairs)

    print(f"\n{'='*70}")
    print(f"Experiment 3.0: CYBERNETIC Bridge Training")
    print(f"  The lattice IS the steersman.")
    print(f"{'='*70}")
    print(f"Model:              {config.model_name}")
    print(f"Rank:               {config.rank}, Channels: {config.n_channels}")
    print(f"Geometry:           {geom_label}")
    print(f"Contrastive:        {'ACTIVE' if contrastive_active else 'DISABLED (no pairs for this n)'}")
    print(f"Steps:              {config.max_steps}")
    print(f"Feedback interval:  every {feedback_interval} steps")
    print(f"Initial contrastive: {initial_contrastive}")
    print(f"Initial spectral:   {initial_spectral}")
    print(f"Spectral target:    {initial_spectral_target}")
    if contrastive_active:
        print(f"Co-planar pairs:    {co_pairs}")
        print(f"Cross-planar pairs: {len(cross_pairs)} pairs")
    print(f"Emanation:          {'ACTIVE' if emanation else 'DISABLED'}")
    print(f"{'='*70}\n")

    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    print(f"Loading {config.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        dtype=torch.bfloat16,
        device_map=device,
    )

    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(
                make_inputs_require_grad
            )

    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    injected = inject_lora(model, config)
    print(f"Injected {len(injected)} LoRA adapters")

    # Emanation architecture: single master bridge emanates into per-layer bridges
    emanation_bridge = None
    if emanation:
        n_layers = len(injected)
        emanation_bridge = EmanationBridge(
            n_channels=config.n_channels,
            n_layers=n_layers,
        ).to(device)
        # Wire each adapter to its emanated bridge
        for layer_idx, (name, lora) in enumerate(injected.items()):
            # Freeze the adapter's own bridge — emanation replaces it
            lora.freeze_bridge()
            # Set external bridge function: this layer's effective_bridge
            # will call emanation.get_bridge(layer_idx) instead of using
            # the local bridge parameter
            idx = layer_idx  # capture in closure
            lora._external_bridge_fn = lambda _idx=idx: emanation_bridge.get_bridge(_idx)
        print(f"Emanation: master bridge ({config.n_channels}x{config.n_channels}) "
              f"→ {n_layers} layer offsets")
        print(f"Emanation coherence: {emanation_bridge.coherence:.4f}")

    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    # Add emanation params if present (they're not part of the model)
    if emanation_bridge is not None:
        trainable_params += sum(
            p.numel() for p in emanation_bridge.parameters()
        )
    total_params = sum(p.numel() for p in model.parameters())
    if emanation_bridge is not None:
        bridge_params = sum(
            p.numel() for p in emanation_bridge.parameters()
        )
    else:
        bridge_params = sum(
            lora.bridge.numel() for lora in injected.values()
            if lora.bridge.requires_grad
        )
    print(f"Trainable: {trainable_params:,} / {total_params:,}")
    print(f"Bridge params: {bridge_params:,}")

    # Separate parameter groups: LoRA A/B vs Bridge
    lora_params = []
    bridge_param_list = []
    for lora in injected.values():
        lora_params.extend([lora.lora_A, lora.lora_B])
        if not emanation and lora.bridge.requires_grad:
            bridge_param_list.append(lora.bridge)

    # Emanation: master + offsets replace per-adapter bridge params
    if emanation_bridge is not None:
        bridge_param_list.append(emanation_bridge.master)
        bridge_param_list.extend(list(emanation_bridge.offsets))

    optimizer = torch.optim.AdamW(
        [
            {"params": lora_params, "lr": config.lr, "name": "lora"},
            {"params": bridge_param_list, "lr": config.lr, "name": "bridge"},
        ],
        weight_decay=0.01,
    )

    total_steps = config.max_steps

    def lr_schedule(step):
        if step < config.warmup_steps:
            return step / max(config.warmup_steps, 1)
        progress = (step - config.warmup_steps) / max(
            total_steps - config.warmup_steps, 1
        )
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, [lr_schedule, lr_schedule]  # one per param group
    )

    # Dataset
    print("Loading Alpaca-cleaned dataset...")
    dataset = AlpacaDataset(tokenizer, max_len=config.max_seq_len)
    val_dataset = AlpacaDataset(
        tokenizer, max_len=config.max_seq_len, max_samples=1000
    )
    print(f"Training examples: {len(dataset)}")

    dataloader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True,
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    # Initialize the Steersman (with optional preset overrides)
    steersman_kwargs = dict(
        base_contrastive_weight=initial_contrastive,
        base_spectral_weight=initial_spectral,
        initial_spectral_target=initial_spectral_target,
    )
    if fiedler_decline_threshold is not None:
        steersman_kwargs["fiedler_decline_threshold"] = fiedler_decline_threshold
    if co_cross_stagnation_band is not None:
        steersman_kwargs["co_cross_stagnation_band"] = co_cross_stagnation_band
    if deviation_rate_threshold is not None:
        steersman_kwargs["deviation_rate_threshold"] = deviation_rate_threshold
    steersman = Steersman(**steersman_kwargs)

    # Training state
    checkpoints: list[dict] = []
    feedback_log: list[dict] = []
    model.train()
    global_step = 0
    accumulation_loss = 0.0
    accumulation_contrastive = 0.0
    accumulation_spectral = 0.0
    steps_since_log = 0
    start_time = time.time()

    # Save step-0 bridges
    for name, lora in injected.items():
        safe = name.replace(".", "_")
        np.save(
            output_dir / f"bridge_step0_{safe}.npy",
            lora.effective_bridge.detach().cpu().numpy(),
        )
    if emanation_bridge is not None:
        np.save(
            output_dir / "bridge_emanation_master_step0.npy",
            emanation_bridge.master.detach().cpu().numpy(),
        )

    # Initial Steersman read (before training begins)
    initial_state = steersman.observe_and_decide(0, injected)
    feedback_log.append(asdict(initial_state))
    print(f"\n  *** STEERSMAN @ step 0 ***")
    print(f"      Fiedler:       {initial_state.fiedler_mean:.5f}")
    print(f"      Spectral gap:  {initial_state.spectral_gap_mean:.5f}")
    print(f"      Deviation:     {initial_state.deviation_mean:.5f}")
    if initial_state.co_cross_ratio is not None:
        print(f"      Co/Cross:      {initial_state.co_cross_ratio:.3f}")
    if emanation_bridge is not None:
        print(f"      Emanation coh: {emanation_bridge.coherence:.4f}")
    print()

    print(f"Training for {total_steps} steps...")
    print(f"Effective batch size: {config.batch_size * config.gradient_accumulation}")
    print()

    done = False
    for epoch in range(config.num_epochs):
        if done:
            break
        for batch_idx, batch in enumerate(dataloader):
            if done:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass — language modeling loss
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            lm_loss = outputs.loss / config.gradient_accumulation

            # Contrastive bridge loss (Steersman-controlled weight)
            c_weight = steersman._contrastive_weight
            if c_weight > 0 and contrastive_active:
                c_loss = contrastive_bridge_loss(
                    injected, co_pairs, cross_pairs
                )
                scaled_c = c_weight * c_loss / config.gradient_accumulation
                accumulation_contrastive += scaled_c.item()
            else:
                scaled_c = 0.0

            # Spectral regularization (Steersman-controlled weight + target)
            s_weight = steersman._spectral_weight
            if s_weight > 0:
                s_loss = spectral_reg_loss(
                    injected,
                    target_fiedler=steersman.spectral_target,
                )
                scaled_s = s_weight * s_loss / config.gradient_accumulation
                accumulation_spectral += scaled_s.item()
            else:
                scaled_s = 0.0

            total_loss = lm_loss
            if isinstance(scaled_c, torch.Tensor):
                total_loss = total_loss + scaled_c
            if isinstance(scaled_s, torch.Tensor):
                total_loss = total_loss + scaled_s

            total_loss.backward()
            accumulation_loss += lm_loss.item()

            if (batch_idx + 1) % config.gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_norm=1.0,
                )

                # Apply Steersman's bridge LR scaling
                bridge_scale = steersman._bridge_lr_scale
                for pg in optimizer.param_groups:
                    if pg.get("name") == "bridge":
                        pg["lr"] = config.lr * lr_schedule(global_step) * bridge_scale

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                steps_since_log += 1

                # Progress display every 100 steps
                if global_step % 100 == 0:
                    avg_lm = accumulation_loss / steps_since_log
                    avg_c = accumulation_contrastive / max(steps_since_log, 1)
                    avg_s = accumulation_spectral / max(steps_since_log, 1)
                    elapsed = time.time() - start_time
                    speed = global_step / elapsed
                    eta = (total_steps - global_step) / speed

                    print(
                        f"  Step {global_step:>5}/{total_steps} | "
                        f"LM: {avg_lm:.4f} | "
                        f"C: {avg_c:.5f}(w={c_weight:.3f}) | "
                        f"S: {avg_s:.5f}(w={s_weight:.3f}) | "
                        f"BLR: {bridge_scale:.2f} | "
                        f"{speed:.1f} step/s | "
                        f"ETA: {eta/60:.0f}m"
                    )

                # ── FEEDBACK LOOP ──
                if (
                    global_step % feedback_interval == 0
                    or global_step >= total_steps
                ):
                    # The Steersman reads the lattice and decides
                    state = steersman.observe_and_decide(global_step, injected)
                    feedback_log.append(asdict(state))

                    # Dynamic bridge temperature annealing (L9)
                    if config.dynamic_bridge:
                        progress = min(global_step / total_steps, 1.0)
                        T = (
                            config.gate_temperature_start
                            + progress * (
                                config.gate_temperature_end
                                - config.gate_temperature_start
                            )
                        )
                        for lora in injected.values():
                            lora.set_gate_temperature(T)

                    # Validation loss
                    val_loss = evaluate(model, val_dataloader, device)

                    avg_lm = accumulation_loss / steps_since_log
                    avg_c = accumulation_contrastive / max(steps_since_log, 1)
                    avg_s = accumulation_spectral / max(steps_since_log, 1)

                    # Checkpoint data
                    cp = {
                        "step": global_step,
                        "train_loss": avg_lm,
                        "val_loss": val_loss,
                        "contrastive_loss": avg_c,
                        "spectral_loss": avg_s,
                        "fiedler_mean": state.fiedler_mean,
                        "fiedler_std": state.fiedler_std,
                        "co_cross_ratio": state.co_cross_ratio,
                        "spectral_gap": state.spectral_gap_mean,
                        "deviation_mean": state.deviation_mean,
                        "deviation_std": state.deviation_std,
                        "grad_eff_rank": state.grad_eff_rank,
                        # Steersman decisions
                        "contrastive_weight": state.contrastive_weight,
                        "spectral_weight": state.spectral_weight,
                        "spectral_target": state.spectral_target,
                        "bridge_lr_scale": state.bridge_lr_scale,
                        "fiedler_trend": state.fiedler_trend,
                        "co_cross_trend": state.co_cross_trend,
                        "deviation_trend": state.deviation_trend,
                        "control_signals": state.control_signals,
                        "wall_time": time.time() - start_time,
                    }
                    # Dynamic bridge metrics (L9)
                    if config.dynamic_bridge:
                        sample_lora = next(iter(injected.values()))
                        cp["gate_temperature"] = sample_lora.gate_temperature()
                        cp["gate_sparsity"] = sample_lora.gate_sparsity()
                    # Emanation metrics
                    if emanation_bridge is not None:
                        cp["emanation_coherence"] = emanation_bridge.coherence
                    checkpoints.append(cp)

                    # Save bridge matrices
                    for name, lora in injected.items():
                        safe = name.replace(".", "_")
                        np.save(
                            output_dir / f"bridge_step{global_step}_{safe}.npy",
                            lora.effective_bridge.detach().cpu().numpy(),
                        )
                    # Save emanation master bridge
                    if emanation_bridge is not None:
                        np.save(
                            output_dir / f"bridge_emanation_master_step{global_step}.npy",
                            emanation_bridge.master.detach().cpu().numpy(),
                        )

                    # Emanation coherence monitoring: if coherence drops
                    # below 0.5, the system is fragmenting — increase
                    # spectral regularization to pull layers back toward master
                    if emanation_bridge is not None:
                        eman_coh = emanation_bridge.coherence
                        if eman_coh < 0.5:
                            boost = min(
                                0.02,
                                steersman.max_spectral - steersman._spectral_weight,
                            )
                            steersman._spectral_weight = min(
                                steersman._spectral_weight + boost,
                                steersman.max_spectral,
                            )
                            state.control_signals["emanation"] = (
                                f"FRAGMENTING (coherence={eman_coh:.4f}), "
                                f"spec_reg +{boost:.4f}"
                            )
                        else:
                            state.control_signals["emanation"] = (
                                f"COHERENT ({eman_coh:.4f})"
                            )

                    # Print Steersman report
                    co_str = (
                        f"{state.co_cross_ratio:.3f}" if state.co_cross_ratio
                        else "N/A"
                    )
                    eman_str = ""
                    if emanation_bridge is not None:
                        eman_str = (
                            f"    Emanation:     {emanation_bridge.coherence:.4f}\n"
                        )
                    print(
                        f"\n  {'='*60}\n"
                        f"  STEERSMAN @ step {global_step}\n"
                        f"  {'='*60}\n"
                        f"  Lattice Health:\n"
                        f"    Fiedler:       {state.fiedler_mean:.5f} (trend: {state.fiedler_trend:+.5f})\n"
                        f"    Spectral gap:  {state.spectral_gap_mean:.5f}\n"
                        f"    Co/Cross:      {co_str} (trend: {state.co_cross_trend:+.5f})\n"
                        f"    Deviation:     {state.deviation_mean:.5f} (trend: {state.deviation_trend:+.5f})\n"
                        f"    Eff rank:      {state.grad_eff_rank:.2f}\n"
                        f"{eman_str}"
                        f"  Loss Components:\n"
                        f"    LM:            {avg_lm:.4f}\n"
                        f"    Val:           {val_loss:.4f}\n"
                        f"    Contrastive:   {avg_c:.5f}\n"
                        f"    Spectral:      {avg_s:.5f}\n"
                        f"  Control Decisions:\n"
                        f"    Contrastive w: {state.contrastive_weight:.4f}\n"
                        f"    Spectral w:    {state.spectral_weight:.4f}\n"
                        f"    Spectral tgt:  {state.spectral_target:.4f}\n"
                        f"    Bridge LR:     {state.bridge_lr_scale:.3f}x\n"
                        f"  Signals:"
                    )
                    for law, msg in state.control_signals.items():
                        print(f"    {law}: {msg}")
                    print(f"  {'='*60}\n")

                    # Reset accumulators
                    accumulation_loss = 0.0
                    accumulation_contrastive = 0.0
                    accumulation_spectral = 0.0
                    steps_since_log = 0

                    # Save intermediate results
                    _save_results(
                        output_dir, config, checkpoints, feedback_log,
                        trainable_params, total_params, bridge_params,
                        injected,
                        emanation_bridge=emanation_bridge,
                    )

                if global_step >= total_steps:
                    done = True

    # Final save
    _save_results(
        output_dir, config, checkpoints, feedback_log,
        trainable_params, total_params, bridge_params,
        injected,
        emanation_bridge=emanation_bridge,
    )

    # Optionally merge LoRA+bridge into base and save as HF model
    if save_merged:
        print("\nMerging LoRA+bridge into base model for lm-eval...")
        merged_dir = merge_and_save(model, injected, output_dir)
        # Also save the tokenizer so lm-eval can load it
        tokenizer.save_pretrained(merged_dir)
        print(f"  Merged model ready at {merged_dir}")

    elapsed = time.time() - start_time
    print(f"\nExperiment 3.0 COMPLETE")
    print(f"Total time: {elapsed/3600:.2f}h")
    print(f"Feedback cycles: {len(feedback_log)}")
    print(f"Results: {output_dir}")

    # Summary: control trajectory
    if len(feedback_log) > 1:
        print(f"\n  Control Trajectory:")
        print(f"  {'Step':>6} | {'Fiedler':>8} | {'Co/Cross':>8} | "
              f"{'Dev':>8} | {'C_wt':>6} | {'S_wt':>6} | {'B_lr':>5}")
        print(f"  {'-'*60}")
        for entry in feedback_log:
            co = f"{entry['co_cross_ratio']:.3f}" if entry.get('co_cross_ratio') else "  N/A "
            print(
                f"  {entry['step']:>6} | "
                f"{entry['fiedler_mean']:>8.5f} | "
                f"{co:>8} | "
                f"{entry['deviation_mean']:>8.5f} | "
                f"{entry['contrastive_weight']:>6.3f} | "
                f"{entry['spectral_weight']:>6.3f} | "
                f"{entry['bridge_lr_scale']:>5.2f}"
            )

    return checkpoints


def _save_results(
    output_dir, config, checkpoints, feedback_log,
    trainable_params, total_params, bridge_params,
    injected,
    emanation_bridge=None,
):
    """Save all results with cybernetic metadata."""
    results = {
        "config": asdict(config),
        "experiment": "3.0",
        "description": "Cybernetic bridge training — the lattice IS the steersman",
        "checkpoints": checkpoints,
        "feedback_log": feedback_log,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "bridge_params": bridge_params,
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save final bridges
    for name, lora in injected.items():
        safe = name.replace(".", "_")
        np.save(
            output_dir / f"bridge_final_{safe}.npy",
            lora.effective_bridge.detach().cpu().numpy(),
        )
    # Save final emanation master bridge
    if emanation_bridge is not None:
        np.save(
            output_dir / "bridge_emanation_master_final.npy",
            emanation_bridge.master.detach().cpu().numpy(),
        )

    # Save adapter state dict (lora_A, bridge, lora_B per layer — compact)
    adapter_state = {}
    for name, lora in injected.items():
        safe = name.replace(".", "_")
        adapter_state[f"{safe}.lora_A"] = lora.lora_A.detach().cpu()
        adapter_state[f"{safe}.lora_B"] = lora.lora_B.detach().cpu()
        adapter_state[f"{safe}.bridge"] = lora.effective_bridge.detach().cpu()
        adapter_state[f"{safe}.scaling"] = torch.tensor(lora.scaling)
        adapter_state[f"{safe}.n_channels"] = torch.tensor(lora.n_channels)
        adapter_state[f"{safe}.rank"] = torch.tensor(lora.rank)
    torch.save(adapter_state, output_dir / "adapter_state.pt")
    print(f"  Saved adapter state ({len(injected)} layers)")


def merge_and_save(model, injected, output_dir):
    """Merge LoRA+bridge deltas into base weights and save as HF model.

    This produces a standard HuggingFace model that lm-eval can load directly.
    The delta for each adapted layer is: scaling * lora_B @ bridge_expanded @ lora_A
    where bridge_expanded is the block-diagonal expansion of the n×n bridge to rank×rank.
    """
    import copy
    merged = copy.deepcopy(model)

    for name, lora in injected.items():
        # Navigate to the LoRAWrappedLinear and get the base layer
        parts = name.split(".")
        parent = merged
        for p in parts[:-1]:
            parent = getattr(parent, p)
        wrapped = getattr(parent, parts[-1])
        base_linear = wrapped.base if hasattr(wrapped, 'base') else wrapped

        # Compute expanded bridge: (rank, rank) block-diagonal
        bridge = lora.effective_bridge.detach().float()  # (C, C)
        C, R = lora.n_channels, lora.channel_size
        bridge_expanded = torch.zeros(lora.rank, lora.rank)
        for i in range(C):
            for j in range(C):
                bridge_expanded[i*R:(i+1)*R, j*R:(j+1)*R] = bridge[i, j] * torch.eye(R)

        # delta_W = scaling * lora_B @ bridge_expanded @ lora_A
        lora_A = lora.lora_A.detach().float()   # (rank, in_features)
        lora_B = lora.lora_B.detach().float()   # (out_features, rank)
        delta_W = lora.scaling * (lora_B @ bridge_expanded @ lora_A)

        # Merge into base weight
        base_linear.weight.data += delta_W.to(base_linear.weight.dtype).to(base_linear.weight.device)

        # Replace the wrapped module with the plain base linear
        setattr(parent, parts[-1], base_linear)

    # Save merged model
    merged_dir = output_dir / "merged_model"
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(merged_dir)
    print(f"  Saved merged model to {merged_dir}")
    return merged_dir


# ── CLI ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 3.0: Cybernetic Bridge Training"
    )
    parser.add_argument(
        "--output", type=str, default="results/exp3",
        help="Output directory",
    )
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
    )
    parser.add_argument(
        "--max-steps", type=int, default=20000,
        help="Total training steps",
    )
    parser.add_argument(
        "--feedback-interval", type=int, default=100,
        help="Steps between Steersman feedback cycles",
    )
    parser.add_argument(
        "--initial-contrastive", type=float, default=0.1,
        help="Initial contrastive loss weight",
    )
    parser.add_argument(
        "--initial-spectral", type=float, default=0.05,
        help="Initial spectral regularization weight",
    )
    parser.add_argument(
        "--spectral-target", type=float, default=0.1,
        help="Initial Fiedler target for spectral reg",
    )
    parser.add_argument(
        "--lr", type=float, default=2e-4,
    )
    parser.add_argument(
        "--batch-size", type=int, default=2,
    )
    parser.add_argument(
        "--gradient-accumulation", type=int, default=8,
    )
    parser.add_argument(
        "--warmup-steps", type=int, default=200,
    )
    parser.add_argument(
        "--checkpoint-steps", type=int, default=100,
    )
    parser.add_argument(
        "--seed", type=int, default=42,
    )
    parser.add_argument(
        "--bridge-mode", type=str, default="identity",
        choices=["identity", "geometric", "corpus", "corpus_coupled"],
        help="Bridge initialization mode",
    )
    parser.add_argument(
        "--n-channels", type=int, default=6,
        help="Number of bridge channels (default 6 = FCC direction pairs)",
    )
    parser.add_argument(
        "--steersman-preset", type=str, default="default",
        choices=["default", "corpus"],
        help="Steersman parameter preset. 'corpus' uses tighter thresholds "
             "calibrated to corpus-coupled bridge dynamics.",
    )
    parser.add_argument(
        "--dynamic-bridge", action="store_true",
        help="Enable input-dependent dynamic bridge gating (L9 prototype)",
    )
    parser.add_argument(
        "--gate-temperature-start", type=float, default=5.0,
        help="Initial gate temperature for dynamic bridge (default 5.0, soft)",
    )
    parser.add_argument(
        "--gate-temperature-end", type=float, default=0.1,
        help="Final gate temperature for dynamic bridge (default 0.1, hard)",
    )
    parser.add_argument(
        "--contrastive-topology", type=str, default="auto",
        choices=["auto", "wrong-labels", "resonance"],
        help="Contrastive pair topology. 'auto' selects based on n-channels "
             "(n=6→RD, n=8→tesseract). 'wrong-labels' uses random partition "
             "of 6 channels (tests whether RD geometry is special). "
             "'resonance' uses prime-threading topology from the corpus "
             "(Sophie Germain, consecutive primes, mod-6 residue).",
    )
    parser.add_argument(
        "--emanation", action="store_true",
        help="Enable emanation architecture: a single master bridge emanates "
             "into per-layer bridges via learned additive offsets. "
             "The One produces without diminishing.",
    )
    parser.add_argument(
        "--save-merged", action="store_true",
        help="After training, merge LoRA+bridge into base model and save as "
             "standard HuggingFace model for lm-eval benchmarking.",
    )
    args = parser.parse_args()

    config = ExperimentConfig(
        name=f"exp3_cybernetic_{args.bridge_mode}",
        rank=24,
        n_channels=args.n_channels,
        bridge_mode=args.bridge_mode,
        bridge_trainable=True,
        model_name=args.model,
        max_steps=args.max_steps,
        lr=args.lr,
        batch_size=args.batch_size,
        gradient_accumulation=args.gradient_accumulation,
        warmup_steps=args.warmup_steps,
        checkpoint_steps=args.checkpoint_steps,
        seed=args.seed,
        dynamic_bridge=args.dynamic_bridge,
        gate_temperature_start=args.gate_temperature_start,
        gate_temperature_end=args.gate_temperature_end,
    )

    # Steersman preset: corpus uses tighter thresholds derived from
    # Fiedler ~0.10 convergence point and corpus-coupled init dynamics
    steersman_kwargs = {}
    if args.steersman_preset == "corpus":
        steersman_kwargs = dict(
            initial_spectral_target=0.1,
            fiedler_decline_threshold=-0.000894,
            co_cross_stagnation_band=0.00957,
            deviation_rate_threshold=0.04677,
        )

    train_cybernetic(
        config,
        Path(args.output),
        feedback_interval=args.feedback_interval,
        contrastive_topology=args.contrastive_topology,
        initial_contrastive=args.initial_contrastive,
        initial_spectral=args.initial_spectral,
        initial_spectral_target=(
            steersman_kwargs.get("initial_spectral_target", args.spectral_target)
        ),
        emanation=args.emanation,
        save_merged=args.save_merged,
        **{k: v for k, v in steersman_kwargs.items()
           if k != "initial_spectral_target"},
    )


if __name__ == "__main__":
    main()
