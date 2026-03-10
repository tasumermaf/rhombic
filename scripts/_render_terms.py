#!/usr/bin/env python3
"""Render individual terminal interactions for the unified video."""
import math, os, sys, shutil
from pathlib import Path

if sys.platform == "win32" and not hasattr(sys.stdout, '_rhombic_wrapped'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    sys.stdout._rhombic_wrapped = True

import numpy as np
from PIL import Image

PREVIEW = "--preview" in sys.argv
BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "scripts"))

if PREVIEW:
    os.environ["RHOMBIC_PREVIEW"] = "1"

import render_terminal as tm

if PREVIEW and tm.W != 1280:
    tm.W, tm.H = 1280, 720

FPS = tm.FPS
TERM_FRAMES = BASE / "assets" / "video" / "terminal_frames"

for idx, interaction in enumerate(tm.INTERACTIONS):
    name = f"terminal_{idx + 1}"
    out_dir = TERM_FRAMES / name
    if out_dir.exists():
        shutil.rmtree(str(out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt_short = interaction["prompt"][:35]
    sys.stdout.write(f"  Terminal {idx + 1}: '{prompt_short}'...\n")
    sys.stdout.flush()

    seq_frames = tm.build_interaction_frames(interaction)

    intro = [([], 0, (f % 24) < 12) for f in range(int(FPS * 0.5))]
    outro = []
    if seq_frames:
        last = seq_frames[-1]
        outro = [(last[0], last[1], (f % 24) < 12) for f in range(int(FPS * 1.0))]

    all_frames = intro + seq_frames + outro
    total = len(all_frames)
    bg = np.array(tm.NUIT, dtype=np.float32)[None, None, :]

    for i, (lines, visible_chars, cursor_on) in enumerate(all_frames):
        img = tm.render_terminal_frame(visible_chars, lines, cursor_on, i)

        if i < 15:
            arr = np.array(img, dtype=np.float32)
            img = Image.fromarray((arr * (i / 15.0) + bg * (1 - i / 15.0)).clip(0, 255).astype(np.uint8))
        if i >= total - 15:
            r = max(0, (total - 1 - i) / 15.0)
            arr = np.array(img, dtype=np.float32)
            img = Image.fromarray((arr * r + bg * (1 - r)).clip(0, 255).astype(np.uint8))

        path = out_dir / f"frame_{i + 1:05d}.png"
        img.save(str(path))

    sys.stdout.write(f"    -> {total} frames ({total / FPS:.1f}s)\n")
    sys.stdout.flush()

sys.stdout.write("  Terminals complete.\n")
sys.stdout.flush()
