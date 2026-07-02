#!/usr/bin/env python3
"""
Particle Simulation Capture — Episode 2 Visual Centerpiece.

Uses Playwright to capture WebGL particle simulations from particles.casberry.in
at 4K resolution. Records to video files for compositing into the final video.

Captures:
  1. Tesseract -> RD particle sim (25s + 5s transitions)
  2. Bridge topology particle sim (12s with ratio slider animation)

Output: assets/video/captures_ep2/tesseract_rd.mp4
        assets/video/captures_ep2/bridge_topo.mp4

Usage:
    python scripts/capture_particles.py
    python scripts/capture_particles.py --preview    # 1080p instead of 4K

Note: Requires playwright installed (`pip install playwright && playwright install chromium`)
      Falls back to synthesized particle fields if Playwright unavailable.
"""

import io
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

BASE = Path(__file__).resolve().parent.parent
CAPTURE_DIR = BASE / "assets" / "video" / "captures_ep2"

PREVIEW = "--preview" in sys.argv
W, H = (1920, 1080) if PREVIEW else (3840, 2160)
FPS = 30

if sys.platform == "win32" and not hasattr(sys.stdout, '_rhombic_wrapped'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stdout._rhombic_wrapped = True
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


# ===================================================================
# Fallback: Synthesized Particle Field (no browser required)
# ===================================================================

NUIT = (8, 6, 32)
GOLD = (220, 201, 100)
FCC_RED = (179, 68, 68)
AZURE = (92, 143, 175)
CUBIC_BLUE = (61, 61, 107)


def synth_tesseract_rd_frame(frame_idx, total_frames):
    """Synthesize a tesseract->RD particle field frame.

    Uses the actual 4D->3D projection math from the particle sim.
    Not as pretty as the WebGL version but mathematically correct.
    """
    img = Image.new("RGB", (W, H), NUIT)
    draw = ImageDraw.Draw(img)

    t = frame_idx / FPS
    cx, cy = W / 2, H / 2
    scale_factor = min(W, H) * 0.3

    # 4D rotation angles
    a1 = t * 0.3
    a2 = t * 0.3 * 0.618

    # 16 tesseract vertices: {-1, +1}^4
    verts_4d = []
    for i in range(16):
        x = 1.0 if (i & 1) else -1.0
        y = 1.0 if (i & 2) else -1.0
        z = 1.0 if (i & 4) else -1.0
        w = 1.0 if (i & 8) else -1.0
        verts_4d.append((x, y, z, w))

    # Rotate in xw and yw planes, then project
    cos1, sin1 = math.cos(a1), math.sin(a1)
    cos2, sin2 = math.cos(a2), math.sin(a2)

    projected = []
    for x, y, z, w in verts_4d:
        # xw rotation
        x2 = x * cos1 - w * sin1
        w2 = x * sin1 + w * cos1
        # yw rotation
        y2 = y * cos2 - w2 * sin2
        w3 = y * sin2 + w2 * cos2
        # Stereographic projection from 4D
        denom = 2.0 - w3 * 0.3
        px = x2 / denom
        py = y2 / denom
        pz = z / denom
        # Screen coords (ignore z for 2D, use for size)
        sx = cx + px * scale_factor
        sy = cy + py * scale_factor
        depth = (pz + 2) / 4  # 0-1 range
        projected.append((sx, sy, depth))

    # Draw edges (32 edges of tesseract)
    rng = np.random.RandomState(42 + frame_idx)
    for i in range(16):
        for j in range(i + 1, 16):
            # Connected if they differ in exactly 1 bit
            diff = i ^ j
            if diff & (diff - 1) == 0:  # power of 2 = 1 bit
                x1, y1, d1 = projected[i]
                x2, y2, d2 = projected[j]
                alpha = int(60 * (d1 + d2) / 2)
                color = (GOLD[0], GOLD[1], GOLD[2])
                draw.line([(x1, y1), (x2, y2)], fill=color, width=max(1, int(2 * W / 3840)))

    # Draw vertex particles (clusters of 30-80 particles per vertex)
    primes_colors = [FCC_RED, AZURE, GOLD, CUBIC_BLUE, (74, 140, 92), (123, 63, 140)]
    for vi, (sx, sy, depth) in enumerate(projected):
        n_particles = int(30 + 50 * depth)
        color = primes_colors[vi % len(primes_colors)]
        for _ in range(n_particles):
            spread = 8 * W / 3840
            px = sx + rng.normal(0, spread)
            py = sy + rng.normal(0, spread)
            r = max(1, int(1.5 * depth * W / 3840))
            if 0 <= px < W and 0 <= py < H:
                draw.ellipse([px - r, py - r, px + r, py + r], fill=color)

    return img


def synth_bridge_topology_frame(frame_idx, total_frames):
    """Synthesize a bridge topology visualization frame.

    Shows blue (co-planar) and red (cross-planar) clusters with
    animated ratio revealing the 22,477:1 asymmetry.
    """
    img = Image.new("RGB", (W, H), NUIT)
    draw = ImageDraw.Draw(img)

    t = frame_idx / FPS
    progress = frame_idx / max(1, total_frames - 1)
    cx, cy = W / 2, H / 2

    rng = np.random.RandomState(123 + frame_idx)

    # Co-planar cluster (blue, large, dominant)
    co_count = int(600 * (0.3 + 0.7 * progress))
    for _ in range(co_count):
        angle = rng.uniform(0, 2 * math.pi)
        radius = rng.exponential(W * 0.12)
        x = cx - W * 0.15 + math.cos(angle) * radius + rng.normal(0, W * 0.02)
        y = cy + math.sin(angle) * radius + rng.normal(0, W * 0.02)
        r = max(1, int(rng.uniform(1, 3) * W / 3840))
        brightness = rng.uniform(0.4, 1.0)
        color = (int(AZURE[0] * brightness), int(AZURE[1] * brightness), int(AZURE[2] * brightness))
        if 0 <= x < W and 0 <= y < H:
            draw.ellipse([x - r, y - r, x + r, y + r], fill=color)

    # Cross-planar cluster (red, tiny, sparse — showing the asymmetry)
    cross_count = max(3, int(co_count / (1 + 200 * progress)))  # Dramatically fewer
    for _ in range(cross_count):
        x = cx + W * 0.25 + rng.normal(0, W * 0.04)
        y = cy + rng.normal(0, W * 0.04)
        r = max(1, int(rng.uniform(1, 2) * W / 3840))
        brightness = rng.uniform(0.5, 1.0)
        color = (int(FCC_RED[0] * brightness), int(FCC_RED[1] * brightness), int(FCC_RED[2] * brightness))
        if 0 <= x < W and 0 <= y < H:
            draw.ellipse([x - r, y - r, x + r, y + r], fill=color)

    # 6 FCC channels as faint connecting lines
    for ch in range(6):
        angle = ch * math.pi / 3 + t * 0.1
        x1 = cx + math.cos(angle) * W * 0.35
        y1 = cy + math.sin(angle) * W * 0.35
        x2 = cx + math.cos(angle + math.pi) * W * 0.35
        y2 = cy + math.sin(angle + math.pi) * W * 0.35
        draw.line([(x1, y1), (x2, y2)], fill=(40, 35, 65), width=max(1, int(W / 3840)))

    # Ratio label (animated count-up)
    if progress > 0.3:
        from PIL import ImageFont
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/bahnschrift.ttf", int(80 * W / 3840))
        except Exception:
            font = ImageFont.load_default()

        ratio = 22477 * min(1.0, (progress - 0.3) / 0.5)
        text = f"{ratio:,.0f}:1"
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        draw.text((cx - tw // 2, int(H * 0.85)), text, fill=GOLD, font=font)

    return img


def render_fallback_captures():
    """Render synthesized particle captures when Playwright is unavailable."""
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)

    # Tesseract -> RD (30 seconds = 25s main + 5s transition)
    print("  Rendering synthesized tesseract->RD (30s)...")
    total_tess = 30 * FPS
    tess_dir = CAPTURE_DIR / "tesseract_frames"
    tess_dir.mkdir(exist_ok=True)
    for i in range(total_tess):
        if i % 100 == 0:
            print(f"    Frame {i}/{total_tess} ({i*100//total_tess}%)", flush=True)
        img = synth_tesseract_rd_frame(i, total_tess)
        img.save(str(tess_dir / f"frame_{i+1:05d}.png"))

    # Bridge topology (12 seconds)
    print("  Rendering synthesized bridge topology (12s)...")
    total_bridge = 12 * FPS
    bridge_dir = CAPTURE_DIR / "bridge_frames"
    bridge_dir.mkdir(exist_ok=True)
    for i in range(total_bridge):
        if i % 100 == 0:
            print(f"    Frame {i}/{total_bridge} ({i*100//total_bridge}%)", flush=True)
        img = synth_bridge_topology_frame(i, total_bridge)
        img.save(str(bridge_dir / f"frame_{i+1:05d}.png"))

    # Encode to MP4
    for name, frame_dir, dur in [
        ("tesseract_rd", tess_dir, 30),
        ("bridge_topo", bridge_dir, 12),
    ]:
        out = CAPTURE_DIR / f"{name}.mp4"
        pattern = (frame_dir / "frame_%05d.png").as_posix()
        for encoder, enc_args in [
            ("h264_nvenc", ["-preset", "p7", "-cq", "16"]),
            ("libx264", ["-crf", "18", "-preset", "slow"]),
        ]:
            cmd = [
                "ffmpeg", "-y",
                "-framerate", str(FPS),
                "-start_number", "1",
                "-i", pattern,
                "-c:v", encoder, *enc_args,
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                out.as_posix(),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode == 0:
                size_mb = out.stat().st_size / (1024 * 1024)
                print(f"  -> {out.name}: {size_mb:.1f}MB ({encoder})")
                break
        else:
            print(f"  ERROR: Failed to encode {name}")


def try_playwright_capture():
    """Attempt browser-based capture of live WebGL particle sims."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("  Playwright not available — using synthesized fallback")
        return False

    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": W, "height": H})

            # Tesseract -> RD
            print("  Capturing tesseract->RD from particles.casberry.in...")
            page.goto("https://particles.casberry.in/", timeout=15000)
            page.wait_for_timeout(3000)  # Let WebGL initialize

            tess_dir = CAPTURE_DIR / "tesseract_frames"
            tess_dir.mkdir(exist_ok=True)
            total = 30 * FPS
            for i in range(total):
                if i % FPS == 0:
                    print(f"    Frame {i}/{total} ({i//FPS}s)", flush=True)
                screenshot = page.screenshot(type="png")
                img = Image.open(io.BytesIO(screenshot))
                img.save(str(tess_dir / f"frame_{i+1:05d}.png"))
                page.wait_for_timeout(int(1000 / FPS))

            browser.close()
    except Exception as e:
        print(f"  Playwright capture failed: {e}")
        return False

    return True


def main():
    print("=" * 60)
    print("  PARTICLE SIMULATION CAPTURE — Episode 2")
    print(f"  Resolution: {W}x{H} @ {FPS}fps")
    print("=" * 60)

    # Try Playwright first, fall back to synthesis
    if not try_playwright_capture():
        print("\n  Using synthesized particle fields (fallback)")
        render_fallback_captures()

    print("\n" + "=" * 60)
    print("  CAPTURE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
