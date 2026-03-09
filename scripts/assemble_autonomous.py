#!/usr/bin/env python3
"""Autonomous video production pipeline for rhombic hackathon demo.

Chains three stages:
  1. Capture Hermes tool outputs via SSH (deterministic JSON)
  2. Render terminal-style video frames with PIL
  3. Encode frames to MP4 with FFmpeg (h264_nvenc on RTX 6000 Ada)

No human intervention required. Run from the rhombic project root:
    python scripts/assemble_autonomous.py

Output: assets/video/rhombic_demo_final.mp4
"""

from __future__ import annotations

import io
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

BASE = Path(__file__).resolve().parent.parent
SCRIPTS = BASE / "scripts"
ASSETS = BASE / "assets" / "video"
FRAMES = ASSETS / "frames"
CAPTURES = ASSETS / "captures"

FPS = 30
OUTPUT = ASSETS / "rhombic_demo_final.mp4"


def banner(msg: str) -> None:
    print(f"\n{'═' * 60}")
    print(f"  {msg}")
    print(f"{'═' * 60}\n")


def check_prerequisites() -> bool:
    """Verify all required tools and assets exist."""
    ok = True

    # FFmpeg
    try:
        r = subprocess.run(
            ["ffmpeg", "-version"], capture_output=True, text=True, timeout=10
        )
        if r.returncode != 0:
            print("ERROR: ffmpeg not found or not working")
            ok = False
        else:
            # Check for nvenc support
            if "nvenc" in r.stdout.lower() or "cuda" in r.stdout.lower():
                print("  ✓ FFmpeg with NVENC support")
            else:
                print("  ⚠ FFmpeg found but NVENC not detected — will try anyway")
    except FileNotFoundError:
        print("ERROR: ffmpeg not in PATH")
        ok = False

    # SSH to hermes
    try:
        r = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=5", "hermes", "echo ok"],
            capture_output=True, text=True, timeout=15,
        )
        if r.returncode == 0 and "ok" in r.stdout:
            print("  ✓ SSH to hermes")
        else:
            print("ERROR: Cannot SSH to hermes")
            ok = False
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("ERROR: SSH connection to hermes failed")
        ok = False

    # Title cards
    for name in ["title_card.png", "closing_card.png"]:
        p = ASSETS / name
        if p.exists():
            print(f"  ✓ {name}")
        else:
            print(f"ERROR: Missing {p}")
            ok = False

    # Python dependencies
    try:
        import PIL  # noqa: F401
        print("  ✓ Pillow")
    except ImportError:
        print("ERROR: Pillow not installed")
        ok = False

    try:
        import numpy  # noqa: F401
        print("  ✓ numpy")
    except ImportError:
        print("ERROR: numpy not installed")
        ok = False

    return ok


def stage_1_capture() -> bool:
    """Stage 1: Capture Hermes tool outputs via SSH."""
    banner("STAGE 1: Capture Hermes Tool Outputs")

    # Import and run the capture module
    sys.path.insert(0, str(SCRIPTS))
    try:
        from capture_hermes import main as capture_main
        capture_main()
    except Exception as e:
        print(f"\nERROR in capture stage: {e}")
        # Check if we have cached captures from a previous run
        combined = CAPTURES / "all_captures.json"
        if combined.exists():
            print("  Using cached captures from previous run")
            return True
        return False

    # Verify outputs
    expected = [
        "act1_lattice_compare.json",
        "act2a_direction_weights.json",
        "act2b_permutation_control.json",
        "act3_explain_mechanism.json",
    ]
    for name in expected:
        if not (CAPTURES / name).exists():
            print(f"ERROR: Missing capture {name}")
            return False
        print(f"  ✓ {name}")

    return True


def stage_2_render() -> int:
    """Stage 2: Render terminal-style video frames."""
    banner("STAGE 2: Render Video Frames")

    # Clean old frames
    if FRAMES.exists():
        shutil.rmtree(str(FRAMES))
    FRAMES.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(SCRIPTS))
    try:
        from render_terminal_video import build_video
        n_frames = build_video(CAPTURES)
    except Exception as e:
        print(f"\nERROR in render stage: {e}")
        import traceback
        traceback.print_exc()
        return 0

    # Verify frames exist
    actual = list(FRAMES.glob("frame_*.png"))
    if len(actual) < 100:
        print(f"ERROR: Only {len(actual)} frames rendered (expected 100+)")
        return 0

    print(f"\n  ✓ {len(actual)} frames rendered")
    return n_frames


def stage_3_encode(n_frames: int) -> bool:
    """Stage 3: Encode frames to MP4 with FFmpeg."""
    banner("STAGE 3: FFmpeg Encode (h264_nvenc)")

    if n_frames == 0:
        print("ERROR: No frames to encode")
        return False

    duration = n_frames / FPS
    print(f"  Input: {n_frames} frames at {FPS}fps = {duration:.1f}s")

    # Build FFmpeg command
    # Try h264_nvenc first (GPU), fall back to libx264 (CPU)
    input_pattern = str(FRAMES / "frame_%05d.png")

    for encoder, label in [("h264_nvenc", "GPU/NVENC"), ("libx264", "CPU/x264")]:
        print(f"  Trying {label} encoder...")

        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(FPS),
            "-i", input_pattern,
            "-c:v", encoder,
        ]

        if encoder == "h264_nvenc":
            cmd += ["-preset", "p7", "-cq", "18"]
        else:
            cmd += ["-preset", "medium", "-crf", "18"]

        cmd += [
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(OUTPUT),
        ]

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600
        )

        if result.returncode == 0:
            print(f"  ✓ Encoded with {label}")
            break
        else:
            print(f"  ✗ {label} failed: {result.stderr[-200:]}")
            if encoder == "libx264":
                return False

    # Verify output
    if not OUTPUT.exists():
        print("ERROR: Output file not created")
        return False

    size_mb = OUTPUT.stat().st_size / (1024 * 1024)
    print(f"\n  Output:   {OUTPUT}")
    print(f"  Size:     {size_mb:.1f} MB")
    print(f"  Duration: {duration:.1f}s")

    # Twitter compatibility check
    if size_mb > 512:
        print("  ⚠ WARNING: Exceeds Twitter 512 MB limit")
    elif duration > 140:
        print("  ⚠ WARNING: Exceeds Twitter 2:20 limit")
    else:
        print("  ✓ Twitter-compatible")

    return True


def stage_4_verify() -> None:
    """Stage 4: Final verification and summary."""
    banner("VERIFICATION")

    if not OUTPUT.exists():
        print("ERROR: No output video")
        return

    # Get video info via ffprobe
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height,r_frame_rate,nb_frames",
                "-show_entries", "format=duration,size",
                "-of", "json",
                str(OUTPUT),
            ],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            import json
            info = json.loads(result.stdout)
            stream = info.get("streams", [{}])[0]
            fmt = info.get("format", {})

            print(f"  Resolution:  {stream.get('width', '?')}×{stream.get('height', '?')}")
            print(f"  Frame rate:  {stream.get('r_frame_rate', '?')}")
            dur = float(fmt.get("duration", 0))
            print(f"  Duration:    {dur:.1f}s")
            size = int(fmt.get("size", 0))
            print(f"  File size:   {size / 1024 / 1024:.1f} MB")
        else:
            print("  (ffprobe failed — skipping detailed verification)")
    except Exception:
        print("  (ffprobe not available — skipping detailed verification)")

    print(f"\n  Final video: {OUTPUT}")


def main():
    """Run the complete autonomous video pipeline."""
    banner("RHOMBIC HACKATHON DEMO — Autonomous Pipeline")
    t0 = time.time()

    # Prerequisites
    print("Checking prerequisites...")
    if not check_prerequisites():
        print("\nPrerequisite check failed. Fix the errors above and retry.")
        sys.exit(1)
    print()

    # Stage 1: Capture
    if not stage_1_capture():
        print("\nStage 1 failed. Cannot proceed without captures.")
        sys.exit(1)

    # Stage 2: Render
    n_frames = stage_2_render()
    if n_frames == 0:
        print("\nStage 2 failed. Cannot proceed without frames.")
        sys.exit(1)

    # Stage 3: Encode
    if not stage_3_encode(n_frames):
        print("\nStage 3 failed. Video not produced.")
        sys.exit(1)

    # Stage 4: Verify
    stage_4_verify()

    elapsed = time.time() - t0
    banner(f"COMPLETE — {elapsed:.0f}s total")
    print(f"  Video: {OUTPUT}")
    print(f"  Ready for hackathon submission.\n")


if __name__ == "__main__":
    main()
