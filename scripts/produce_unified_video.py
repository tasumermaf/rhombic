#!/usr/bin/env python3
"""
TASUMER MAF — Unified Production Video Pipeline.

Interleaves three renderers into the cybernetic circuit demo:
  1. Ray-traced RD logo animation (crystalline intro)
  2. Motion graphics slides (4K business-class number pitches)
  3. Terminal interactions (Strongbad-style Hermes Agent sessions)

Structure:
  LOGO INTRO → THESIS slide → TERMINAL(compare) → NUMBERS slide →
  TERMINAL(weights) → TERMINAL(permutation) → VISION slide →
  TERMINAL(mechanism) → CTA slide

Each headline fact gets its slick animated presentation, then the
observer-steersman circuit proves it live.

Output: assets/video/tasumer_maf_unified.mp4

Usage:
    python scripts/produce_unified_video.py
    python scripts/produce_unified_video.py --skip-renders    # Use existing frames
    python scripts/produce_unified_video.py --preview         # 720p fast
"""

import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path

if sys.platform == "win32" and not hasattr(sys.stdout, '_rhombic_wrapped'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stdout._rhombic_wrapped = True
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

BASE = Path(__file__).resolve().parent.parent
ASSETS = BASE / "assets" / "video"
LOGO_FRAMES = ASSETS / "logo_frames"
SLIDE_FRAMES = ASSETS / "slide_frames"
TERM_FRAMES = ASSETS / "terminal_frames"
SEGMENT_DIR = ASSETS / "segments"
FINAL_OUTPUT = ASSETS / "tasumer_maf_unified.mp4"

FPS = 30
XFADE_SEC = 0.5  # crossfade between segments


def banner(msg):
    print(f"\n{'=' * 64}")
    print(f"  {msg}")
    print(f"{'=' * 64}\n")


def run_script(script_name, extra_args=None):
    cmd = [sys.executable, str(BASE / "scripts" / script_name)]
    if extra_args:
        cmd.extend(extra_args)
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(BASE), timeout=3600)
    return result.returncode == 0


def count_frames(frame_dir):
    if not frame_dir.exists():
        return 0
    return len(list(frame_dir.glob("frame_*.png")))


def encode_frames(frame_dir, output_path, start=1, count=None, pattern=None):
    """Encode a range of frames to MP4."""
    if pattern is None:
        # Auto-detect pattern from first frame
        first = sorted(frame_dir.glob("frame_*.png"))[0].name if any(frame_dir.glob("frame_*.png")) else None
        if first and len(first.replace("frame_", "").replace(".png", "")) == 5:
            pattern = "frame_%05d.png"
        else:
            pattern = "frame_%04d.png"

    input_pattern = (frame_dir / pattern).as_posix()

    for encoder, enc_args in [
        ("h264_nvenc", ["-preset", "p7", "-cq", "16", "-profile:v", "high"]),
        ("libx264", ["-crf", "18", "-preset", "slow", "-profile:v", "high"]),
    ]:
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(FPS),
            "-start_number", str(start),
            "-i", input_pattern,
        ]
        if count:
            cmd += ["-frames:v", str(count)]
        cmd += [
            "-c:v", encoder,
        ] + enc_args + [
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            output_path.as_posix(),
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode == 0:
                size_mb = output_path.stat().st_size / (1024 * 1024)
                dur = get_duration(output_path)
                print(f"    -> {output_path.name} ({size_mb:.1f}MB, {dur:.1f}s, {encoder})")
                return True
            print(f"    {encoder} failed: {result.stderr[-200:]}")
        except Exception as e:
            print(f"    {encoder} error: {e}")
    return False


def posix(p):
    """Convert path to forward-slash string for ffmpeg on Windows."""
    return Path(p).as_posix() if hasattr(p, 'as_posix') else str(p).replace('\\', '/')


def get_duration(path):
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", posix(path)],
            capture_output=True, text=True, timeout=10)
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def xfade_concat(segments, output_path):
    """
    Concatenate video segments with xfade transitions.
    Uses filter_complex chain for multiple inputs.
    """
    if len(segments) == 0:
        return False
    if len(segments) == 1:
        import shutil
        shutil.copy2(str(segments[0]), str(output_path))
        return True

    # Build the filter graph for N inputs with N-1 xfades
    inputs = []
    for seg in segments:
        inputs += ["-i", posix(seg)]

    # Chain xfades: [0][1]xfade -> [v1], [v1][2]xfade -> [v2], ...
    durations = [get_duration(s) for s in segments]
    filter_parts = []
    offsets = []

    # Calculate cumulative offset for each xfade
    cumulative = 0.0
    for i in range(len(segments) - 1):
        offset = cumulative + durations[i] - XFADE_SEC
        offsets.append(offset)
        cumulative = offset  # After xfade, the combined duration = offset + dur[i+1]

    # Build filter chain
    if len(segments) == 2:
        filter_str = (
            f"[0:v][1:v]xfade=transition=fade:duration={XFADE_SEC}"
            f":offset={offsets[0]:.3f},format=yuv420p[v]"
        )
    else:
        # First pair
        filter_str = (
            f"[0:v][1:v]xfade=transition=fade:duration={XFADE_SEC}"
            f":offset={offsets[0]:.3f}[v1]"
        )
        # Middle pairs
        for i in range(1, len(segments) - 2):
            filter_str += (
                f";[v{i}][{i + 1}:v]xfade=transition=fade:duration={XFADE_SEC}"
                f":offset={offsets[i]:.3f}[v{i + 1}]"
            )
        # Last pair
        last_i = len(segments) - 2
        filter_str += (
            f";[v{last_i}][{last_i + 1}:v]xfade=transition=fade:duration={XFADE_SEC}"
            f":offset={offsets[last_i]:.3f},format=yuv420p[v]"
        )

    cmd = (
        ["ffmpeg", "-y"] + inputs +
        ["-filter_complex", filter_str, "-map", "[v]",
         "-c:v", "h264_nvenc", "-preset", "p7", "-cq", "16",
         "-profile:v", "high", "-movflags", "+faststart",
         posix(output_path)]
    )

    print(f"  Stitching {len(segments)} segments with xfade...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode == 0:
            size_mb = output_path.stat().st_size / (1024 * 1024)
            dur = get_duration(output_path)
            print(f"  → {output_path.name} ({size_mb:.1f}MB, {dur:.1f}s)")
            return True
        else:
            print(f"  xfade chain failed: {result.stderr[-500:]}")
            # Fallback: simple concat without transitions
            return simple_concat(segments, output_path)
    except Exception as e:
        print(f"  xfade error: {e}")
        return simple_concat(segments, output_path)


def simple_concat(segments, output_path):
    """Fallback: concat demuxer without transitions."""
    list_file = SEGMENT_DIR / "concat_list.txt"
    with open(list_file, "w") as f:
        for seg in segments:
            f.write(f"file '{posix(seg)}'\n")

    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", posix(list_file),
        "-c:v", "h264_nvenc", "-preset", "p7", "-cq", "16",
        "-movflags", "+faststart",
        posix(output_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode == 0:
            size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"  Concat fallback: {output_path.name} ({size_mb:.1f}MB)")
            list_file.unlink(missing_ok=True)
            return True
    except Exception as e:
        print(f"  Concat error: {e}")
    list_file.unlink(missing_ok=True)
    return False


def verify_output(path):
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height,r_frame_rate",
             "-show_entries", "format=duration,size",
             "-of", "json", posix(path)],
            capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            info = json.loads(result.stdout)
            stream = info.get("streams", [{}])[0]
            fmt = info.get("format", {})
            print(f"  Resolution:  {stream.get('width', '?')}x{stream.get('height', '?')}")
            print(f"  Frame rate:  {stream.get('r_frame_rate', '?')}")
            print(f"  Duration:    {float(fmt.get('duration', 0)):.1f}s")
            print(f"  File size:   {int(fmt.get('size', 0)) / 1024 / 1024:.1f} MB")
    except Exception:
        pass


# ===================================================================
# Slide Renderer — Individual motion graphics scenes
# ===================================================================

def render_individual_slides(preview=False):
    """
    Render each motion graphics scene as separate frame sequences.
    Shells out to avoid importlib conflicts with stdio wrappers.
    """
    extra = ["--preview"] if preview else []

    script = BASE / "scripts" / "_render_slides.py"
    # Write without f-string to avoid escaping issues
    content = r'''#!/usr/bin/env python3
"""Render individual slide scenes for the unified video."""
import json, os, sys
from pathlib import Path

# Prevent double-wrapping: only wrap if not already wrapped
if sys.platform == "win32" and not hasattr(sys.stdout, '_rhombic_wrapped'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    sys.stdout._rhombic_wrapped = True

PREVIEW = "--preview" in sys.argv
BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "scripts"))

# Patch the module globals BEFORE import by pre-setting env
if PREVIEW:
    os.environ["RHOMBIC_PREVIEW"] = "1"

# Import renders — this module also wraps stdio, but we guard above
import render_demo_production as dm

# Override resolution if preview (the module reads PREVIEW from sys.argv)
if PREVIEW and dm.W != 1280:
    dm.W, dm.H = 1280, 720

FPS = dm.FPS
SLIDE_FRAMES = BASE / "assets" / "video" / "slide_frames"
captures = BASE / "assets" / "video" / "captures"
act1 = json.loads((captures / "act1_lattice_compare.json").read_text())
act3 = json.loads((captures / "act3_explain_mechanism.json").read_text())

scenes = [
    ("thesis",  dm.render_scene_thesis,  7.0,  {}),
    ("numbers", dm.render_scene_numbers, 10.0, {"data": act1}),
    ("vision",  dm.render_scene_vision,  8.0,  {}),
    ("cta",     dm.render_scene_cta,     7.0,  {}),
]

for name, renderer, duration, kwargs in scenes:
    out_dir = SLIDE_FRAMES / name
    out_dir.mkdir(parents=True, exist_ok=True)
    n_frames = int(duration * FPS)
    sys.stdout.write(f"  Rendering slide '{name}' ({duration}s, {n_frames} frames)...\n")
    sys.stdout.flush()
    for f in range(n_frames):
        img = renderer(f, n_frames, **kwargs)
        path = out_dir / f"frame_{f + 1:05d}.png"
        img.save(str(path))
    sys.stdout.write(f"    -> {n_frames} frames\n")
    sys.stdout.flush()
sys.stdout.write("  Slides complete.\n")
sys.stdout.flush()
'''
    script.write_text(content, encoding="utf-8")

    cmd = [sys.executable, str(script)] + extra
    print(f"  Running slide renderer...")
    result = subprocess.run(cmd, cwd=str(BASE), timeout=600)
    script.unlink(missing_ok=True)
    return result.returncode == 0


# ===================================================================
# Terminal Renderer — Individual interaction sequences
# ===================================================================

def render_individual_terminals(preview=False):
    """
    Render each terminal interaction as a separate frame sequence.
    Shells out to avoid importlib conflicts.
    """
    extra = ["--preview"] if preview else []

    script = BASE / "scripts" / "_render_terms.py"
    content = r'''#!/usr/bin/env python3
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
'''
    script.write_text(content, encoding="utf-8")

    cmd = [sys.executable, str(script)] + extra
    print(f"  Running terminal renderer...")
    result = subprocess.run(cmd, cwd=str(BASE), timeout=1800)
    script.unlink(missing_ok=True)
    return result.returncode == 0


# ===================================================================
# Main Pipeline
# ===================================================================

def main():
    banner("TASUMER MAF — Unified Production Video")
    t0 = time.time()

    skip_renders = "--skip-renders" in sys.argv
    preview = "--preview" in sys.argv
    extra = ["--preview"] if preview else []

    SEGMENT_DIR.mkdir(parents=True, exist_ok=True)

    # ── STAGE 1: Render all components ──
    if not skip_renders:
        # Logo — always use existing 4K frames (GPU render is expensive)
        banner("STAGE 1a: Logo Animation")
        n_logo = count_frames(LOGO_FRAMES)
        if n_logo < 100 and not preview:
            run_script("raytrace_rd_logo.py", extra)
        else:
            print(f"  Using existing logo frames ({n_logo})")

        # Slides
        banner("STAGE 1b: Motion Graphics Slides")
        render_individual_slides(preview)

        # Terminal
        banner("STAGE 1c: Terminal Interactions")
        render_individual_terminals(preview)
    else:
        print("  Using existing rendered frames")

    # ── STAGE 2: Encode each segment ──
    banner("STAGE 2: Encode Segments")

    # Segment order defines the final video structure
    # (name, frame_dir, pattern, start_number)
    segment_order = [
        ("logo",         LOGO_FRAMES,                "frame_%04d.png", 0),
        ("slide_thesis", SLIDE_FRAMES / "thesis",    "frame_%05d.png", 1),
        ("term_1",       TERM_FRAMES / "terminal_1", "frame_%05d.png", 1),
        ("slide_numbers",SLIDE_FRAMES / "numbers",   "frame_%05d.png", 1),
        ("term_2",       TERM_FRAMES / "terminal_2", "frame_%05d.png", 1),
        ("term_3",       TERM_FRAMES / "terminal_3", "frame_%05d.png", 1),
        ("slide_vision", SLIDE_FRAMES / "vision",    "frame_%05d.png", 1),
        ("term_4",       TERM_FRAMES / "terminal_4", "frame_%05d.png", 1),
        ("slide_cta",    SLIDE_FRAMES / "cta",       "frame_%05d.png", 1),
    ]

    segment_files = []
    for name, frame_dir, pattern, start_num in segment_order:
        n = count_frames(frame_dir)
        if n == 0:
            print(f"  WARNING: No frames for '{name}' in {frame_dir}, skipping")
            continue

        out = SEGMENT_DIR / f"{name}.mp4"
        print(f"  Encoding '{name}' ({n} frames, {n / FPS:.1f}s)...")
        if encode_frames(frame_dir, out, start=start_num, count=n, pattern=pattern):
            segment_files.append(out)

    if not segment_files:
        print("  ERROR: No segments encoded!")
        sys.exit(1)

    # ── STAGE 3: Stitch ──
    banner("STAGE 3: Stitch Final Video")
    xfade_concat(segment_files, FINAL_OUTPUT)

    # ── STAGE 4: Web version ──
    banner("STAGE 4: Web-Optimized 1080p")
    web_output = BASE / "website" / "demo.mp4"
    if FINAL_OUTPUT.exists():
        cmd = [
            "ffmpeg", "-y",
            "-i", posix(FINAL_OUTPUT),
            "-vf", "scale=1920:1080",
            "-c:v", "libx264", "-crf", "23", "-preset", "slow",
            "-profile:v", "high", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", "-an",
            posix(web_output),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            size_mb = web_output.stat().st_size / (1024 * 1024)
            print(f"  → {web_output} ({size_mb:.1f}MB)")

    # ── STAGE 5: Verify ──
    banner("VERIFICATION")
    if FINAL_OUTPUT.exists():
        verify_output(FINAL_OUTPUT)

    elapsed = time.time() - t0
    banner(f"COMPLETE — {elapsed / 60:.1f} minutes")
    print(f"  4K:   {FINAL_OUTPUT}")
    print(f"  Web:  {web_output}")
    print()


if __name__ == "__main__":
    main()
