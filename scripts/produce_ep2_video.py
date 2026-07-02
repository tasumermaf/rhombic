#!/usr/bin/env python3
"""
TASUMER MAF — Episode 2: "The Briefing" Production Pipeline.

Orchestrates all Episode 2 renderers into the final video:
  1. Render motion graphics slides (Executive register)
  2. Render terminal interactions (Nerd register)
  3. Capture/synthesize particle simulations
  4. Encode each segment to MP4
  5. Stitch with crossfade transitions
  6. Mix 8-layer audio
  7. Mux audio + video
  8. Generate 1080p web version

Two visual registers:
  - Executive: serif fonts, GOLD/AVORIO on NUIT, particle fields
  - Nerd: terminal window, Consolas mono, green prompt, inline vizs

In Act III, the registers merge.

Output: assets/video/tasumer_maf_ep2_av.mp4

Usage:
    python scripts/produce_ep2_video.py
    python scripts/produce_ep2_video.py --skip-renders   # Use existing frames
    python scripts/produce_ep2_video.py --preview         # 720p fast
    python scripts/produce_ep2_video.py --skip-audio      # Video only
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
SLIDE_FRAMES = ASSETS / "frames_ep2" / "slides"
TERM_FRAMES = ASSETS / "frames_ep2" / "terminals"
CAPTURE_DIR = ASSETS / "captures_ep2"
SEGMENT_DIR = ASSETS / "segments_ep2"
LOGO_FRAMES = ASSETS / "logo_frames"  # Reuse Ep1 logo

FINAL_VIDEO = ASSETS / "tasumer_maf_ep2_video.mp4"
FINAL_AV = ASSETS / "tasumer_maf_ep2_av.mp4"
WEB_OUTPUT = ASSETS / "tasumer_maf_ep2_web.mp4"

FPS = 30
XFADE_SEC = 0.5


def banner(msg):
    print(f"\n{'=' * 64}")
    print(f"  {msg}")
    print(f"{'=' * 64}\n")


def run_script(name, extra_args=None):
    cmd = [sys.executable, str(BASE / "scripts" / name)]
    if extra_args:
        cmd.extend(extra_args)
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(BASE), timeout=7200)
    return result.returncode == 0


def count_frames(d):
    if not d.exists():
        return 0
    return len(list(d.glob("frame_*.png")))


def posix(p):
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


def encode_frames(frame_dir, output_path, start=1, count=None, pattern=None):
    """Encode PNG frame sequence to MP4."""
    if pattern is None:
        first = sorted(frame_dir.glob("frame_*.png"))
        if first and len(first[0].name.replace("frame_", "").replace(".png", "")) == 5:
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


def get_resolution(path):
    """Get video resolution as (width, height)."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "stream=width,height",
             "-of", "csv=p=0", posix(path)],
            capture_output=True, text=True, timeout=10)
        parts = result.stdout.strip().split(",")
        return int(parts[0]), int(parts[1])
    except Exception:
        return 0, 0


def xfade_concat(segments, output_path):
    """Concatenate video segments with crossfade transitions."""
    if len(segments) == 0:
        return False
    if len(segments) == 1:
        import shutil
        shutil.copy2(str(segments[0]), str(output_path))
        return True

    inputs = []
    for seg in segments:
        inputs += ["-i", posix(seg)]

    durations = [get_duration(s) for s in segments]

    # Detect target resolution (most common among segments)
    resolutions = [get_resolution(s) for s in segments]
    from collections import Counter
    target_w, target_h = Counter(resolutions).most_common(1)[0][0]

    # Build scale filters for inputs that differ from target
    scale_filters = []
    scaled_labels = []
    for i, (w, h) in enumerate(resolutions):
        if (w, h) != (target_w, target_h):
            label = f"s{i}"
            scale_filters.append(
                f"[{i}:v]scale={target_w}:{target_h}:flags=lanczos,"
                f"setsar=1,format=yuv420p[{label}]"
            )
            scaled_labels.append((i, label))
        else:
            scaled_labels.append((i, None))

    def inp(idx):
        """Return the label for input idx (scaled or raw)."""
        _, label = scaled_labels[idx]
        return f"[{label}]" if label else f"[{idx}:v]"

    offsets = []
    cumulative = 0.0
    for i in range(len(segments) - 1):
        offset = cumulative + durations[i] - XFADE_SEC
        offsets.append(offset)
        cumulative = offset

    # Build xfade chain using inp() for potentially-scaled inputs
    xfade_parts = []
    if len(segments) == 2:
        xfade_parts.append(
            f"{inp(0)}{inp(1)}xfade=transition=fade:duration={XFADE_SEC}"
            f":offset={offsets[0]:.3f},format=yuv420p[v]"
        )
    else:
        xfade_parts.append(
            f"{inp(0)}{inp(1)}xfade=transition=fade:duration={XFADE_SEC}"
            f":offset={offsets[0]:.3f}[v1]"
        )
        for i in range(1, len(segments) - 2):
            xfade_parts.append(
                f"[v{i}]{inp(i + 1)}xfade=transition=fade:duration={XFADE_SEC}"
                f":offset={offsets[i]:.3f}[v{i + 1}]"
            )
        last_i = len(segments) - 2
        xfade_parts.append(
            f"[v{last_i}]{inp(last_i + 1)}xfade=transition=fade:duration={XFADE_SEC}"
            f":offset={offsets[last_i]:.3f},format=yuv420p[v]"
        )

    filter_str = ";".join(scale_filters + xfade_parts)

    if any(label for _, label in scaled_labels):
        print(f"  Normalizing resolutions to {target_w}x{target_h}")

    # Use libx264 for complex xfade chains — NVENC struggles with long filter graphs.
    # Try NVENC first for <=4 segments, libx264 for larger chains.
    encoders = (
        [("h264_nvenc", ["-preset", "p7", "-cq", "16", "-profile:v", "high"])]
        if len(segments) <= 4
        else [("libx264", ["-crf", "18", "-preset", "slow", "-profile:v", "high"])]
    )

    print(f"  Stitching {len(segments)} segments with xfade...")
    for encoder, enc_args in encoders:
        cmd = (
            ["ffmpeg", "-y"] + inputs +
            ["-filter_complex", filter_str, "-map", "[v]",
             "-c:v", encoder, *enc_args,
             "-pix_fmt", "yuv420p", "-movflags", "+faststart",
             posix(output_path)]
        )
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
            if result.returncode == 0:
                size_mb = output_path.stat().st_size / (1024 * 1024)
                dur = get_duration(output_path)
                print(f"  -> {output_path.name} ({size_mb:.1f}MB, {dur:.1f}s, {encoder})")
                return True
            else:
                print(f"  xfade failed ({encoder}): {result.stderr[-300:]}")
        except Exception as e:
            print(f"  xfade error ({encoder}): {e}")

    return simple_concat(segments, output_path)


def simple_concat(segments, output_path):
    """Fallback: concat without transitions."""
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
# Segment Order — The Final Video Structure
# ===================================================================

# Maps segment name to (frame_dir, pattern, start_number)
# Order defines the video interleaving of Executive and Nerd registers

SEGMENT_ORDER = [
    # ACT I — "Status Update" (0:00–1:15)
    ("logo",             LOGO_FRAMES,                     "frame_%04d.png", 0),
    ("slide_ep2_title",  SLIDE_FRAMES / "ep2_title",      "frame_%05d.png", 1),
    ("slide_status",     SLIDE_FRAMES / "status",         "frame_%05d.png", 1),
    ("term_1",           TERM_FRAMES / "terminal_1",      "frame_%05d.png", 1),
    ("slide_scale",      SLIDE_FRAMES / "scale",          "frame_%05d.png", 1),
    ("term_2",           TERM_FRAMES / "terminal_2",      "frame_%05d.png", 1),
    ("particle_tess",    CAPTURE_DIR / "tesseract_frames", "frame_%05d.png", 1),

    # ACT II — "The Evidence Pile" (1:15–2:30)
    ("particle_bridge",  CAPTURE_DIR / "bridge_frames",   "frame_%05d.png", 1),
    ("slide_blood",      SLIDE_FRAMES / "blood_test",     "frame_%05d.png", 1),
    ("term_3",           TERM_FRAMES / "terminal_3",      "frame_%05d.png", 1),
    ("slide_negative",   SLIDE_FRAMES / "negative",       "frame_%05d.png", 1),
    ("term_4",           TERM_FRAMES / "terminal_4",      "frame_%05d.png", 1),

    # ACT III — "The Pattern" (2:30–3:30)
    ("slide_convergence", SLIDE_FRAMES / "convergence",   "frame_%05d.png", 1),
    ("slide_rd_sound",   SLIDE_FRAMES / "rd_sound_viz",   "frame_%05d.png", 1),
    ("term_5",           TERM_FRAMES / "terminal_5",      "frame_%05d.png", 1),

    # CODA (3:30–3:50)
    ("slide_tagline",    SLIDE_FRAMES / "tagline",        "frame_%05d.png", 1),
    ("slide_cta",        SLIDE_FRAMES / "cta",            "frame_%05d.png", 1),
]


# ===================================================================
# Main Pipeline
# ===================================================================

def main():
    banner("TASUMER MAF — Episode 2: The Briefing")
    t0 = time.time()

    skip_renders = "--skip-renders" in sys.argv
    skip_audio = "--skip-audio" in sys.argv
    preview = "--preview" in sys.argv
    extra = ["--preview"] if preview else []

    SEGMENT_DIR.mkdir(parents=True, exist_ok=True)

    # ── STAGE 1: Render all components ──
    if not skip_renders:
        banner("STAGE 1a: Synthesize Audio Assets")
        run_script("synthesize_isochronic.py")
        run_script("process_angelic_keys.py")

        banner("STAGE 1b: Capture Particle Simulations")
        run_script("capture_particles.py", extra)

        banner("STAGE 1c: Motion Graphics Slides")
        run_script("render_ep2_slides.py", extra)

        banner("STAGE 1d: Terminal Interactions")
        run_script("render_ep2_terminals.py", extra)
    else:
        print("  Using existing rendered frames (--skip-renders)")

    # ── STAGE 2: Encode each segment ──
    banner("STAGE 2: Encode Segments")

    segment_files = []
    for name, frame_dir, pattern, start_num in SEGMENT_ORDER:
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

    print(f"\n  Total segments: {len(segment_files)}")
    total_dur = sum(get_duration(s) for s in segment_files)
    print(f"  Total duration (pre-xfade): {total_dur:.1f}s")

    # ── STAGE 3: Stitch ──
    banner("STAGE 3: Stitch Final Video")
    xfade_concat(segment_files, FINAL_VIDEO)

    # ── STAGE 4: Audio ──
    if not skip_audio:
        banner("STAGE 4: Audio Mix")
        run_script("mix_audio_ep2.py")

        # Mux audio + video
        banner("STAGE 5: Mux Audio + Video")
        audio_path = BASE / "assets" / "audio" / "master_mix_ep2.wav"
        if audio_path.exists() and FINAL_VIDEO.exists():
            cmd = [
                "ffmpeg", "-y",
                "-i", posix(FINAL_VIDEO),
                "-i", posix(audio_path),
                "-c:v", "copy",
                "-c:a", "aac", "-b:a", "192k",
                "-map", "0:v:0", "-map", "1:a:0",
                "-shortest",
                "-movflags", "+faststart",
                posix(FINAL_AV),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode == 0:
                size_mb = FINAL_AV.stat().st_size / (1024 * 1024)
                print(f"  -> {FINAL_AV.name} ({size_mb:.1f}MB)")
            else:
                print(f"  Mux failed: {result.stderr[-300:]}")
        else:
            print("  WARNING: Audio or video file missing for mux")
    else:
        print("  Skipping audio (--skip-audio)")

    # ── STAGE 6: Web version ──
    banner("STAGE 6: Web-Optimized 1080p")
    source = FINAL_AV if FINAL_AV.exists() else FINAL_VIDEO
    if source.exists():
        cmd = [
            "ffmpeg", "-y",
            "-i", posix(source),
            "-vf", "scale=1920:1080",
            "-c:v", "libx264", "-crf", "23", "-preset", "slow",
            "-profile:v", "high", "-pix_fmt", "yuv420p",
        ]
        if FINAL_AV.exists():
            cmd += ["-c:a", "aac", "-b:a", "128k"]
        else:
            cmd += ["-an"]
        cmd += ["-movflags", "+faststart", posix(WEB_OUTPUT)]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            size_mb = WEB_OUTPUT.stat().st_size / (1024 * 1024)
            print(f"  -> {WEB_OUTPUT.name} ({size_mb:.1f}MB)")

    # ── VERIFICATION ──
    banner("VERIFICATION")
    for label, path in [("4K", FINAL_VIDEO), ("4K+Audio", FINAL_AV), ("1080p Web", WEB_OUTPUT)]:
        if path.exists():
            print(f"\n  [{label}] {path.name}:")
            verify_output(path)

    elapsed = time.time() - t0
    banner(f"COMPLETE — {elapsed / 60:.1f} minutes")
    print(f"  4K Video:    {FINAL_VIDEO}")
    print(f"  4K+Audio:    {FINAL_AV}")
    print(f"  1080p Web:   {WEB_OUTPUT}")
    print()


if __name__ == "__main__":
    main()
