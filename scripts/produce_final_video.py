#!/usr/bin/env python3
"""
TASUMER MAF — Final Video Production Pipeline.

Runs the complete pipeline:
  1. Ray-traced logo animation (4K SSAA)
  2. Motion graphics demo (4K)
  3. FFmpeg stitch with crossfade

Output: assets/video/tasumer_maf_final.mp4

Usage:
    python scripts/produce_final_video.py              # Full production
    python scripts/produce_final_video.py --skip-logo   # Skip logo render (use existing)
    python scripts/produce_final_video.py --skip-demo   # Skip demo render (use existing)
    python scripts/produce_final_video.py --preview      # 720p fast
"""

import io
import os
import subprocess
import sys
import time
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

BASE = Path(__file__).resolve().parent.parent
ASSETS = BASE / "assets" / "video"
LOGO_FRAMES = ASSETS / "logo_frames"
DEMO_FRAMES = ASSETS / "demo_frames"
FINAL_OUTPUT = ASSETS / "tasumer_maf_final.mp4"
LOGO_OUTPUT = ASSETS / "tasumer_maf_logo.mp4"
DEMO_OUTPUT = ASSETS / "tasumer_maf_demo.mp4"
COMBINED_FRAMES = ASSETS / "combined_frames"

FPS = 30
XFADE_FRAMES = 15  # crossfade between logo and demo


def banner(msg):
    print(f"\n{'=' * 64}")
    print(f"  {msg}")
    print(f"{'=' * 64}\n")


def run_script(script_name, extra_args=None):
    """Run a Python script from the scripts directory."""
    cmd = [sys.executable, str(BASE / "scripts" / script_name)]
    if extra_args:
        cmd.extend(extra_args)

    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(BASE), timeout=1800)
    return result.returncode == 0


def count_frames(frame_dir):
    """Count PNG frames in a directory."""
    if not frame_dir.exists():
        return 0
    return len(list(frame_dir.glob("frame_*.png")))


def encode_sequence(frame_dir, output_path, pattern="frame_%04d.png"):
    """Encode a frame sequence to MP4 with NVENC."""
    n = count_frames(frame_dir)
    if n == 0:
        print(f"  ERROR: No frames in {frame_dir}")
        return False

    print(f"  Encoding {n} frames from {frame_dir}...")
    input_pattern = str(frame_dir / pattern)

    for encoder, enc_args in [
        ("h264_nvenc", ["-preset", "p7", "-cq", "16", "-profile:v", "high"]),
        ("libx264", ["-crf", "16", "-profile:v", "high"]),
    ]:
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(FPS),
            "-i", input_pattern,
            "-c:v", encoder,
        ] + enc_args + [
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(output_path),
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode == 0:
                size_mb = output_path.stat().st_size / (1024 * 1024)
                print(f"  Encoded: {output_path.name} ({size_mb:.1f} MB, {encoder})")
                return True
            print(f"  {encoder} failed, trying next...")
        except Exception as e:
            print(f"  {encoder} error: {e}")

    return False


def stitch_videos(logo_mp4, demo_mp4, output_path):
    """Concatenate logo and demo videos with a crossfade transition."""
    xfade_sec = XFADE_FRAMES / FPS

    # Get logo duration for xfade offset
    probe_cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "csv=p=0",
        str(logo_mp4),
    ]
    try:
        result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=30)
        logo_duration = float(result.stdout.strip())
    except Exception:
        logo_duration = 10.0  # fallback

    offset = logo_duration - xfade_sec

    # FFmpeg xfade filter
    cmd = [
        "ffmpeg", "-y",
        "-i", str(logo_mp4),
        "-i", str(demo_mp4),
        "-filter_complex",
        f"[0:v][1:v]xfade=transition=fade:duration={xfade_sec}:offset={offset},format=yuv420p[v]",
        "-map", "[v]",
        "-c:v", "h264_nvenc",
        "-preset", "p7",
        "-cq", "16",
        "-profile:v", "high",
        "-movflags", "+faststart",
        str(output_path),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode == 0:
            size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"  Final: {output_path} ({size_mb:.1f} MB)")
            return True
        else:
            print(f"  xfade failed: {result.stderr[-300:]}")
            # Fallback: simple concat
            print("  Trying simple concat...")
            return concat_simple(logo_mp4, demo_mp4, output_path)
    except Exception as e:
        print(f"  Stitch error: {e}")
        return concat_simple(logo_mp4, demo_mp4, output_path)


def concat_simple(v1, v2, output):
    """Simple concatenation without transition."""
    list_file = ASSETS / "concat_list.txt"
    list_file.write_text(f"file '{v1.name}'\nfile '{v2.name}'\n")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(list_file),
        "-c:v", "h264_nvenc",
        "-preset", "p7", "-cq", "16",
        "-movflags", "+faststart",
        str(output),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=600, cwd=str(ASSETS))
        if result.returncode == 0:
            size_mb = output.stat().st_size / (1024 * 1024)
            print(f"  Concat: {output} ({size_mb:.1f} MB)")
            list_file.unlink(missing_ok=True)
            return True
        print(f"  Concat failed: {result.stderr[-200:]}")
    except Exception as e:
        print(f"  Concat error: {e}")
    list_file.unlink(missing_ok=True)
    return False


def verify_output(path):
    """Print video info via ffprobe."""
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate",
            "-show_entries", "format=duration,size",
            "-of", "json",
            str(path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            import json
            info = json.loads(result.stdout)
            stream = info.get("streams", [{}])[0]
            fmt = info.get("format", {})
            w = stream.get("width", "?")
            h = stream.get("height", "?")
            fps = stream.get("r_frame_rate", "?")
            dur = float(fmt.get("duration", 0))
            size = int(fmt.get("size", 0))
            print(f"  Resolution:  {w}x{h}")
            print(f"  Frame rate:  {fps}")
            print(f"  Duration:    {dur:.1f}s")
            print(f"  File size:   {size / 1024 / 1024:.1f} MB")

            if dur > 140 or size > 512 * 1024 * 1024:
                print("  WARNING: May exceed Twitter limits")
            else:
                print("  Twitter-compatible")
    except Exception:
        pass


def main():
    banner("TASUMER MAF -- Production Video Pipeline")
    t0 = time.time()

    skip_logo = "--skip-logo" in sys.argv
    skip_demo = "--skip-demo" in sys.argv
    preview = "--preview" in sys.argv
    extra = ["--preview"] if preview else []

    # Stage 1: Logo
    if not skip_logo:
        banner("STAGE 1: Ray-Traced Logo Animation")
        if not run_script("raytrace_rd_logo.py", extra):
            print("Logo render failed!")
            sys.exit(1)
    else:
        print("Skipping logo render (using existing frames)")

    n_logo = count_frames(LOGO_FRAMES)
    print(f"  Logo frames: {n_logo}")

    # Stage 2: Demo
    if not skip_demo:
        banner("STAGE 2: Motion Graphics Demo")
        if not run_script("render_demo_production.py", extra):
            print("Demo render failed!")
            sys.exit(1)
    else:
        print("Skipping demo render (using existing frames)")

    n_demo = count_frames(DEMO_FRAMES)
    print(f"  Demo frames: {n_demo}")

    # Stage 3: Encode each part
    banner("STAGE 3: Encode")

    if n_logo > 0:
        encode_sequence(LOGO_FRAMES, LOGO_OUTPUT, "frame_%04d.png")
    if n_demo > 0:
        encode_sequence(DEMO_FRAMES, DEMO_OUTPUT, "frame_%05d.png")

    # Stage 4: Stitch
    banner("STAGE 4: Stitch Final Video")

    if LOGO_OUTPUT.exists() and DEMO_OUTPUT.exists():
        stitch_videos(LOGO_OUTPUT, DEMO_OUTPUT, FINAL_OUTPUT)
    elif LOGO_OUTPUT.exists():
        # Logo only
        import shutil
        shutil.copy2(str(LOGO_OUTPUT), str(FINAL_OUTPUT))
        print(f"  Logo-only output: {FINAL_OUTPUT}")
    elif DEMO_OUTPUT.exists():
        import shutil
        shutil.copy2(str(DEMO_OUTPUT), str(FINAL_OUTPUT))
        print(f"  Demo-only output: {FINAL_OUTPUT}")
    else:
        print("  ERROR: No video files to stitch")
        sys.exit(1)

    # Stage 5: Verify
    banner("VERIFICATION")
    if FINAL_OUTPUT.exists():
        verify_output(FINAL_OUTPUT)

    elapsed = time.time() - t0
    banner(f"COMPLETE -- {elapsed / 60:.1f} minutes")
    print(f"  Final video: {FINAL_OUTPUT}")
    print(f"  Ready for hackathon submission.\n")


if __name__ == "__main__":
    main()
