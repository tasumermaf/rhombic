"""
Master Audio Mix — Harmony of the Spheres + Voice + SFX.

Layers all audio assets onto the video timeline:
  - Harmony of the Spheres (full during slides, attenuated during terminals)
  - ElevenLabs voice lines at precise timestamps
  - Mechanical keyboard SFX during terminal typing phases
  - Terminal CRT hum during terminal segments
  - Transition effects between segments
  - Humorous volume ducking (boardroom muzak fades OUT when terminal
    takes over, fades back IN with corporate certainty)

Output: assets/audio/master_mix.wav
"""

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, filtfilt, resample
from pathlib import Path
import subprocess
import random

SR = 44100
BASE = Path(__file__).resolve().parent.parent
AUDIO = BASE / "assets" / "audio"
OUT = AUDIO / "master_mix.wav"

XFADE = 0.5

# ── Timeline (from produce_unified_video.py segment structure) ──
SEGMENTS = [
    ("logo",         10.0,  "slide"),
    ("slide_thesis",  7.0,  "slide"),
    ("term_1",       22.07, "terminal"),
    ("slide_numbers",10.0,  "slide"),
    ("term_2",       24.33, "terminal"),
    ("term_3",       21.73, "terminal"),
    ("slide_vision",  8.0,  "slide"),
    ("term_4",       26.33, "terminal"),
    ("slide_cta",     7.0,  "slide"),
]

# Calculate absolute start times
def compute_starts():
    cumulative = 0.0
    starts = []
    for i, (name, dur, typ) in enumerate(SEGMENTS):
        starts.append(cumulative)
        if i < len(SEGMENTS) - 1:
            cumulative += dur - XFADE
    return starts

STARTS = compute_starts()
TOTAL_DUR = 132.5  # Match video duration

# ── Voice Events ──
VOICE_EVENTS = [
    # (filename, absolute_time, volume)
    ("slide1_thesis",  STARTS[1] + 1.5,  0.85),
    ("slide2_numbers", STARTS[3] + 1.5,  0.85),
    ("slide3_vision",  STARTS[6] + 1.5,  0.85),
    ("term1_reaction", STARTS[2] + 15.0, 0.70),
    ("term2_reaction", STARTS[4] + 16.0, 0.70),
    ("term3_reaction", STARTS[5] + 14.0, 0.70),
    ("term4_tagline",  STARTS[7] + 21.0, 0.80),
]

# ── Terminal typing phases (approximate: prompt appears, user "types") ──
# Each terminal has a prompt phase where text appears character by character
TYPING_EVENTS = [
    # (start_time, duration, density) — density = clicks per second
    (STARTS[2] + 1.5,  3.0, 8),   # term_1 prompt
    (STARTS[4] + 1.5,  4.0, 7),   # term_2 prompt
    (STARTS[5] + 1.5,  3.5, 9),   # term_3 prompt
    (STARTS[7] + 1.5,  3.0, 7),   # term_4 prompt
]

# ── Transition points (segment boundaries) ──
TRANSITION_TIMES = [s for s in STARTS[1:]]  # Every segment boundary


def load_wav(path):
    """Load WAV, return mono float64 at SR."""
    rate, data = wavfile.read(str(path))
    if data.dtype == np.int16:
        data = data.astype(np.float64) / 32767.0
    elif data.dtype == np.int32:
        data = data.astype(np.float64) / 2147483647.0
    elif data.dtype == np.float32:
        data = data.astype(np.float64)

    # Convert stereo to mono if needed, or keep stereo
    if len(data.shape) > 1:
        return rate, data
    return rate, data


def load_mp3(path):
    """Load MP3 via ffmpeg, return mono float64 at SR."""
    cmd = [
        "ffmpeg", "-y", "-i", str(path),
        "-f", "wav", "-acodec", "pcm_s16le",
        "-ar", str(SR), "-ac", "1",
        "pipe:1"
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=30)
    if result.returncode != 0:
        print(f"  WARNING: Failed to decode {path}")
        return np.zeros(SR)  # 1 second of silence

    # Parse WAV from pipe
    raw = result.stdout
    # WAV header is 44 bytes for standard PCM
    if len(raw) < 44:
        return np.zeros(SR)
    # Find data chunk
    data_offset = raw.find(b'data')
    if data_offset < 0:
        return np.zeros(SR)
    data_size = int.from_bytes(raw[data_offset+4:data_offset+8], 'little')
    audio_data = raw[data_offset+8:data_offset+8+data_size]
    samples = np.frombuffer(audio_data, dtype=np.int16).astype(np.float64) / 32767.0
    return samples


def resample_to_sr(data, orig_sr):
    """Resample audio to target SR."""
    if orig_sr == SR:
        return data
    ratio = SR / orig_sr
    new_len = int(len(data) * ratio)
    return resample(data, new_len)


def overlay(track_L, track_R, mono_signal, start_s, volume=1.0, pan=0.5):
    """Overlay a mono signal onto stereo tracks at a given time."""
    start_idx = int(start_s * SR)
    end_idx = min(start_idx + len(mono_signal), len(track_L))
    actual_len = end_idx - start_idx
    if actual_len <= 0:
        return
    sig = mono_signal[:actual_len] * volume
    track_L[start_idx:end_idx] += sig * (1 - pan)
    track_R[start_idx:end_idx] += sig * pan


def overlay_stereo(track_L, track_R, stereo, start_s, volume=1.0):
    """Overlay a stereo signal onto stereo tracks at a given time."""
    start_idx = int(start_s * SR)
    end_idx = min(start_idx + stereo.shape[0], len(track_L))
    actual_len = end_idx - start_idx
    if actual_len <= 0:
        return
    track_L[start_idx:end_idx] += stereo[:actual_len, 0] * volume
    track_R[start_idx:end_idx] += stereo[:actual_len, 1] * volume


def build_harmony_envelope(total_samples):
    """
    Build a volume envelope for the harmony track.
    Full volume during slides, duck to ~25% during terminals.
    Smooth crossfades at transitions for humorous effect —
    the corporate muzak fades out reluctantly as the terminal hacker
    takes over, then fades back in with corporate certainty.
    """
    env = np.ones(total_samples)
    fade_samples = int(SR * 1.5)  # 1.5s fades

    for (name, dur, typ), start in zip(SEGMENTS, STARTS):
        seg_start = int(start * SR)
        seg_end = min(int((start + dur) * SR), total_samples)

        if typ == "terminal":
            # Duck the harmony during terminal segments
            # But keep a little — the muzak never fully goes away
            duck_level = 0.15
            # Apply duck to the segment region
            env[seg_start:seg_end] = duck_level

            # Smooth fade-out at terminal entry
            fade_start = max(seg_start - fade_samples, 0)
            fade_len = seg_start - fade_start
            if fade_len > 0:
                env[fade_start:seg_start] = np.linspace(1.0, duck_level, fade_len)

            # Smooth fade-in at terminal exit
            fade_end = min(seg_end + fade_samples, total_samples)
            fade_len = fade_end - seg_end
            if fade_len > 0:
                env[seg_end:fade_end] = np.linspace(duck_level, 1.0, fade_len)

    return env


def generate_typing_clicks(duration_s, density):
    """Generate a stream of random typing clicks."""
    n_clicks = int(duration_s * density)
    total_samples = int(duration_s * SR)
    result = np.zeros(total_samples)

    # Load click templates
    click_files = sorted((AUDIO / "sfx").glob("typing_*.wav"))
    clicks = []
    for cf in click_files:
        _, data = wavfile.read(str(cf))
        if data.dtype == np.int16:
            data = data.astype(np.float64) / 32767.0
        clicks.append(data)

    if not clicks:
        return result

    for _ in range(n_clicks):
        pos = random.randint(0, max(0, total_samples - SR // 10))
        click = random.choice(clicks)
        # Slight volume variation
        vol = random.uniform(0.3, 0.7)
        end = min(pos + len(click), total_samples)
        actual = end - pos
        result[pos:end] += click[:actual] * vol

    return result


def main():
    total_samples = int(TOTAL_DUR * SR)
    track_L = np.zeros(total_samples)
    track_R = np.zeros(total_samples)

    # ── Layer 1: Harmony of the Spheres ──
    print("Loading Harmony of the Spheres...")
    harmony_path = AUDIO / "music" / "harmony_of_spheres.wav"
    h_sr, harmony = load_wav(harmony_path)
    if len(harmony.shape) > 1:
        # Stereo
        if h_sr != SR:
            harmony_L = resample_to_sr(harmony[:, 0], h_sr)
            harmony_R = resample_to_sr(harmony[:, 1], h_sr)
            harmony = np.column_stack([harmony_L, harmony_R])
        # Trim to video length
        harm_len = min(harmony.shape[0], total_samples)
        harmony = harmony[:harm_len]
    else:
        if h_sr != SR:
            harmony = resample_to_sr(harmony, h_sr)
        harm_len = min(len(harmony), total_samples)
        harmony = np.column_stack([harmony[:harm_len], harmony[:harm_len]])

    # Build volume envelope (full during slides, ducked during terminals)
    envelope = build_harmony_envelope(harm_len)
    harmony_vol = 0.65  # Overall harmony volume
    track_L[:harm_len] += harmony[:, 0] * envelope * harmony_vol
    track_R[:harm_len] += harmony[:, 1] * envelope * harmony_vol
    print(f"  Harmony: {harm_len / SR:.1f}s, ducking at terminal segments")

    # ── Layer 2: Terminal CRT Hum ──
    print("Loading terminal hum...")
    hum_path = AUDIO / "sfx" / "terminal_hum.wav"
    if hum_path.exists():
        h_sr, hum = load_wav(hum_path)
        if len(hum.shape) > 1:
            hum = hum[:, 0]
        if h_sr != SR:
            hum = resample_to_sr(hum, h_sr)

        for (name, dur, typ), start in zip(SEGMENTS, STARTS):
            if typ == "terminal":
                seg_len = int(dur * SR)
                hum_seg = hum[:min(seg_len, len(hum))]
                # Fade in/out
                fade = min(int(SR * 0.5), len(hum_seg) // 4)
                hum_seg = hum_seg.copy()
                hum_seg[:fade] *= np.linspace(0, 1, fade)
                hum_seg[-fade:] *= np.linspace(1, 0, fade)
                overlay(track_L, track_R, hum_seg, start, volume=0.6, pan=0.5)
        print("  Terminal hum placed at all terminal segments")

    # ── Layer 3: Voice Lines ──
    print("Placing voice lines...")
    for vname, vtime, vvol in VOICE_EVENTS:
        vpath = AUDIO / "voice" / f"{vname}.mp3"
        if not vpath.exists():
            print(f"  WARNING: Missing {vpath}")
            continue
        voice = load_mp3(vpath)
        # Gentle fade in/out on voice
        fade = min(int(SR * 0.05), len(voice) // 4)
        voice[:fade] *= np.linspace(0, 1, fade)
        voice[-fade:] *= np.linspace(1, 0, fade)

        # Terminal voices slightly left-panned, slide voices centered
        pan = 0.4 if "term" in vname else 0.5
        overlay(track_L, track_R, voice, vtime, volume=vvol, pan=pan)
        print(f"  {vname} at {vtime:.1f}s ({len(voice)/SR:.1f}s)")

    # ── Layer 4: Typing SFX ──
    print("Generating typing clicks...")
    for t_start, t_dur, t_density in TYPING_EVENTS:
        clicks = generate_typing_clicks(t_dur, t_density)
        overlay(track_L, track_R, clicks, t_start, volume=0.4, pan=0.45)
    print(f"  {len(TYPING_EVENTS)} typing sequences placed")

    # ── Layer 5: Transition Effects ──
    print("Placing transition effects...")
    whoosh_path = AUDIO / "sfx" / "whoosh.wav"
    scratch_path = AUDIO / "sfx" / "scratch.wav"

    if whoosh_path.exists():
        _, whoosh = load_wav(whoosh_path)
        if len(whoosh.shape) > 1:
            whoosh = whoosh[:, 0]

    if scratch_path.exists():
        _, scratch = load_wav(scratch_path)
        if len(scratch.shape) > 1:
            scratch = scratch[:, 0]

    for i, t in enumerate(TRANSITION_TIMES):
        # Alternate between whoosh and scratch
        if i % 2 == 0 and whoosh_path.exists():
            overlay(track_L, track_R, whoosh, t - 0.3, volume=0.25, pan=0.5)
        elif scratch_path.exists():
            overlay(track_L, track_R, scratch, t - 0.2, volume=0.2, pan=0.5)
    print(f"  {len(TRANSITION_TIMES)} transitions placed")

    # ── Master: Normalize + Limiter ──
    print("Mastering...")
    peak = max(np.max(np.abs(track_L)), np.max(np.abs(track_R)))
    if peak > 0:
        # Normalize to -1dB headroom
        target = 0.89  # ~-1dB
        gain = target / peak
        track_L *= gain
        track_R *= gain

    # Gentle fade in at start, fade out at end
    fade_in = int(SR * 2)
    fade_out = int(SR * 3)
    track_L[:fade_in] *= np.linspace(0, 1, fade_in)
    track_R[:fade_in] *= np.linspace(0, 1, fade_in)
    track_L[-fade_out:] *= np.linspace(1, 0, fade_out)
    track_R[-fade_out:] *= np.linspace(1, 0, fade_out)

    # Soft clip
    track_L = np.clip(track_L, -1.0, 1.0)
    track_R = np.clip(track_R, -1.0, 1.0)

    # Write
    stereo = np.column_stack([track_L, track_R])
    wav_data = (stereo * 32767).astype(np.int16)
    wavfile.write(str(OUT), SR, wav_data)
    size_kb = OUT.stat().st_size // 1024
    print(f"\n-> {OUT.name}: {TOTAL_DUR:.1f}s stereo, {size_kb}KB")

    # ── Mux with Video ──
    print("\nMuxing audio with video...")
    video_path = BASE / "assets" / "video" / "tasumer_maf_unified.mp4"
    output_av = BASE / "assets" / "video" / "tasumer_maf_unified_av.mp4"

    if not video_path.exists():
        print(f"  WARNING: Video not found at {video_path}")
        return

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path.as_posix(),
        "-i", OUT.as_posix(),
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        "-map", "0:v:0", "-map", "1:a:0",
        "-shortest",
        "-movflags", "+faststart",
        output_av.as_posix(),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode == 0:
        size_mb = output_av.stat().st_size / (1024 * 1024)
        print(f"-> {output_av.name}: {size_mb:.1f}MB (4K with audio)")

        # Also generate web version (1080p)
        web_output = BASE / "website" / "demo.mp4"
        cmd_web = [
            "ffmpeg", "-y",
            "-i", output_av.as_posix(),
            "-vf", "scale=1920:1080",
            "-c:v", "h264_nvenc", "-preset", "p7", "-cq", "24",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
            web_output.as_posix(),
        ]
        result_web = subprocess.run(cmd_web, capture_output=True, text=True, timeout=120)
        if result_web.returncode == 0:
            web_mb = web_output.stat().st_size / (1024 * 1024)
            print(f"-> {web_output.name}: {web_mb:.1f}MB (1080p web)")
        else:
            print(f"  Web encode failed: {result_web.stderr[-300:]}")
    else:
        print(f"  Mux failed: {result.stderr[-300:]}")


if __name__ == "__main__":
    main()
