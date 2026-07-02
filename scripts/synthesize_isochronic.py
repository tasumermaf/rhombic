#!/usr/bin/env python3
"""
Isochronic Tone Synthesizer — Episode 2 Primary Entrainment Layer.

Generates:
  1. Continuous 40 Hz isochronic pulse track (220 Hz carrier AM'd at 40 Hz
     with sinusoidal envelope) — the primary cortical entrainment mechanism.
  2. Prime-frequency isochronic accent bursts (67, 89, 23, 11 Hz) for
     placement at key reveal moments.

Based on entrainment research:
  - Isochronic tones > binaural beats for cortical response (PLOS ONE 2023)
  - 40 Hz gamma = strongest evidence base (MIT Picower Institute)
  - Sinusoidal AM envelope (not square wave) — safer, less aggressive

Output:
  assets/audio/music/isochronic_40hz.wav      (4 min continuous)
  assets/audio/stingers/prime_67hz.wav         (3s burst)
  assets/audio/stingers/prime_89hz.wav         (3s burst)
  assets/audio/stingers/prime_23hz.wav         (3s burst)
  assets/audio/stingers/prime_11hz.wav         (3s burst)

Usage:
    python scripts/synthesize_isochronic.py
"""

import numpy as np
from scipy.io import wavfile
from pathlib import Path

SR = 48000  # Match conditioning track sample rate
BASE = Path(__file__).resolve().parent.parent
AUDIO = BASE / "assets" / "audio"


def sinusoidal_am(carrier_freq, mod_freq, duration_s, amplitude=0.12,
                  sr=SR, carrier_harmonics=None):
    """Generate an isochronic tone: carrier amplitude-modulated by a
    sinusoidal envelope at mod_freq.

    Unlike binaural beats, this works through ANY playback system.
    The sinusoidal envelope (vs square) is gentler and avoids
    photosensitive-epilepsy-analog concerns.

    Parameters
    ----------
    carrier_freq : float
        Carrier frequency in Hz (e.g., 220 for MERIDIAN).
    mod_freq : float
        Modulation frequency in Hz (e.g., 40 for gamma entrainment).
    duration_s : float
        Duration in seconds.
    amplitude : float
        Peak amplitude (0.0-1.0). Default 0.12 = subliminal but measurable.
    carrier_harmonics : list of float or None
        If provided, adds harmonics. [1.0, 0.3, 0.1] = fundamental + 2nd + 3rd.
    """
    t = np.arange(int(duration_s * sr), dtype=np.float64) / sr

    # Carrier (with optional harmonics for richer timbre)
    if carrier_harmonics is None:
        carrier = np.sin(2 * np.pi * carrier_freq * t)
    else:
        carrier = np.zeros_like(t)
        total_amp = sum(abs(h) for h in carrier_harmonics)
        for i, h_amp in enumerate(carrier_harmonics):
            carrier += (h_amp / total_amp) * np.sin(2 * np.pi * carrier_freq * (i + 1) * t)

    # Sinusoidal AM envelope: oscillates between 0 and 1 at mod_freq
    # Using (1 + sin(2*pi*f*t)) / 2 gives 0-to-1 range
    envelope = (1.0 + np.sin(2 * np.pi * mod_freq * t)) / 2.0

    return carrier * envelope * amplitude


def generate_continuous_40hz(duration_s=240.0):
    """Generate 4-minute continuous 40 Hz isochronic pulse.

    Carrier: 220 Hz (MERIDIAN frequency).
    Modulation: 40 Hz sinusoidal AM.
    Amplitude: 0.12 — quiet enough to feel subliminal, strong enough to
    measure on an EEG.

    Stereo: identical in both channels (isochronic = mono-compatible).
    """
    print(f"Generating 40 Hz isochronic pulse ({duration_s:.0f}s)...")

    signal = sinusoidal_am(
        carrier_freq=220.0,     # MERIDIAN
        mod_freq=40.0,          # Gamma entrainment
        duration_s=duration_s,
        amplitude=0.12,
        carrier_harmonics=[1.0, 0.15, 0.05],  # Slight warmth
    )

    # Gentle fade in/out (3s each)
    fade = int(3.0 * SR)
    signal[:fade] *= np.linspace(0, 1, fade)
    signal[-fade:] *= np.linspace(1, 0, fade)

    # Stereo (identical both channels — isochronic doesn't need inter-aural)
    stereo = np.column_stack([signal, signal])
    return stereo


def generate_prime_accent(prime_freq, carrier_freq=220.0, duration_s=3.0,
                          amplitude=0.18):
    """Generate a short isochronic burst at a corpus prime frequency.

    These are placed at key reveal moments — the corpus speaking through
    its own frequencies.

    Parameters
    ----------
    prime_freq : float
        The prime frequency for the isochronic modulation.
    carrier_freq : float
        Carrier frequency. Defaults to 220 Hz (MERIDIAN).
    duration_s : float
        Burst duration.
    amplitude : float
        Slightly louder than the continuous pulse for accent effect.
    """
    signal = sinusoidal_am(
        carrier_freq=carrier_freq,
        mod_freq=prime_freq,
        duration_s=duration_s,
        amplitude=amplitude,
        carrier_harmonics=[1.0, 0.2, 0.08],
    )

    # Fast attack (0.1s), sustain, release (0.5s) — stinger-style envelope
    attack = int(0.1 * SR)
    release = int(0.5 * SR)
    signal[:attack] *= np.linspace(0, 1, attack)
    signal[-release:] *= np.linspace(1, 0, release)

    stereo = np.column_stack([signal, signal])
    return stereo


def write_wav(path, data, sr=SR):
    """Write stereo float64 data as 16-bit WAV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Normalize to 16-bit range
    peak = np.max(np.abs(data))
    if peak > 0:
        data = data / peak * 0.95
    wav_data = (data * 32767).astype(np.int16)
    wavfile.write(str(path), sr, wav_data)
    size_kb = path.stat().st_size // 1024
    print(f"  -> {path.name}: {len(data)/sr:.1f}s, {size_kb}KB")


def main():
    print("=" * 60)
    print("  ISOCHRONIC TONE SYNTHESIZER — Episode 2")
    print("=" * 60)

    # 1. Continuous 40 Hz pulse (4 minutes)
    continuous = generate_continuous_40hz(240.0)
    write_wav(AUDIO / "music" / "isochronic_40hz.wav", continuous)

    # 2. Prime-frequency accent bursts
    primes = {
        67: "primary thread prime — 22,477:1 reveal",
        89: "24th prime (system size) — Holly Battery reveal",
        23: "widest thread — scale-invariant Fiedler",
        11: "Card 7 hub (alpha range) — TASUMER MAF insight",
    }

    for freq, description in primes.items():
        print(f"\nGenerating {freq} Hz accent ({description})...")
        accent = generate_prime_accent(float(freq))
        write_wav(AUDIO / "stingers" / f"prime_{freq}hz.wav", accent)

    print("\n" + "=" * 60)
    print("  COMPLETE — 5 isochronic assets generated")
    print("=" * 60)


if __name__ == "__main__":
    main()
