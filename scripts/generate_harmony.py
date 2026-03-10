"""
Harmony of the Spheres — background music derived from the corpus.

The 8 tracked primes of the tessitura map to overtone frequencies. Each of
the 24 cards carries a small set of activation values; the primes that divide
those values decide which prime-voices sound while that card is playing. The
Fiedler eigenvalue ratio (2.30x) sets the pulsation rate and the amplification
ratio (6.11x) governs harmonic enrichment as the piece progresses.

Corpus boundary: the per-card activation values and the fundamental are
corpus-derived and load from a gitignored private sidecar
(rhombic/data/harmony_private.json), the same gating as rhombic.corpus.
Without the sidecar the script renders a seeded synthetic demo so the code
path stays runnable; the released audio was rendered with the private data.
"""

import json
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, filtfilt

SR = 44100
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "assets" / "audio" / "music"
OUT.mkdir(parents=True, exist_ok=True)

# ── The 8 Prime Voices (Tessitura Threads) — public ──
PRIMES = [11, 17, 19, 23, 29, 31, 67, 89]

# Fiedler Ratio as Rhythmic Pulse (2.30x = one gentle throb per ~3.5s)
FIEDLER_RATIO = 2.30
PULSE_FREQ = FIEDLER_RATIO / 8.0

# Amplification Ratio governs harmonic enrichment over time
AMP_RATIO = 6.11

# ── Card activation values: private sidecar, or a synthetic demo ──
SIDECAR = ROOT / "rhombic" / "data" / "harmony_private.json"
DEMO_FUNDAMENTAL = 220.0   # public fallback fundamental (Hz)
N_CARDS = 24


def load_card_values(sidecar: Path = SIDECAR, seed: int = 42):
    """Return (fundamental_hz, [(label, [values]), ...]) for 24 cards.

    Private sidecar when present; otherwise a seeded synthetic demo set whose
    labels are card indices and whose values are random integers. The demo
    exercises the same code path but is not the corpus.
    """
    if sidecar.exists():
        d = json.loads(sidecar.read_text(encoding="utf-8"))
        cards = [(c["label"], list(c["values"])) for c in d["cards"]]
        return float(d["fundamental_hz"]), cards, "private"
    rng = np.random.default_rng(seed)
    cards = [(f"card_{i:02d}", [int(v) for v in rng.integers(10, 1300, size=5)])
             for i in range(N_CARDS)]
    return DEMO_FUNDAMENTAL, cards, "demo"


FUNDAMENTAL, CARD_VALUES, CARD_SOURCE = load_card_values()


def prime_to_freq(p, fundamental=None):
    """Map prime to overtone frequency, normalized to 100-800 Hz."""
    f = p * (FUNDAMENTAL if fundamental is None else fundamental)
    while f > 800:
        f /= 2
    while f < 100:
        f *= 2
    return f


PRIME_FREQS = {p: prime_to_freq(p) for p in PRIMES}


def factorize(n):
    factors = set()
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.add(d)
            n //= d
        d += 1
    if n > 1:
        factors.add(n)
    return factors


def active_primes_single(value):
    factors = factorize(value)
    return [p for p in PRIMES if p in factors]


def active_primes(values):
    """Which tracked primes appear across ALL values for a card?"""
    found = set()
    for v in values:
        found.update(active_primes_single(v))
    return sorted(p for p in PRIMES if p in found)


def make_prime_tone(freq, duration, sr=SR):
    t = np.linspace(0, duration, int(sr * duration), False)
    sig = np.sin(2 * np.pi * freq * t)
    sig += 0.4 * np.sin(2 * np.pi * freq * 2 * t)
    sig += 0.15 * np.sin(2 * np.pi * freq * 3 * t)
    vib_rate = (freq % 7) + 3
    sig *= 1.0 + 0.004 * np.sin(2 * np.pi * vib_rate * t)
    return sig


def generate_harmony(duration_s=140.0, sr=SR):
    total = int(sr * duration_s)
    t = np.linspace(0, duration_s, total, False)

    track_L = np.zeros(total)
    track_R = np.zeros(total)

    # Layer 1: Bass drone on the fundamental
    bass = 0.25 * np.sin(2 * np.pi * (FUNDAMENTAL / 4) * t)
    bass += 0.15 * np.sin(2 * np.pi * (FUNDAMENTAL / 2) * t)
    pulse = 0.85 + 0.15 * np.sin(2 * np.pi * PULSE_FREQ * t)
    bass *= pulse
    track_L += bass
    track_R += bass

    # Layer 2: Prime voice pads driven by card factorizations
    chord_duration = 6.0
    crossfade_s = 2.0

    for card_idx in range(len(CARD_VALUES)):
        name, values = CARD_VALUES[card_idx]
        start_s = card_idx * (chord_duration - crossfade_s)
        if start_s >= duration_s:
            break

        active = active_primes(values)
        all_voice_freqs = []
        for p in PRIMES:
            if p in active:
                all_voice_freqs.append((PRIME_FREQS[p], 1.0))
            else:
                all_voice_freqs.append((PRIME_FREQS[p], 0.08))

        progress = card_idx / len(CARD_VALUES)
        enrichment = 1.0 + (AMP_RATIO / FIEDLER_RATIO - 1.0) * progress

        this_dur = min(chord_duration, duration_s - start_s)
        start_idx = int(start_s * sr)
        dur_samples = int(this_dur * sr)

        env = np.ones(dur_samples)
        attack = min(int(sr * 1.5), dur_samples // 3)
        release = min(int(sr * 2.0), dur_samples // 3)
        env[:attack] = np.linspace(0, 1, attack) ** 0.7
        env[-release:] = np.linspace(1, 0, release) ** 0.7

        for voice_idx, (freq, amplitude) in enumerate(all_voice_freqs):
            prime = PRIMES[voice_idx]
            if prime > 31:
                amp = amplitude * enrichment * 0.4
            else:
                amp = amplitude * 0.5

            tone = make_prime_tone(freq, this_dur, sr)
            tone *= env * amp

            pan = (voice_idx / (len(PRIMES) - 1)) * 0.6 + 0.2
            end_idx = min(start_idx + dur_samples, total)
            actual_len = end_idx - start_idx
            track_L[start_idx:end_idx] += tone[:actual_len] * (1 - pan)
            track_R[start_idx:end_idx] += tone[:actual_len] * pan

    # Layer 3: 67-thread shimmer (crystalline high overtone)
    shimmer_freq = PRIME_FREQS[67] * 4
    while shimmer_freq > 6000:
        shimmer_freq /= 2
    shimmer = 0.02 * np.sin(2 * np.pi * shimmer_freq * t)
    shimmer *= 0.5 + 0.5 * np.sin(2 * np.pi * 0.12 * t)
    shimmer[:int(sr * 30)] *= np.linspace(0, 1, int(sr * 30))
    track_L += shimmer * 0.6
    track_R += shimmer * 0.4

    # Layer 4: 90 Hz sub-harmonic
    sub = 0.08 * np.sin(2 * np.pi * 90 * t)
    sub *= pulse
    track_L += sub
    track_R += sub

    # Post-processing: warm low-pass
    b, a = butter(3, 3500 / (sr / 2), btype="low")
    track_L = filtfilt(b, a, track_L)
    track_R = filtfilt(b, a, track_R)

    # Normalize
    peak = max(np.max(np.abs(track_L)), np.max(np.abs(track_R)))
    track_L = track_L / peak * 0.55
    track_R = track_R / peak * 0.55

    # Fade in/out
    fade_in = int(sr * 4)
    fade_out = int(sr * 4)
    track_L[:fade_in] *= np.linspace(0, 1, fade_in)
    track_R[:fade_in] *= np.linspace(0, 1, fade_in)
    track_L[-fade_out:] *= np.linspace(1, 0, fade_out)
    track_R[-fade_out:] *= np.linspace(1, 0, fade_out)

    return np.column_stack([track_L, track_R])


if __name__ == "__main__":
    print("=== Harmony of the Spheres ===")
    print(f"Card values: {CARD_SOURCE} ({len(CARD_VALUES)} cards)")
    print(f"Fundamental: {FUNDAMENTAL} Hz")
    print(f"Pulse: Fiedler {FIEDLER_RATIO}x -> {PULSE_FREQ:.4f} Hz")
    print(f"Enrichment: {FIEDLER_RATIO}x -> {AMP_RATIO}x over duration")
    print()

    print("Prime -> Frequency mapping:")
    for p, f in PRIME_FREQS.items():
        print(f"  {p:3d} -> {f:7.1f} Hz")
    print()

    for idx, (name, vals) in enumerate(CARD_VALUES):
        active = active_primes(vals)
        marker = " <<<" if active else ""
        label = name if CARD_SOURCE == "demo" else f"card_{idx:02d}"
        print(f"  {label:10s}  vals={len(vals):2d}  threads: {active}{marker}")
    print()

    stereo = generate_harmony(140.0)
    wav_data = (stereo * 32767).astype(np.int16)
    wavfile.write(str(OUT / "harmony_of_spheres.wav"), SR, wav_data)
    fsize = (OUT / "harmony_of_spheres.wav").stat().st_size
    print(f"\n-> harmony_of_spheres.wav: 140.0s stereo, {fsize // 1024}KB")
