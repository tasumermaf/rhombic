# Rung 3: Signal Processing — Raw Results

## Method

Both lattices fill [0,1]³ with density-matched sample counts. The same
RBF interpolator (thin-plate spline, zero smoothing) reconstructs the
signal at 600-800 random evaluation points well inside the boundary
(margin = 0.15). The reconstructor is topology-agnostic — any quality
difference comes purely from sample arrangement.

Isotropic test signal:
```
f(x,y,z) = sin(ωx)cos(ωy)sin(ωz) + 0.5cos(ωr) + 0.3sin(ω(x+y+z)/√3)
```
where ω = 2πf and r = √(x²+y²+z²). Signal power is spherically
symmetric in frequency space — this is exactly the class of signals
where FCC's closer-to-spherical Voronoi cell geometry should
provide the greatest spatial sampling advantage.

Frequencies sweep from 10% to 100% of the cubic Nyquist frequency.

## Scale 1: ~216 samples

Cubic: 216, FCC: 216, Eval: 800

| Freq | Cubic PSNR | FCC PSNR | Δ PSNR | MSE Ratio | Winner |
|------|-----------|---------|--------|-----------|--------|
| 0.3  | 60.59     | 61.64   | +1.05  | 0.78      | FCC    |
| 0.6  | 53.99     | 56.45   | +2.46  | 0.57      | FCC    |
| 0.9  | 46.57     | 50.39   | +3.82  | 0.41      | FCC    |
| 1.2  | 34.09     | 40.21   | +6.12  | 0.24      | FCC    |
| 1.5  | 26.00     | 31.11   | +5.11  | 0.31      | FCC    |
| 1.8  | 22.61     | 24.62   | +2.01  | 0.63      | FCC    |
| 2.1  | 20.30     | 21.54   | +1.23  | 0.75      | FCC    |
| 2.4  | 17.13     | 15.74   | -1.38  | 1.37      | Cubic  |
| 2.7  | 16.99     | 15.01   | -1.98  | 1.58      | Cubic  |
| 3.0  | 16.71     | 14.89   | -1.82  | 1.52      | Cubic  |

Isotropy at freq=1.2: Cubic=0.000217, FCC=0.000043 (ratio: 0.20)

## Scale 2: ~512 samples

Cubic: 512, FCC: 512, Eval: 800

| Freq | Cubic PSNR | FCC PSNR | Δ PSNR | MSE Ratio | Winner |
|------|-----------|---------|--------|-----------|--------|
| 0.4  | 68.52     | 75.77   | +7.25  | 0.19      | FCC    |
| 0.8  | 62.11     | 68.45   | +6.34  | 0.23      | FCC    |
| 1.2  | 50.37     | 52.82   | +2.45  | 0.57      | FCC    |
| 1.6  | 36.21     | 39.61   | +3.40  | 0.46      | FCC    |
| 2.0  | 27.60     | 29.65   | +2.06  | 0.62      | FCC    |
| 2.4  | 22.53     | 24.41   | +1.88  | 0.65      | FCC    |
| 2.8  | 19.90     | 17.56   | -2.33  | 1.71      | Cubic  |
| 3.2  | 18.20     | 15.03   | -3.18  | 2.08      | Cubic  |
| 3.6  | 16.43     | 14.39   | -2.04  | 1.60      | Cubic  |
| 4.0  | 17.18     | 15.20   | -1.97  | 1.58      | Cubic  |

Isotropy at freq=1.6: Cubic=0.000083, FCC=0.000004 (ratio: 0.05)

## Scale 3: ~1000 samples

Cubic: 1000, FCC: 1000, Eval: 600

| Freq | Cubic PSNR | FCC PSNR | Δ PSNR | MSE Ratio | Winner |
|------|-----------|---------|--------|-----------|--------|
| 0.5  | 72.62     | 82.59   | +9.97  | 0.10      | FCC    |
| 1.0  | 64.60     | 70.31   | +5.71  | 0.27      | FCC    |
| 1.5  | 50.75     | 52.79   | +2.03  | 0.63      | FCC    |
| 2.0  | 37.43     | 38.71   | +1.28  | 0.75      | FCC    |
| 2.5  | 28.16     | 27.89   | -0.27  | 1.06      | ~tie   |
| 3.0  | 23.35     | 32.70   | +9.35  | 0.12      | FCC    |
| 3.5  | 18.97     | 15.20   | -3.77  | 2.38      | Cubic  |
| 4.0  | 19.39     | 15.85   | -3.55  | 2.26      | Cubic  |
| 4.5  | 17.57     | 14.85   | -2.72  | 1.87      | Cubic  |
| 5.0  | 16.80     | 14.18   | -2.63  | 1.83      | Cubic  |

Isotropy at freq=2.0: Cubic=0.000014, FCC=0.000002 (ratio: 0.11)

## Summary

| Metric | 216 samples | 512 samples | 1000 samples |
|--------|------------|------------|-------------|
| Peak FCC advantage | +6.12 dB | +7.25 dB | +9.97 dB |
| Best MSE ratio | 0.24× | 0.19× | 0.10× |
| Isotropy ratio | 0.20× | 0.05× | 0.11× |
| Crossover frequency | ~80% Nyquist | ~70% Nyquist | ~50% Nyquist |
| FCC wins at | 7/10 freqs | 6/10 freqs | 6/10 freqs |

## Reproduction

```bash
python -m rhombic.signal            # default suite (216, 512)
python scripts/generate_rung3_dashboard.py  # full dashboard with 1000
```
