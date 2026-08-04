"""
Verify the Director's paired non-inferiority design by simulation rather than
taking the table on trust. Loop norm: corrections travel both ways.

Design: same held-out case answered by student and teacher, blind-scored
correct/incorrect by the PI. Paired binary outcome -> McNemar-type
non-inferiority on the difference of correlated proportions.

H0: p_student - p_teacher <= -delta   (student worse by more than the margin)
H1: p_student - p_teacher  > -delta
one-sided alpha = 0.05, target power 0.80.
"""
import random
from math import sqrt

def trial(n_items, disc_rate, win_frac_among_disc, delta, rng):
    b = c = 0                      # b: student wins, c: teacher wins
    for _ in range(n_items):
        if rng.random() < disc_rate:
            if rng.random() < win_frac_among_disc: b += 1
            else: c += 1
    n_disc = b + c
    if n_disc == 0:
        return False
    d = (b - c) / n_items                       # observed difference
    se = sqrt(n_disc) / n_items                 # standard error of the difference
    z = (d + delta) / se                        # non-inferiority statistic
    return z > 1.645                            # one-sided alpha = 0.05

def power(n_items, disc_rate, win_frac, delta, reps=40000, seed=20260804):
    rng = random.Random(seed)
    return sum(trial(n_items, disc_rate, win_frac, delta, rng) for _ in range(reps)) / reps

print("Paired non-inferiority power, 40,000 sims per cell, one-sided alpha=0.05")
print()
print(f"{'items':>6} {'disc%':>6} {'true win%':>10} {'margin':>7} {'power':>7}")
for n in (120, 155, 200, 250):
    for dr in (0.40,):
        for wf in (0.50, 0.45):
            for delta in (0.15,):
                p = power(n, dr, wf, delta)
                print(f"{n:>6} {dr*100:>5.0f}% {wf*100:>9.0f}% {delta*100:>6.0f}pp {p:>7.3f}")
print()
print("Sensitivity to the discordance assumption at 200 items, 15pp, true win 50%:")
for dr in (0.20, 0.30, 0.40, 0.50, 0.60):
    print(f"  discordance {dr*100:>3.0f}%  ->  power {power(200, dr, 0.50, 0.15):.3f}")
print()
print("What a 10pp margin would actually cost (true win 50%, 40% discordance):")
for n in (200, 300, 380, 450):
    print(f"  {n:>4} items -> power {power(n, 0.40, 0.50, 0.10):.3f}")
