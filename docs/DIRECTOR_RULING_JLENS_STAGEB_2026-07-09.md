# Director's Ruling — Level-B J-lens Stage-B Defaults

**Date:** July 9, 2026
**From:** the Director · **To:** Meridian (cc: PI)
**Re:** the four pinned Stage-B defaults of `asset1_jacobian_lens.py`, deferred in the A3 ruling, now on the clock
**Verified against:** repo `main` at `a703cc27`. I read the tool source and confirmed the two load-bearing interlocks before ruling. All four defaults: **APPROVED for reportable use**, with one condition that gates the arbiter role specifically (not the construction work, which is null-class and proceeds).

## What I verified in source

- **The Level-A/Level-B probe+sketch identity is import-enforced, not re-implemented.** Line 114: `from asset1_vocab_signature import (...)` with the "shared probes/sketch" note; the sketch uses `vocab_sketch` stream tag 72, same as Level A. This is what makes the A-vs-B comparison a clean isolation of mid-network propagation rather than two loosely-related readouts, and it is enforced by import, so it cannot silently drift. This is the single most important property for the arbiter role and it holds.
- **The GPU/transformers interlock is real.** transformers is lazy-imported inside `estimate_lenses` (line 236), and `--estimate` refuses without `--i-have-gpu-and-bank-is-complete` checked before any import; module import never initializes CUDA. Stage C is additionally `require_complete_bank`-gated. The non-reportable construction pilot on Hermes is correctly outside what this ruling governs.

## The four defaults — APPROVED

1. **32 sha256-pinned neutral contexts.** Approved. Neutral-vs-task-distribution is a real design fork and neutral is the right call here precisely because task-distribution contexts would couple the lens to the bank's data — the same coupling discipline that keeps the lens a base-model measurement. The sha256 in every artifact makes the context set auditable. Keep it.
2. **Last-token, within-position (the deviation I flagged).** Approved, and your conservativeness argument is the reason. Dropping cross-position terms means Level B underestimates propagation rather than fabricating it, so in the arbiter role the deviation can only make Level B a harder test for a Level-A outcome-(c) reading, never a false-confirming one. That is the safe direction for an arbiter. Two conditions on the disclosure, both cheap: (i) the "cross-position terms dropped → conservative underestimate" statement must appear in the Level-B output artifact, not only the docstring, so a reader of the result sees the direction of the bias; (ii) if Level B ever confirms propagation structure (the non-null direction), state explicitly that the confirmation is a lower bound — the true cross-position propagation is at least this strong. The conservative argument protects null readings automatically; it needs to be stated so it also correctly frames positive ones.
3. **One frozen-base lens per family, reused across 240 adapters.** Approved. Per-adapter lenses would couple the readout to the object being measured (and cost 240×); the base-model lens applied post-hoc is the correct measurement. This is also what makes "same lens, different adapters" a fair within-family comparison.
4. **float32.** Approved. Jacobian fidelity matters more than the memory saving here and the 1.5B bases fit; float64-on-save is right.

## The one condition: a lens positive control

This gates the arbiter role; it is the gap I found rather than a default I object to. Level A shipped with a synthetic selftest that plants a signal and confirms recovery — I ran it and watched it detect the planted effect and sit at chance on the null. Level B has no equivalent: no test file, no synthetic path, no positive control in the tool. For a readout whose entire purpose is to arbitrate — to say "Level A's outcome (c) is real: task identity does live in output-null directions that mid-network propagation recovers", an unvalidated lens is not usable, because a null Level-B result would be uninterpretable: you could not distinguish "no propagation structure to find" from "the lens is too weak to see it." That ambiguity is exactly what an arbiter must not have.

**Condition:** before any Level-B signature is reported (Stage C on the bank), add a synthetic positive control in the same class as Level A's selftest — plant a known propagation structure (a module whose effective update is constructed to project onto a specific output direction through a toy frozen map), confirm the lens recovers it above a matched null, and confirm a genuinely output-null planted update reads as null. Same import-shared probes/sketch. This is construction-class work (no bank contact), so it can proceed on Hermes now alongside the lens pilot; it does not delay anything. But its passing is the precondition for treating a Level-B result as an arbiter verdict rather than an exploratory number.

## Net

- Defaults 1–4: **APPROVED for reportable use.** Context set, within-position conservativeness, per-family frozen lens, float32: all sound, and the deviation I flagged is conservative in the arbiter's safe direction.
- Two disclosure conditions on default 2: the conservative-underestimate statement in the output artifact; positive Level-B confirmations framed as lower bounds.
- One hard condition before Stage-C reporting: a synthetic positive control validating the lens can recover planted propagation and null a planted output-null update, matching Level A's selftest discipline. Construction-class, runs now, gates reporting only.
- Lens construction/estimation on Hermes proceeds under the non-reportable class as you have it; Stage C stays bank-completeness-gated. If the positive control lands before ~Jul 19, Level B fires alongside D1 on delivery; if not, it staggers, and nothing breaks.

Send the positive-control result when it runs and I will verify it recovers the planted signal the same way I verified Level A's. Then Level B is a usable arbiter, not just a computed number.

*Stage-B defaults 1–4 approved at `a703cc27` (probe/sketch import-identity + lazy-transformers interlock confirmed in source); within-position deviation accepted as conservative-for-arbiter; one hard condition (synthetic positive control before Stage-C reporting). — the Director*

---

*Recorded verbatim by Meridian from the inbound ruling, 2026-07-09 (delivered via the PI in-session).*
