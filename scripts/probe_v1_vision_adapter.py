"""PROBE-V1 — do Macro Meridian's image records survive the adapter path?

    STATUS: PREPARED, NOT RUN. NO TINKER TOKENS HAVE BEEN SPENT ON THIS.
    The PI ruled 2026-08-04 that no fine-tuning and no Tinker spend happens
    while the dataset methodology is being built. This script is therefore an
    instrument waiting for authorization, not a result. Its ``--dry-run`` path
    makes zero API calls and produced the visual-token table below; nothing
    else in it has been executed.

The Director's condition on base-model commitment: the method report's §7
concedes vision through the adapter path is untested, and a natively-multimodal
MoE is the right architecture *on paper* only. This answers the mechanical
question cheaply, before any base is chosen.

RUNS UNDER THE ``tinker`` CONDA ENV (py3.12):
    C:\\miniconda3\\envs\\tinker\\python.exe

IP BOUNDARY — READ THIS BEFORE CHANGING THE DATA SOURCE
-------------------------------------------------------
Macro Meridian's corpus is one person's food photographs and body data. It is
gitignored and local, and F085 in the Asset-1 ladder is the deferred
IP-boundary harness gap in a *less* sensitive domain.

So this probe sends **no real record**. It reconstructs the record SHAPE —
the same message layout, the same system/user/assistant roles, the same
answer schema, and image dimensions taken from the real archive's metadata
(width/height, not content) — with synthetic imagery. That answers "does a
record of this shape serialize, train, and export" exactly, which is the
question asked.

What it therefore does NOT establish: that a real photograph's content trains
usefully. That needs a real-data run and an explicit authorization from the PI
that has not been given and is not assumed here.

WHAT IS MEASURED
----------------
1. The visual-token rule, re-verified against the server rather than recalled
   (deliberately send a wrong ``expected_tokens`` and read the true count back).
2. Whether a mixed text+image ``ModelInput`` is accepted for training at all.
3. Whether the loss moves on a handful of optimizer steps.
4. Whether the adapter exports afterwards.
5. What a real training resolution costs in visual tokens — the archive holds
   4000x3000 camera originals, which is 11,750 tokens per image at the 32px
   rule, so resolution is a budget decision rather than a property of the file.

BUDGET
------
Hard abort at $2.00 projected. The account cap is ~$30 shared with every other
Tinker use; the Asset-1 signal pilot spent $2.687 total for six rank-32 runs on
an 8B, so this is deliberately smaller than that.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import tinker
from tinker import types
from PIL import Image, ImageDraw

# ---------------------------------------------------------------- configuration

# Smallest multimodal model on the roster (measured 2026-08-03: the roster
# declares no modality, so this list is empirical, not documented).
BASE_MODEL = "Qwen/Qwen3.6-27B"
RANK = 16                     # behavioural update, not style transfer
LEARNING_RATE = 1e-4
HARD_ABORT_USD = 2.00
# Qwen3-8B metered at $0.44/M train tokens (verified 2026-07-30). A 27B costs
# more per token; this rate is used ONLY for the guard, deliberately optimistic
# so the guard trips early rather than late. The billing API is the truth.
GUARD_RATE_USD_PER_MTOK = 2.00

OUT = Path(__file__).resolve().parents[1] / "results" / "probe-v1-vision"

# Real archive geometry, read from the JPEG headers of Macro Meridian's image
# store on 2026-08-04. Dimensions only — no pixel data leaves that machine.
ARCHIVE_GEOMETRY = [
    {"w": 4000, "h": 3000, "n": 7, "note": "camera originals, archived for training"},
    {"w": 520, "h": 760, "n": 1, "note": "screenshot/label capture"},
]

# What we would actually train at. The originals are archived precisely so this
# stays a choice; see the token table this script prints.
TRAIN_W, TRAIN_H = 1024, 768


def visual_tokens(w: int, h: int) -> int:
    """One token per 32px block, ROUNDED TO NEAREST (measured, not floored)."""
    return round(w / 32) * round(h / 32)


# ------------------------------------------------------------------ synthetic data

def synth_image(w: int, h: int, seed: int) -> bytes:
    """A deterministic non-photographic image at the given geometry.

    Content is irrelevant to the question — serialization and token accounting
    depend on dimensions and format, not on what is depicted. Deliberately not
    food, so nothing here resembles the real corpus even by accident.
    """
    img = Image.new("RGB", (w, h), (18 + seed * 7 % 60, 22, 30))
    d = ImageDraw.Draw(img)
    step = max(16, w // 12)
    for i in range(0, w, step):
        shade = (40 + (i + seed * 13) % 180, 60 + (i * 3 + seed * 5) % 150, 90)
        d.rectangle([i, (i + seed) % max(1, h // 2), i + step // 2, h], fill=shade)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return buf.getvalue()


# The real record layout, reproduced. Macro Meridian writes system + user +
# assistant per interaction, the assistant turn being raw JSON in the analyze
# schema. Lengths here are representative of the real prompts (~1.5k tokens of
# system rules), so the text:image token ratio is realistic.
SYSTEM = (
    "You are Meridian. Report what you read or infer; never decide what the "
    "numbers mean. Grams for macros, integer kcal. totalCarbs is TOTAL "
    "carbohydrate with nothing subtracted. Micronutrients come from food "
    "composition science first and the label last. Omit what you do not know: "
    "absent is not zero. Set measured TRUE only when the AMOUNT came from an "
    "actual measurement rather than from judgement of the picture. "
) * 6

USER = (
    "Identify this and return the schema. No note given — estimate from the "
    "image alone. Remaining: 820 kcal, 12.0 g carbs, 71 g protein to goal."
)

ASSISTANT = json.dumps({
    "items": [{
        "name": "Synthetic probe item", "serving": "1 unit (100 g)", "kcal": 210,
        "totalCarbs": 3.1, "fiber": 1.2, "otherCarbs": 0, "fat": 17.4,
        "protein": 11.0, "netEligible": False, "measured": False,
        "micros": {"sodiumMg": 380, "potassiumMg": 240, "magnesiumMg": 28},
        "confidence": "medium",
        "assumptions": "Probe record. Sized from block geometry, not from a scale.",
    }],
    "note": "",
})


@dataclass
class Result:
    started: str
    base_model: str = BASE_MODEL
    rank: int = RANK
    token_rule_checks: list = field(default_factory=list)
    accepted_mixed_input: bool | None = None
    losses: list = field(default_factory=list)
    adapter_exported: bool | None = None
    train_tokens: int = 0
    projected_usd: float = 0.0
    aborted: str | None = None
    notes: list = field(default_factory=list)


def build_datum(tc, img_bytes: bytes, n_vis: int) -> tuple[types.Datum, int]:
    """One supervised record: [system+user text][image][assistant answer].

    ``to_ints()`` refuses a mixed text+image ModelInput, so the supervised span
    is computed from the KNOWN LAYOUT rather than recovered from the assembled
    input. ``target_tokens`` aligns 1:1 with input positions and has length n,
    not n-1 — the server does the shift. Both facts were measured 2026-08-03
    and both cost time when assumed.
    """
    tok = tc.get_tokenizer()
    prompt_ids = tok.encode(f"<|system|>{SYSTEM}\n<|user|>{USER}\n", add_special_tokens=False)
    answer_ids = tok.encode(f"<|assistant|>{ASSISTANT}", add_special_tokens=False)

    mi = types.ModelInput(chunks=[
        types.EncodedTextChunk(tokens=prompt_ids),
        types.ImageChunk(
            data=base64.b64encode(img_bytes).decode(),
            format="jpeg",
            expected_tokens=n_vis,
        ),
        types.EncodedTextChunk(tokens=answer_ids),
    ])

    n_total = len(prompt_ids) + n_vis + len(answer_ids)
    # Supervise the answer only (completion-only masking): everything before the
    # answer span is context, including every visual position.
    n_ctx = len(prompt_ids) + n_vis
    targets = [0] * n_ctx + answer_ids
    weights = [0.0] * n_ctx + [1.0] * len(answer_ids)
    # Mean-normalise so the reported loss is per-token and the gradient scale is
    # independent of batch size; the backend cross-entropy is an unnormalised sum.
    n_loss = len(answer_ids)
    weights = [w / n_loss for w in weights]

    datum = types.Datum(
        model_input=mi,
        loss_fn_inputs={"target_tokens": targets, "weights": weights},
    )
    return datum, n_total


def check_token_rule(tc, res: Result) -> None:
    """The server is the oracle. Send a wrong count and read the true one back."""
    for (w, h) in [(512, 512), (TRAIN_W, TRAIN_H), (520, 760)]:
        predicted = visual_tokens(w, h)
        img = synth_image(w, h, seed=1)
        wrong = predicted + 7                     # deliberately wrong
        entry = {"w": w, "h": h, "predicted_by_rule": predicted}
        try:
            mi = types.ModelInput(chunks=[
                types.ImageChunk(data=base64.b64encode(img).decode(),
                                 format="jpeg", expected_tokens=wrong),
            ])
            tc.forward_backward(
                [types.Datum(model_input=mi,
                             loss_fn_inputs={"target_tokens": [0] * (wrong),
                                             "weights": [0.0] * (wrong)})],
                "cross_entropy",
            ).result()
            entry["server"] = f"ACCEPTED a deliberately wrong count ({wrong}) — rule NOT enforced"
        except Exception as e:  # noqa: BLE001 - the message is the measurement
            entry["server"] = str(e)[:400]
        res.token_rule_checks.append(entry)
        print(f"  {w}x{h}: rule says {predicted}; server said → {entry['server'][:160]}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--records", type=int, default=4)
    ap.add_argument("--skip-token-check", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the token budget table and exit without calling the API.")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    res = Result(started=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    print("PROBE-V1 — vision through the adapter path")
    print(f"  base {BASE_MODEL} · rank {RANK} · hard abort ${HARD_ABORT_USD:.2f}")
    print()
    print("Archive geometry (real dimensions, synthetic content):")
    for g in ARCHIVE_GEOMETRY:
        vt = visual_tokens(g["w"], g["h"])
        print(f"  {g['n']}x {g['w']}x{g['h']:<5} = {vt:>6,} visual tokens   {g['note']}")
    print(f"  training at {TRAIN_W}x{TRAIN_H} = {visual_tokens(TRAIN_W, TRAIN_H):,} tokens "
          f"({visual_tokens(4000,3000)/visual_tokens(TRAIN_W,TRAIN_H):.0f}x cheaper than native)")
    print()
    res.notes.append(
        f"native 4000x3000 = {visual_tokens(4000,3000)} visual tokens; "
        f"train res {TRAIN_W}x{TRAIN_H} = {visual_tokens(TRAIN_W,TRAIN_H)}; "
        f"ratio {visual_tokens(4000,3000)/visual_tokens(TRAIN_W,TRAIN_H):.1f}x"
    )

    if args.dry_run:
        (OUT / "probe_v1_dryrun.json").write_text(json.dumps(asdict(res), indent=2))
        print("dry run — no API calls made")
        return 0

    sc = tinker.ServiceClient()
    tc = sc.create_lora_training_client(base_model=BASE_MODEL, rank=RANK)

    if not args.skip_token_check:
        print("Token-rule check (server is the oracle):")
        check_token_rule(tc, res)
        print()

    n_vis = visual_tokens(TRAIN_W, TRAIN_H)
    batch, total_tokens = [], 0
    for i in range(args.records):
        d, n = build_datum(tc, synth_image(TRAIN_W, TRAIN_H, seed=i), n_vis)
        batch.append(d)
        total_tokens += n
    print(f"Built {len(batch)} records, {total_tokens:,} tokens/step "
          f"({n_vis:,} of them visual per record)")

    print("Training:")
    for step in range(args.steps):
        projected = (res.train_tokens + total_tokens) / 1e6 * GUARD_RATE_USD_PER_MTOK
        if projected > HARD_ABORT_USD:
            res.aborted = f"budget guard at step {step}: projected ${projected:.2f}"
            print(f"  ABORT — {res.aborted}")
            break
        try:
            fwd = tc.forward_backward(batch, "cross_entropy").result()
            tc.optim_step(types.AdamParams(learning_rate=LEARNING_RATE)).result()
        except Exception as e:  # noqa: BLE001
            res.accepted_mixed_input = False
            res.aborted = f"step {step}: {type(e).__name__}: {str(e)[:500]}"
            print(f"  FAILED — {res.aborted}")
            break
        res.accepted_mixed_input = True
        res.train_tokens += total_tokens
        loss = None
        try:
            lo = fwd.loss_fn_outputs
            loss = float(sum(float(x["loss:sum"]) for x in lo)) if lo else None
        except Exception:
            try:
                loss = float(fwd.metrics.get("loss:sum"))
            except Exception:
                pass
        res.losses.append(loss)
        print(f"  step {step}: loss {loss}")

    res.projected_usd = res.train_tokens / 1e6 * GUARD_RATE_USD_PER_MTOK

    if res.accepted_mixed_input:
        try:
            saved = tc.save_weights_for_sampler(name="probe-v1-final").result()
            res.adapter_exported = True
            res.notes.append(f"adapter path: {getattr(saved, 'path', str(saved))[:200]}")
            print(f"  adapter exported: {getattr(saved, 'path', saved)}")
        except Exception as e:  # noqa: BLE001
            res.adapter_exported = False
            res.notes.append(f"export failed: {type(e).__name__}: {str(e)[:300]}")
            print(f"  export FAILED: {e}")

    (OUT / "probe_v1_result.json").write_text(json.dumps(asdict(res), indent=2))
    print()
    print(f"train tokens {res.train_tokens:,} · guard-estimate ${res.projected_usd:.4f}")
    print(f"written: {OUT / 'probe_v1_result.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
