"""BM-001 progress report — runs on the RunPod pod."""
import json
import sys
from pathlib import Path

base = Path("/workspace/rhombic/results")

# Config A
a = json.load(open(base / "BM-001-standard-lora/results.json"))
ac = a["checkpoints"][-1]
print("=" * 60)
print("Config A: Standard LoRA (COMPLETE)")
print("=" * 60)
print(f"  Final train loss: {ac['train_loss']:.4f}")
print(f"  Final val loss:   {ac['val_loss']:.4f}")
print(f"  Wall time:        {ac['wall_time']/3600:.2f}h")
print()

# Loss curve A (sampled)
print("  Loss curve (every 1000 steps):")
for cp in a["checkpoints"]:
    if cp["step"] % 1000 == 0 or cp["step"] == a["checkpoints"][-1]["step"]:
        print(f"    step {cp['step']:>5}: train={cp['train_loss']:.4f}  val={cp['val_loss']:.4f}")
print()

# Config B
try:
    b = json.load(open(base / "BM-001-telora/results.json"))
    bc = b["checkpoints"][-1]
    pct = bc["step"] / 10000 * 100
    print("=" * 60)
    print(f"Config B: TeLoRA (step {bc['step']}/10000 = {pct:.0f}%)")
    print("=" * 60)
    print(f"  Current train loss: {bc['train_loss']:.4f}")
    print(f"  Current val loss:   {bc['val_loss']:.4f}")
    print(f"  Wall time:          {bc['wall_time']/3600:.2f}h")
    co = bc.get("co_cross_ratio")
    print(f"  Fiedler mean:       {bc['fiedler_mean']:.5f}")
    print(f"  Co/cross ratio:     {co:.1f}" if co else "  Co/cross ratio:     N/A")
    print(f"  Deviation mean:     {bc['deviation_mean']:.5f}")
    print()

    # Head-to-head at matched steps
    a_by_step = {cp["step"]: cp for cp in a["checkpoints"]}
    b_by_step = {cp["step"]: cp for cp in b["checkpoints"]}
    common = sorted(set(a_by_step) & set(b_by_step))

    if common:
        print("=" * 60)
        print("HEAD-TO-HEAD COMPARISON")
        print("=" * 60)
        print(f"  {'Step':>6}  {'Std Train':>10}  {'TeL Train':>10}  {'Delta':>8}  {'Std Val':>10}  {'TeL Val':>10}  {'Delta':>8}")
        print("  " + "-" * 72)
        for s in common:
            at = a_by_step[s]["train_loss"]
            bt = b_by_step[s]["train_loss"]
            av = a_by_step[s]["val_loss"]
            bv = b_by_step[s]["val_loss"]
            dt = bt - at
            dv = bv - av
            print(f"  {s:>6}  {at:>10.4f}  {bt:>10.4f}  {dt:>+8.4f}  {av:>10.4f}  {bv:>10.4f}  {dv:>+8.4f}")

        # Summary at latest common step
        last = common[-1]
        at = a_by_step[last]["train_loss"]
        bt = b_by_step[last]["train_loss"]
        av = a_by_step[last]["val_loss"]
        bv = b_by_step[last]["val_loss"]
        print()
        print(f"  At step {last}:")
        if bt < at:
            print(f"    TeLoRA train loss BETTER by {at-bt:.4f}")
        elif bt > at:
            print(f"    TeLoRA train loss WORSE by {bt-at:.4f}")
        else:
            print(f"    Train loss identical")
        if bv < av:
            print(f"    TeLoRA val loss BETTER by {av-bv:.4f}")
        elif bv > av:
            print(f"    TeLoRA val loss WORSE by {bv-av:.4f}")
        else:
            print(f"    Val loss identical")
    print()

    # Bridge diagnostics
    print("  Bridge diagnostics (sampled):")
    for cp in b["checkpoints"]:
        if cp["step"] % 500 == 0 or cp["step"] == bc["step"]:
            co = cp.get("co_cross_ratio")
            co_s = f"{co:>8.1f}" if co else "     N/A"
            print(f"    step {cp['step']:>5}: fiedler={cp['fiedler_mean']:.5f}  co/cross={co_s}  dev={cp['deviation_mean']:.5f}")

except FileNotFoundError:
    print("Config B: Not started yet")
except Exception as e:
    print(f"Config B: Error reading results — {e}")

# Benchmarks
print()
for name in ["base.json", "standard-lora.json", "telora.json"]:
    p = base / "BM-001" / name
    if p.exists():
        print(f"Benchmark {name}: EXISTS")
    else:
        print(f"Benchmark {name}: pending")
