# -*- coding: utf-8 -*-
"""gpu_guard — cooperative GPU arbitration for the shared RTX 6000 Ada.

Three instruments, per MERIDIAN_PROGRAM_PLAN_2026-08-04 §2:

  preflight(needed_gb)   block until free VRAM >= needed_gb + MARGIN_GB.
  pause_gate()           block while C:\\gpu-claims\\PAUSE_RESEARCH exists.
  claim(...)/release()   advertise an active research run in
                         C:\\gpu-claims\\RESEARCH_ACTIVE.json so other
                         instances (Security & Access / ComfyUI) can see it.

Design rule: a refused start is a log line; an OOM is a failure. Runners call
    with gpu_guard.guarded(purpose=..., needed_gb=..., expected_min=...):
        <launch training>
between EVERY run, so the worst-case wait for another user is one run.

The claims directory lives OUTSIDE all repos and is world-writable by every
local instance. Protocol doc: C:\\gpu-claims\\PROTOCOL.md.
"""
import contextlib
import json
import os
import subprocess
import time
from pathlib import Path

CLAIMS = Path(r"C:\gpu-claims")
ACTIVE = CLAIMS / "RESEARCH_ACTIVE.json"
PAUSE = CLAIMS / "PAUSE_RESEARCH"
MARGIN_GB = 6.0
POLL_S = 60


def _log(msg):
    print(f"[gpu_guard] {time.strftime('%H:%M:%S')} {msg}", flush=True)


def free_vram_gb():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True).stdout
    used, total = (float(x) for x in out.strip().split("\n")[0].split(","))
    return (total - used) / 1024.0


def pause_gate():
    """Block while the PAUSE_RESEARCH sentinel exists."""
    while PAUSE.exists():
        _log(f"PAUSE_RESEARCH present ({PAUSE}); waiting {POLL_S}s")
        time.sleep(POLL_S)


def preflight(needed_gb):
    """Block until free VRAM >= needed_gb + MARGIN_GB. Never OOM-launch."""
    while True:
        free = free_vram_gb()
        if free >= needed_gb + MARGIN_GB:
            _log(f"preflight OK: free {free:.1f} GB >= "
                 f"{needed_gb:.1f}+{MARGIN_GB:.0f} GB")
            return
        _log(f"preflight WAIT: free {free:.1f} GB < "
             f"{needed_gb:.1f}+{MARGIN_GB:.0f} GB; polling {POLL_S}s")
        time.sleep(POLL_S)


def claim(purpose, needed_gb, expected_min):
    CLAIMS.mkdir(exist_ok=True)
    ACTIVE.write_text(json.dumps({
        "owner": "meridian-research",
        "purpose": purpose,
        "needed_gb": needed_gb,
        "expected_min": expected_min,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pid": os.getpid(),
    }, indent=1), encoding="utf-8")
    _log(f"claimed: {purpose} (~{expected_min} min, {needed_gb} GB)")


def release():
    with contextlib.suppress(FileNotFoundError):
        ACTIVE.unlink()
    _log("released")


@contextlib.contextmanager
def guarded(purpose, needed_gb, expected_min):
    """The one entry point runners should use, between every run."""
    pause_gate()
    preflight(needed_gb)
    claim(purpose, needed_gb, expected_min)
    try:
        yield
    finally:
        release()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="gpu_guard status / test")
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args()
    print(f"free_vram_gb = {free_vram_gb():.1f}")
    print(f"pause        = {PAUSE.exists()}")
    print(f"active       = {ACTIVE.read_text() if ACTIVE.exists() else 'none'}")
