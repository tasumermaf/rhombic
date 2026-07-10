#!/usr/bin/env bash
# XR-001 externalization pilot -> E-5 bifurcation sweep chain (Hermes).
# 2026-07-10. Launch detached from the repo root:
#   nohup bash scripts/hermes_xr001_e5_chain.sh > results/chain_master.log 2>&1 &
# Kill-safe: both stages are manifest-resumable; rerunning this script skips
# completed work. The 12h timeout on the pilot is a hang guard only — the
# pilot projects to ~3h; E-5 does not depend on the pilot's outcome.
set -u
cd "$(dirname "$0")/.." || exit 1
PY="${PY:-/home/timm156/miniforge3/envs/captioning/bin/python}"
export PYTHONUNBUFFERED=1
# THIS tree's rhombic package must win over any stale editable install
# elsewhere in the env (Hermes carries a March `pip install -e ~/rhombic`).
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

XR=results/XR-001-externalization-pilot
E5=results/E-5-bifurcation
mkdir -p "$XR" "$E5"

echo "[chain] XR-001 run start $(date -u +%FT%TZ)"
timeout 12h "$PY" scripts/xr001_externalization_pilot.py run \
  --tasks "$XR/tasks.json" \
  --models gemma3:4b,qwen3:14b,qwen3-coder:30b \
  --regimes R0,R1,R2 \
  --outdir "$XR" >> "$XR/run.log" 2>&1
xr_rc=$?
echo "[chain] XR-001 run exit=$xr_rc $(date -u +%FT%TZ)"

# --allow-partial only PERMITS an interim look: on a fully COMPLETE bank the
# output carries no partial banner and is the confirmatory analysis.
"$PY" scripts/xr001_externalization_pilot.py score \
  --tasks "$XR/tasks.json" --outdir "$XR" >> "$XR/run.log" 2>&1 &&
"$PY" scripts/xr001_externalization_pilot.py analyze \
  --results "$XR/results.json" --outdir "$XR" --allow-partial \
  >> "$XR/run.log" 2>&1
echo "[chain] XR-001 score/analyze exit=$? $(date -u +%FT%TZ)"

# Drain Ollama from VRAM before training claims the GPU.
for m in gemma3:4b qwen3:14b qwen3-coder:30b; do
  curl -s http://localhost:11434/api/chat \
    -d "{\"model\":\"$m\",\"keep_alive\":0,\"messages\":[]}" \
    >/dev/null 2>&1 || true
done
used=99999
for _ in $(seq 1 40); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  [ "${used:-99999}" -lt 1500 ] && break
  sleep 30
done
if [ "${used:-99999}" -ge 1500 ]; then
  # Launching 15 doomed runs against a wedged GPU wastes days: abort loudly.
  echo "[chain] ABORT: GPU never drained (used=${used}MiB after 20min) — E-5 NOT started $(date -u +%FT%TZ)"
  exit 1
fi
echo "[chain] GPU drained (used=${used}MiB) $(date -u +%FT%TZ)"

echo "[chain] E-5 sweep start $(date -u +%FT%TZ)"
"$PY" scripts/e5_bifurcation_sweep.py run \
  --outdir "$E5" --python "$PY" --steps 3000 >> "$E5/chain.log" 2>&1
e5_rc=$?
echo "[chain] E-5 run exit=$e5_rc $(date -u +%FT%TZ)"
"$PY" scripts/e5_bifurcation_sweep.py endpoints --outdir "$E5" \
  >> "$E5/chain.log" 2>&1
"$PY" scripts/e5_bifurcation_sweep.py summarize --outdir "$E5" \
  >> "$E5/chain.log" 2>&1
echo "[chain] chain complete $(date -u +%FT%TZ)"
