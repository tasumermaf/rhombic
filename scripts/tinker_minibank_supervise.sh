#!/usr/bin/env bash
# E-T4 mini-bank shard supervisor.
#
# Watches the three training shards. A shard whose log has not advanced for
# STALE_SECS is considered stalled and is RESTARTED (its ledger persists, and
# --skip-existing means completed runs are never redone, so a restart is
# lossless apart from the in-progress run).
#
# Per the Director's standing instruction (2026-08-04): more than MAX_FIRINGS
# restarts on the SAME shard is the §6.3 stop-rule escalation — the supervisor
# stops intervening on that shard, writes ESCALATE.marker, and exits. It does
# not loop on a failing step.
#
# Note the distinction this script maintains deliberately: a stall that
# RESOLVES ON ITS OWN is not a firing. Both stall episodes observed before this
# supervisor existed self-recovered and required no restart.

set -u
BANK=/c/falco/rhombic/results/tinker-minibank
LOGDIR=/c/tmp
STALE_SECS=${STALE_SECS:-480}
MAX_FIRINGS=${MAX_FIRINGS:-2}
PY=/c/miniconda3/envs/tinker/python.exe
MARKER="$BANK/ESCALATE.marker"
COUNTS="$BANK/logs/shard_firings.txt"

declare -A PRIOR=( [0]=19.891300 [1]=19.891300 [2]=19.439700 )
declare -A ENVEL=( [0]=7.213600  [1]=7.213600  [2]=7.665200 )
declare -A FIRED=( [0]=0 [1]=0 [2]=0 )

mkdir -p "$BANK/logs"

while true; do
  n=$(ls "$BANK"/*_d*_i*/run_record.json 2>/dev/null | wc -l)
  [ "$n" -ge 54 ] && { echo "SUPERVISOR: all 54 runs trained"; break; }

  alive_any=0
  for k in 0 1 2; do
    log="$LOGDIR/bank_shard$k.log"
    [ -f "$log" ] || continue
    pid=$(powershell.exe -NoProfile -Command "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*--shard $k --n-shards 3*' }).ProcessId" 2>/dev/null | tr -d '\r' | head -1)
    [ -n "$pid" ] && alive_any=1

    # A finished shard exits cleanly; that is not a stall.
    if [ -z "$pid" ]; then
      if grep -q "budget\] FINAL" "$log" 2>/dev/null; then continue; fi
    fi

    age=$(( $(date +%s) - $(stat -c %Y "$log") ))
    if [ "$age" -gt "$STALE_SECS" ]; then
      if [ "${FIRED[$k]}" -ge "$MAX_FIRINGS" ]; then
        echo "ESCALATE: shard $k stalled again after ${FIRED[$k]} restarts (age ${age}s). Per the standing instruction this is the §6.3 stop rule — not restarting."
        echo "shard $k firings=${FIRED[$k]} age=${age}s at $(date +%H:%M:%S)" > "$MARKER"
        exit 2
      fi
      FIRED[$k]=$(( FIRED[$k] + 1 ))
      echo "WATCHDOG FIRING #${FIRED[$k]} on shard $k (log stale ${age}s) at $(date +%H:%M:%S)"
      [ -n "$pid" ] && powershell.exe -NoProfile -Command "Stop-Process -Id $pid -Force" >/dev/null 2>&1
      sleep 5
      ( cd /c/falco/rhombic && nohup "$PY" scripts/tinker_minibank_train.py \
          --shard "$k" --n-shards 3 --skip-existing --defer-export \
          --ledger "results/tinker-minibank/spend_ledger_shard$k.json" \
          --prior-usd "${PRIOR[$k]}" --shard-budget-usd "${ENVEL[$k]}" \
          >> "$log" 2>&1 & )
      echo "  shard $k relaunched"
    fi
    echo "shard $k firings=${FIRED[$k]}" >> "$COUNTS"
  done

  if [ "$alive_any" = "0" ]; then
    echo "SUPERVISOR: no shard processes alive; $n/54 trained"
    break
  fi
  sleep 60
done
