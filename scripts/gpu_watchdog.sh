#!/bin/bash
# GPU Watchdog — runs queued experiments automatically
#
# USAGE:
#   bash scripts/gpu_watchdog.sh           # Run next queued experiment for this GPU
#   bash scripts/gpu_watchdog.sh --loop    # Keep watching, run experiments as GPU frees up
#   bash scripts/gpu_watchdog.sh --status  # Show queue status
#
# The watchdog:
#   1. Reads experiment_queue.json
#   2. Finds the highest-priority 'queued' experiment for this GPU
#   3. Marks it 'running', launches it
#   4. On completion, marks it 'done' with results summary
#   5. In --loop mode, picks the next experiment and repeats

set -e

SCRIPT_DIR="$(pwd)/scripts"
QUEUE="scripts/experiment_queue.json"
LOG="results/watchdog.log"
GPU_NAME="local"  # Change to 'hermes' on the Hermes machine

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"
}

# Check dependencies
if ! command -v python &>/dev/null; then
    echo "Python not found"
    exit 1
fi
if ! command -v jq &>/dev/null; then
    # Fall back to Python for JSON manipulation
    JQ_CMD="python"
else
    JQ_CMD="jq"
fi

show_status() {
    python -c "
import json
with open('$QUEUE') as f:
    q = json.load(f)
print(f\"{'ID':<20} {'Status':<10} {'Priority':<8} {'GPU':<8} {'Description'}\")
print('-' * 90)
for exp in sorted(q['experiments'], key=lambda x: x['priority']):
    print(f\"{exp['id']:<20} {exp['status']:<10} {exp['priority']:<8} {exp['gpu']:<8} {exp['description'][:40]}\")
"
}

get_next_experiment() {
    python -c "
import json
with open('$QUEUE') as f:
    q = json.load(f)
queued = [e for e in q['experiments'] if e['status'] == 'queued' and e['gpu'] == '$GPU_NAME']
if queued:
    best = min(queued, key=lambda x: x['priority'])
    print(best['id'])
"
}

get_command() {
    local exp_id="$1"
    python -c "
import json
with open('$QUEUE') as f:
    q = json.load(f)
for e in q['experiments']:
    if e['id'] == '$exp_id':
        print(e['command'])
        break
"
}

mark_status() {
    local exp_id="$1"
    local new_status="$2"
    python -c "
import json
with open('$QUEUE') as f:
    q = json.load(f)
for e in q['experiments']:
    if e['id'] == '$exp_id':
        e['status'] = '$new_status'
        break
with open('$QUEUE', 'w') as f:
    json.dump(q, f, indent=2)
"
}

mark_done() {
    local exp_id="$1"
    local exit_code="$2"
    python -c "
import json, datetime
with open('$QUEUE') as f:
    q = json.load(f)
for e in q['experiments']:
    if e['id'] == '$exp_id':
        e['status'] = 'done' if $exit_code == 0 else 'failed'
        e['completed_at'] = datetime.datetime.now().isoformat()
        e['exit_code'] = $exit_code
        break
with open('$QUEUE', 'w') as f:
    json.dump(q, f, indent=2)
"
}

run_next() {
    local exp_id
    exp_id=$(get_next_experiment)

    if [ -z "$exp_id" ]; then
        log "No queued experiments for GPU=$GPU_NAME"
        return 1
    fi

    local cmd
    cmd=$(get_command "$exp_id")

    log "=== Starting experiment: $exp_id ==="
    log "Command: $cmd"
    mark_status "$exp_id" "running"

    # Run the experiment (write to log file, not pipe, to avoid SIGPIPE on terminal death)
    local exit_code=0
    local exp_log="results/${exp_id}.log"
    eval "$cmd" > "$exp_log" 2>&1 || exit_code=$?
    cat "$exp_log" >> "$LOG"

    mark_done "$exp_id" "$exit_code"

    if [ $exit_code -eq 0 ]; then
        log "=== Experiment $exp_id COMPLETED SUCCESSFULLY ==="
    else
        log "=== Experiment $exp_id FAILED (exit code $exit_code) ==="
    fi

    return 0
}

# Main
case "${1:-}" in
    --status)
        show_status
        ;;
    --loop)
        log "=== Watchdog started in loop mode (GPU=$GPU_NAME) ==="
        while true; do
            if run_next; then
                log "Checking for next experiment..."
                sleep 5
            else
                log "Queue empty. Sleeping 5 minutes..."
                sleep 300
            fi
        done
        ;;
    *)
        # Run single next experiment
        if ! run_next; then
            echo "No experiments queued for GPU=$GPU_NAME"
            show_status
            exit 0
        fi
        ;;
esac
