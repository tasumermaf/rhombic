"""Check status of all Holly multi-seed RunPod pods."""
import runpod
import json
import subprocess
import sys

import os
runpod.api_key = os.environ["RUNPOD_API_KEY"]

PODS = [
    {"id": "ku57em9ko4ac2x", "name": "ctrl-s42", "config": "control", "seed": 42},
    {"id": "aybdjh0chqm2g4", "name": "ctrl-s123", "config": "control", "seed": 123},
    {"id": "7trt449enqipmr", "name": "ctrl-s456", "config": "control", "seed": 456},
    {"id": "nagjuyp22j8tqw", "name": "rhombi-s42", "config": "rhombi", "seed": 42},
    {"id": "upyyxx6ocaeuxt", "name": "rhombi-s123", "config": "rhombi", "seed": 123},
    {"id": "r1ao6zjz02yfug", "name": "rhombi-s456", "config": "rhombi", "seed": 456},
]

def check_ssh(pod_id):
    """Try to get last log line via SSH."""
    host = f"root@{pod_id}-ssh.proxy.runpod.net"
    try:
        result = subprocess.run(
            ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=5",
             host, "tail -1 /workspace/deploy.log 2>/dev/null || echo 'no log'"],
            capture_output=True, text=True, timeout=10
        )
        return result.stdout.strip() if result.returncode == 0 else "SSH failed"
    except Exception:
        return "SSH timeout"


print(f"\n{'Name':<14} {'ID':<18} {'Status':<10} {'Uptime':>8} {'$/hr':>6}  {'Last Log'}")
print("-" * 90)

total_cost = 0
for p in PODS:
    try:
        pod = runpod.get_pod(p['id'])
        status = pod.get('desiredStatus', '?')
        uptime = pod.get('uptimeSeconds', 0)
        cost = pod.get('costPerHr', 0)
        total_cost += float(cost) if cost else 0

        # Try SSH if pod is up
        log_line = ""
        if uptime and uptime > 30:
            log_line = check_ssh(p['id'])

        print(f"{p['name']:<14} {p['id']:<18} {status:<10} {uptime:>6}s  ${cost:>5}  {log_line[:40]}")
    except Exception as e:
        print(f"{p['name']:<14} {p['id']:<18} ERROR: {e}")

print("-" * 90)
print(f"Total: ${total_cost:.2f}/hr")

if "--terminate" in sys.argv:
    print("\nTerminating all pods...")
    for p in PODS:
        try:
            runpod.terminate_pod(p['id'])
            print(f"  Terminated {p['name']}")
        except Exception as e:
            print(f"  Failed {p['name']}: {e}")
