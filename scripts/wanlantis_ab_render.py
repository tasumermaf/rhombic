"""
Wanlantis A/B Render — Automated ComfyUI comparison
Generates identical video clips with Standard LoRA and TeLoRA,
using the ComfyUI API.

Requires: ComfyUI running at http://127.0.0.1:8188
Model: Wan2_1-T2V-14B_fp8_e4m3fn_scaled_KJ.safetensors
LoRAs: wanlantis_standard_ab.safetensors, wanlantis_telora_ab.safetensors

Usage:
  python scripts/wanlantis_ab_render.py              # Run both renders
  python scripts/wanlantis_ab_render.py --start-comfy # Start ComfyUI first
"""

import json
import time
import sys
import os
import urllib.request
import urllib.error
import subprocess
import argparse
from pathlib import Path

COMFY_URL = "http://127.0.0.1:8188"
COMFY_ROOT = Path("C:/ComfyUI")
OUTPUT_DIR = Path("C:/falco/rhombic/results/wanlantis-ab")

# Two prompts that test different aspects of the LoRA's learned style
PROMPTS = [
    "atlantis style ancient stone steps ascending through terraced levels with ornate carved statues among lush vegetation",
    "atlantis style golden sunlight streaming through crystal domes reflecting off still water pools in a subterranean temple",
    "atlantis style massive carved stone archway with intricate spiral patterns overlooking a misty underwater canyon",
]

LORAS = {
    "standard": "wanlantis_standard_ab.safetensors",
    "telora": "wanlantis_telora_ab.safetensors",
}

MODEL_PATH = "WanVideo\\fp8_scaled_kj\\T2V\\Wan2_1-T2V-14B_fp8_e4m3fn_scaled_KJ.safetensors"
T5_MODEL = "umt5-xxl-enc-bf16.safetensors"
VAE_MODEL = "Wan2_1_VAE_bf16.safetensors"

SEED = 42
WIDTH = 832
HEIGHT = 480
NUM_FRAMES = 81  # ~5.4 sec at 15fps
FPS = 15
STEPS = 30
GUIDANCE = 5.0
LORA_STRENGTH = 1.0


def build_workflow(prompt: str, lora_path: str, seed: int, output_prefix: str) -> dict:
    """Build a ComfyUI API workflow for T2V + LoRA generation.

    Node signatures verified against WanVideoWrapper source (Mar 25, 2026):
    - WanVideoTextEncode: positive_prompt, negative_prompt, t5, force_offload
    - WanVideoSampler: model, text_embeds, image_embeds, steps, cfg, shift, seed,
                        force_offload, scheduler, riflex_freq_index
    - WanVideoDecode: vae, samples, enable_vae_tiling, tile_x, tile_y, ...
    - WanVideoVAELoader: model_name, precision
    """
    workflow = {
        # T5 text encoder
        "1": {
            "class_type": "LoadWanVideoT5TextEncoder",
            "inputs": {
                "model_name": T5_MODEL,
                "precision": "bf16",
                "load_device": "offload_device",
                "quantization": "disabled",
            }
        },
        # Encode prompt (positive + empty negative)
        "2": {
            "class_type": "WanVideoTextEncode",
            "inputs": {
                "positive_prompt": prompt,
                "negative_prompt": "",
                "t5": ["1", 0],
                "force_offload": True,
            }
        },
        # Empty embeds (T2V mode — no image input)
        "3": {
            "class_type": "WanVideoEmptyEmbeds",
            "inputs": {
                "width": WIDTH,
                "height": HEIGHT,
                "num_frames": NUM_FRAMES,
            }
        },
        # Select LoRA (merge_loras=False required for fp8_scaled models)
        "5": {
            "class_type": "WanVideoLoraSelectMulti",
            "inputs": {
                "lora_0": lora_path,
                "strength_0": LORA_STRENGTH,
                "lora_1": "none",
                "strength_1": 1,
                "lora_2": "none",
                "strength_2": 1,
                "lora_3": "none",
                "strength_3": 1,
                "lora_4": "none",
                "strength_4": 1,
                "merge_loras": False,
            }
        },
        # Load base model + LoRA (passed via optional lora input, not SetLoRAs)
        "4": {
            "class_type": "WanVideoModelLoader",
            "inputs": {
                "model": MODEL_PATH,
                "base_precision": "fp16",
                "quantization": "fp8_e4m3fn_scaled",
                "load_device": "offload_device",
                "attention_mode": "sdpa",
                "lora": ["5", 0],
            }
        },
        # Load VAE
        "10": {
            "class_type": "WanVideoVAELoader",
            "inputs": {
                "model_name": VAE_MODEL,
                "precision": "bf16",
            }
        },
        # Sampler
        "7": {
            "class_type": "WanVideoSampler",
            "inputs": {
                "model": ["4", 0],
                "text_embeds": ["2", 0],
                "image_embeds": ["3", 0],
                "steps": STEPS,
                "cfg": GUIDANCE,
                "shift": 8.0,
                "seed": seed,
                "force_offload": True,
                "scheduler": "unipc",
                "riflex_freq_index": 0,
            }
        },
        # Decode
        "8": {
            "class_type": "WanVideoDecode",
            "inputs": {
                "vae": ["10", 0],
                "samples": ["7", 0],
                "enable_vae_tiling": True,
                "tile_x": 272,
                "tile_y": 272,
                "tile_stride_x": 144,
                "tile_stride_y": 128,
            }
        },
        # Save video
        "9": {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "images": ["8", 0],
                "frame_rate": FPS,
                "loop_count": 0,
                "filename_prefix": output_prefix,
                "format": "video/h264-mp4",
                "pingpong": False,
                "save_output": True,
            }
        },
    }
    return workflow


def comfy_running() -> bool:
    """Check if ComfyUI is responding."""
    try:
        req = urllib.request.Request(f"{COMFY_URL}/system_stats")
        resp = urllib.request.urlopen(req, timeout=5)
        return resp.status == 200
    except (urllib.error.URLError, TimeoutError, ConnectionError):
        return False


def queue_prompt(workflow: dict) -> str:
    """Queue a workflow via the ComfyUI API. Returns prompt_id."""
    payload = json.dumps({"prompt": workflow}).encode()
    req = urllib.request.Request(
        f"{COMFY_URL}/prompt",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        resp = urllib.request.urlopen(req)
        data = json.loads(resp.read())
        return data.get("prompt_id", "unknown")
    except urllib.error.HTTPError as e:
        error_body = e.read().decode()
        print(f"  API Error {e.code}: {error_body[:500]}")
        raise


def wait_for_completion(prompt_id: str, timeout: int = 1800) -> bool:
    """Poll the ComfyUI history until the prompt completes."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            req = urllib.request.Request(f"{COMFY_URL}/history/{prompt_id}")
            resp = urllib.request.urlopen(req, timeout=10)
            data = json.loads(resp.read())
            if prompt_id in data:
                outputs = data[prompt_id].get("outputs", {})
                status = data[prompt_id].get("status", {})
                if status.get("completed", False) or outputs:
                    return True
                if status.get("status_str") == "error":
                    print(f"  ERROR: Prompt {prompt_id} failed")
                    print(f"  {json.dumps(status, indent=2)}")
                    return False
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(10)
        elapsed = int(time.time() - start)
        print(f"  Waiting... {elapsed}s elapsed", end="\r")
    print(f"\n  TIMEOUT after {timeout}s")
    return False


def start_comfyui():
    """Start ComfyUI in a subprocess."""
    print("Starting ComfyUI...")
    proc = subprocess.Popen(
        [sys.executable, str(COMFY_ROOT / "main.py"), "--listen", "127.0.0.1"],
        cwd=str(COMFY_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    # Wait for it to be ready
    for _ in range(120):
        if comfy_running():
            print("ComfyUI is ready!")
            return proc
        time.sleep(2)
    print("ERROR: ComfyUI did not start within 240 seconds")
    proc.kill()
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Wanlantis A/B Render")
    parser.add_argument("--start-comfy", action="store_true", help="Start ComfyUI first")
    parser.add_argument("--prompts", type=int, default=2, help="Number of prompts to render (1-3)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Check model exists (primary on F: via junction, overflow on E:)
    model_rel = MODEL_PATH.replace("\\", "/")
    model_locations = [
        COMFY_ROOT / "models" / "diffusion_models" / model_rel,
        Path("E:/AI-Models") / model_rel,
    ]
    if not any(p.exists() for p in model_locations):
        print(f"ERROR: T2V model not found at any of:")
        for p in model_locations:
            print(f"  {p}")
        print("Download it first: huggingface-cli download Kijai/WanVideo_comfy_fp8_scaled T2V/Wan2_1-T2V-14B_fp8_e4m3fn_scaled_KJ.safetensors")
        sys.exit(1)
    else:
        found = next(p for p in model_locations if p.exists())
        print(f"Model found: {found}")

    comfy_proc = None
    if args.start_comfy:
        comfy_proc = start_comfyui()
    elif not comfy_running():
        print("ComfyUI not running. Start it or use --start-comfy")
        print("Attempting to start...")
        comfy_proc = start_comfyui()

    prompts_to_use = PROMPTS[:args.prompts]
    results = {}

    for prompt_idx, prompt in enumerate(prompts_to_use):
        print(f"\n{'='*60}")
        print(f"Prompt {prompt_idx + 1}/{len(prompts_to_use)}: {prompt[:60]}...")
        print(f"{'='*60}")

        for lora_name, lora_path in LORAS.items():
            prefix = f"wanlantis_ab_{lora_name}_p{prompt_idx}"
            print(f"\n  Rendering: {lora_name} (seed={SEED})")

            workflow = build_workflow(prompt, lora_path, SEED, prefix)

            # Save workflow for inspection
            wf_path = OUTPUT_DIR / f"workflow_{lora_name}_p{prompt_idx}.json"
            with open(wf_path, "w") as f:
                json.dump(workflow, f, indent=2)
            print(f"  Workflow saved: {wf_path}")

            prompt_id = queue_prompt(workflow)
            print(f"  Queued: {prompt_id}")

            success = wait_for_completion(prompt_id, timeout=1800)
            results[f"{lora_name}_p{prompt_idx}"] = {
                "success": success,
                "prompt_id": prompt_id,
                "prompt": prompt,
                "lora": lora_name,
            }

            if success:
                print(f"  DONE: {lora_name} prompt {prompt_idx}")
            else:
                print(f"  FAILED: {lora_name} prompt {prompt_idx}")

    # Summary
    print(f"\n{'='*60}")
    print("A/B RENDER SUMMARY")
    print(f"{'='*60}")
    for key, res in results.items():
        status = "OK" if res["success"] else "FAIL"
        print(f"  [{status}] {key}: {res['prompt'][:50]}...")

    # Save results
    summary_path = OUTPUT_DIR / "render_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSummary: {summary_path}")
    print(f"Videos in: C:\\ComfyUI\\output\\")

    if comfy_proc:
        print("\nComfyUI still running. Kill manually when done.")


if __name__ == "__main__":
    main()
