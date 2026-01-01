#!/usr/bin/env python3
"""
Step 3 (FINAL V6): run_diffusion_reference_v6.py

Updates:
1. CRITICAL FIX: Loads IP-Adapter BEFORE enabling CPU offload.
   (Fixes 'Tensor on device meta' error).
2. LOGIC: Keeps the smart manifest logic from V5.
"""

import json
import os
import argparse
import shutil
import gc
import torch
from PIL import Image, ImageFilter
from diffusers import AutoPipelineForInpainting
from tqdm import tqdm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def flush_memory():
    gc.collect()
    torch.cuda.empty_cache()


def get_reference_image(input_dir):
    """Finds best reference by looking at Step 1 manifest."""
    manifest_path = os.path.join(input_dir, "manifest.json")
    if not os.path.exists(manifest_path): return None

    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    # 1. Prefer NOVEL_VIEW
    candidates = [e for e in manifest if e.get("decision") == "NOVEL_VIEW"]
    # 2. Fallback to NONE
    if not candidates:
        candidates = [e for e in manifest if e.get("decision") == "NONE"]

    if not candidates: return None

    candidates.sort(key=lambda x: x.get("score", 0), reverse=True)
    best = candidates[0]

    # Try finding the file
    paths = [
        os.path.join(input_dir, "processed_train", best["filename"]),
        os.path.join(input_dir, "images", best["filename"]),
        os.path.join(input_dir, best["filename"])
    ]
    for p in paths:
        if os.path.exists(p):
            print(f"🌟 Identity Anchor Found: {best['filename']} (Score: {best.get('score', 0)})")
            return Image.open(p).convert("RGB")
    return None


def process_repairs(repair_manifest, args, pipe, ref_image, final_dir):
    repair_queue = [x for x in repair_manifest if x.get("decision") == "REPAIR"]

    if not repair_queue:
        print("✅ No images marked as REPAIR in the manifest.")
        return

    print(f"   Processing {len(repair_queue)} REPAIR images...")
    generator = torch.Generator(device="cpu").manual_seed(42)

    for entry in tqdm(repair_queue):
        filename = entry.get("original_filename", entry["filename"])

        # Source Image
        possible_src = [
            os.path.join(args.input_dir, "processed_train", filename),
            os.path.join(args.input_dir, "images", filename),
            os.path.join(args.input_dir, filename)
        ]
        src_path = next((p for p in possible_src if os.path.exists(p)), None)

        # Mask Path
        base_name = os.path.splitext(filename)[0]
        mask_path = os.path.join(args.out_dir, "masks", f"{base_name}.png")

        if not src_path or not os.path.exists(mask_path):
            print(f"   ⚠️ Skipping {filename} (Missing Image or Mask)")
            continue

        try:
            image = Image.open(src_path).convert("RGB")
            # Resize to multiple of 8
            w, h = image.size
            image = image.resize((w - w % 8, h - h % 8), Image.Resampling.LANCZOS)

            mask = Image.open(mask_path).convert("L")
            mask = mask.resize((w - w % 8, h - h % 8), Image.Resampling.LANCZOS)
            mask = mask.filter(ImageFilter.MaxFilter(9))

            result = pipe(
                prompt="high quality, realistic, 8k, seamless",
                ip_adapter_image=ref_image,
                negative_prompt="blur, noise, artifacts, distorted",
                image=image,
                mask_image=mask,
                guidance_scale=7.0,
                num_inference_steps=30,
                strength=1.00,
                generator=generator
            ).images[0]

            result.save(os.path.join(final_dir, filename))

        except Exception as e:
            print(f"   ❌ Error {filename}: {e}")


def main(args):
    print(f"🔍 Hardware: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # 1. Load Step 2 Manifest
    step2_manifest_path = os.path.join(args.out_dir, "inpainting_manifest.json")
    if not os.path.exists(step2_manifest_path):
        raise FileNotFoundError("Step 2 Manifest missing.")

    with open(step2_manifest_path, 'r') as f:
        step2_manifest = json.load(f)

    # 2. Load Step 1 Manifest
    step1_manifest_path = os.path.join(args.input_dir, "manifest.json")
    step1_manifest = []
    if os.path.exists(step1_manifest_path):
        with open(step1_manifest_path, 'r') as f:
            step1_manifest = json.load(f)

    # Output Dir
    res_name = os.path.basename(os.path.normpath(args.input_dir))
    final_dir = os.path.join(args.input_dir, f"final_results_{res_name}")
    os.makedirs(final_dir, exist_ok=True)

    # PHASE 0: COPY CLEAN IMAGES
    print(f"\n🔹 PHASE 0: Copying clean images...")
    repair_filenames = {x.get("original_filename", x["filename"]) for x in step2_manifest}

    copied = 0
    for entry in step1_manifest:
        fname = entry["filename"]
        if fname not in repair_filenames:
            possible_src = [
                os.path.join(args.input_dir, "processed_train", fname),
                os.path.join(args.input_dir, "images", fname),
                os.path.join(args.input_dir, fname)
            ]
            src = next((p for p in possible_src if os.path.exists(p)), None)
            if src:
                shutil.copy2(src, os.path.join(final_dir, fname))
                copied += 1
    print(f"   Copied {copied} clean images.")

    # PHASE 1: REPAIR
    print(f"\n🔹 PHASE 1: Initializing IP-Adapter...")
    try:
        # 1. Load Pipe FIRST
        pipe = AutoPipelineForInpainting.from_pretrained(
            "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        ).to(DEVICE)

        # 2. Load Adapter SECOND (While pipe is still fully accessible)
        # Note: If this crashes RAM, we are in trouble, but this is the correct order for Logic.
        pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")
        pipe.set_ip_adapter_scale(0.7)

        # 3. Enable Offload LAST (Only now do we pack it away to save RAM)
        if DEVICE.type == 'cuda':
            print("   Activating RAM Saver (Sequential Offload)...")
            pipe.enable_sequential_cpu_offload()

    except Exception as e:
        print(f"❌ Load Error: {e}")
        return

    ref_img = get_reference_image(args.input_dir)
    if not ref_img:
        print("❌ Critical: No reference image found.")
        return

    process_repairs(step2_manifest, args, pipe, ref_img, final_dir)
    print(f"\n✅ Done. Output: {final_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    main(args)