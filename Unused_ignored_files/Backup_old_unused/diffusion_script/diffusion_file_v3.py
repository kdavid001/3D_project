#!/usr/bin/env python3
"""
Step 3 (FINAL FIXED): run_diffusion_final.py

Fixes applied:
1. JSON COMPATIBILITY: Reads manifest as a List (output of Step 1) instead of Dict.
2. FLAT KEYS: Accesses 'decision' directly (not nested in 'screening').
3. PATH FIX: Points to 'processed_train' instead of 'images'.
4. MASK EXPANSION: Dilates mask by 9px to fix gray blobs.
"""

import json
import os
import argparse
import shutil
import gc
import torch
from PIL import Image, ImageFilter
from diffusers import AutoPipelineForInpainting, AutoPipelineForImage2Image
from tqdm import tqdm

# ----------------------------
# Config
# ----------------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Apple Silicon (M1/M2) Support
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")


def get_name_from_path(path):
    return os.path.basename(os.path.normpath(path))


def resize_to_multiple_of_8(img):
    w, h = img.size
    w = w - (w % 8)
    h = h - (h % 8)
    if w != img.size[0] or h != img.size[1]:
        return img.resize((w, h), Image.Resampling.LANCZOS)
    return img


def flush_memory():
    gc.collect()
    torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


# ----------------------------
# PHASE 1: REPAIR (Inpainting)
# ----------------------------
def process_inpainting_phase(manifest, args, final_output_dir):
    print(f"\n🔹 PHASE 1: Loading INPAINTING Model...")

    # Load SDXL Inpainting
    pipe = AutoPipelineForInpainting.from_pretrained(
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    ).to(DEVICE)

    # Optimizations
    if DEVICE.type == 'cuda':
        pipe.enable_model_cpu_offload()

    if hasattr(pipe, 'watermarker'): pipe.watermarker = None
    generator = torch.Generator(device="cpu").manual_seed(42)  # CPU generator is safer for MPS

    print("   Processing REPAIR images...")

    # FIX 1 & 2: Iterate list directly & Access flat keys
    repair_items = [entry for entry in manifest if entry.get("decision") == "REPAIR"]

    if not repair_items:
        print("   ⚠️ No images marked for REPAIR. Skipping Phase 1.")
        return

    for entry in tqdm(repair_items):
        filename = entry["filename"]

        # FIX 3: Point to 'processed_train' where Step 1 saved the copies
        original_img_path = os.path.join(args.input_dir, "processed_train", filename)
        save_path = os.path.join(final_output_dir, filename)

        # Mask name usually matches filename (png)
        base_name = os.path.splitext(filename)[0]
        mask_path = os.path.join(args.out_dir, "masks", f"{base_name}.png")

        if not os.path.exists(mask_path):
            print(f"   ⚠️ Mask missing for {filename}, skipping.")
            continue
        if not os.path.exists(original_img_path):
            print(f"   ⚠️ Original image missing for {filename}, skipping.")
            continue

        try:
            image = Image.open(original_img_path).convert("RGB")
            image = resize_to_multiple_of_8(image)

            mask = Image.open(mask_path).convert("L")
            mask = resize_to_multiple_of_8(mask)

            # --- CRITICAL FIX: EXPAND MASK ---
            # MaxFilter(9) expands white area by 9 pixels to cover gray edges
            mask = mask.filter(ImageFilter.MaxFilter(9))

            # Run Inpainting
            result = pipe(
                prompt="high quality, sharp focus, realistic texture, 8k, photorealistic",
                negative_prompt="blur, noise, artifacts, distorted, seams, gray patches, mask",
                image=image,
                mask_image=mask,
                guidance_scale=8.0,
                num_inference_steps=30,
                strength=1.00,  # 1.0 = Fully replace masked area
                generator=generator
            ).images[0]

            result.save(save_path)

        except Exception as e:
            print(f"   ❌ Error on {filename}: {e}")

    del pipe
    flush_memory()


# ----------------------------
# PHASE 2: NOVEL VIEW (Optional)
# ----------------------------
def process_img2img_phase(manifest, args, final_output_dir):
    # FIX 1 & 2: Iterate list directly & Access flat keys
    novel_items = [entry for entry in manifest if entry.get("decision") == "NOVEL_VIEW"]
    if not novel_items:
        print("   ⚠️ No images selected for NOVEL_VIEW. Skipping Phase 2.")
        return

    print(f"\n🔹 PHASE 2: Loading IMG2IMG Model (Base)...")

    pipe = AutoPipelineForImage2Image.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    ).to(DEVICE)

    if DEVICE.type == 'cuda':
        pipe.enable_model_cpu_offload()
    if hasattr(pipe, 'watermarker'): pipe.watermarker = None
    generator = torch.Generator(device="cpu").manual_seed(99)

    print("   Processing NOVEL_VIEW images...")

    for entry in tqdm(novel_items):
        filename = entry["filename"]
        # FIX 3: Point to 'processed_train'
        original_img_path = os.path.join(args.input_dir, "processed_train", filename)

        if not os.path.exists(original_img_path): continue

        try:
            image = Image.open(original_img_path).convert("RGB")
            image = resize_to_multiple_of_8(image)

            # 1. Copy Original (Keep the original view too)
            shutil.copy2(original_img_path, os.path.join(final_output_dir, filename))

            # 2. Generate Variation
            result = pipe(
                prompt="same scene, slightly rotated camera angle, 3d shift, highly detailed",
                negative_prompt="watermark, blur, ugly, text",
                image=image,
                strength=0.55,
                guidance_scale=8.0,
                num_inference_steps=40,
                generator=generator
            ).images[0]

            # 3. Save
            base_name, ext = os.path.splitext(filename)
            novel_save_path = os.path.join(final_output_dir, f"{base_name}_novel{ext}")
            result.save(novel_save_path)

        except Exception as e:
            print(f"   ❌ Error on {filename}: {e}")

    del pipe
    flush_memory()


def main(args):
    resultname = get_name_from_path(args.input_dir)
    print(f"🔍 Checking Hardware...")
    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        print(f"✅ Apple Silicon (MPS) Detected")
    else:
        print("⚠️ Warning: CPU Mode (Very Slow)")

    # Load Manifest
    manifest_path = os.path.join(args.input_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found at {manifest_path}")

    with open(manifest_path, "r") as f:
        manifest = json.load(f)  # This is a LIST

    final_output_dir = os.path.join(args.input_dir, f"final_results_{resultname}")
    os.makedirs(final_output_dir, exist_ok=True)

    # PHASE 0: COPY "NONE" (Clean Images)
    print(f"\n🔹 PHASE 0: Copying clean images...")
    # FIX 1 & 2: Iterate list directly & Access flat keys
    # Note: 'NONE' essentially means "processed but not selected for Repair or Novel View"
    # In v25 logic, almost everything is REPAIR or NOVEL_VIEW, but safety check remains.
    for entry in manifest:
        if entry.get("decision") == "NONE":
            src = os.path.join(args.input_dir, "processed_train", entry["filename"])
            dst = os.path.join(final_output_dir, entry["filename"])
            if os.path.exists(src):
                shutil.copy2(src, dst)

    # Run Phases
    process_inpainting_phase(manifest, args, final_output_dir)
    process_img2img_phase(manifest, args, final_output_dir)

    print(f"\n✅ All done! Results saved to: {final_output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # input_dir contains: processed_train/, manifest.json
    parser.add_argument("--input_dir", type=str, required=True, help="Path to 'output' from Step 1")
    # out_dir contains: masks/
    parser.add_argument("--out_dir", type=str, required=True, help="Path to 'inpainting_ready' from Step 2")
    args = parser.parse_args()
    main(args)