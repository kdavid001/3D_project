"""
Step 3 (FINAL V2): run_diffusion_final.py

Fixes:
1. GRAY BLOB FIX: Automatically dilates (expands) the mask by 9 pixels to ensure
   it fully covers any gray patches in the source image.
2. NOVEL VIEW FIX: Switched from 'Refiner' (texture only) to 'SDXL Base' (geometry aware).
   This restores the ability to rotate/shift camera angles.
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

# ----------------------------
# PHASE 1: REPAIR (Inpainting)
# ----------------------------
def process_inpainting_phase(manifest, args, final_output_dir):
    print(f"\n🔹 PHASE 1: Loading INPAINTING Model...")

    # SDXL Inpainting (Standard)
    pipe = AutoPipelineForInpainting.from_pretrained(
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    ).to(DEVICE)

    pipe.enable_model_cpu_offload()
    if hasattr(pipe, 'watermarker'): pipe.watermarker = None

    generator = torch.Generator(device=DEVICE).manual_seed(42)
    print("   Processing REPAIR images...")

    for entry in tqdm(manifest["images"]):
        if entry["screening"]["decision"] != "REPAIR": continue

        filename = entry["filename"]
        # Try to find file in step1_dir first, but if user overwrote it, we accept it
        original_img_path = os.path.join(args.step1_dir, "images", filename)
        save_path = os.path.join(final_output_dir, filename)

        mask_filename = os.path.splitext(filename)[0] + ".png"
        mask_path = os.path.join(args.step2_dir, "masks", mask_filename)

        if not os.path.exists(mask_path) or not os.path.exists(original_img_path):
            continue

        try:
            image = Image.open(original_img_path).convert("RGB")
            image = resize_to_multiple_of_8(image)

            mask = Image.open(mask_path).convert("L")
            mask = resize_to_multiple_of_8(mask)

            # --- CRITICAL FIX: EXPAND MASK ---
            # We dilate the mask to ensure it is strictly LARGER than the gray blob.
            # MaxFilter(9) grows the white area by radius 9.
            mask = mask.filter(ImageFilter.MaxFilter(9))

            # Run Inpainting with 1.0 strength (Destroy & Rebuild)
            result = pipe(
                prompt="high quality, sharp focus, realistic texture, 8k, photorealistic",
                negative_prompt="blur, noise, artifacts, distorted, seams, gray patches, mask",
                image=image,
                mask_image=mask,
                guidance_scale=8.0,
                num_inference_steps=30,
                strength=1.00,
                generator=generator
            ).images[0]

            result.save(save_path)

        except Exception as e:
            print(f"   ❌ Error on {filename}: {e}")

    del pipe
    flush_memory()

# ----------------------------
# PHASE 2: NOVEL VIEW (Img2Img)
# ----------------------------
def process_img2img_phase(manifest, args, final_output_dir):
    print(f"\n🔹 PHASE 2: Loading IMG2IMG Model (Base)...")

    # SWITCHED: Using 'base-1.0' instead of 'refiner' to allow geometry changes
    pipe = AutoPipelineForImage2Image.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    ).to(DEVICE)

    pipe.enable_model_cpu_offload()
    if hasattr(pipe, 'watermarker'): pipe.watermarker = None

    generator = torch.Generator(device=DEVICE).manual_seed(99)
    print("   Processing NOVEL_VIEW images...")

    for entry in tqdm(manifest["images"]):
        if entry["screening"]["decision"] != "NOVEL_VIEW": continue

        filename = entry["filename"]
        original_img_path = os.path.join(args.step1_dir, "images", filename)

        if not os.path.exists(original_img_path): continue

        try:
            image = Image.open(original_img_path).convert("RGB")
            image = resize_to_multiple_of_8(image)

            # 1. Copy Original
            main_save_path = os.path.join(final_output_dir, filename)
            shutil.copy2(original_img_path, main_save_path)

            # 2. Generate Variation
            # Using Base 1.0 allows us to actually rotate the camera
            result = pipe(
                prompt="same scene, slightly rotated camera angle, 3d shift, new perspective, highly detailed",
                negative_prompt="watermark, blur, ugly, different objects, text",
                image=image,
                strength=0.55, # High enough to shift angle, low enough to keep identity
                guidance_scale=8.0,
                num_inference_steps=40,
                generator=generator
            ).images[0]

            # 3. Save as NEW file
            base_name, ext = os.path.splitext(filename)
            novel_save_path = os.path.join(final_output_dir, f"{base_name}_novel{ext}")
            result.save(novel_save_path)

        except Exception as e:
            print(f"   ❌ Error on {filename}: {e}")

    del pipe
    flush_memory()

def main(args):
    print(f"🔍 Checking GPU...")
    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("❌ NO GPU DETECTED. Exiting.")
        return

    # Load Manifest
    manifest_path = os.path.join(args.step1_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError("Manifest not found")

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    final_output_dir = os.path.join(args.step1_dir, "final_results_sdxl")
    os.makedirs(final_output_dir, exist_ok=True)

    # PHASE 0: COPY "NONE"
    print(f"\n🔹 PHASE 0: Copying clean images...")
    for entry in manifest["images"]:
        if entry["screening"]["decision"] == "NONE":
            src = os.path.join(args.step1_dir, "images", entry["filename"])
            dst = os.path.join(final_output_dir, entry["filename"])
            if os.path.exists(src):
                shutil.copy2(src, dst)

    process_inpainting_phase(manifest, args, final_output_dir)
    process_img2img_phase(manifest, args, final_output_dir)

    print(f"\n✅ All done! Results saved to: {final_output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step1_dir", type=str, required=True)
    parser.add_argument("--step2_dir", type=str, required=True)
    args = parser.parse_args()
    main(args)