"""
Step 3 (UPGRADED): run_diffusion_sdxl.py - High-Quality SDXL Processing

Usage:
    python run_diffusion_sdxl.py --step1_dir ./preprocessed --step2_dir ./inpainting_ready

Differences from previous version:
- Uses 'diffusers/stable-diffusion-xl-1.0-inpainting-0.1' (Much higher quality)
- Native 1024x1024 support (No tiling artifacts)
- Uses 'refiner' logic for extra sharpness (Optional but recommended)

Problem: Causes exploding RAM on google colab as both repair and novel model are 13GB + in total and GPU is 15Gb max
only use this when there is enough RAM on GPU solution: Sequential processing -> check the main diffusion_script_V2.py,
The Solution: Sequential ProcessingWe cannot load both models at the same time. We must use a Relay Race
strategy:Load Inpainting Model -> Process all "Repair" images -> Delete Model from Memory.Load
Img2Img Model -> Process all "Novel View" images -> Delete Model from Memory.This ensures you
never use more than ~7GB at a time, keeping you safe from crashing."""

import json
import os
import argparse
import shutil
import torch
from PIL import Image
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


def main(args):
    print(f"🚀 Initializing SDXL Models on {DEVICE}...")

    # 1. Load SDXL Pipelines (AutoPipeline handles the complex XL loading for us)
    #    We use the dedicated Inpainting model for SDXL 1.0

    # INPAINTING MODEL
    pipe_inpaint = AutoPipelineForInpainting.from_pretrained(
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        torch_dtype=torch.float16,
        variant="fp16",
        matermaker = None,
    ).to(DEVICE)

    # IMG2IMG MODEL (Reuses the same base if possible, but we load explicit for safety)
    pipe_img2img \
        = AutoPipelineForImage2Image.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",  # Using refiner for img2img gives great detail
        torch_dtype=torch.float16,
        variant="fp16",
        watermarker = None,
    ).to(DEVICE)

    # Enable Memory Optimizations (Critical for SDXL on Colab T4)
    pipe_inpaint.enable_model_cpu_offload()
    pipe_img2img.enable_model_cpu_offload()

    # 2. Load Manifest
    manifest_path = os.path.join(args.step1_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Cannot find manifest at {manifest_path}")

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    # 3. Output Directory
    final_output_dir = os.path.join(args.step1_dir, "final_results_sdxl")
    os.makedirs(final_output_dir, exist_ok=True)

    print(f"📂 Processing {len(manifest['images'])} images with SDXL...")

    for entry in tqdm(manifest["images"]):
        filename = entry["filename"]
        decision = entry["screening"]["decision"]

        original_img_path = os.path.join(args.step1_dir, "images", filename)
        main_save_path = os.path.join(final_output_dir, filename)

        if not os.path.exists(original_img_path):
            continue

        try:
            # CASE 1: NONE (Copy)
            if decision == "NONE":
                shutil.copy2(original_img_path, main_save_path)
                continue

            # Load Image
            image = Image.open(original_img_path).convert("RGB")
            # SDXL prefers 1024x1024. If your images are smaller, this might be slow.
            image = resize_to_multiple_of_8(image)
            generator = torch.Generator(device=DEVICE).manual_seed(99)

            # CASE 2: REPAIR (SDXL Inpainting)
            if decision == "REPAIR":
                mask_filename = os.path.splitext(filename)[0] + ".png"
                mask_path = os.path.join(args.step2_dir, "masks", mask_filename)

                if not os.path.exists(mask_path):
                    shutil.copy2(original_img_path, main_save_path)
                    continue

                mask = Image.open(mask_path).convert("L")
                mask = resize_to_multiple_of_8(mask)

                # SDXL Inpainting Inference
                result = pipe_inpaint(
                    prompt="high quality, sharp focus, realistic texture, 8k, photorealistic",
                    negative_prompt="blur, noise, artifacts, distorted, seams, cartoon, painting",
                    image=image,
                    mask_image=mask,
                    guidance_scale=8.0,
                    num_inference_steps=25,  # SDXL needs fewer steps than 1.5
                    strength=0.99,
                    generator=generator
                ).images[0]

                result.save(main_save_path)

            # CASE 3: NOVEL VIEW (SDXL Refiner Img2Img)
            elif decision == "NOVEL_VIEW":
                # Copy original
                shutil.copy2(original_img_path, main_save_path)

                # Generate Variation
                result = pipe_img2img(
                    prompt="same scene, highly detailed, photorealistic, 8k, cinematic lighting",
                    image=image,
                    strength=0.25,  # Keep it subtle
                    guidance_scale=7.5,
                    num_inference_steps=25,
                    generator=generator
                ).images[0]

                # Save new file
                base_name, ext = os.path.splitext(filename)
                novel_save_path = os.path.join(final_output_dir, f"{base_name}_novel{ext}")
                result.save(novel_save_path)

        except Exception as e:
            print(f"❌ Error {filename}: {e}")
            continue

    print(f"\n✅ SDXL Processing Complete!")


if __name__ == "__main__":
    print(f"🔍 Checking GPU availability...")
    if torch.cuda.is_available():
        print(f"✅ GPU Detected: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")
        DEVICE = torch.device('cuda')
    else:
        print("❌ NO GPU DETECTED. The script will run on CPU (Very Slow).")
        print("   Troubleshooting: Check 'nvidia-smi' or reinstall torch with CUDA support.")
        DEVICE = torch.device('cpu')
    parser = argparse.ArgumentParser()
    parser.add_argument("--step1_dir", type=str, required=True)
    parser.add_argument("--step2_dir", type=str, required=True)
    args = parser.parse_args()
    main(args)