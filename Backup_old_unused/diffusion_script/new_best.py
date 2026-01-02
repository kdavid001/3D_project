#!/usr/bin/env python3
"""
COMBINED PIPELINE V3: run_full_generation_v3.py

UPDATES:
1. NUKE PROTOCOL: Wipes temp folders at start to remove "Ghost Files" (852x1920 errors).
2. DEBUG PRINTS: Shows you exactly what size the images are during processing.
3. REMBG + CENTERING: Keeps the "Smart Clean" logic to prevent double objects.
"""

import sys
import os
import shutil  # <--- Added for Nuke Protocol
import gc

# --- PATCH FOR UPSCALER ---
try:
    import torchvision.transforms.functional_tensor
except ImportError:
    try:
        import torchvision.transforms.functional as functional

        sys.modules["torchvision.transforms.functional_tensor"] = functional
    except ImportError:
        pass
# --------------------------

import argparse
import json
import torch
import cv2
import numpy as np
import io
from PIL import Image
from diffusers import DiffusionPipeline
from tqdm import tqdm
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from rembg import remove


def get_name_from_path(path):
    return os.path.basename(os.path.normpath(path))


def flush_memory():
    gc.collect()
    torch.cuda.empty_cache()


# ==========================================
# PHASE 0: PRE-PROCESSING (The Fix)
# ==========================================
def process_for_zero123(pil_image):
    """
    1. Removes background.
    2. Centers object on a 512x512 gray square.
    """
    # A. Remove Background
    img_byte_arr = io.BytesIO()
    pil_image.save(img_byte_arr, format='PNG')
    img_bytes = img_byte_arr.getvalue()

    output_bytes = remove(img_bytes)
    no_bg_image = Image.open(io.BytesIO(output_bytes)).convert("RGBA")

    # B. Center and Pad to Square
    canvas_size = 512
    canvas = Image.new("RGB", (canvas_size, canvas_size), (127, 127, 127))

    w, h = no_bg_image.size
    scale = (canvas_size * 0.85) / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized_obj = no_bg_image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    x = (canvas_size - new_w) // 2
    y = (canvas_size - new_h) // 2

    canvas.paste(resized_obj, (x, y), resized_obj)
    return canvas


# ==========================================
# PHASE 1: SYNTHESIS HELPER
# ==========================================
def crop_zero123_grid(grid_img, base_filename, output_dir):
    w, h = grid_img.size
    view_w = w // 3
    view_h = h // 2
    count = 0
    generated_files = []

    for row in range(2):
        for col in range(3):
            left = col * view_w
            top = row * view_h
            right = left + view_w
            bottom = top + view_h
            view = grid_img.crop((left, top, right, bottom))

            clean_name = os.path.splitext(base_filename)[0]
            save_name = f"synth_{clean_name}_v{count}.png"
            save_path = os.path.join(output_dir, save_name)
            view.save(save_path)
            generated_files.append(save_name)
            count += 1
    return generated_files


def run_synthesis_phase(input_dir, temp_dir, candidates):
    print(f"\n🔹 PHASE 1: Cleaning & Synthesizing (Zero123++)...")

    print("   Loading Generation Model...")
    pipeline = DiffusionPipeline.from_pretrained(
        "sudo-ai/zero123plus-v1.2",
        custom_pipeline="sudo-ai/zero123plus-pipeline",
        torch_dtype=torch.float16
    ).to("cuda")

    total_images = 0

    for entry in tqdm(candidates, desc="Generating Swarm"):
        filename = entry["filename"]
        paths = [
            os.path.join(input_dir, "processed_train", filename),
            os.path.join(input_dir, "images", filename),
            os.path.join(input_dir, filename)
        ]
        img_path = next((p for p in paths if os.path.exists(p)), None)
        if not img_path: continue

        input_img = Image.open(img_path).convert("RGB")

        # CLEAN & CENTER
        clean_input = process_for_zero123(input_img)

        # Debug Check: Verify size is 512x512
        if clean_input.size != (512, 512):
            print(f"⚠️ Warning: Resizing failed for {filename}. Got {clean_input.size}")
            continue

        clean_input.save(os.path.join(temp_dir, f"anchor_{filename}"))
        total_images += 1

        # Generate Views
        result_grid = pipeline(clean_input, num_inference_steps=75).images[0]
        new_files = crop_zero123_grid(result_grid, filename, temp_dir)
        total_images += len(new_files)

    del pipeline
    flush_memory()
    return total_images


# ==========================================
# PHASE 2: UPSCALE HELPER
# ==========================================
def run_upscale_phase(temp_dir, final_dir):
    print(f"\n🔹 PHASE 2: High-Res Upscaling (Real-ESRGAN)...")

    if not os.path.exists('weights/RealESRGAN_x4plus.pth'):
        os.makedirs('weights', exist_ok=True)
        torch.hub.download_url_to_file(
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
            'weights/RealESRGAN_x4plus.pth'
        )

    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    upsampler = RealESRGANer(
        scale=4,
        model_path='weights/RealESRGAN_x4plus.pth',
        model=model,
        tile=400,
        tile_pad=10,
        pre_pad=0,
        half=True,
        gpu_id=0
    )

    valid_exts = ('.png', '.jpg', '.jpeg')
    files = [f for f in os.listdir(temp_dir) if f.lower().endswith(valid_exts)]

    for filename in tqdm(files, desc="Upscaling"):
        img_path = os.path.join(temp_dir, filename)
        try:
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if img is None: continue

            h, w = img.shape[:2]

            # --- DEBUG CHECK ---
            # If an image is 852x1920, it is garbage. Skip it.
            if h > 1000 and w < 1000:
                # Vertical video frame detected. This is a ghost file.
                # print(f"   ⚠️ Skipping Ghost File: {filename} ({w}x{h})")
                continue

            if w < 1000:
                output, _ = upsampler.enhance(img, outscale=4)
            else:
                output = img

            cv2.imwrite(os.path.join(final_dir, filename), output)
        except Exception as e:
            print(f"❌ Error {filename}: {e}")


# ==========================================
# MAIN
# ==========================================
def main(args):
    print(f"🔍 Hardware: {torch.cuda.get_device_name(0)}")

    dataset_name = get_name_from_path(args.input_dir)
    temp_dir = os.path.join(args.out_dir, f"temp_synth_{dataset_name}")
    final_dir = os.path.join(args.out_dir, f"final_dataset_{dataset_name}")

    # --- NUKE PROTOCOL: DELETE OLD DATA ---
    if os.path.exists(temp_dir):
        print(f"🧹 Nuking old temp directory: {temp_dir}")
        shutil.rmtree(temp_dir)
    if os.path.exists(final_dir):
        print(f"🧹 Nuking old final directory: {final_dir}")
        shutil.rmtree(final_dir)

    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(final_dir, exist_ok=True)

    manifest_path = os.path.join(args.input_dir, "manifest.json")
    if not os.path.exists(manifest_path): raise FileNotFoundError("Manifest missing.")
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    candidates = [e for e in manifest if e.get("decision") in ["NOVEL_VIEW", "NONE"]]
    candidates.sort(key=lambda x: x.get("score", 0), reverse=True)
    candidates = candidates[:20]

    if not candidates:
        print("❌ No clean images found.")
        return

    print(f"🚀 Starting V3 Pipeline (Nuke Old Data + Auto-Clean)...")
    run_synthesis_phase(args.input_dir, temp_dir, candidates)
    run_upscale_phase(temp_dir, final_dir)

    print(f"\n✅✅ PIPELINE V3 COMPLETE!")
    print(f"📂 Final Dataset: {final_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    main(args)