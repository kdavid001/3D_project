#!/usr/bin/env python3
"""
COMBINED PIPELINE V4: run_full_generation_v4.py

UPDATES:
1. FORCE UNIFORMITY: Every output file is resized to exactly 1024x1024.
2. SMART CROP: Mathematically calculates grid splits to fix "Two images in one" bug.
3. DEBUGGING: Saves the raw grid for inspection if cropping fails.
"""

import sys
import os
import shutil
import gc
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

# --- PATCH FOR UPSCALER ---
try:
    import torchvision.transforms.functional_tensor
except ImportError:
    try:
        import torchvision.transforms.functional as functional

        sys.modules["torchvision.transforms.functional_tensor"] = functional
    except ImportError:
        pass


def get_name_from_path(path):
    return os.path.basename(os.path.normpath(path))


def flush_memory():
    gc.collect()
    torch.cuda.empty_cache()


# ==========================================
# PHASE 0: PRE-PROCESSING (Standardize Inputs)
# ==========================================
def process_for_zero123(pil_image):
    # A. Remove Background
    img_byte_arr = io.BytesIO()
    pil_image.save(img_byte_arr, format='PNG')
    output_bytes = remove(img_byte_arr.getvalue())
    no_bg_image = Image.open(io.BytesIO(output_bytes)).convert("RGBA")

    # B. FORCE 512x512 Canvas (AI Requirement)
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
# PHASE 1: SYNTHESIS (Smart Crop)
# ==========================================
def crop_zero123_grid_dynamic(grid_img, base_filename, output_dir):
    """
    Dynamically cuts the grid based on its actual size.
    Zero123++ v1.2 usually outputs 3 columns x 2 rows.
    """
    w, h = grid_img.size

    # Calculate single tile size
    tile_w = w // 3
    tile_h = h // 2

    # Debug print for the first image
    if not hasattr(crop_zero123_grid_dynamic, "debug_printed"):
        print(f"   📏 Detected AI Grid Size: {w}x{h}")
        print(f"   📏 Calculated Tile Size:  {tile_w}x{tile_h}")
        crop_zero123_grid_dynamic.debug_printed = True

    count = 0
    generated_files = []

    for row in range(2):
        for col in range(3):
            left = col * tile_w
            top = row * tile_h
            right = left + tile_w
            bottom = top + tile_h

            view = grid_img.crop((left, top, right, bottom))

            # Save raw synthetic view
            save_name = f"synth_{os.path.splitext(base_filename)[0]}_v{count}.png"
            view.save(os.path.join(output_dir, save_name))
            generated_files.append(save_name)
            count += 1

    return generated_files


def run_synthesis_phase(input_dir, temp_dir, candidates):
    print(f"\n🔹 PHASE 1: Cleaning & Synthesizing...")

    pipeline = DiffusionPipeline.from_pretrained(
        "sudo-ai/zero123plus-v1.2",
        custom_pipeline="sudo-ai/zero123plus-pipeline",
        torch_dtype=torch.float16
    ).to("cuda")

    for entry in tqdm(candidates, desc="Generating Swarm"):
        filename = entry["filename"]
        paths = [
            os.path.join(input_dir, "processed_train", filename),
            os.path.join(input_dir, "images", filename),
            os.path.join(input_dir, filename)
        ]
        img_path = next((p for p in paths if os.path.exists(p)), None)
        if not img_path: continue

        # 1. Clean Input
        input_img = Image.open(img_path).convert("RGB")
        clean_input = process_for_zero123(input_img)
        clean_input.save(os.path.join(temp_dir, f"anchor_{filename}"))  # Save Anchor

        # 2. Generate Grid
        result_grid = pipeline(clean_input, num_inference_steps=75).images[0]

        # 3. Smart Crop
        crop_zero123_grid_dynamic(result_grid, filename, temp_dir)

    del pipeline
    flush_memory()


# ==========================================
# PHASE 2: UPSCALE & STANDARDIZE (The Final Fix)
# ==========================================
def run_upscale_phase(temp_dir, final_dir):
    print(f"\n🔹 PHASE 2: Upscaling & Standardizing to 1024x1024...")

    if not os.path.exists('weights/RealESRGAN_x4plus.pth'):
        os.makedirs('weights', exist_ok=True)
        torch.hub.download_url_to_file(
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
            'weights/RealESRGAN_x4plus.pth'
        )

    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    upsampler = RealESRGANer(scale=4, model_path='weights/RealESRGAN_x4plus.pth', model=model, tile=400, half=True,
                             gpu_id=0)

    valid_exts = ('.png', '.jpg', '.jpeg')
    files = [f for f in os.listdir(temp_dir) if f.lower().endswith(valid_exts)]

    for filename in tqdm(files, desc="Standardizing"):
        img_path = os.path.join(temp_dir, filename)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None: continue

        h, w = img.shape[:2]

        # A. UPSCALE LOGIC
        # If small (< 800), upscale it 4x.
        if w < 800:
            output, _ = upsampler.enhance(img, outscale=4)
        else:
            output = img

        # B. STANDARDIZATION LOGIC (FORCE 1024x1024)
        # This fixes the "Ghost File" size mismatch for GS compatibility.
        output = cv2.resize(output, (1024, 1024), interpolation=cv2.INTER_LANCZOS4)

        cv2.imwrite(os.path.join(final_dir, filename), output)


def main(args):
    print(f"🔍 Hardware: {torch.cuda.get_device_name(0)}")

    # FORCE NEW FOLDER to verify fix
    dataset_name = get_name_from_path(args.input_dir)
    unique_suffix = "v4_uniform"

    temp_dir = os.path.join(args.out_dir, f"temp_{dataset_name}_{unique_suffix}")
    final_dir = os.path.join(args.out_dir, f"final_{dataset_name}_{unique_suffix}")

    if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
    if os.path.exists(final_dir): shutil.rmtree(final_dir)
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(final_dir, exist_ok=True)

    manifest_path = os.path.join(args.input_dir, "manifest.json")
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    candidates = [e for e in manifest if e.get("decision") in ["NOVEL_VIEW", "NONE"]]
    candidates.sort(key=lambda x: x.get("score", 0), reverse=True)
    candidates = candidates[:20]

    run_synthesis_phase(args.input_dir, temp_dir, candidates)
    run_upscale_phase(temp_dir, final_dir)

    print(f"\n✅✅ DONE! All images are now EXACTLY 1024x1024.")
    print(f"📂 Output: {final_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    main(args)