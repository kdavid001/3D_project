"""
COMBINED PIPELINE V7 (FIXED LAYOUT): run_full_generation_v7.py

UPDATES:
1. LAYOUT FIX: Adjusted slicing for 2 Columns x 3 Rows (2x3).
2. DEBUG SAVE: Still saves the full grid so you can double-check.
3. FORCE UNIFORMITY: 1024x1024 output.
"""

import sys
import os

# ==========================================
# 🚨 CRITICAL PATCH 🚨
# ==========================================
try:
    import torchvision.transforms.functional_tensor
except ImportError:
    try:
        import torchvision.transforms.functional as functional

        sys.modules["torchvision.transforms.functional_tensor"] = functional
    except ImportError:
        pass
# ==========================================

import shutil
import gc
import argparse
import json
import torch
import cv2
import numpy as np
from PIL import Image
from diffusers import DiffusionPipeline
from tqdm import tqdm
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer


def get_name_from_path(path):
    return os.path.basename(os.path.normpath(path))


def flush_memory():
    gc.collect()
    torch.cuda.empty_cache()


# ==========================================
# PHASE 0: PRE-PROCESSING
# ==========================================
def process_for_zero123(pil_image):
    canvas_size = 512
    canvas = Image.new("RGB", (canvas_size, canvas_size), (0, 0, 0))  # Black BG

    w, h = pil_image.size
    scale = (canvas_size * 0.85) / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized_obj = pil_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    x = (canvas_size - new_w) // 2
    y = (canvas_size - new_h) // 2

    canvas.paste(resized_obj, (x, y))
    return canvas


# ==========================================
# PHASE 1: SYNTHESIS (FIXED FOR 2x3 GRID)
# ==========================================
def crop_zero123_grid_dynamic(grid_img, base_filename, output_dir):
    w, h = grid_img.size

    # --- 🚨 THE FIX IS HERE 🚨 ---
    # Layout: 2 Columns, 3 Rows
    tile_w = w // 2
    tile_h = h // 3
    # -----------------------------

    count = 0
    generated_files = []

    # Iterate: 3 Rows down, 2 Columns across
    for row in range(3):
        for col in range(2):
            # Calculate coordinates
            left = col * tile_w
            top = row * tile_h
            right = left + tile_w
            bottom = top + tile_h

            view = grid_img.crop((left, top, right, bottom))

            save_name = f"synth_{os.path.splitext(base_filename)[0]}_v{count}.png"
            view.save(os.path.join(output_dir, save_name))
            generated_files.append(save_name)
            count += 1

    return generated_files


def run_synthesis_phase(input_dir, temp_dir, candidates):
    print(f"\n🔹 PHASE 1: Synthesizing (2x3 Grid Layout)...")

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

        input_img = Image.open(img_path).convert("RGB")
        clean_input = process_for_zero123(input_img)
        clean_input.save(os.path.join(temp_dir, f"anchor_{filename}"))

        result_grid = pipeline(clean_input, num_inference_steps=75).images[0]

        # Save Debug Grid
        result_grid.save(os.path.join(temp_dir, f"FULL_GRID_{filename}"))

        # Cut using NEW 2x3 logic
        crop_zero123_grid_dynamic(result_grid, filename, temp_dir)

    del pipeline
    flush_memory()


# ==========================================
# PHASE 2: UPSCALE & STANDARDIZE
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
        # Skip Full Grids
        if "FULL_GRID" in filename: continue

        img_path = os.path.join(temp_dir, filename)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None: continue

        h, w = img.shape[:2]

        if w < 800:
            output, _ = upsampler.enhance(img, outscale=4)
        else:
            output = img

        output = cv2.resize(output, (1024, 1024), interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(os.path.join(final_dir, filename), output)


def main(args):
    print(f"🔍 Hardware: {torch.cuda.get_device_name(0)}")

    dataset_name = get_name_from_path(args.input_dir)
    unique_suffix = "fixed"

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

    print(f"\n✅✅ DONE! (Used 2x3 Grid Layout)")
    print(f"📂 Output: {final_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    main(args)