#!/usr/bin/env python3
"""
COMBINED PIPELINE V2.0 (Zero123++ Edition)
HYBRID DATA ARCHITECTURE PATCHES APPLIED

MODES:
1. --mode synthesis (DEFAULT): Runs Zero-123++ (2x3 Grid, Black BG).
2. --mode restoration: Runs ControlNet logic (Fixes natural images, Keeps BG).
"""

import sys

print("🚀 SCRIPT BOOT SEQUENCE STARTED...")
import os
import shutil
import gc
import argparse
import json
import torch
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from rembg import remove as rembg_remove

# Diffusers imports
from diffusers import (
    DiffusionPipeline,
    StableDiffusionControlNetImg2ImgPipeline,
    ControlNetModel,
    UniPCMultistepScheduler
)

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


def get_name_from_path(path):
    return os.path.basename(os.path.normpath(path))


def flush_memory():
    gc.collect()
    torch.cuda.empty_cache()


# ==========================================
# PHASE 0: PRE-PROCESSING (Synthesis Only)
# ==========================================
def process_for_zero123(pil_image):
    """
    Applies black background and centering.
    ONLY used for Synthesis Mode.
    """
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
# PHASE 1A: SYNTHESIS (Zero-123++ Logic)
# ==========================================
def crop_zero123_grid_dynamic(grid_img, base_filename, output_dir):
    w, h = grid_img.size
    # Layout: 2 Columns, 3 Rows
    tile_w = w // 2
    tile_h = h // 3

    count = 0
    generated_files = []

    # Iterate: 3 Rows down, 2 Columns across
    for row in range(3):
        for col in range(2):
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
    print(f"\n🔹 PHASE 1: Synthesizing (Zero-123++ Mode)...")

    pipeline = DiffusionPipeline.from_pretrained(
        "sudo-ai/zero123plus-v1.2",
        custom_pipeline="sudo-ai/zero123plus-pipeline",
        torch_dtype=torch.float16
    ).to("cuda")

    for entry in tqdm(candidates, desc="Generating Swarm"):
        filename = entry["filename"]
        raw_fallback = input_dir.replace("output_processed", "output_train")

        # 🟢 ARCHITECTURE PATCH: Added robust path hunting including "input" folder
        paths = [
            os.path.join(input_dir, "processed_train", filename),
            os.path.join(raw_fallback, "train", filename),
            os.path.join(input_dir, "images", filename),
            os.path.join(input_dir, "input", filename),
            os.path.join(input_dir, filename)
        ]
        img_path = next((p for p in paths if os.path.exists(p)), None)
        if not img_path: continue

        input_img = Image.open(img_path).convert("RGB")

        # V7 Logic: Force black background
        clean_input = process_for_zero123(input_img)
        clean_input.save(os.path.join(temp_dir, f"anchor_{filename}"))

        result_grid = pipeline(clean_input, num_inference_steps=75).images[0]
        result_grid.save(os.path.join(temp_dir, f"FULL_GRID_{filename}"))
        generated_files = crop_zero123_grid_dynamic(result_grid, filename, temp_dir)

        # Remove grey Zero123++ backgrounds from synth views — anchor is untouched
        for synth_name in generated_files:
            synth_path = os.path.join(temp_dir, synth_name)
            img_rgba = Image.open(synth_path).convert("RGBA")
            subject = rembg_remove(img_rgba)
            canvas = Image.new("RGBA", subject.size, (0, 0, 0, 255))
            canvas.paste(subject, (0, 0), subject)
            canvas.convert("RGB").save(synth_path)

    del pipeline
    flush_memory()


# ==========================================
# PHASE 1B: RESTORATION (ControlNet Logic)
# ==========================================
def run_restoration_phase(input_dir, temp_dir, candidates, prompt):
    print(f"\n🔹 PHASE 1: Restoring (ControlNet Tile Mode)...")
    print(f"   Prompt: '{prompt}'")

    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11f1e_sd15_tile",
        torch_dtype=torch.float16
    )

    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None
    ).to("cuda")

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    success_count = 0

    for i, entry in enumerate(tqdm(candidates, desc="Restoring Images")):
        filename = entry["filename"]
        raw_fallback = input_dir.replace("output_processed", "output_train")


        possible_paths = [
            os.path.join(input_dir, "processed_train", filename),
            os.path.join(raw_fallback, "train", filename),
            os.path.join(input_dir, "images", filename),
            os.path.join(input_dir, "input", filename),
            os.path.join(input_dir, filename)
        ]

        img_path = next((p for p in possible_paths if os.path.exists(p)), None)

        if not img_path:
            if i == 0:
                print(f"\n❌ CRITICAL ERROR: Could not find image '{filename}'")
            continue

        original_image = Image.open(img_path).convert("RGB")

        w, h = original_image.size
        new_w = (w // 8) * 8
        new_h = (h // 8) * 8
        if new_w != w or new_h != h:
            original_image = original_image.resize((new_w, new_h))

        clean_image = pipe(
            prompt,
            image=original_image,
            control_image=original_image,
            negative_prompt="blur, noise, grain, low resolution, distorted, plastic, cartoon",
            num_inference_steps=30,
            strength=0.35,
            guidance_scale=7.0
        ).images[0]

        save_name = f"restored_{filename}"
        if not save_name.lower().endswith(".png"):
            save_name = os.path.splitext(save_name)[0] + ".png"

        clean_image.save(os.path.join(temp_dir, save_name))
        success_count += 1

    # To prevent OOM Error
    del pipe
    del controlnet
    flush_memory()


# ==========================================
# PHASE 2: UPSCALE (Shared)
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
        if "FULL_GRID" in filename: continue  # Skip debug grids
        # 🟢 ARCHITECTURE PATCH: Anchor files are safely passed through to ESRGAN!

        img_path = os.path.join(temp_dir, filename)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None: continue

        h, w = img.shape[:2]

        if w < 1024 or h < 1024:
            output, _ = upsampler.enhance(img, outscale=4)
        else:
            output = img

        output = cv2.resize(output, (1024, 1024), interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(os.path.join(final_dir, filename), output)


def main(args):
    print(f"🔍 Hardware: {torch.cuda.get_device_name(0)}")
    print(f"⚙️  MODE: {args.mode.upper()}")

    dataset_name = get_name_from_path(args.input_dir)
    unique_suffix = "fixed" if args.mode == "synthesis" else "restored"

    temp_dir = os.path.join(args.out_dir, f"temp_{dataset_name}_{unique_suffix}")
    # 🟢 ARCHITECTURE PATCH: Both modes target the exact same output folder
    final_dir = os.path.join(args.out_dir, f"final_{dataset_name}_run")

    # Safely clear temp folder for current run
    if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    # 🟢 ARCHITECTURE PATCH: Strictly APPEND to the final folder, do NOT delete it.
    os.makedirs(final_dir, exist_ok=True)

    # --- CANDIDATE SELECTION LOGIC ---
    manifest_path = os.path.join(args.input_dir, "manifest.json")
    candidates = []
    good_candidates = []

    if os.path.exists(manifest_path):
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)

        if args.mode == "synthesis":
            candidates = [e for e in manifest if e.get("decision") in ["NOVEL_VIEW", "NONE"]]
            candidates.sort(key=lambda x: x.get("score", 0), reverse=True)
            candidates = candidates[:20]

        elif args.mode == "restoration":
            target_decisions = ["REPAIR", "BAD", "DISCARD", "blur"]
            candidates = [e for e in manifest if e.get("decision") in target_decisions]
            good_candidates = [e for e in manifest if e.get("decision") not in target_decisions]
            if not candidates:
                candidates = manifest
                good_candidates = []

    else:
        # 🟢 ARCHITECTURE PATCH: Included "input" in the fallback search paths
        search_path = os.path.join(args.input_dir, "images")
        if not os.path.exists(search_path):
            search_path = os.path.join(args.input_dir, "input")
            if not os.path.exists(search_path):
                search_path = args.input_dir

        raw_files = [f for f in os.listdir(search_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        candidates = [{"filename": f} for f in raw_files]

    print(f"✅ Selected {len(candidates)} candidates.")

    # --- PASS-THROUGH GOOD IMAGES (Restoration mode only) ---
    # Images whose decision is not REPAIR/BAD/DISCARD/blur are already fine.
    # Copy them straight into temp_dir as anchor_ files so the upscaler picks them up.
    if good_candidates:
        print(f"✅ Passing through {len(good_candidates)} good images as anchors...")
        for entry in good_candidates:
            filename = entry["filename"]
            raw_fallback = args.input_dir.replace("output_processed", "output_train")
            possible_paths = [
                os.path.join(args.input_dir, "processed_train", filename),
                os.path.join(raw_fallback, "train", filename),
                os.path.join(args.input_dir, "images", filename),
                os.path.join(args.input_dir, "input", filename),
                os.path.join(args.input_dir, filename)
            ]
            img_path = next((p for p in possible_paths if os.path.exists(p)), None)
            if not img_path:
                continue
            stem = os.path.splitext(filename)[0]
            shutil.copy2(img_path, os.path.join(temp_dir, f"anchor_{stem}.png"))

    # --- EXECUTION ---
    if len(candidates) > 0:
        if args.mode == "synthesis":
            run_synthesis_phase(args.input_dir, temp_dir, candidates)
        elif args.mode == "restoration":
            run_restoration_phase(args.input_dir, temp_dir, candidates, args.prompt)

        # Both modes run the Upscaler
        run_upscale_phase(temp_dir, final_dir)
    else:
        print("❌ No candidates found.")

    print(f"\n✅✅ DONE! (Mode: {args.mode})")
    print(f"📂 Output appended securely to: {final_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--mode", choices=["synthesis", "restoration"], default="synthesis",
                        help="Choose 'synthesis' for Zero-123 (V7) or 'restoration' for ControlNet.")
    parser.add_argument("--prompt", type=str, default="high quality photo, detailed, sharp focus, 8k",
                        help="Prompt for Restoration mode only.")

    args = parser.parse_args()
    main(args)
