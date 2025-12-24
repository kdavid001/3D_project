#!/usr/bin/env python3
"""
Step 1: load_images.py - Image quality screening with PyIQA (BRISQUE)

Usage:
    python load_images.py --input_dir ./raw_images --out_dir ./preprocessed --target_size 1024 --resize_mode pad

Decision Logic (BRISQUE Score 0-100, Lower is Better):
- Score > 21.0: BAD     -> Flag for REPAIR (Inpainting)
- Score < 15.0: PERFECT -> Flag for NOVEL_VIEW (Img2Img Variations)
- Score 15-21:  GOOD    -> Keep as is (NONE)
"""

import os
import json
import argparse
from PIL import Image, ImageOps
import numpy as np
from tqdm import tqdm
import torch
import pyiqa

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png"}

# -------------------------------------------------------------------------
# GLOBAL MODEL INITIALIZATION
# -------------------------------------------------------------------------
print("⏳ Loading PyIQA (BRISQUE) model...")
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

try:
    # Initialize metric once
    IQA_MODEL = pyiqa.create_metric('brisque', device=DEVICE)
    print(f"✅ Model loaded on {DEVICE}")
except Exception as e:
    print(f"❌ Error loading PyIQA model: {e}")
    exit(1)


def is_image_file(filename):
    return os.path.splitext(filename.lower())[1] in SUPPORTED_EXTS


def load_images_list(input_dir):
    files = sorted([
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if is_image_file(f)
    ])
    return files


def resize_image(img: Image.Image, target_size, mode="pad"):
    if mode == "none" or target_size is None:
        return img

    if isinstance(target_size, int):
        target_w = target_h = target_size
    else:
        target_w, target_h = target_size

    if mode == "stretch":
        return img.resize((target_w, target_h), Image.LANCZOS)

    w, h = img.size
    if mode == "pad":
        img.thumbnail((target_w, target_h), Image.LANCZOS)
        pad_w = target_w - img.width
        pad_h = target_h - img.height
        left = pad_w // 2
        top = pad_h // 2
        right = pad_w - left
        bottom = pad_h - top
        return ImageOps.expand(img, border=(left, top, right, bottom), fill=(0, 0, 0))
    elif mode == "crop":
        min_side = min(w, h)
        left = (w - min_side) // 2
        top = (h - min_side) // 2
        cropped = img.crop((left, top, left + min_side, top + min_side))
        return cropped.resize((target_w, target_h), Image.LANCZOS)
    else:
        raise ValueError("Unknown resize mode: " + str(mode))


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def get_image_score(image_path):
    """
    Computes the BRISQUE score.
    Returns: float (0.0 best - 100.0 worst)
    """
    with torch.no_grad():
        score = IQA_MODEL(image_path)
    return score.item()


def screen_image_brisque(score):
    """
    Decision Logic for Pipeline
    """
    # --- THRESHOLDS ---
    # > 21 is clearly bad.
    # < 15 is exceptionally high quality (candidates for novel view generation).
    BAD_QUALITY_THRESHOLD = 21.0
    PERFECT_QUALITY_THRESHOLD = 15.0

    flags = []
    decision = "NONE"

    if score > BAD_QUALITY_THRESHOLD:
        flags.append("poor_quality_brisque")
        decision = "REPAIR"
    elif score < PERFECT_QUALITY_THRESHOLD:
        flags.append("perfect_quality_brisque")
        decision = "NOVEL_VIEW"

    return {
        "needs_diffusion": decision != "NONE",
        "flags": flags,
        "decision": decision,
        "score": score
    }


def process_images(args):
    """Main processing function"""
    input_dir = args.input_dir
    out_dir = args.out_dir
    target_size = args.target_size
    resize_mode = args.resize_mode

    potential_dir = os.path.join(input_dir, "images")
    if os.path.exists(potential_dir):
        input_images_dir = potential_dir
    else:
        input_images_dir = input_dir

    ensure_dir(out_dir)
    images_out_dir = os.path.join(out_dir, "images")
    ensure_dir(images_out_dir)

    files = load_images_list(input_images_dir)
    if not files:
        print(f"No images found in {input_images_dir}. Supported: {SUPPORTED_EXTS}")
        return

    manifest = {"images": []}

    print(f"Processing {len(files)} images...")

    for path in tqdm(files, desc="Analyzing"):
        base = os.path.basename(path)
        try:
            # 1. Load and Resize
            img = Image.open(path).convert("RGB")
            proc_img = resize_image(img, target_size, resize_mode) if target_size else img

            # 2. Save processed image
            out_path = os.path.join(images_out_dir, base)
            proc_img.save(out_path)

            # 3. Compute Score
            brisque_score = get_image_score(out_path)

            # 4. Make Decision
            screening_result = screen_image_brisque(brisque_score)

            entry = {
                "filename": base,
                "quality_score": brisque_score,
                "screening": screening_result
            }

            manifest["images"].append(entry)

        except Exception as e:
            print(f"Error processing {base}: {e}")
            continue

    # Count stats
    num_repair = sum(1 for e in manifest["images"] if e["screening"]["decision"] == "REPAIR")
    num_novel = sum(1 for e in manifest["images"] if e["screening"]["decision"] == "NOVEL_VIEW")
    num_none = len(manifest["images"]) - num_repair - num_novel

    print(f"\n{'=' * 60}")
    print(f"Screening Results (BRISQUE Metric):")
    print(f"  Total images: {len(manifest['images'])}")
    print(f"  REPAIR (> 21.0):      {num_repair} (Bad quality)")
    print(f"  NOVEL_VIEW (< 15.0):  {num_novel}  (Perfect quality)")
    print(f"  NONE (15.0 - 21.0):   {num_none}   (Standard quality)")
    print(f"{'=' * 60}\n")

    # Save manifest
    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"✓ Manifest saved to: {manifest_path}")
    print(f"✓ Processed images saved to: {images_out_dir}")

    if num_repair > 0:
        print(f"\n[!] Run prepare_masks.py to generate masks for the {num_repair} REPAIR images.")

    if num_novel > 0:
        print(f"[!] {num_novel} images marked for NOVEL_VIEW generation.")

    return manifest


def test_single_image(image_path, target_size=None, resize_mode="pad"):
    """
    Test a single image with PyIQA
    """
    print(f"\n{'=' * 60}")
    print(f"SINGLE IMAGE ANALYSIS (PyIQA BRISQUE)")
    print(f"{'=' * 60}")
    print(f"Image: {image_path}\n")

    try:
        img = Image.open(image_path).convert("RGB")
        print(f"Original Size: {img.size}")

        proc_img = resize_image(img, target_size, resize_mode) if target_size else img

        # Temp save for inference
        temp_path = "temp_test_image_iqa.png"
        proc_img.save(temp_path)

        score = get_image_score(temp_path)

        if os.path.exists(temp_path):
            os.remove(temp_path)

        result = screen_image_brisque(score)

        print("\n" + "-" * 60)
        print(f"QUALITY SCORE: {score:.4f}")
        print("-" * 60)
        print("Scale: 0 (Best) to 100 (Worst)")
        print("Thresholds: <15 (Perfect), >21 (Bad)")

        print(f"\nDecision: {result['decision']}")

        if result['decision'] == "REPAIR":
            print("❌ Status: BAD QUALITY")
            print("   Action: Needs Diffusion Repair")
        elif result['decision'] == "NOVEL_VIEW":
            print("✨ Status: PERFECT QUALITY")
            print("   Action: Generate Novel Views")
        else:
            print("✅ Status: GOOD")
            print("   Action: Keep as is")

        print(f"\n{'=' * 60}\n")

    except Exception as e:
        print(f"❌ Error analyzing image: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="Image quality screening using PyIQA")
    parser.add_argument("--input_dir", help="Folder with raw images")
    parser.add_argument("--out_dir", help="Folder to write output")
    parser.add_argument("--target_size", type=int, default=None, help="Resize to square size (e.g. 1024)")
    parser.add_argument("--resize_mode", choices=["pad", "crop", "stretch", "none"], default="pad")

    parser.add_argument("--test", action="store_true", help="Test mode: analyze a single image")
    parser.add_argument("--test_image", help="Path to single image to test")

    args = parser.parse_args()

    if args.test:
        if not args.test_image:
            print("❌ --test_image required when using --test")
            return
        test_single_image(args.test_image, args.target_size, args.resize_mode)
    else:
        if not args.input_dir or not args.out_dir:
            print("❌ --input_dir and --out_dir required for batch processing")
            return
        process_images(args)


if __name__ == "__main__":
    main()