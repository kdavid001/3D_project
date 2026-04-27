#!/usr/bin/env python3
"""
Step 2 (METADATA FIX): prepare_masks.py

Update:
- Now copies 'decision', 'score', and 'note' to the output manifest.
- This ensures Step 3 knows exactly what to do without guessing.
"""

import os
import json
import re
import argparse
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm

# --- CONFIGURATION ---
SAT_THRESHOLD = 240
EXP_MAX_THRESHOLD = 245
EXP_MIN_THRESHOLD = 15


def prepare_mask_for_inpainting(mask_pil):
    mask = np.array(mask_pil).astype("float32") / 255.0
    mask_binary = (mask > 0.5).astype("float32")
    return Image.fromarray((mask_binary * 255).astype("uint8"))


def create_masked_image(image_pil, mask_pil):
    image = np.array(image_pil).astype("float32") / 255.0
    mask = np.array(mask_pil).astype("float32") / 255.0
    mask_binary = (mask > 0.5).astype("float32")
    mask_3c = np.repeat(mask_binary[:, :, None], 3, axis=2)
    fill_value = 0.5
    masked_image = image * (1.0 - mask_3c) + fill_value * mask_3c
    masked_image = np.clip(masked_image, 0.0, 1.0)
    return Image.fromarray((masked_image * 255).astype("uint8"))


def resize_to_multiple_of_8(pil_img):
    w, h = pil_img.size
    new_w = w - (w % 8)
    new_h = h - (h % 8)
    if new_w == w and new_h == h: return pil_img
    return pil_img.resize((new_w, new_h), Image.BICUBIC)


def generate_hybrid_mask(img_rgb, sensitivity=30, expand_iterations=4):
    # 1. Restoration
    clean = cv2.fastNlMeansDenoisingColored(img_rgb, None, 10, 10, 7, 21)
    diff = cv2.absdiff(img_rgb, clean)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
    _, mask_restoration = cv2.threshold(diff_gray, sensitivity, 255, cv2.THRESH_BINARY)

    # 2. Radiometric
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    _, mask_sat = cv2.threshold(s, SAT_THRESHOLD, 255, cv2.THRESH_BINARY)
    _, mask_bright = cv2.threshold(v, EXP_MAX_THRESHOLD, 255, cv2.THRESH_BINARY)
    _, mask_dark = cv2.threshold(v, EXP_MIN_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    mask_radio = cv2.bitwise_or(mask_sat, mask_bright)
    mask_radio = cv2.bitwise_or(mask_radio, mask_dark)

    # 3. Combine
    final_mask = cv2.bitwise_or(mask_restoration, mask_radio)

    # 4. Background Exclusion (Ignore Black Void)
    lower_black = np.array([0, 0, 0], dtype=np.uint8)
    upper_black = np.array([2, 2, 2], dtype=np.uint8)
    bg_mask = cv2.inRange(img_rgb, lower_black, upper_black)
    final_mask = cv2.bitwise_and(final_mask, cv2.bitwise_not(bg_mask))

    # 5. Cleanup
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel_small)
    kernel_med = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel_med)

    if expand_iterations > 0:
        kernel_expand = np.ones((3, 3), np.uint8)
        final_mask = cv2.dilate(final_mask, kernel_expand, iterations=expand_iterations)

    return final_mask


def process_images(args):
    if not os.path.exists(args.manifest):
        print(f"❌ Manifest not found: {args.manifest}")
        return

    with open(args.manifest, 'r') as f:
        manifest = json.load(f)

    # Filter for REPAIR images
    flagged_images = [img for img in manifest if img.get('decision') == "REPAIR"]

    if not flagged_images:
        print("✅ No images marked for REPAIR. Exiting.")
        return

    print(f"\n{'=' * 60}")
    print(f"Processing {len(flagged_images)} flagged images...")
    print(f"{'=' * 60}\n")
    out_img_dir = os.path.join(args.output, "masked_images")
    out_mask_dir = os.path.join(args.output, "masks")
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_mask_dir, exist_ok=True)

    results = []

    for entry in tqdm(flagged_images, desc="Generating masks"):
        try:
            filename = entry['filename']
            base_name = os.path.splitext(filename)[0]
            out_filename = f"{base_name}.png"

            # Find Source Image
            base_dir = os.path.dirname(args.manifest)
            possible_paths = [
                os.path.join(base_dir, "processed_train", filename),
                os.path.join(base_dir, filename)
            ]
            img_path = next((p for p in possible_paths if os.path.exists(p)), None)

            if not img_path: continue

            # Generate Mask
            img_pil = Image.open(img_path).convert("RGB")
            img_pil = resize_to_multiple_of_8(img_pil)
            img_np = np.array(img_pil)

            mask = generate_hybrid_mask(img_np, sensitivity=args.sensitivity, expand_iterations=args.expand)

            # Fallback for empty masks
            if np.sum(mask > 0) / mask.size < 0.01:
                mask = generate_hybrid_mask(img_np, sensitivity=10, expand_iterations=6)

            # Save
            mask_pil = Image.fromarray(mask)
            mask_processed = prepare_mask_for_inpainting(mask_pil)
            mask_processed.save(os.path.join(out_mask_dir, out_filename))

            masked_img_pil = create_masked_image(img_pil, mask_processed)
            masked_img_pil.save(os.path.join(out_img_dir, out_filename))

            # --- METADATA PASS-THROUGH ---
            results.append({
                "filename": out_filename,
                "original_filename": filename,
                "mask_path": os.path.join("masks", out_filename),
                "decision": entry.get("decision", "REPAIR"),  # <--- PASSING DECISION
                "score": entry.get("score", 0),  # <--- PASSING SCORE
                "note": entry.get("note", "")
            })

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    with open(os.path.join(args.output, "inpainting_manifest.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Done! Generated inpainting_manifest.json with {len(results)} entries.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sensitivity", type=int, default=30)
    parser.add_argument("--expand", type=int, default=4)
    args = parser.parse_args()
    process_images(args)