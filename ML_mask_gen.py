#!/usr/bin/env python3
"""
Step 2: prepare_masks.py - Auto-generate masks using Restoration Difference

Usage:
    python prepare_masks.py --manifest ./preprocessed/manifest.json --output ./inpainting_ready --sensitivity 30

Method: "Restoration Difference"
1. Creates a "Clean" version of the image using strong denoising/smoothing.
2. Subtracts Clean from Original to find "Defects" (Noise, Artifacts, Blur).
3. Thresholds this difference to create the Mask.
"""

import os
import json
import argparse
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm


def prepare_mask_for_inpainting(mask_pil):
    """Prepare mask for SD inpainting model (binary)"""
    mask = np.array(mask_pil).astype("float32") / 255.0
    mask_binary = (mask > 0.5).astype("float32")
    return Image.fromarray((mask_binary * 255).astype("uint8"))


def create_masked_image(image_pil, mask_pil, fill_mode="gray"):
    """Create masked image for SD inpainting conditioning"""
    image = np.array(image_pil).astype("float32") / 255.0
    mask = np.array(mask_pil).astype("float32") / 255.0

    mask_binary = (mask > 0.5).astype("float32")
    mask_3c = np.repeat(mask_binary[:, :, None], 3, axis=2)

    # Fill mode logic
    if fill_mode == "gray":
        fill_value = 0.5
    elif fill_mode == "noise":
        fill_value = np.random.rand(*image.shape)
    else:
        fill_value = 0.5

    masked_image = image * (1.0 - mask_3c) + fill_value * mask_3c
    masked_image = np.clip(masked_image, 0.0, 1.0)
    return Image.fromarray((masked_image * 255).astype("uint8"))


def visualize_mask_overlay(image_pil, mask_pil, alpha=0.6):
    """Red overlay visualization"""
    image = np.array(image_pil).astype("float32") / 255.0
    mask = np.array(mask_pil).astype("float32") / 255.0
    mask_binary = (mask > 0.5).astype("float32")

    overlay = image.copy()
    overlay[:, :, 0] = np.clip(overlay[:, :, 0] + mask_binary * 0.7, 0, 1)

    result = image * (1 - alpha) + overlay * alpha
    return Image.fromarray((np.clip(result, 0, 1) * 255).astype("uint8"))


def resize_to_multiple_of_8(pil_img):
    w, h = pil_img.size
    new_w = w - (w % 8)
    new_h = h - (h % 8)
    if new_w == w and new_h == h: return pil_img
    return pil_img.resize((new_w, new_h), Image.BICUBIC)


# -------------------------------------------------------------------------
# THE CORE ALGORITHM: RESTORATION DIFFERENCE
# -------------------------------------------------------------------------
def generate_restoration_mask(img_rgb, sensitivity=25, expand_iterations=4):
    """
    Generates a mask by comparing the original image to a "Restored" version.

    Args:
        img_rgb: Input image (numpy array)
        sensitivity: Difference threshold (Lower = More sensitive/Larger Mask)
        expand_iterations: How much to grow the mask to cover edges
    """
    # 1. Create "Restored" Reference
    # We use Non-Local Means Denoising to act as our "Blind Restoration" model.
    # It aggressively smooths noise and textures while trying to keep edges.
    # h=10 is strength (higher = smoother = bigger difference = bigger mask)
    clean = cv2.fastNlMeansDenoisingColored(img_rgb, None, 10, 10, 7, 21)

    # 2. Calculate Difference (Artifact Map)
    # This highlights Grain, JPEG blocks, and Sensor Noise
    diff = cv2.absdiff(img_rgb, clean)

    # Convert difference to grayscale intensity
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)

    # 3. Create Binary Mask
    # If the difference is high (pixel was noisy), mark it white.
    # sensitivity is the cutoff (0-255).
    _, mask = cv2.threshold(diff_gray, sensitivity, 255, cv2.THRESH_BINARY)

    # 4. Clean Up (Morphology)
    # Remove tiny speckles (noise that is too small to care about)
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)

    # Fill small holes inside big blobs
    kernel_med = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_med)

    # 5. Expand (Dilate)
    # Noise/Blur usually has a "halo" around it. We expand the mask to catch it.
    if expand_iterations > 0:
        kernel_expand = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask, kernel_expand, iterations=expand_iterations)

    return mask


def process_images(args):
    """Main Loop with Deduplication"""
    if not os.path.exists(args.manifest):
        print(f"❌ Manifest not found: {args.manifest}")
        return

    with open(args.manifest, 'r') as f:
        manifest = json.load(f)

    # Filter for REPAIR images
    flagged_images = [img for img in manifest['images']
                      if img.get('screening', {}).get('decision') == "REPAIR"]

    # Deduplicate (Keep the one with most info/flags if duplicates exist)
    unique_images = {}
    for entry in flagged_images:
        unique_images[entry['filename']] = entry
    processing_queue = list(unique_images.values())

    print(f"\n{'=' * 60}")
    print(f"Processing {len(processing_queue)} images using Restoration Difference")
    print(f"Sensitivity: {args.sensitivity} (Lower = More Aggressive)")
    print(f"{'=' * 60}\n")

    # Setup directories
    out_img_dir = os.path.join(args.output, "masked_images")
    out_mask_dir = os.path.join(args.output, "masks")
    out_vis_dir = os.path.join(args.output, "visualizations")
    for d in [out_img_dir, out_mask_dir, out_vis_dir]:
        os.makedirs(d, exist_ok=True)

    results = []
    processed = 0

    for entry in tqdm(processing_queue, desc="Generating masks"):
        try:
            filename = entry['filename']
            # Force PNG output
            base_name = os.path.splitext(filename)[0]
            out_filename = f"{base_name}.png"

            # Locate image
            base_dir = os.path.dirname(args.manifest)
            possible_paths = [
                os.path.join(base_dir, "images", filename),
                os.path.join(base_dir, filename)
            ]
            img_path = next((p for p in possible_paths if os.path.exists(p)), None)

            if not img_path: continue

            # Load
            img_pil = Image.open(img_path).convert("RGB")
            img_pil = resize_to_multiple_of_8(img_pil)
            img_np = np.array(img_pil)

            # --- GENERATE MASK (New Method) ---
            mask = generate_restoration_mask(
                img_np,
                sensitivity=args.sensitivity,
                expand_iterations=args.expand
            )

            # --- FALLBACK CHECK ---
            # If mask is empty but BRISQUE complained, perform global fallback?
            # Or just lower sensitivity?
            coverage = np.sum(mask > 0) / mask.size
            if coverage < 0.05:  # Less than 5% coverage
                print(f"\n  ⚠️ Low coverage ({coverage:.1%}) for {filename}. Boosting sensitivity...")
                # Retry with higher aggression (lower threshold)
                mask = generate_restoration_mask(img_np, sensitivity=args.sensitivity - 10,
                                                 expand_iterations=args.expand + 2)

            # Save
            mask_pil = Image.fromarray(mask)
            mask_processed = prepare_mask_for_inpainting(mask_pil)
            mask_processed.save(os.path.join(out_mask_dir, out_filename))

            masked_img_pil = create_masked_image(img_pil, mask_processed)
            masked_img_pil.save(os.path.join(out_img_dir, out_filename))

            vis_pil = visualize_mask_overlay(img_pil, mask_processed)
            vis_pil.save(os.path.join(out_vis_dir, out_filename))

            results.append({
                "filename": out_filename,
                "mask_path": os.path.join("masks", out_filename),
                "masked_image_path": os.path.join("masked_images", out_filename),
                "flags": entry['screening']['flags']
            })
            processed += 1

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    # Save output manifest
    with open(os.path.join(args.output, "inpainting_manifest.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Done! {processed} masks generated.")
    print(f"Outputs at: {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="Path to manifest.json")
    parser.add_argument("--output", default="./inpainting_ready", help="Output folder")
    # Sensitivity: Lower number (e.g. 15) = detect subtle noise. Higher (e.g. 50) = only heavy artifacts.
    parser.add_argument("--sensitivity", type=int, default=30, help="Diff threshold (0-255). Lower = Bigger Mask.")
    parser.add_argument("--expand", type=int, default=4, help="Dilate iterations")

    args = parser.parse_args()
    process_images(args)