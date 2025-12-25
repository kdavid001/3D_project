#!/usr/bin/env python3
"""
Step 2: prepare_masks.py - Auto-generate masks from BRISQUE & Structural Flags

Usage:
    # Test single image:
    python prepare_masks.py --manifest ./preprocessed/manifest.json --test --test_image image1.jpg

    # Process all flagged images:
    python prepare_masks.py --manifest ./preprocessed/manifest.json --output ./inpainting_ready

What it does:
- Reads manifest.json from Step 1
- Handles 'poor_quality_brisque' flags
- Hybrid Strategy: Uses local detectors (blur/texture) to find defects
- Fallback: If no defects found but score is bad -> Masks entire image
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


# --- LOCAL DETECTORS (The "Locators") ---
def blur_mask(gray, thresh=150):
    """Finds blurry regions (laplacian variance < thresh)"""
    # Gaussian blur to remove noise before checking edges
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    lap = cv2.Laplacian(gray_blur, cv2.CV_32F)
    # Regions with low variance (few edges) are candidates
    # We invert the magnitude: low edge strength -> high mask value
    mag = np.abs(lap)
    mask = mag < (thresh / 255.0)  # Normalizing threshold approx
    return mask.astype(np.uint8) * 255


def low_texture_mask(gray, std_thresh=0.02, window_size=15):
    """Finds flat/low-texture regions"""
    mean = cv2.GaussianBlur(gray, (window_size, window_size), 0)
    sq_mean = cv2.GaussianBlur(gray ** 2, (window_size, window_size), 0)
    local_var = sq_mean - mean ** 2
    local_var = np.maximum(local_var, 0)
    mask = local_var < std_thresh
    return mask.astype(np.uint8) * 255


def saturation_mask(img_rgb, low=0.1, high=0.9):
    """Finds clipped highlights or crushed shadows"""
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1] / 255.0
    val = hsv[:, :, 2] / 255.0
    # Mask areas with extreme saturation or extreme brightness
    mask = (sat > high) | (val > 0.95) | (val < 0.05)
    return mask.astype(np.uint8) * 255


def full_image_mask(shape):
    """Returns a white mask covering the whole image"""
    return np.ones((shape[0], shape[1]), dtype=np.uint8) * 255


def clean_mask(mask, min_area=500):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    clean = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean[labels == i] = 255
    return clean


def generate_hybrid_mask(img_rgb, flags, blur_thresh=100, texture_thresh=0.02):
    """
    HYBRID LOGIC:
    1. If 'poor_quality_brisque' is set:
       - Run ALL local detectors (blur, texture, saturation) to find defects.
       - Combine them.
       - If combined mask is empty -> Fallback to FULL MASK.
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    masks = []

    print(f"    - Processing flags: {flags}")

    # Check for the global BRISQUE flag
    is_brisque_bad = "poor_quality_brisque" in flags

    # 1. Run Local Detectors
    # We run these if specifically flagged OR if BRISQUE is bad (to find *where* it's bad)

    # Check Blur
    if is_brisque_bad or "blur" in flags:
        print(f"    - Running blur detector...")
        masks.append(blur_mask(gray, thresh=blur_thresh))

    # Check Texture/Noise
    if is_brisque_bad or "low_texture_ratio" in flags:
        print(f"    - Running texture detector...")
        masks.append(low_texture_mask(gray, std_thresh=texture_thresh))

    # Check Saturation/Exposure
    if is_brisque_bad or "saturation_ratio" in flags:
        print(f"    - Running saturation detector...")
        masks.append(saturation_mask(img_rgb))

    # 2. Combine Masks
    if not masks:
        final_mask = np.zeros_like(gray, dtype=np.uint8)
    else:
        final_mask = np.maximum.reduce(masks)
        final_mask = clean_mask(final_mask, min_area=1000)

    # 3. BRISQUE Fallback Strategy
    # If BRISQUE says it's bad, but our detectors found nothing (empty mask),
    # it likely means the issue is subtle or global (like compression artifacts).
    # In this case, we mask the WHOLE image to force regeneration.
    coverage = np.sum(final_mask > 0) / final_mask.size

    if is_brisque_bad and coverage < 0.10:  # If less than 10% covered
        print("    ⚠️  Local detectors found little/no defects, but BRISQUE score is bad.")
        print("    -> Falling back to FULL IMAGE MASK for global repair.")
        final_mask = full_image_mask(gray.shape)

    return final_mask


def process_images(args):
    """Main Loop"""
    if not os.path.exists(args.manifest):
        print(f"❌ Manifest not found: {args.manifest}")
        return

    with open(args.manifest, 'r') as f:
        manifest = json.load(f)

    # Filter for REPAIR images
    flagged_images = [img for img in manifest['images']
                      if img.get('screening', {}).get('decision') == "REPAIR"]

    print(f"Found {len(flagged_images)} images flagged for REPAIR.")

    # Setup directories
    out_img_dir = os.path.join(args.output, "masked_images")
    out_mask_dir = os.path.join(args.output, "masks")
    out_vis_dir = os.path.join(args.output, "visualizations")
    for d in [out_img_dir, out_mask_dir, out_vis_dir]:
        os.makedirs(d, exist_ok=True)

    results = []

    for entry in tqdm(flagged_images):
        try:
            filename = entry['filename']
            # Locate image (check both root and /images subfolder)
            base_dir = os.path.dirname(args.manifest)
            possible_paths = [
                os.path.join(base_dir, "images", filename),
                os.path.join(base_dir, filename)
            ]
            img_path = next((p for p in possible_paths if os.path.exists(p)), None)

            if not img_path:
                print(f"Skipping missing image: {filename}")
                continue

            # Load
            img_pil = Image.open(img_path).convert("RGB")
            img_pil = resize_to_multiple_of_8(img_pil)
            img_np = np.array(img_pil)

            # Generate Mask
            mask = generate_hybrid_mask(
                img_np,
                entry['screening']['flags'],
                blur_thresh=args.blur_thresh,
                texture_thresh=args.texture_thresh
            )

            # Save
            mask_pil = Image.fromarray(mask)

            # 1. Save Mask (Binary)
            mask_processed = prepare_mask_for_inpainting(mask_pil)
            mask_processed.save(os.path.join(out_mask_dir, filename))

            # 2. Save Masked Image (Gray filled)
            masked_img_pil = create_masked_image(img_pil, mask_processed)
            masked_img_pil.save(os.path.join(out_img_dir, filename))

            # 3. Visualization
            vis_pil = visualize_mask_overlay(img_pil, mask_processed)
            vis_pil.save(os.path.join(out_vis_dir, filename))

            results.append({
                "filename": filename,
                "mask_path": os.path.join("masks", filename),
                "masked_image_path": os.path.join("masked_images", filename)
            })

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    # Save output manifest
    with open(os.path.join(args.output, "inpainting_manifest.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Processing complete. {len(results)} masks generated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="Path to manifest.json")
    parser.add_argument("--output", default="./inpainting_ready", help="Output folder")
    parser.add_argument("--blur_thresh", type=int, default=120, help="Sensitivity for blur")
    parser.add_argument("--texture_thresh", type=float, default=0.01, help="Sensitivity for texture")

    # Test single mode support
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--test_image")

    args = parser.parse_args()

    if args.test and args.test_image:
        # Simple shim for test mode
        class MockArgs:
            manifest = args.manifest
            output = "./test_output"
            blur_thresh = args.blur_thresh
            texture_thresh = args.texture_thresh


        process_images(MockArgs())
    else:
        process_images(args)