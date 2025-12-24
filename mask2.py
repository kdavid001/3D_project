#!/usr/bin/env python3
"""
Step 2: prepare_masks.py - Generate masks and prepare for SD inpainting

Usage:
    # Test single image first:
    python prepare_masks.py --manifest ./preprocessed/manifest.json --test --test_image image1.jpg

    # Process all flagged images:
    python prepare_masks.py --manifest ./preprocessed/manifest.json --output ./inpainting_ready

What it does:
- Reads manifest.json from step 1
- Generates masks ONLY for images flagged as needing repair
- Improves mask quality with cleanup and expansion
- Creates masked images for SD inpainting
- Outputs ready-to-use dataset for diffusion model
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
    elif fill_mode == "black":
        fill_value = 0.0
    elif fill_mode == "white":
        fill_value = 1.0
    elif fill_mode == "noise":
        fill_value = np.random.rand(*image.shape)
    else:
        fill_value = 0.5

    if fill_mode == "noise":
        masked_image = image * (1.0 - mask_3c) + fill_value * mask_3c
    else:
        masked_image = image * (1.0 - mask_3c) + fill_value * mask_3c

    masked_image = np.clip(masked_image, 0.0, 1.0)
    return Image.fromarray((masked_image * 255).astype("uint8"))


def visualize_mask_overlay(image_pil, mask_pil, alpha=0.6):
    """Create visualization with red overlay on masked regions"""
    image = np.array(image_pil).astype("float32") / 255.0
    mask = np.array(mask_pil).astype("float32") / 255.0
    mask_binary = (mask > 0.5).astype("float32")

    overlay = image.copy()
    overlay[:, :, 0] = np.clip(overlay[:, :, 0] + mask_binary * 0.7, 0, 1)

    result = image * (1 - alpha) + overlay * alpha
    result = np.clip(result, 0, 1)

    return Image.fromarray((result * 255).astype("uint8"))


def resize_to_multiple_of_8(pil_img):
    """Resize image to dimensions divisible by 8 (required for VAE)"""
    w, h = pil_img.size
    new_w = w - (w % 8)
    new_h = h - (h % 8)
    if new_w == w and new_h == h:
        return pil_img
    return pil_img.resize((new_w, new_h), Image.BICUBIC)


# IMPROVED MASK GENERATION FUNCTIONS
def blur_mask(gray, thresh=100):
    """Detect blurry regions - HIGHER threshold = less sensitive"""
    gray_uint8 = (gray * 255).astype(np.uint8)
    gray_uint8 = cv2.GaussianBlur(gray_uint8, (5, 5), 0)

    lap = cv2.Laplacian(gray_uint8, cv2.CV_32F)
    mag = np.abs(lap)

    mask = mag < thresh
    return mask.astype(np.uint8) * 255


def low_texture_mask(gray, std_thresh=0.02, window_size=15):
    """Detect low texture regions - HIGHER threshold = less sensitive"""
    mean = cv2.GaussianBlur(gray, (window_size, window_size), 0)
    sq_mean = cv2.GaussianBlur(gray ** 2, (window_size, window_size), 0)
    local_var = sq_mean - mean ** 2
    local_var = np.maximum(local_var, 0)

    mask = local_var < std_thresh
    return mask.astype(np.uint8) * 255


def saturation_mask(gray, low=0.02, high=0.98):
    """Detect over/under saturated regions"""
    mask = (gray < low) | (gray > high)
    return mask.astype(np.uint8) * 255


def clean_mask(mask, min_area=500):
    """Remove small disconnected regions"""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    clean = np.zeros_like(mask)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            clean[labels == i] = 255

    return clean


def morphological_cleanup(mask, close_kernel=7, open_kernel=5):
    """Apply morphological operations to smooth mask"""
    kernel_close = np.ones((close_kernel, close_kernel), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    kernel_open = np.ones((open_kernel, open_kernel), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)

    return mask


def expand_mask_boundaries(mask, iterations=3):
    """Expand mask boundaries to ensure full coverage"""
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(mask, kernel, iterations=iterations)
    return dilated


def generate_repair_mask(
        img_rgb,
        flags,
        blur_thresh=100,
        texture_thresh=0.02,
        min_region_area=500,
        expand_iterations=3
):
    """
    Generate improved mask for inpainting.

    Key improvements:
    - MUCH higher thresholds (less sensitive)
    - Better cleanup (remove noise)
    - Moderate expansion (ensure coverage)
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    masks = []

    if "blur" in flags:
        masks.append(blur_mask(gray, thresh=blur_thresh))

    if "low_texture_ratio" in flags:
        masks.append(low_texture_mask(gray, std_thresh=texture_thresh))

    if "saturation_ratio" in flags:
        masks.append(saturation_mask(gray))

    if not masks:
        return np.zeros_like(gray, dtype=np.uint8)

    # Combine masks
    final_mask = np.maximum.reduce(masks)

    # Clean up
    final_mask = clean_mask(final_mask, min_area=min_region_area)
    final_mask = morphological_cleanup(final_mask, close_kernel=7, open_kernel=5)

    # Expand boundaries
    if expand_iterations > 0:
        final_mask = expand_mask_boundaries(final_mask, iterations=expand_iterations)

    # Ensure binary
    final_mask = (final_mask > 127).astype(np.uint8) * 255

    return final_mask


def test_single_image(manifest_path, image_filename, output_dir="./mask_test"):
    """Test mask generation on a single image"""
    os.makedirs(output_dir, exist_ok=True)

    # Load manifest
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    # Find image entry
    entry = None
    for e in manifest["images"]:
        if e["filename"] == image_filename:
            entry = e
            break

    if entry is None:
        print(f"❌ Image {image_filename} not found in manifest")
        return

    if entry["screening"]["decision"] != "REPAIR":
        print(f"⚠️  Image {image_filename} is not flagged for REPAIR")
        print(f"   Decision: {entry['screening']['decision']}")
        print(f"   Flags: {entry['screening']['flags']}")
        return

    # Get paths
    base_dir = os.path.dirname(manifest_path)
    image_path = os.path.join(base_dir, "images", entry["filename"])

    print(f"\n{'=' * 60}")
    print(f"Testing mask generation for: {image_filename}")
    print(f"Flags: {entry['screening']['flags']}")
    print(f"{'=' * 60}\n")

    # Load image
    image = Image.open(image_path).convert("RGB")
    image = resize_to_multiple_of_8(image)
    img_rgb = np.array(image)

    # Test different thresholds
    print("Testing different blur thresholds:")
    for blur_t in [50, 100, 150, 200]:
        mask = generate_repair_mask(
            img_rgb,
            entry['screening']['flags'],
            blur_thresh=blur_t,
            texture_thresh=0.02,
            min_region_area=500,
            expand_iterations=3
        )

        coverage = np.sum(mask > 127) / mask.size * 100
        status = "✓ GOOD" if coverage < 30 else ("○ OK" if coverage < 50 else "⚠️  TOO HIGH")
        print(f"  blur_thresh={blur_t:3d}: {coverage:5.1f}% coverage  {status}")

        Image.fromarray(mask).save(os.path.join(output_dir, f"mask_blur{blur_t}.png"))

        # Save visualization
        mask_pil = Image.fromarray(mask)
        vis = visualize_mask_overlay(image, mask_pil, alpha=0.6)
        vis.save(os.path.join(output_dir, f"overlay_blur{blur_t}.png"))

    print(f"\n✓ Test outputs saved to: {output_dir}")
    print("Review the overlay images - red shows what will be inpainted")


def process_all_images(manifest_path, output_dir, blur_thresh=100, texture_thresh=0.02):
    """Process all flagged images and generate masks"""

    # Load manifest
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    base_dir = os.path.dirname(manifest_path)
    images_dir = os.path.join(base_dir, "images")

    # Create output directories
    masked_images_dir = os.path.join(output_dir, "masked_images")
    masks_dir = os.path.join(output_dir, "masks")
    vis_dir = os.path.join(output_dir, "visualizations")

    os.makedirs(masked_images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    # Filter to only REPAIR images
    repair_images = [e for e in manifest["images"] if e["screening"]["decision"] == "REPAIR"]

    if not repair_images:
        print("❌ No images flagged for REPAIR in manifest")
        return

    print(f"\n{'=' * 60}")
    print(f"Processing {len(repair_images)} images flagged for REPAIR")
    print(f"Blur threshold: {blur_thresh}")
    print(f"Texture threshold: {texture_thresh}")
    print(f"{'=' * 60}\n")

    output_manifest = []
    processed_count = 0
    skipped_count = 0

    for entry in tqdm(repair_images, desc="Generating masks"):
        filename = entry["filename"]
        image_path = os.path.join(images_dir, filename)

        try:
            # Load image
            image = Image.open(image_path).convert("RGB")
            image = resize_to_multiple_of_8(image)
            img_rgb = np.array(image)

            # Generate mask
            mask = generate_repair_mask(
                img_rgb,
                entry['screening']['flags'],
                blur_thresh=blur_thresh,
                texture_thresh=texture_thresh,
                min_region_area=500,
                expand_iterations=3
            )

            # Check coverage
            coverage = np.sum(mask > 127) / mask.size

            if coverage == 0:
                print(f"⚠️  Skipping {filename}: empty mask")
                skipped_count += 1
                continue

            if coverage > 0.8:
                print(f"⚠️  Skipping {filename}: {coverage * 100:.1f}% coverage (too high)")
                skipped_count += 1
                continue

            # Prepare mask and masked image
            mask_pil = Image.fromarray(mask)
            mask_processed = prepare_mask_for_inpainting(mask_pil)
            masked_image = create_masked_image(image, mask_processed, fill_mode="gray")

            # Save outputs
            base_name = os.path.splitext(filename)[0]
            masked_image.save(os.path.join(masked_images_dir, f"{base_name}.png"))
            mask_processed.save(os.path.join(masks_dir, f"{base_name}.png"))

            # Visualization
            vis = visualize_mask_overlay(image, mask_processed, alpha=0.6)
            vis.save(os.path.join(vis_dir, f"{base_name}_overlay.png"))

            # Add to output manifest
            output_manifest.append({
                "filename": filename,
                "original_image": image_path,
                "masked_image": os.path.join(masked_images_dir, f"{base_name}.png"),
                "mask": os.path.join(masks_dir, f"{base_name}.png"),
                "flags": entry['screening']['flags'],
                "mask_coverage": float(coverage)
            })

            processed_count += 1

        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")
            skipped_count += 1
            continue

    # Save output manifest
    output_manifest_path = os.path.join(output_dir, "inpainting_manifest.json")
    with open(output_manifest_path, 'w') as f:
        json.dump(output_manifest, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"✓ Processing complete!")
    print(f"  Processed: {processed_count} images")
    print(f"  Skipped: {skipped_count} images")
    print(f"\nOutputs:")
    print(f"  - Masked images: {masked_images_dir}")
    print(f"  - Masks: {masks_dir}")
    print(f"  - Visualizations: {vis_dir}")
    print(f"  - Manifest: {output_manifest_path}")

    if processed_count > 0:
        avg_coverage = np.mean([m['mask_coverage'] for m in output_manifest])
        print(f"\nAverage mask coverage: {avg_coverage * 100:.1f}%")

    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(description="Generate masks for flagged images")
    parser.add_argument("--manifest", required=True, help="Path to manifest.json from step 1")
    parser.add_argument("--output", default="./inpainting_ready", help="Output directory")
    parser.add_argument("--test", action="store_true", help="Test mode: generate masks for one image")
    parser.add_argument("--test_image", help="Filename to test (requires --test)")
    parser.add_argument("--blur_thresh", type=int, default=60,
                        help="Blur threshold (higher = less sensitive)")
    parser.add_argument("--texture_thresh", type=float, default=0.02,
                        help="Texture threshold (higher = less sensitive)")

    args = parser.parse_args()

    if args.test:
        if not args.test_image:
            print("❌ --test_image required when using --test")
            return
        test_single_image(args.manifest, args.test_image, output_dir="./mask_test")
    else:
        process_all_images(
            args.manifest,
            args.output,
            blur_thresh=args.blur_thresh,
            texture_thresh=args.texture_thresh
        )


if __name__ == "__main__":
    main()