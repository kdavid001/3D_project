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
- IMPROVED: Better mask generation with cleanup, expansion, and edge handling
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
    if new_w == w and new_h == h:
        return pil_img
    return pil_img.resize((new_w, new_h), Image.BICUBIC)


# --- IMPROVED LOCAL DETECTORS ---
def blur_mask(gray, thresh=100):
    """
    Detect blurry regions using Laplacian variance.
    HIGHER threshold = less sensitive (larger values needed to be considered sharp).
    """
    gray_uint8 = (gray * 255).astype(np.uint8)
    # Pre-blur to reduce noise
    gray_uint8 = cv2.GaussianBlur(gray_uint8, (5, 5), 0)

    # Laplacian detects edges (high variance = sharp, low variance = blurry)
    lap = cv2.Laplacian(gray_uint8, cv2.CV_32F)
    mag = np.abs(lap)

    # Areas with magnitude below threshold are blurry
    mask = mag < thresh
    return mask.astype(np.uint8) * 255


def low_texture_mask(gray, std_thresh=0.02, window_size=15):
    """
    Detect low texture/detail regions.
    HIGHER threshold = less sensitive.
    """
    mean = cv2.GaussianBlur(gray, (window_size, window_size), 0)
    sq_mean = cv2.GaussianBlur(gray ** 2, (window_size, window_size), 0)
    local_var = sq_mean - mean ** 2
    local_var = np.maximum(local_var, 0)  # Clip negative values

    mask = local_var < std_thresh
    return mask.astype(np.uint8) * 255


def saturation_mask(img_rgb, low=0.02, high=0.98):
    """
    Detect over/under saturated regions (clipped highlights/shadows).
    Works on grayscale by checking extreme brightness values.
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    mask = (gray < low) | (gray > high)
    return mask.astype(np.uint8) * 255


def full_image_mask(shape):
    """Returns a white mask covering the whole image"""
    return np.ones((shape[0], shape[1]), dtype=np.uint8) * 255


# --- IMPROVED MASK PROCESSING ---
def clean_mask(mask, min_area=500):
    """
    Remove small disconnected regions (noise).
    Keeps only regions with area >= min_area pixels.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    clean = np.zeros_like(mask)
    for i in range(1, num_labels):  # Skip background (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            clean[labels == i] = 255

    return clean


def morphological_cleanup(mask, close_kernel=7, open_kernel=5):
    """
    Apply morphological operations to smooth mask.
    - Closing: Fills small holes inside masked regions
    - Opening: Removes small noise/protrusions
    """
    # Closing: dilate then erode (fills holes)
    kernel_close = np.ones((close_kernel, close_kernel), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    # Opening: erode then dilate (removes noise)
    kernel_open = np.ones((open_kernel, open_kernel), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)

    return mask


def expand_mask_boundaries(mask, iterations=5):
    """
    Expand mask boundaries to ensure full coverage of defect edges.
    This helps the model blend properly at boundaries.
    """
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(mask, kernel, iterations=iterations)
    return dilated


def feather_mask_edges(mask, feather_radius=5):
    """
    Optional: Apply Gaussian blur to mask edges for smoother transitions.
    Creates soft boundaries instead of hard edges.
    """
    mask_float = mask.astype(np.float32) / 255.0
    kernel_size = feather_radius * 2 + 1
    blurred = cv2.GaussianBlur(mask_float, (kernel_size, kernel_size), 0)
    return (blurred * 255).astype(np.uint8)


def generate_hybrid_mask(
        img_rgb,
        flags,
        blur_thresh=100,
        texture_thresh=0.02,
        min_region_area=500,
        expand_iterations=5,
        feather_radius=0,
        fallback_threshold=0.10
):
    """
    IMPROVED HYBRID LOGIC with better mask processing:

    1. If 'poor_quality_brisque' flag exists:
       - Run ALL local detectors (blur, texture, saturation)
       - Combine their masks
       - Apply advanced cleanup and expansion
       - If combined mask < 10% coverage -> Fallback to FULL MASK

    2. Better mask quality through:
       - Connected component filtering (remove noise)
       - Morphological cleanup (smooth edges, fill holes)
       - Boundary expansion (ensure full coverage)
       - Optional edge feathering (smooth transitions)
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    masks = []

    print(f"    Flags: {flags}")

    # Check for global BRISQUE flag
    is_brisque_bad = any(f in flags for f in ["poor_quality_brisque", "fair_quality_brisque"])

    # 1. Run Local Detectors
    if is_brisque_bad or "blur" in flags:
        print(f"      → Running blur detector (thresh={blur_thresh})...")
        blur_m = blur_mask(gray, thresh=blur_thresh)
        blur_coverage = np.sum(blur_m > 127) / blur_m.size
        print(f"         Found {blur_coverage * 100:.1f}% blur regions")
        if blur_coverage > 0.01:  # Only add if significant
            masks.append(blur_m)

    if is_brisque_bad or "low_texture_ratio" in flags:
        print(f"      → Running texture detector (thresh={texture_thresh})...")
        texture_m = low_texture_mask(gray, std_thresh=texture_thresh)
        texture_coverage = np.sum(texture_m > 127) / texture_m.size
        print(f"         Found {texture_coverage * 100:.1f}% low-texture regions")
        if texture_coverage > 0.01:
            masks.append(texture_m)

    if is_brisque_bad or "saturation_ratio" in flags:
        print(f"      → Running saturation detector...")
        sat_m = saturation_mask(img_rgb)
        sat_coverage = np.sum(sat_m > 127) / sat_m.size
        print(f"         Found {sat_coverage * 100:.1f}% saturated regions")
        if sat_coverage > 0.01:
            masks.append(sat_m)

    # 2. Combine Masks
    if not masks:
        print(f"      ⚠️  No detector masks generated")
        final_mask = np.zeros_like(gray, dtype=np.uint8)
    else:
        print(f"      → Combining {len(masks)} detector mask(s)...")
        final_mask = np.maximum.reduce(masks)

        # 3. IMPROVED CLEANUP PIPELINE
        print(f"      → Cleaning mask (removing regions < {min_region_area} pixels)...")
        final_mask = clean_mask(final_mask, min_area=min_region_area)

        print(f"      → Morphological cleanup (smoothing edges, filling holes)...")
        final_mask = morphological_cleanup(final_mask, close_kernel=7, open_kernel=5)

        # 4. Expand boundaries for better coverage
        if expand_iterations > 0:
            print(f"      → Expanding boundaries ({expand_iterations} iterations)...")
            final_mask = expand_mask_boundaries(final_mask, iterations=expand_iterations)

        # 5. Optional edge feathering
        if feather_radius > 0:
            print(f"      → Feathering edges (radius={feather_radius})...")
            final_mask = feather_mask_edges(final_mask, feather_radius=feather_radius)

    # 6. Check coverage
    coverage = np.sum(final_mask > 127) / final_mask.size
    print(f"      Final mask coverage: {coverage * 100:.1f}%")

    # 7. BRISQUE Fallback Strategy
    if is_brisque_bad and coverage < fallback_threshold:
        print(f"      ⚠️  BRISQUE says poor quality but coverage < {fallback_threshold * 100}%")
        print(f"      → Fallback: Creating FULL IMAGE MASK for global repair")
        final_mask = full_image_mask(gray.shape)

    # Ensure binary
    final_mask = (final_mask > 127).astype(np.uint8) * 255

    return final_mask


def test_single_image(args):
    """Test mask generation on a single image with parameter exploration"""
    with open(args.manifest, 'r') as f:
        manifest = json.load(f)

    # Find the test image
    entry = next((img for img in manifest['images'] if img['filename'] == args.test_image), None)

    if not entry:
        print(f"❌ Image {args.test_image} not found in manifest")
        return

    if entry['screening']['decision'] != "REPAIR":
        print(f"⚠️  Warning: {args.test_image} is not flagged as REPAIR")
        print(f"   Decision: {entry['screening']['decision']}")
        print(f"   Continuing anyway for testing...\n")

    # Locate image
    base_dir = os.path.dirname(args.manifest)
    possible_paths = [
        os.path.join(base_dir, "images", args.test_image),
        os.path.join(base_dir, args.test_image)
    ]
    img_path = next((p for p in possible_paths if os.path.exists(p)), None)

    if not img_path:
        print(f"❌ Image not found at any of: {possible_paths}")
        return

    print(f"\n{'=' * 60}")
    print(f"Testing mask generation: {args.test_image}")
    print(f"Flags: {entry['screening']['flags']}")
    if 'score' in entry['screening']:
        print(f"BRISQUE Score: {entry['screening']['score']:.2f}")
    print(f"{'=' * 60}\n")

    # Load image
    img_pil = Image.open(img_path).convert("RGB")
    img_pil = resize_to_multiple_of_8(img_pil)
    img_np = np.array(img_pil)

    # Test with different thresholds
    test_configs = [
        {"blur_thresh": 80, "texture_thresh": 0.01, "expand": 3},
        {"blur_thresh": 100, "texture_thresh": 0.02, "expand": 5},
        {"blur_thresh": 150, "texture_thresh": 0.03, "expand": 7},
    ]

    os.makedirs("./mask_test", exist_ok=True)

    for i, config in enumerate(test_configs, 1):
        print(
            f"\n--- Test {i}: blur={config['blur_thresh']}, texture={config['texture_thresh']}, expand={config['expand']} ---")

        mask = generate_hybrid_mask(
            img_np,
            entry['screening']['flags'],
            blur_thresh=config['blur_thresh'],
            texture_thresh=config['texture_thresh'],
            min_region_area=500,
            expand_iterations=config['expand'],
            feather_radius=0
        )

        mask_pil = Image.fromarray(mask)
        mask_processed = prepare_mask_for_inpainting(mask_pil)

        # Save outputs
        mask_processed.save(f"./mask_test/mask_config{i}.png")

        masked_img = create_masked_image(img_pil, mask_processed)
        masked_img.save(f"./mask_test/masked_config{i}.png")

        vis = visualize_mask_overlay(img_pil, mask_processed)
        vis.save(f"./mask_test/overlay_config{i}.png")

    print(f"\n✓ Test outputs saved to ./mask_test/")
    print(f"  Compare overlay_config1/2/3.png to choose best parameters")


def process_images(args):
    """Main processing loop"""
    if not os.path.exists(args.manifest):
        print(f"❌ Manifest not found: {args.manifest}")
        return

    with open(args.manifest, 'r') as f:
        manifest = json.load(f)

    # Filter for REPAIR images
    flagged_images = [img for img in manifest['images']
                      if img.get('screening', {}).get('decision') == "REPAIR"]

    print(f"\n{'=' * 60}")
    print(f"Found {len(flagged_images)} images flagged for REPAIR")
    print(f"Parameters:")
    print(f"  - Blur threshold: {args.blur_thresh}")
    print(f"  - Texture threshold: {args.texture_thresh}")
    print(f"  - Min region area: {args.min_area} pixels")
    print(f"  - Expand iterations: {args.expand}")
    print(f"  - Feather radius: {args.feather}")
    print(f"{'=' * 60}\n")

    # Setup directories
    out_img_dir = os.path.join(args.output, "masked_images")
    out_mask_dir = os.path.join(args.output, "masks")
    out_vis_dir = os.path.join(args.output, "visualizations")
    for d in [out_img_dir, out_mask_dir, out_vis_dir]:
        os.makedirs(d, exist_ok=True)

    results = []
    processed = 0
    skipped = 0

    for entry in tqdm(flagged_images, desc="Generating masks"):
        try:
            filename = entry['filename']

            # Locate image
            base_dir = os.path.dirname(args.manifest)
            possible_paths = [
                os.path.join(base_dir, "images", filename),
                os.path.join(base_dir, filename)
            ]
            img_path = next((p for p in possible_paths if os.path.exists(p)), None)

            if not img_path:
                print(f"\n  ⚠️  Skipping {filename}: not found")
                skipped += 1
                continue

            # Load
            img_pil = Image.open(img_path).convert("RGB")
            img_pil = resize_to_multiple_of_8(img_pil)
            img_np = np.array(img_pil)

            print(f"\n  {filename}:")

            # Generate Mask
            mask = generate_hybrid_mask(
                img_np,
                entry['screening']['flags'],
                blur_thresh=args.blur_thresh,
                texture_thresh=args.texture_thresh,
                min_region_area=args.min_area,
                expand_iterations=args.expand,
                feather_radius=args.feather
            )

            # Save outputs
            mask_pil = Image.fromarray(mask)
            mask_processed = prepare_mask_for_inpainting(mask_pil)
            mask_processed.save(os.path.join(out_mask_dir, filename))

            masked_img_pil = create_masked_image(img_pil, mask_processed)
            masked_img_pil.save(os.path.join(out_img_dir, filename))

            vis_pil = visualize_mask_overlay(img_pil, mask_processed)
            vis_pil.save(os.path.join(out_vis_dir, filename))

            results.append({
                "filename": filename,
                "mask_path": os.path.join("masks", filename),
                "masked_image_path": os.path.join("masked_images", filename),
                "flags": entry['screening']['flags']
            })

            processed += 1

        except Exception as e:
            print(f"\n  ❌ Error processing {filename}: {e}")
            import traceback
            traceback.print_exc()
            skipped += 1

    # Save output manifest
    with open(os.path.join(args.output, "inpainting_manifest.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"✅ Processing complete!")
    print(f"  Processed: {processed} images")
    print(f"  Skipped: {skipped} images")
    print(f"\nOutputs:")
    print(f"  - Masked images: {out_img_dir}")
    print(f"  - Masks: {out_mask_dir}")
    print(f"  - Visualizations: {out_vis_dir}")
    print(f"  - Manifest: {os.path.join(args.output, 'inpainting_manifest.json')}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="Path to manifest.json")
    parser.add_argument("--output", default="./inpainting_ready", help="Output folder")
    parser.add_argument("--blur_thresh", type=int, default=100, help="Blur sensitivity (higher=less sensitive)")
    parser.add_argument("--texture_thresh", type=float, default=0.02,
                        help="Texture sensitivity (higher=less sensitive)")
    parser.add_argument("--min_area", type=int, default=500, help="Min region size to keep (pixels)")
    parser.add_argument("--expand", type=int, default=5, help="Boundary expansion iterations")
    parser.add_argument("--feather", type=int, default=0, help="Edge feathering radius (0=hard edges)")

    # Test mode
    parser.add_argument("--test", action="store_true", help="Test mode: try multiple configs")
    parser.add_argument("--test_image", help="Filename to test")

    args = parser.parse_args()

    if args.test:
        if not args.test_image:
            print("❌ --test_image required for test mode")
            print("Usage: python prepare_masks.py --manifest manifest.json --test --test_image image.jpg")
        else:
            test_single_image(args)
    else:
        process_images(args)