#!/usr/bin/env python3
"""
SINGLE_ABSOLUTE
Step 1: load_images.py - Image quality screening WITHOUT mask generation

z_thresh=1.5 #Detection Sensitivity Threshold
Usage:
    python load_images.py --input_dir ./raw_images --out_dir ./preprocessed --target_size 1024 --resize_mode pad

What it does:
- Loads and resizes images
- Computes quality metrics per image
- Screens images to determine which need diffusion processing
- Writes manifest.json with metadata (NO MASKS GENERATED HERE)
"""

import os
import json
import argparse
from PIL import Image, ImageOps
import numpy as np
from tqdm import tqdm
import cv2

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png"}


def is_image_file(filename):
    return os.path.splitext(filename.lower())[1] in SUPPORTED_EXTS


def load_images_list(input_dir):
    files = sorted([
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if is_image_file(f)
    ])
    return files


def pil_to_np(img):
    arr = np.array(img)
    if arr.ndim == 2:
        arr = np.expand_dims(arr, axis=-1)
    return arr


def resize_image(img: Image.Image, target_size, mode="pad"):
    """Resize image with different strategies"""
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


# Quality metric functions
def laplacian_variance(gray):
    gray_u8 = (gray * 255).astype(np.uint8)
    return cv2.Laplacian(gray_u8, cv2.CV_64F).var()


def image_entropy(gray):
    hist = np.histogram(gray, bins=256, range=(0, 1))[0]
    prob = hist / (hist.sum() + 1e-8)
    prob = prob[prob > 0]
    return float(-np.sum(prob * np.log2(prob)))


def edge_density(gray):
    edges = cv2.Canny((gray * 255).astype(np.uint8), 50, 150)
    return edges.mean() / 255.0


def saturation_ratio(gray, low=0.02, high=0.98):
    return np.mean((gray < low) | (gray > high))


# def motion_blur_score(gray):
#     sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
#     sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
#     mag = np.sqrt(sobelx ** 2 + sobely ** 2)
#     return float(mag.mean())


def low_texture_ratio(gray, thresh=0.01):
    return np.mean(gray.std(axis=0) < thresh)


def compute_quality_metrics(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    return {
        "blur": laplacian_variance(gray),
        # "motion_blur": motion_blur_score(gray),
        "entropy": image_entropy(gray),
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "edge_density": edge_density(gray),
        "saturation_ratio": saturation_ratio(gray),
        "low_texture_ratio": low_texture_ratio(gray),
    }


def compute_dataset_stats(metrics_list):
    stats = {}
    for key in metrics_list[0]:
        values = np.array([m[key] for m in metrics_list])
        stats[key] = {
            "mean": float(values.mean()),
            "std": float(values.std() + 1e-6),
        }
    return stats


def screen_image_absolute(metrics):
    """
    Screen image using ABSOLUTE thresholds.
    Good for: single image testing, consistent quality standards.
    """
    flags = []
    decision = "NONE"

    # ABSOLUTE THRESHOLDS (tuned for general photography)
    BLUR_MIN = 100  # Laplacian variance < 100 = blurry
    # MOTION_BLUR_MAX = 0.311  # Motion blur > 0.08 = significant motion
    ENTROPY_MIN = 6.0  # Entropy < 6.0 = low detail
    BRIGHTNESS_MIN = 0.15  # Too dark
    BRIGHTNESS_MAX = 0.85  # Too bright
    CONTRAST_MIN = 0.1  # Too flat
    EDGE_DENSITY_MIN = 0.1  # Too smooth
    SATURATION_EXTREME = 0.1  # Over/under saturated pixels > 10%
    LOW_TEXTURE_MAX = 0.15  # Low texture regions > 15%

    # Check each metric

    # CHANGE THIS IN YOUR SCRIPT
    # Only flag as blurry if the image actually HAS detail to loose.
    # If edge_density is tiny (e.g. < 0.02), it's likely just a flat image, not blurry.
    if metrics["blur"] < BLUR_MIN and metrics["edge_density"] > 0.1:
        flags.append("blur")

    # if metrics["motion_blur"] > MOTION_BLUR_MAX:
    #     flags.append("motion_blur")

    if metrics["saturation_ratio"] > SATURATION_EXTREME:
        flags.append("saturation_ratio")

    if metrics["low_texture_ratio"] > LOW_TEXTURE_MAX:
        flags.append("low_texture_ratio")

    if metrics["brightness"] < BRIGHTNESS_MIN or metrics["brightness"] > BRIGHTNESS_MAX:
        flags.append("brightness_extreme")

    if metrics["contrast"] < CONTRAST_MIN:
        flags.append("low_contrast")

    # Decision logic , removed "motion_blur"
    if any(f in flags for f in ["blur", "low_texture_ratio", "saturation_ratio"]):
        decision = "REPAIR"
    elif metrics["entropy"] > ENTROPY_MIN and metrics["edge_density"] > EDGE_DENSITY_MIN:
        decision = "NOVEL_VIEW"

    return {
        "needs_diffusion": decision != "NONE",
        "flags": flags,
        "decision": decision,
        "method": "absolute"
    }


def screen_image_relative(metrics, dataset_stats, z_thresh=1.5):
    """
    Screen image using RELATIVE thresholds (z-scores).
    Good for: finding outliers within a dataset.
    """
    # , "motion_blur"
    flags = []
    STRUCTURAL_KEYS = {
        "blur", "edge_density",
        "entropy", "saturation_ratio", "low_texture_ratio"
    }

    for k, v in metrics.items():
        z = abs(v - dataset_stats[k]["mean"]) / dataset_stats[k]["std"]
        if z > z_thresh:
            flags.append(k)

    decision = "NONE"

    # REPAIR conditions , "motion_blur"
    if any(f in flags for f in ["blur", "low_texture_ratio", "saturation_ratio"]):
        decision = "REPAIR"
    # NOVEL VIEW conditions
    elif (
            metrics["entropy"] > dataset_stats["entropy"]["mean"] and
            metrics["edge_density"] > dataset_stats["edge_density"]["mean"]
    ):
        decision = "NOVEL_VIEW"

    return {
        "needs_diffusion": decision != "NONE",
        "flags": flags,
        "decision": decision,
        "method": "relative"
    }


def screen_image_hybrid(metrics, dataset_stats=None, z_thresh=1.5):
    """
    HYBRID: Use absolute thresholds first, then refine with dataset-relative.
    This is the RECOMMENDED approach.
    """
    # First pass: absolute thresholds
    absolute_result = screen_image_absolute(metrics)

    # If we don't have dataset stats, return absolute result
    if dataset_stats is None:
        return absolute_result

    # Second pass: relative thresholds
    relative_result = screen_image_relative(metrics, dataset_stats, z_thresh)

    # Combine flags (union of both methods)
    combined_flags = list(set(absolute_result["flags"] + relative_result["flags"]))

    # Decision priority:
    # 1. If absolute says REPAIR, trust it (catches objectively bad images)
    # 2. If relative says REPAIR, consider it (catches dataset outliers)
    # 3. Otherwise, use stricter of the two

    if absolute_result["decision"] == "REPAIR":
        final_decision = "REPAIR"
    elif relative_result["decision"] == "REPAIR":
        final_decision = "REPAIR"
    elif absolute_result["decision"] == "NOVEL_VIEW" or relative_result["decision"] == "NOVEL_VIEW":
        final_decision = "NOVEL_VIEW"
    else:
        final_decision = "NONE"

    return {
        "needs_diffusion": final_decision != "NONE",
        "flags": combined_flags,
        "decision": final_decision,
        "method": "hybrid",
        "absolute_flags": absolute_result["flags"],
        "relative_flags": relative_result["flags"]
    }


def screen_image(metrics, dataset_stats, z_thresh, min_flags):
    """
    Screen image to determine if it needs diffusion processing.

    DEPRECATED: Use screen_image_hybrid() instead for better results.
    Kept for backward compatibility.
    """
    return screen_image_hybrid(metrics, dataset_stats, z_thresh)


def process_images(args):
    """Main processing function - NO MASK GENERATION"""
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
    quality_list = []

    # Process all images
    for path in tqdm(files, desc="Processing images"):
        base = os.path.basename(path)
        try:
            img = Image.open(path).convert("RGB")
            proc_img = resize_image(img, target_size, resize_mode) if target_size else img
            np_img = pil_to_np(proc_img)

            out_path = os.path.join(images_out_dir, base)
            proc_img.save(out_path)
            quality = compute_quality_metrics(np_img)

            entry = {
                "filename": base,
                "quality": quality,
            }

            manifest["images"].append(entry)
            quality_list.append(quality)

        except Exception as e:
            print(f"Error processing {base}: {e}")
            continue

    # Compute dataset statistics
    dataset_stats = compute_dataset_stats(quality_list)

    # Screen images
    for entry in manifest["images"]:
        decision = screen_image(
            entry["quality"],
            dataset_stats,
            z_thresh=1.5
            ,
            min_flags=2
        )
        entry["screening"] = decision

    # Count flagged images
    num_repair = sum(1 for e in manifest["images"] if e["screening"]["decision"] == "REPAIR")
    num_novel = sum(1 for e in manifest["images"] if e["screening"]["decision"] == "NOVEL_VIEW")

    print(f"\n{'=' * 60}")
    print(f"Screening Results:")
    print(f"  Total images: {len(manifest['images'])}")
    print(f"  Flagged for REPAIR: {num_repair}")
    print(f"  Flagged for NOVEL_VIEW: {num_novel}")
    print(f"  No processing needed: {len(manifest['images']) - num_repair - num_novel}")
    print(f"{'=' * 60}\n")

    # Save manifest
    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"✓ Manifest saved to: {manifest_path}")
    print(f"✓ Processed images saved to: {images_out_dir}")
    print(f"\nNext step: Run prepare_masks.py to generate masks for flagged images")

    return manifest


def test_single_image(image_path, target_size=None, resize_mode="pad"):
    """
    Test a single image to see if it needs diffusion processing.
    Returns detailed analysis without saving anything.
    """
    print(f"\n{'=' * 60}")
    print(f"SINGLE IMAGE ANALYSIS")
    print(f"{'=' * 60}")
    print(f"Image: {image_path}\n")

    try:
        # Load image
        img = Image.open(image_path).convert("RGB")
        original_size = img.size
        print(f"Original size: {original_size[0]}x{original_size[1]}")

        # Resize if needed
        proc_img = resize_image(img, target_size, resize_mode) if target_size else img
        if proc_img.size != original_size:
            print(f"Resized to: {proc_img.size[0]}x{proc_img.size[1]}")

        np_img = pil_to_np(proc_img)

        # Compute quality metrics
        print("\n" + "-" * 60)
        print("QUALITY METRICS:")
        print("-" * 60)
        quality = compute_quality_metrics(np_img)

        # Print metrics in readable format
        print(f"  Blur (Laplacian variance): {quality['blur']:.2f}")
        # print(f"  Motion blur score:         {quality['motion_blur']:.4f}")
        print(f"  Image entropy:             {quality['entropy']:.2f}")
        print(f"  Brightness:                {quality['brightness']:.3f}")
        print(f"  Contrast (std):            {quality['contrast']:.3f}")
        print(f"  Edge density:              {quality['edge_density']:.4f}")
        print(f"  Saturation ratio:          {quality['saturation_ratio']:.4f}")
        print(f"  Low texture ratio:         {quality['low_texture_ratio']:.4f}")

        # Create dummy dataset stats for comparison (typical values)
        # These are rough estimates - in real use, we need the full dataset
        dummy_stats = {
            "blur": {"mean": 500, "std": 300},
            # "motion_blur": {"mean": 0.05, "std": 0.02},
            "entropy": {"mean": 7.0, "std": 0.5},
            "brightness": {"mean": 0.5, "std": 0.15},
            "contrast": {"mean": 0.2, "std": 0.05},
            "edge_density": {"mean": 0.1, "std": 0.05},
            "saturation_ratio": {"mean": 0.05, "std": 0.03},
            "low_texture_ratio": {"mean": 0.1, "std": 0.05},
        }

        # Screen the image
        decision = screen_image(quality, dummy_stats, z_thresh=1.5, min_flags=2)

        print("\n" + "-" * 60)
        print("SCREENING ANALYSIS:")
        print("-" * 60)

        # Show z-scores for each metric
        print("\nZ-scores (how far from typical):")
        for key in quality.keys():
            z = abs(quality[key] - dummy_stats[key]["mean"]) / dummy_stats[key]["std"]
            flag_marker = "🚩" if z > 1.5 else "  "
            print(f"  {flag_marker} {key:20s}: z={z:.2f}")

        print(f"\nFlags detected: {decision['flags'] if decision['flags'] else 'None'}")

        # Detailed explanation
        print("\n" + "-" * 60)
        print("DECISION & EXPLANATION:")
        print("-" * 60)

        if decision["decision"] == "REPAIR":
            print("✓ Decision: REPAIR (Inpainting needed)")
            print("\nReasons:")
            if "blur" in decision["flags"]:
                print("  • Image has BLURRY regions (low Laplacian variance)")
                print("    → Inpainting will regenerate sharp details")
            # if "motion_blur" in decision["flags"]:
            #     print("  • Image has MOTION BLUR")
            #     print("    → Inpainting will remove blur artifacts")
            if "low_texture_ratio" in decision["flags"]:
                print("  • Image has LOW TEXTURE areas")
                print("    → Inpainting will add detail to flat regions")
            if "saturation_ratio" in decision["flags"]:
                print("  • Image has OVER/UNDER SATURATED pixels")
                print("    → Inpainting will fix clipped regions")

            print("\nNext steps:")
            print("  1. Run prepare_masks.py to generate inpainting mask")
            print("  2. Use SD inpainting model to repair damaged regions")

        elif decision["decision"] == "NOVEL_VIEW":
            print("✓ Decision: NOVEL_VIEW (Image-to-image needed)")
            print("\nReasons:")
            print("  • High entropy + high edge density")
            print("  • Image has interesting structure but may need enhancement")
            print("\nNext steps:")
            print("  1. Use img2img diffusion to enhance image quality")
            print("  2. No mask needed - process entire image")

        else:
            print("✓ Decision: NONE (No processing needed)")
            print("\nReasons:")
            print("  • Image quality is acceptable")
            print("  • No significant anomalies detected")
            print("\nNext steps:")
            print("  • Image can be used as-is")

        print(f"\n{'=' * 60}\n")

        return {
            "quality": quality,
            "screening": decision,
            "original_size": original_size,
            "processed_size": proc_img.size
        }

    except Exception as e:
        print(f"❌ Error analyzing image: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description="Image quality screening (no mask generation)")
    parser.add_argument("--input_dir", help="Folder with raw images (for batch processing)")
    parser.add_argument("--out_dir", help="Folder to write preprocessed images and manifest (for batch processing)")
    parser.add_argument("--target_size", type=int, default=None,
                        help="If set, resize images to this square size (e.g. 1024)")
    parser.add_argument("--resize_mode", choices=["pad", "crop", "stretch", "none"], default="pad",
                        help="How to resize while preserving aspect")

    # Single image test mode
    parser.add_argument("--test", action="store_true", help="Test mode: analyze a single image")
    parser.add_argument("--test_image", help="Path to single image to test (requires --test)")

    args = parser.parse_args()

    if args.test:
        if not args.test_image:
            print("❌ --test_image required when using --test")
            print("Usage: python load_images.py --test --test_image path/to/image.jpg")
            return

        test_single_image(args.test_image, args.target_size, args.resize_mode)

    else:
        if not args.input_dir or not args.out_dir:
            print("❌ --input_dir and --out_dir required for batch processing")
            print("\nUsage:")
            print("  Batch:  python load_images.py --input_dir ./images --out_dir ./output")
            print("  Test:   python load_images.py --test --test_image path/to/image.jpg")
            return

        process_images(args)


if __name__ == "__main__":
    main()