#!/usr/bin/env python3
"""
corrupt_dataset_v3.py - The "Sensor/Image Failure" Generator

Updates:
1. REMOVED: Black Holes (Geometry Artifacts).
2. ADDED: Radiometric Corruptions (Saturation, Exposure Extremes).
3. KEEPS: Optical Corruptions (Defocus Blur, Noise).
4. KEEPS: Sector Deletion (Missing Views logic).
5. ADDED: Single Image Auto-Detect (Generates isolated failure modes for thesis collages).
"""
import json
import os
import shutil
import random
import argparse
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

SUPPORTED_EXTS = (".png", ".jpg", ".jpeg")


# -------------------------
# SENSOR FAILURE FUNCTIONS
# -------------------------
def apply_saturation_boost(img):
    """
    Simulates "Deep Fried" sensor clipping.
    Boosts Saturation to extreme levels.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    # Boost saturation by 2x to 4x
    factor = random.uniform(2.0, 4.0)
    hsv[:, :, 1] = hsv[:, :, 1] * factor
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def apply_exposure_failure(img):
    """
    Simulates Shutter Failure.
    Either "Flash Bang" (Blown Out) or "Lens Cap" (Underexposed).
    """
    mode = random.choice(["over", "under"])

    if mode == "over":
        # Flash Bang: 2x to 5x brightness
        factor = random.uniform(2.5, 5.0)
    else:
        # Underexposed: 0.1x to 0.3x brightness
        factor = random.uniform(0.1, 0.3)

    return np.clip(img * factor, 0, 255).astype(np.uint8)


def apply_defocus_blur(img):
    """
    Simulates Missed Focus (Optical Blur).
    Uses Gaussian Blur (softer than Motion Blur).
    """
    k_size = random.choice([7, 11, 15, 21])  # Odd numbers only
    return cv2.GaussianBlur(img, (k_size, k_size), 0)


def apply_iso_noise(img):
    """
    Simulates High ISO Grain (Sensor Noise).
    """
    row, col, ch = img.shape
    mean = 0
    sigma = random.uniform(30, 80)  # Heavy grain
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy = img.astype(np.float32) + gauss
    return np.clip(noisy, 0, 255).astype(np.uint8)


def corrupt_image(img):
    """
    Randomly selects a 'Sensor Failure' mode.
    """
    # Equal chance for any failure
    corruption_type = random.choice(["saturation", "exposure", "blur", "noise"])

    if corruption_type == "saturation":
        return apply_saturation_boost(img)
    elif corruption_type == "exposure":
        return apply_exposure_failure(img)
    elif corruption_type == "blur":
        return apply_defocus_blur(img)
    elif corruption_type == "noise":
        return apply_iso_noise(img)

    return img


def get_name_from_path(path):
    return os.path.basename(os.path.normpath(path))


def copy_if_exists(src, dst):
    if os.path.exists(src):
        shutil.copytree(src, dst, dirs_exist_ok=True)


# -------------------------
# MAIN PIPELINE
# -------------------------
def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)

    # ==========================================
    # NEW: SINGLE IMAGE AUTO-DETECT LOGIC
    # ==========================================
    if os.path.isfile(args.clean_dir):
        print(f"📷 Single image mode detected: {args.clean_dir}")
        os.makedirs(args.out_dir, exist_ok=True)

        img = cv2.imread(args.clean_dir)
        if img is None:
            raise Exception(f"❌ Could not read image {args.clean_dir}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        fname = os.path.basename(args.clean_dir)
        name, ext = os.path.splitext(fname)

        # 1. Generate standard random corruption based on probability
        if random.random() > args.corrupt_prob:
            out_img = img
            out_path = os.path.join(args.out_dir, f"{name}_clean{ext}")
            print(f"Status: Saved Clean (Failed corrupt_prob roll)")
        else:
            out_img = corrupt_image(img)
            out_path = os.path.join(args.out_dir, f"{name}_random_corrupt{ext}")
            print(f"Status: Saved Random Corruption")
        Image.fromarray(out_img).save(out_path)

        # 2. Thesis Collage Generator (Always generates all 4 variants)
        print("\n🎓 Thesis Mode: Generating all isolated failure modes for visual reporting...")
        Image.fromarray(apply_saturation_boost(img)).save(os.path.join(args.out_dir, f"{name}_deepfried{ext}"))
        Image.fromarray(apply_exposure_failure(img)).save(os.path.join(args.out_dir, f"{name}_exposure{ext}"))
        Image.fromarray(apply_defocus_blur(img)).save(os.path.join(args.out_dir, f"{name}_blur{ext}"))
        Image.fromarray(apply_iso_noise(img)).save(os.path.join(args.out_dir, f"{name}_noise{ext}"))

        print(f"✔ All single-image variations saved to {args.out_dir}")
        return
    # ==========================================
    # END SINGLE IMAGE LOGIC
    # ==========================================

    # Original Directory Logic
    file_name = get_name_from_path(args.clean_dir)

    # 1. Setup Input Directory
    if args.use_geometry:
        potential_dir = os.path.join(args.clean_dir, "train")
        input_images_dir = potential_dir if os.path.exists(potential_dir) else args.clean_dir
    else:
        potential_dir = os.path.join(args.clean_dir, "images")
        input_images_dir = potential_dir if os.path.exists(potential_dir) else args.clean_dir

    if not os.path.exists(input_images_dir):
        raise Exception(f"❌ No images found in {input_images_dir}")

    # 2. Setup Output Directory
    out_train_dir = os.path.join(args.out_dir, f"{file_name}/train")
    if os.path.exists(out_train_dir):
        shutil.rmtree(out_train_dir)
    os.makedirs(out_train_dir)

    # 3. Setup Logic: Identify "Banned" Sector (Missing Views)
    banned_filenames = set()
    out_other_dir = os.path.join(args.out_dir, f"{file_name}")
    copy_if_exists(os.path.join(args.clean_dir, "sparse"), os.path.join(out_other_dir, "sparse"))

    json_path = os.path.join(args.clean_dir, "transforms_train.json")

    if args.use_geometry and os.path.exists(json_path):
        print(f"📐 Found JSON at {json_path}. Analyzing geometry for Sector Deletion...")

        # Copy JSON to output
        shutil.copy(json_path, os.path.join(args.out_dir, f"{file_name}/transforms_train.json"))

        with open(json_path, 'r') as f:
            data = json.load(f)

        for frame in data['frames']:
            fpath = frame['file_path']
            fname = os.path.basename(fpath)
            if not fname.lower().endswith(SUPPORTED_EXTS):
                fname += ".png"

            # Check Geometry (Negative X Sector)
            matrix = frame['transform_matrix']
            x_pos = matrix[0][3]

            if x_pos < -0.5:
                banned_filenames.add(fname)

        print(f"💀 'Sector Deletion' active. Banned {len(banned_filenames)} images based on position.")
    else:
        print("⚠️ No JSON found or geometry disabled. Using purely random deletion.")

    # 4. Processing Loop
    image_files = [f for f in os.listdir(input_images_dir) if f.lower().endswith(SUPPORTED_EXTS)]
    print(f"Processing {len(image_files)} images...")

    processed_count = 0

    for fname in tqdm(image_files):
        src_path = os.path.join(input_images_dir, fname)
        dst_path = os.path.join(out_train_dir, fname)

        # --- STEP A: DELETION LOGIC (Missing Views) ---
        if fname in banned_filenames:
            continue
        if fname not in banned_filenames and random.random() < args.delete_prob:
            continue

        # --- STEP B: CORRUPTION LOGIC (Sensor Failure) ---
        img = cv2.imread(src_path)
        if img is None: continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if random.random() > args.corrupt_prob:
            # SAVE CLEAN
            Image.fromarray(img).save(dst_path)
        else:
            # SAVE CORRUPTED (Sensor Failure)
            corrupted = corrupt_image(img)
            Image.fromarray(corrupted).save(dst_path)

        processed_count += 1

    print(f"\n✔ Done. Saved {processed_count} images to {out_train_dir}")
    print(f"✔ Dataset ready for 'Radiometric Robustness' testing.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean_dir", required=True, help="Path to input dataset root OR a single image file")
    parser.add_argument("--out_dir", required=True, help="Path to output root")
    parser.add_argument("--use_geometry", action='store_true', help="If set, reads JSON and deletes Negative-X sector.")
    parser.add_argument("--corrupt_prob", type=float, default=0.5, help="Chance to corrupt an image (0.0 to 1.0)")
    parser.add_argument("--delete_prob", type=float, default=0.0, help="Chance to randomly delete EXTRA images")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    main(args)
