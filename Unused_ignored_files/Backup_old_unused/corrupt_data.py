#!/usr/bin/env python3
"""
corrupt_dataset_v2.py

Fixed logic for:
1. Guaranteed Sector Deletion (using JSON geometry)
2. Random Robustness Corruption (Blur, Noise, etc.)
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
# Corruption functions (Unchanged)
# -------------------------
def random_mask(h, w):
    mask = np.zeros((h, w), dtype=np.uint8)
    num_regions = random.randint(1, 4)
    for _ in range(num_regions):
        rw = random.randint(w // 10, w // 3)
        rh = random.randint(h // 10, h // 3)
        x = random.randint(0, w - rw)
        y = random.randint(0, h - rh)
        mask[y:y + rh, x:x + rw] = 255
    return mask


def apply_blackout(img, mask):
    img[mask == 255] = 0
    return img


def apply_motion_blur(img):
    k = random.choice([5, 9, 15])
    kernel = np.zeros((k, k))
    kernel[k // 2, :] = np.ones(k)
    kernel /= k
    return cv2.filter2D(img, -1, kernel)


def apply_noise(img):
    noise = np.random.normal(0, 20, img.shape)
    noisy = img + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def apply_exposure(img):
    factor = random.uniform(0.4, 1.8)
    return np.clip(img * factor, 0, 255).astype(np.uint8)


def corrupt_image(img):
    h, w, _ = img.shape
    mask = random_mask(h, w)
    corruption_type = random.choice(["mask", "blur", "noise", "exposure"])
    corrupted = img.copy()

    if corruption_type == "mask":
        corrupted = apply_blackout(corrupted, mask)
    elif corruption_type == "blur":
        corrupted = apply_motion_blur(corrupted)
    elif corruption_type == "noise":
        corrupted = apply_noise(corrupted)
    elif corruption_type == "exposure":
        corrupted = apply_exposure(corrupted)

    return corrupted, mask


def get_name_from_path(path):
    return os.path.basename(os.path.normpath(path))


def copy_if_exists(src, dst):
    if os.path.exists(src):
        shutil.copytree(src, dst, dirs_exist_ok=True)


# -------------------------
# Main pipeline
# -------------------------
def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    file_name = get_name_from_path(args.clean_dir)

    # 1. Setup Input Directory
    if args.use_geometry:  # Use if nerf synthetic data
        potential_dir = os.path.join(args.clean_dir, "train")
        if os.path.exists(potential_dir):
            input_images_dir = potential_dir
        else:
            # Fallback if user points directly to an images folder
            input_images_dir = args.clean_dir
    else:  # Use if taking in Normal Images
        potential_dir = os.path.join(args.clean_dir, "images")
        if os.path.exists(potential_dir):
            input_images_dir = potential_dir
        else:
            input_images_dir = args.clean_dir

    if not os.path.exists(input_images_dir):
        raise Exception(f"❌ No images found in {input_images_dir}")

    # 2. Setup Output Directory
    # We maintain the structure: output_folder/train/
    out_train_dir = os.path.join(args.out_dir, f"{file_name}/train")
    if os.path.exists(out_train_dir):
        shutil.rmtree(out_train_dir)
    os.makedirs(out_train_dir)

    # 3. Setup Logic: Identify "Banned" Sector (The Hole)
    banned_filenames = set()
    # directory for other_files
    out_other_dir = os.path.join(args.out_dir, f"{file_name}")
    copy_if_exists(
        os.path.join(args.clean_dir, "sparse"),
        os.path.join(out_other_dir, "sparse"),
    )
    # If user provided a path to the root folder (e.g., "nerf_synthetic/lego")
    # we look for the JSON to do smart deletion.
    json_path = os.path.join(args.clean_dir, "transforms_train.json")

    if args.use_geometry and os.path.exists(json_path):
        print(f"📐 Found JSON at {json_path}. Analyzing geometry for Sector Deletion...")

        # Copy JSON to output so the pipeline still works
        shutil.copy(json_path, os.path.join(args.out_dir, f"{file_name}/transforms_train.json"))

        with open(json_path, 'r') as f:
            data = json.load(f)

        for frame in data['frames']:
            # Extract filename (e.g., "./train/r_0" -> "r_0.png")
            fpath = frame['file_path']
            fname = os.path.basename(fpath)
            if not fname.lower().endswith(SUPPORTED_EXTS):
                fname += ".png"  # NeRF JSONs usually omit extension

            # Check Geometry (Negative X Sector)
            matrix = frame['transform_matrix']
            x_pos = matrix[0][3]

            # CONSTRAINT: Delete Left Side (X < -0.5)
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

        # --- STEP A: DELETION LOGIC ---

        # 1. Geometry Check (The Big Hole)
        if fname in banned_filenames:
            # print(f"Deleted (Sector): {fname}")
            continue

            # 2. Random Check (The Sparse Data simulation)
        # If NOT using geometry, we rely on this.
        # If using geometry, we might want to disable this or keep it low.
        if fname not in banned_filenames and random.random() < args.delete_prob:
            # print(f"Deleted (Random): {fname}")
            continue

        # --- STEP B: CORRUPTION LOGIC ---

        # Read Image
        img = cv2.imread(src_path)
        if img is None: continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Randomly decide to Corrupt or Keep Clean
        if random.random() > args.corrupt_prob:
            # SAVE CLEAN
            Image.fromarray(img).save(dst_path)
        else:
            # SAVE CORRUPTED
            corrupted, _ = corrupt_image(img)
            Image.fromarray(corrupted).save(dst_path)

        processed_count += 1

    print(f"\n✔ Done. Saved {processed_count} images to {out_train_dir}")
    print(f"✔ JSON copied (if found). Dataset is ready for Gaussian Splatting.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Path to the ROOT folder (e.g., "nerf_synthetic/lego")
    parser.add_argument("--clean_dir", required=True,
                        help="Path to input dataset root (containing 'train' and 'transforms_train.json')")

    # Path to where you want the result
    parser.add_argument("--out_dir", required=True, help="Path to output root")

    # Switch to enable Geometric Deletion
    parser.add_argument("--use_geometry", action='store_true', help="If set, reads JSON and deletes Negative-X sector.")

    # Probabilities
    parser.add_argument("--corrupt_prob", type=float, default=0.5, help="Chance to blur/noise an image (0.0 to 1.0)")
    parser.add_argument("--delete_prob", type=float, default=0.0,
                        help="Chance to randomly delete EXTRA images (keep 0.0 if using geometry)")

    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    main(args)
