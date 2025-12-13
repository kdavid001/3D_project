#!/usr/bin/env python3
"""
corrupt_dataset.py

This script is just to use to test on already perfected image datase0t

This is image and Poses just for research not real world application

Dependencies:
pip install pillow numpy opencv-python tqdm
"""

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
# Corruption functions
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


# -------------------------
# Main pipeline
# -------------------------

def corrupt_image(img):
    h, w, _ = img.shape
    mask = random_mask(h, w)

    corruption_type = random.choice([
        "mask", "blur", "noise", "exposure"
    ])

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


def copy_if_exists(src, dst):
    if os.path.exists(src):
        shutil.copytree(src, dst, dirs_exist_ok=True)


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Try "images/" subfolder, fallback to the folder itself
    potential_dir = os.path.join(args.clean_dir, "images")
    if os.path.exists(potential_dir):
        clean_images = potential_dir
    else:
        clean_images = args.clean_dir

    if not os.path.exists(clean_images):
        raise Exception(f"No images found in {args.clean_dir}")

    # Output directories
    corrupted_images = os.path.join(args.out_dir, "images")
    mask_dir = os.path.join(args.mask_dir)

    # Recreate corrupted images folder
    if os.path.exists(corrupted_images):
        shutil.rmtree(corrupted_images)
    os.makedirs(corrupted_images)

    # Ensure mask directory exists
    os.makedirs(mask_dir, exist_ok=True)

    # Copy sparse folder if it exists (COLMAP poses)
    copy_if_exists(
        os.path.join(args.clean_dir, "sparse"),
        os.path.join(args.out_dir, "sparse")
    )
    # Get all supported image files
    image_files = [
        f for f in os.listdir(clean_images)
        if f.lower().endswith(SUPPORTED_EXTS)
    ]

    print(f"Processing {len(image_files)} images...")

    for fname in tqdm(image_files):
        src_path = os.path.join(clean_images, fname)
        dst_path = os.path.join(corrupted_images, fname)
        mask_path = os.path.join(mask_dir, fname)

        # Read image once
        img = cv2.imread(src_path)
        if img is None:
            print(f"Warning: could not read {src_path}, skipping.")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Step 1: Randomly delete image
        if random.random() < args.delete_prob:
            # print(f"Skipping image (deleted): {fname}")
            continue

        # Step 2: Randomly decide whether to corrupt
        if random.random() > args.corrupt_prob:
            # Keep image as-is
            Image.fromarray(img).save(dst_path)
            empty_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
            Image.fromarray(empty_mask).save(mask_path)
            continue

        # Step 3: Apply corruption
        corrupted, mask = corrupt_image(img)
        Image.fromarray(corrupted).save(dst_path)
        Image.fromarray(mask).save(mask_path)

    print("✔ Corrupted dataset created successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean_dir", required=True,
                        help="Path to clean dataset root")
    parser.add_argument("--out_dir", required=True,
                        help="Path to output corrupted dataset root")
    parser.add_argument("--mask_dir", required=True,
                        help="Where to save corruption masks")
    parser.add_argument("--corrupt_prob", type=float, default=0.5,
                        help="Probability of corrupting an image")
    parser.add_argument("--delete_prob", type=float, default=0.2,
                        help="Probability of deleting an image")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    main(args)
