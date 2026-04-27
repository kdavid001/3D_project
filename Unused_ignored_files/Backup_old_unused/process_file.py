#!/usr/bin/env python3
"""
load_images_v22.py - The "Radiometric Screener"
(Replaces Geometry Checks with Color/Exposure Integrity)

Logic:
1. SATURATION Check: Flags "Deep Fried" (Neon) or "Dead" (Grayscale) images.
2. EXPOSURE Check: Flags "Blown Out" (All White) or "Pitch Black" images.
3. QUALITY Check: Uses MUSIQ for Blur/Noise.

Usage:
    python load_images_v22.py --input_dir ./data --mode synthetic --debug
"""

import os
import json
import argparse
import shutil
import torch
import pyiqa
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png"}

THRESHOLDS = {
    "natural": {
        "BAD_LIMIT": 40.0,  # MUSIQ Score
        "GOOD_LIMIT": 70.0,
        "USE_CROP": False,
        "CHECK_COLOR": True,  # Enable Radiometric Checks
        "SAT_MAX_AVG": 170.0,  # Max average saturation (0-255)
        "SAT_CLIPPED": 0.05,  # Max % of pixels allowed to be fully saturated (255)
        "EXP_MIN_AVG": 30.0,  # Minimum brightness
        "EXP_MAX_AVG": 220.0  # Maximum brightness
    },
    "synthetic": {
        "BAD_LIMIT": 65.0,
        "GOOD_LIMIT": 71.0,
        "USE_CROP": True,
        "CHECK_COLOR": True,
        "SAT_MAX_AVG": 200.0,  # Synthetic can be more colorful
        "SAT_CLIPPED": 0.10,  # Allow 10% clipped pixels
        "EXP_MIN_AVG": 10.0,  # Synthetic backgrounds are black, so low avg is ok
        "EXP_MAX_AVG": 230.0
    }
}

# --- MODEL LOADER ---
print("⏳ Initializing MUSIQ Model...")
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

try:
    IQA_MODEL = pyiqa.create_metric('musiq', device=DEVICE)
    print(f"✅ MUSIQ Loaded on {DEVICE}")
except Exception as e:
    print(f"❌ Error: {e}")
    exit(1)


def check_radiometrics(img_path, settings, debug_dir=None, save_name=None):
    """
    Checks for Saturation and Exposure anomalies.
    """
    img = cv2.imread(img_path)
    if img is None: return False, "None"

    # Convert to HSV for Saturation/Value analysis
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Mask out the black background (for synthetic) so it doesn't skew stats
    # We only analyze pixels that are NOT (0,0,0)
    mask = np.all(img != [0, 0, 0], axis=2)
    valid_pixels = np.sum(mask)

    if valid_pixels == 0: return False, "Empty Image"

    # --- 1. SATURATION CHECK ---
    # Metric A: Average Saturation
    avg_sat = np.mean(s[mask])

    # Metric B: Clipped Saturation (Percentage of pixels at 255)
    clipped_pixels = np.sum((s > 254) & mask)
    clipped_ratio = clipped_pixels / valid_pixels

    if avg_sat > settings['SAT_MAX_AVG']:
        return True, f"Oversaturated (Avg: {avg_sat:.1f})"

    if clipped_ratio > settings['SAT_CLIPPED']:
        return True, f"Neon/Fried ({clipped_ratio:.1%} pixels clipped)"

    # --- 2. EXPOSURE CHECK ---
    # Metric A: Average Brightness
    avg_val = np.mean(v[mask])

    if avg_val > settings['EXP_MAX_AVG']:
        return True, f"Overexposed (Avg: {avg_val:.1f})"

    # Only check underexposure if not synthetic (synthetic backgrounds skew this)
    # or rely on the masked average.
    if avg_val < settings['EXP_MIN_AVG']:
        return True, f"Underexposed (Avg: {avg_val:.1f})"

    # --- VISUAL DEBUG ---
    if debug_dir:
        fname = save_name if save_name else os.path.basename(img_path)

        # Create a heatmap of Saturation to show where the "Fried" parts are
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.title("Original")
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title("Saturation Heatmap")
        plt.imshow(s, cmap='inferno')
        plt.colorbar()
        plt.axis('off')

        save_path = os.path.join(debug_dir, fname)
        plt.savefig(save_path)
        plt.close()

    return False, "Pass"


def crop_to_content(img_path):
    img = cv2.imread(img_path)
    if img is None: return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return img_path
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    pad = 10
    h_img, w_img = img.shape[:2]
    x = max(0, x - pad);
    y = max(0, y - pad)
    w = min(w_img - x, w + 2 * pad);
    h = min(h_img - y, h + 2 * pad)
    cropped = img[y:y + h, x:x + w]
    temp_path = img_path.replace(".png", "_temp_crop.png")
    cv2.imwrite(temp_path, cropped)
    return temp_path


def get_quality_score(img_path, use_crop):
    target = img_path
    if use_crop:
        cropped = crop_to_content(img_path)
        if cropped: target = cropped
    with torch.no_grad():
        score = IQA_MODEL(target)
    if use_crop and target != img_path and os.path.exists(target):
        os.remove(target)
    return score.item()


def process_batch(args):
    input_dir = args.input_dir
    mode = args.mode
    settings = THRESHOLDS[mode]
    search_dir = os.path.join(input_dir, "train")
    if not os.path.exists(search_dir): search_dir = input_dir

    files = [f for f in os.listdir(search_dir) if f.lower().endswith(tuple(SUPPORTED_EXTS))]
    files.sort()

    if not os.path.exists(args.out_dir): os.makedirs(args.out_dir)
    processed_dir = os.path.join(args.out_dir, "processed_train")
    if not os.path.exists(processed_dir): os.makedirs(processed_dir)

    debug_dir = None
    if args.debug:
        debug_dir = os.path.join(args.out_dir, "debug_visuals")
        if not os.path.exists(debug_dir): os.makedirs(debug_dir)

    print(f"📂 Scanning {len(files)} images...")

    manifest_data = []

    for fname in tqdm(files):
        img_path = os.path.join(search_dir, fname)
        decision = "NONE"
        note = ""
        score = 0.0

        try:
            # 1. Check Radiometrics (Saturation/Exposure)
            if settings['CHECK_COLOR']:
                is_bad_color, reason = check_radiometrics(img_path, settings, debug_dir)
                if is_bad_color:
                    decision = "REPAIR"
                    note = reason

            # 2. Check Quality (Blur/Noise) - Only if color is okay
            if decision == "NONE":
                score = get_quality_score(img_path, settings['USE_CROP'])
                score = round(score, 2)
                if score < settings['BAD_LIMIT']:
                    decision = "REPAIR"
                    note = f"Blurry/Noisy (Score {score})"

            shutil.copy(img_path, os.path.join(processed_dir, fname))
            manifest_data.append({"filename": fname, "score": score, "decision": decision, "note": note})

        except Exception as e:
            print(f"⚠️ Error {fname}: {e}")

    with open(os.path.join(args.out_dir, "manifest.json"), 'w') as f:
        json.dump(manifest_data, f, indent=2)

    repairs = len([x for x in manifest_data if x['decision'] == 'REPAIR'])
    print(f"\n📊 SUMMARY: Detected {repairs} / {len(files)}")
    print(f"✅ Processed images: {processed_dir}")


def test_single_image(image_path, mode, debug=False, out_dir="./output"):
    if not os.path.exists(image_path): return print("❌ Error: Not found")
    settings = THRESHOLDS[mode]
    print(f"\n🔎 Analyzing: {image_path}")

    debug_dir = out_dir if debug else None
    if debug_dir and not os.path.exists(debug_dir): os.makedirs(debug_dir)

    # 1. Radiometrics
    if settings['CHECK_COLOR']:
        is_bad_color, reason = check_radiometrics(image_path, settings, debug_dir, "debug_single.png")
        print(f"🎨 Radiometric Check: {reason}")
        if is_bad_color:
            print("🚩 Result: REPAIR")
            if debug: print(f"   🐛 Visual saved to {os.path.join(out_dir, 'debug_single.png')}")
            return

    # 2. Quality
    score = get_quality_score(image_path, settings['USE_CROP'])
    print(f"📊 Quality Score: {score:.2f}")
    if score < settings['BAD_LIMIT']:
        print("🚩 Result: REPAIR (Blurry)")
    else:
        print("✅ Result: PASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["natural", "synthetic"], default="synthetic")
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--test_image", type=str)
    parser.add_argument("--out_dir", type=str, default="./output")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.test_image:
        test_single_image(args.test_image, args.mode, args.debug, args.out_dir)
    elif args.input_dir:
        process_batch(args)
    else:
        print("❌ Provide --input_dir or --test_image")