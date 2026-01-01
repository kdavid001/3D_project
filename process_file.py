#!/usr/bin/env python3
"""
load_images_v25.py - The "Three-Pillar" Screener & Router
(Checks Saturation + Exposure + Quality)

Updates:
1. Unified 'is_fail' logic: Checks Color OR Light OR Quality.
2. AUTO-ROUTING:
   - FAILED images -> tagged "REPAIR"
   - PASSED images -> tagged "NOVEL_VIEW" (Ready for augmentation)
3. Debug Visuals now show MUSIQ score on the image.

Usage:
    python load_images_v25.py --input_dir ./data --mode synthetic --debug
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

# Prevent crashes on headless servers
plt.switch_backend('Agg')

# --- CONFIGURATION ---
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png"}

THRESHOLDS = {
    "natural": {
        "BAD_LIMIT": 40.0,  # MUSIQ Score < 40 = FAIL (Blur/Noise)
        "GOOD_LIMIT": 70.0,
        "USE_CROP": False,
        "CHECK_COLOR": True,
        "SAT_MAX_AVG": 170.0,
        "SAT_CLIPPED": 0.05,
        "EXP_MIN_AVG": 30.0,
        "EXP_MAX_AVG": 220.0
    },
    "synthetic": {
        "BAD_LIMIT": 65.0,  # Stricter for synthetic
        "GOOD_LIMIT": 71.0,
        "USE_CROP": True,
        "CHECK_COLOR": True,
        "SAT_MAX_AVG": 180.0,
        "SAT_CLIPPED": 0.08,
        "EXP_MIN_AVG": 10.0,
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


def analyze_image(img_path, settings, debug_dir=None, save_name=None):
    """
    Checks ALL 3 Failure Modes:
    1. Saturation (Fried)
    2. Exposure (Blown/Dark)
    3. Quality (Blur/Noise via MUSIQ)
    """
    img = cv2.imread(img_path)
    if img is None: return False, "None", 0.0

    # --- 1. GET MUSIQ SCORE (Blur/Noise) ---
    score = get_quality_score(img_path, settings['USE_CROP'])
    score = round(score, 2)

    is_qual_fail = score < settings['BAD_LIMIT']
    qual_reason = f"Low Quality ({score} < {settings['BAD_LIMIT']})" if is_qual_fail else "Quality Pass"

    # --- 2. RADIOMETRIC CHECKS (Sat/Exp) ---
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    mask = np.all(img != [0, 0, 0], axis=2)  # Ignore background
    valid_pixels = np.sum(mask)

    if valid_pixels == 0: return False, "Empty", 0.0

    # Saturation
    avg_sat = np.mean(s[mask])
    clipped_pixels = np.sum((s >= 250) & mask)
    clipped_ratio = clipped_pixels / valid_pixels

    is_sat_fail = False
    sat_reason = "Sat Pass"
    if avg_sat > settings['SAT_MAX_AVG']:
        is_sat_fail = True
        sat_reason = f"Oversaturated (Avg {avg_sat:.1f})"
    elif clipped_ratio > settings['SAT_CLIPPED']:
        is_sat_fail = True
        sat_reason = f"Neon Clip ({clipped_ratio:.1%})"

    # Exposure
    avg_val = np.mean(v[mask])
    is_exp_fail = False
    exp_reason = "Exp Pass"
    if avg_val > settings['EXP_MAX_AVG']:
        is_exp_fail = True
        exp_reason = f"Overexposed ({avg_val:.1f})"
    elif avg_val < settings['EXP_MIN_AVG']:
        is_exp_fail = True
        exp_reason = f"Underexposed ({avg_val:.1f})"

    # --- 3. FINAL DECISION ---
    # Fails if ANY of the 3 checks fail
    is_fail = is_sat_fail or is_exp_fail or is_qual_fail

    # Priority for Reason String: Quality -> Sat -> Exp
    final_reason = "Pass"
    if is_fail:
        if is_qual_fail:
            final_reason = qual_reason
        elif is_sat_fail:
            final_reason = sat_reason
        elif is_exp_fail:
            final_reason = exp_reason

    # --- VISUAL DEBUG ---
    if debug_dir:
        fname = save_name if save_name else os.path.basename(img_path)
        fig, ax = plt.subplots(1, 3, figsize=(18, 5))

        # Panel 1: Original + Pass/Fail Status
        status_color = 'red' if is_fail else 'green'
        ax[0].set_title(f"{'FAIL' if is_fail else 'PASS'}\n{final_reason}",
                        color=status_color, fontweight='bold', fontsize=12)
        ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax[0].axis('off')

        # Panel 2: Saturation Heatmap
        ax[1].set_title(f"Sat Clipping: {clipped_ratio:.1%}")
        im = ax[1].imshow(s, cmap='inferno', vmin=0, vmax=255)
        plt.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)
        ax[1].axis('off')

        # Panel 3: Exposure Histogram
        ax[2].set_title(f"Exposure (Avg: {avg_val:.1f})")
        ax[2].hist(v[mask].ravel(), bins=256, range=[0, 256], color='gray', alpha=0.8)
        # Threshold Lines
        ax[2].axvline(settings['EXP_MIN_AVG'], color='red', linestyle='--', label='Min')
        ax[2].axvline(settings['EXP_MAX_AVG'], color='red', linestyle='--', label='Max')
        ax[2].axvline(avg_val, color='blue', linestyle='-', label='Avg')
        ax[2].set_xlim([0, 256])
        ax[2].legend()
        ax[2].grid(True, alpha=0.3)

        save_path = os.path.join(debug_dir, fname)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    return is_fail, final_reason, score


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

        try:
            # UNIFIED CHECK (Color + Light + Blur/Noise)
            is_fail, reason, score = analyze_image(img_path, settings, debug_dir)

            # --- AUTO ROUTING LOGIC ---
            if is_fail:
                decision = "REPAIR"
                note = reason
            else:
                decision = "NOVEL_VIEW"
                note = "High Quality - Selected for Novel View"

            # Copy processed image
            shutil.copy(img_path, os.path.join(processed_dir, fname))
            manifest_data.append({"filename": fname, "score": score, "decision": decision, "note": note})

        except Exception as e:
            print(f"⚠️ Error {fname}: {e}")

    with open(os.path.join(args.out_dir, "manifest.json"), 'w') as f:
        json.dump(manifest_data, f, indent=2)

    repairs = len([x for x in manifest_data if x['decision'] == 'REPAIR'])
    novels = len([x for x in manifest_data if x['decision'] == 'NOVEL_VIEW'])

    print(f"\n📊 SUMMARY:")
    print(f"   🔴 REPAIR:     {repairs}")
    print(f"   🟢 NOVEL_VIEW: {novels}")
    print(f"   ⚪ TOTAL:      {len(files)}")
    print(f"✅ Processed images: {processed_dir}")


def test_single_image(image_path, mode, debug=False, out_dir="./output"):
    if not os.path.exists(image_path): return print("❌ Error: Not found")
    settings = THRESHOLDS[mode]
    print(f"\n🔎 Analyzing: {image_path}")

    debug_dir = out_dir if debug else None
    if debug_dir and not os.path.exists(debug_dir): os.makedirs(debug_dir)

    # UNIFIED CHECK
    is_fail, reason, score = analyze_image(image_path, settings, debug_dir, "debug_single.png")

    print(f"📊 MUSIQ Score: {score}")
    if is_fail:
        print(f"🚩 Result: REPAIR ({reason})")
    else:
        print(f"✅ Result: NOVEL_VIEW (Pass)")

    if debug:
        print(f"🐛 Visual saved to {os.path.join(out_dir, 'debug_single.png')}")


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