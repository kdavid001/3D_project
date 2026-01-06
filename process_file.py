#!/usr/bin/env python3
"""
load_images_legend.py - Fixed for Natural Images with BETTER GRAPHS
(Includes Legends, Axis Labels, and Your Custom Thresholds)
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

# --- CONFIGURATION (YOUR EXACT SETTINGS) ---
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png"}

THRESHOLDS = {
    "natural": {
        "USE_CROP": False,
        "BAD_LIMIT": 40.0,
        "SAT_CLIPPED": 0.05,
        "SAT_MAX_AVG": 180.0,
        "EXP_MIN_AVG": 45.0,  # As requested
        "EXP_MAX_AVG": 250.0,
        "CONTRAST_MIN": 10.0,
        "CAST_LIMIT": 60.0
    },
    "synthetic": {
        "USE_CROP": True,
        # 1. QUALITY (MUSIQ)
        "BAD_LIMIT": 65.0,
        # 2. SATURATION (Neon Check)
        "SAT_MAX_AVG": 160.0,
        "SAT_CLIPPED": 0.02,
        # 3. EXPOSURE (Light Check)
        "EXP_MIN_AVG": 45.0,
        "EXP_MAX_AVG": 230.0,
        # 4. CONTRAST (Flatness Check)
        "CONTRAST_MIN": 25.0,
        # 5. COLOR CAST (Tint Check)
        "CAST_LIMIT": 50.0
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
    img = cv2.imread(img_path)
    if img is None: return False, "None", 0.0

    # --- 1. QUALITY ---
    score = get_quality_score(img_path, settings['USE_CROP'])
    score = round(score, 2)
    is_qual_fail = score < settings['BAD_LIMIT']
    qual_reason = f"Blurry ({score})" if is_qual_fail else "Sharp"

    # --- 2. RADIOMETRICS ---
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = np.all(img != [0, 0, 0], axis=2)
    valid_pixels = np.sum(mask)
    if valid_pixels == 0: return False, "Empty", 0.0

    # Saturation
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    avg_sat = np.mean(s[mask])
    clipped_ratio = np.sum((s >= 250) & mask) / valid_pixels
    is_sat_fail = avg_sat > settings['SAT_MAX_AVG'] or clipped_ratio > settings['SAT_CLIPPED']
    sat_reason = "Neon" if is_sat_fail else "Color OK"

    # Exposure
    v = hsv[:, :, 2]
    avg_val = np.mean(v[mask])
    is_exp_fail = avg_val > settings['EXP_MAX_AVG'] or avg_val < settings['EXP_MIN_AVG']
    exp_reason = "Bad Light" if is_exp_fail else "Light OK"

    # Contrast
    contrast_val = np.std(v[mask])
    is_cont_fail = contrast_val < settings['CONTRAST_MIN']
    cont_reason = "Flat" if is_cont_fail else "Contrasty"

    # --- 3. COLOR CAST ---
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b_chan = cv2.split(lab)
    avg_a = np.mean(a[mask]) - 128
    avg_b = np.mean(b_chan[mask]) - 128
    cast_score = np.sqrt(avg_a ** 2 + avg_b ** 2)
    is_cast_fail = cast_score > settings['CAST_LIMIT']
    cast_reason = f"Tinted ({cast_score:.1f})" if is_cast_fail else "Neutral"

    # --- 4. FINAL DECISION ---
    is_fail = is_sat_fail or is_exp_fail or is_qual_fail or is_cont_fail or is_cast_fail

    final_reason = "Pass"
    if is_fail:
        if is_cast_fail:
            final_reason = cast_reason
        elif is_qual_fail:
            final_reason = qual_reason
        elif is_sat_fail:
            final_reason = sat_reason
        elif is_exp_fail:
            final_reason = exp_reason
        elif is_cont_fail:
            final_reason = cont_reason

    # --- VISUAL DEBUG (NOW WITH LEGENDS) ---
    if debug_dir:
        fname = save_name if save_name else os.path.basename(img_path)
        fig = plt.figure(figsize=(16, 6))
        gs = fig.add_gridspec(1, 3)

        # Panel 1: Image
        ax1 = fig.add_subplot(gs[0, 0])
        status_color = 'red' if is_fail else 'green'
        ax1.set_title(f"{'REJECT' if is_fail else 'PASS'}: {final_reason}\nScore: {score}",
                      color=status_color, fontweight='bold', fontsize=14)
        ax1.imshow(img_rgb)
        ax1.axis('off')

        # Panel 2: RGB Histogram WITH LEGEND
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.set_title(f"Color Balance (Cast Score: {cast_score:.1f})")

        # Plot lines with labels
        colors = ('r', 'g', 'b')
        labels = ('Red Channel', 'Green Channel', 'Blue Channel')

        for i, color in enumerate(colors):
            hist = cv2.calcHist([img], [i], None, [256], [1, 256])
            ax2.plot(hist, color=color, linewidth=2, alpha=0.8, label=labels[i])
            ax2.fill_between(range(256), hist.flatten(), color=color, alpha=0.1)

        ax2.set_xlim([0, 256])
        ax2.grid(True, alpha=0.3)
        ax2.set_xlabel("Pixel Brightness (0=Dark, 255=Bright)")
        ax2.set_ylabel("Pixel Count")

        # Add the Legend!
        ax2.legend(loc='upper right', fontsize=9)

        if is_cast_fail:
            ax2.text(128, ax2.get_ylim()[1] * 0.8, "❌ UNBALANCED",
                     color='red', ha='center', fontweight='bold')

        # Panel 3: Stats
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.axis('off')
        ax3.set_title("Inspection Report")
        metrics = [
            ("Quality", score, settings['BAD_LIMIT'], ">"),
            ("Saturation", clipped_ratio * 100, settings['SAT_CLIPPED'] * 100, "<"),
            ("Brightness", avg_val, settings['EXP_MIN_AVG'], ">"),
            ("Contrast", contrast_val, settings['CONTRAST_MIN'], ">"),
            ("Color Tint", cast_score, settings['CAST_LIMIT'], "<")
        ]
        y_pos = 0.9
        for name, val, thresh, op in metrics:
            pass_metric = (val > thresh) if op == ">" else (val < thresh)
            icon = "✅" if pass_metric else "❌"
            text_color = "black" if pass_metric else "red"
            ax3.text(0.1, y_pos, f"{icon} {name}", fontsize=12, fontweight='bold')
            ax3.text(0.6, y_pos, f"{val:.1f}  (Limit {thresh:.1f})", fontsize=12, color=text_color)
            y_pos -= 0.15

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
            is_fail, reason, score = analyze_image(img_path, settings, debug_dir)
            decision = "REPAIR" if is_fail else "NOVEL_VIEW"
            note = reason

            if not is_fail:
                shutil.copy(img_path, os.path.join(processed_dir, fname))

            manifest_data.append({"filename": fname, "score": score, "decision": decision, "note": note})

        except Exception as e:
            print(f"⚠️ Error {fname}: {e}")

    with open(os.path.join(args.out_dir, "manifest.json"), 'w') as f:
        json.dump(manifest_data, f, indent=2)

    # --- STATISTICAL SUMMARY ---
    total_files = len(files)
    processed_count = len(manifest_data)
    repair_count = sum(1 for item in manifest_data if item['decision'] == 'REPAIR')
    novel_count = sum(1 for item in manifest_data if item['decision'] == 'NOVEL_VIEW')
    pass_rate = (novel_count / total_files) * 100 if total_files > 0 else 0

    print("\n" + "=" * 40)
    print(f"📊 FINAL DATASET REPORT")
    print("=" * 40)
    print(f"   ⚪ TOTAL IMAGES:      {total_files}")
    print(f"   🟢 PASSED (Novel):    {novel_count} ({pass_rate:.1f}%)")
    print(f"   🔴 FAILED (Repair):   {repair_count}")
    print("-" * 40)
    print(f"✅ Clean dataset: {processed_dir}")
    if args.debug:
        print(f"🐛 Debug Charts:  {debug_dir}")
    print("=" * 40 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["natural", "synthetic"], default="synthetic")
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--test_image", type=str)
    parser.add_argument("--out_dir", type=str, default="./output")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.test_image:
        analyze_image(args.test_image, THRESHOLDS[args.mode], args.out_dir, "debug_test.png")
    elif args.input_dir:
        process_batch(args)
    else:
        print("❌ Provide --input_dir or --test_image")