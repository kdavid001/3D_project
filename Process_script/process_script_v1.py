#!/usr/bin/env python3
"""
load_images_v9.py - Geometry-Aware Screening
(Solves the "Hole Blending" issue using Convex Hulls)

Logic:
1. Finds the object.
2. Wraps a 'Convex Hull' (rubber band) around it.
3. Checks for Black Pixels INSIDE that hull.
   - Outside Black = Background (Ignore)
   - Inside Black  = Corruption (Flag!)

Usage:
    python load_images_v9.py --input_dir ./data --mode synthetic
"""

import os
import json
import argparse
import torch
import pyiqa
import cv2
import numpy as np
from tqdm import tqdm

# --- CONFIGURATION ---
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png"}

THRESHOLDS = {
    "natural": {
        "BAD_LIMIT": 40.0,
        "GOOD_LIMIT": 70.0,
        "USE_CROP": False,
        "CHECK_HOLES": True,
        "HOLE_THRESHOLD": 0.05,
        "SAT_MIN": 5.0,
        "SAT_MAX": 170.0
    },
    "synthetic": {
        "BAD_LIMIT": 65.0,
        "GOOD_LIMIT": 71.0,
        "USE_CROP": True,
        "CHECK_HOLES": True,  # <--- ENABLED NOW (Using Geometry Logic)
        "HOLE_THRESHOLD": 0.21,  # If >2% of the object's internal area is missing -> REPAIR
        "SAT_MIN": 0.0,
        "SAT_MAX": 255.0
    }
}

# --- MODEL LOADER ---
print("⏳ Initializing MUSIQ Model...")
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
try:
    IQA_MODEL = pyiqa.create_metric('musiq', device=DEVICE)
    print(f"✅ MUSIQ Loaded on {DEVICE}")
except Exception as e:
    print(f"❌ Error: {e}")
    exit(1)


def check_saturation(img_path, min_sat, max_sat):
    """Checks for extreme saturation"""
    img = cv2.imread(img_path)
    if img is None: return "PASS", 0.0
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s_channel = hsv[:, :, 1]
    mask = np.all(img != [0, 0, 0], axis=2)
    if np.sum(mask) == 0: return "PASS", 0.0
    avg_sat = np.mean(s_channel[mask])
    if avg_sat > max_sat:
        return "HIGH", avg_sat
    elif avg_sat < min_sat:
        return "LOW", avg_sat
    return "PASS", avg_sat


def detect_holes_geometry(img_path, threshold, mode):
    """
    Advanced Hole Detection:
    Distinguishes 'Background Black' from 'Corruption Black' using Convex Hulls.
    """
    img = cv2.imread(img_path)
    if img is None: return False, 0.0

    # 1. Create a Binary Mask of the Object (Everything NOT black)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)

    # 2. Find Contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return False, 0.0  # Empty image

    # 3. Find the Largest Object (The Lego Truck)
    largest_contour = max(contours, key=cv2.contourArea)

    # 4. Calculate Convex Hull (The 'Rubber Band' around the object)
    hull = cv2.convexHull(largest_contour)

    # 5. Create a Mask of the Hull (Solid white shape)
    hull_mask = np.zeros_like(gray)
    cv2.drawContours(hull_mask, [hull], -1, 255, thickness=cv2.FILLED)

    # 6. LOGIC:
    # The 'hull_mask' represents where the object SHOULD be solid.
    # The 'binary' represents where pixels ACTUALLY exist.
    # Difference = HOLES.

    # Logic: (Inside Hull) AND (Is Black)
    # hull_mask is 255 where object should be.
    # binary is 0 where pixel is black.
    # We want pixels where hull_mask is 255 BUT binary is 0.
    holes = cv2.bitwise_xor(hull_mask, binary)
    holes = cv2.bitwise_and(holes, hull_mask)  # Ensure we only look inside the hull

    # Calculate Ratio of holes relative to the object size
    hull_area = np.sum(hull_mask > 0)
    if hull_area == 0: return False, 0.0

    hole_pixels = np.sum(holes > 0)
    ratio = hole_pixels / hull_area

    # If simple natural mode, fallback to global black counting if needed,
    # but this geometry method is superior for objects.
    return ratio > threshold, ratio


def crop_to_content(img_path):
    """Removes black void background"""
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


def test_single_image(image_path, mode):
    if not os.path.exists(image_path):
        return print(f"❌ Error 🗂️{image_path} does not exist")
    settings = THRESHOLDS[mode]
    print(f"\n🔎 Analyzing: {image_path}")
    print(f"⚙️  Mode: {mode.upper()}")

    # 1. Geometry Integrity Check
    if settings['CHECK_HOLES']:
        has_holes, ratio = detect_holes_geometry(image_path, settings['HOLE_THRESHOLD'], mode)
        print(f"⚫ Internal Hole Ratio: {ratio:.1%}")
        if has_holes:
            print(f"🚩 Result: REPAIR (Corrupted)")
            print(f"   -> Reason: {ratio:.1%} of the object's interior is missing.")
            return

    # 2. Saturation
    sat_status, sat_val = check_saturation(image_path, settings['SAT_MIN'], settings['SAT_MAX'])
    print(f"🎨 Saturation Level: {sat_val:.1f}")
    if sat_status == "HIGH":
        print(f"🚩 Result: REPAIR (Oversaturated)")
        return

    # 3. Quality (AI)
    score = get_quality_score(image_path, settings['USE_CROP'])
    print(f"📊 MUSIQ Quality: {score:.2f}")
    if score < settings['BAD_LIMIT']:
        print(f"🚩 Result: REPAIR (Blurry/Noisy)")
    elif score > settings['GOOD_LIMIT']:
        print(f"✨ Result: NOVEL_VIEW (Perfect)")
    else:
        print(f"✅ Result: PASS")


def process_batch(args):
    input_dir = args.input_dir
    mode = args.mode
    settings = THRESHOLDS[mode]

    search_dir = os.path.join(input_dir, "train")
    if not os.path.exists(search_dir): search_dir = input_dir
    if not os.path.exists(search_dir): return

    files = [f for f in os.listdir(search_dir) if f.lower().endswith(tuple(SUPPORTED_EXTS))]
    files.sort()

    print(f"📂 Scanning {len(files)} images...")
    if not os.path.exists(args.out_dir): os.makedirs(args.out_dir)

    manifest_data = []

    for fname in tqdm(files):
        img_path = os.path.join(search_dir, fname)
        decision = "NONE"
        note = ""
        score = 0.0

        try:
            # 1. Geometry Integrity
            if settings['CHECK_HOLES']:
                has_holes, ratio = detect_holes_geometry(img_path, settings['HOLE_THRESHOLD'], mode)
                if has_holes:
                    decision = "REPAIR"
                    note = f"Geometry Hole ({ratio:.1%})"

            # 2. Saturation
            if decision == "NONE":
                sat_status, sat_val = check_saturation(img_path, settings['SAT_MIN'], settings['SAT_MAX'])
                if sat_status == "HIGH":
                    decision = "REPAIR"
                    note = f"Oversaturated ({sat_val:.1f})"

            # 3. AI Quality
            if decision == "NONE":
                score = get_quality_score(img_path, settings['USE_CROP'])
                score = round(score, 2)

                if score < settings['BAD_LIMIT']:
                    decision = "REPAIR"
                    note = "Blurry/Noisy (MUSIQ)"
                elif score > settings['GOOD_LIMIT']:
                    decision = "NOVEL_VIEW"

            manifest_data.append({
                "filename": fname,
                "score": score,
                "decision": decision,
                "note": note
            })

        except Exception as e:
            print(f"⚠️ Error {fname}: {e}")

    with open(os.path.join(args.out_dir, "manifest.json"), 'w') as f:
        json.dump(manifest_data, f, indent=2)

    repairs = len([x for x in manifest_data if x['decision'] == 'REPAIR'])
    print(f"\n✅ Done. Flagged {repairs} images for repair.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["natural", "synthetic"], default="synthetic")
    parser.add_argument("--test_image", type=str)
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--out_dir", type=str, default="./output")
    args = parser.parse_args()

    if args.test_image:
        test_single_image(args.test_image, args.mode)
    elif args.input_dir:
        process_batch(args)
    else:
        print("❌ Provide --input_dir or --test_image")