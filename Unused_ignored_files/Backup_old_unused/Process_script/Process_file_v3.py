#!/usr/bin/env python3
"""
load_images_v14_fixed.py - v14 Logic + Visual Debugging + Saving

Features:
1. Saves processed images to ./output/train/
2. DEBUG: Saves images with GREEN (Detected), RED (Rejected), and BLUE (Hull) outlines.
3. Fixes 'Manifest Reasons' not saving correctly.

Usage:
    Batch:  python load_images_v14_fixed.py --input_dir ./data --mode synthetic --debug
    Single: python load_images_v14_fixed.py --test_image ./data/r_73.png --mode synthetic --debug
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
        "CHECK_HOLES": True,
        "HOLE_THRESHOLD": 0.01,  # Ignore holes smaller than 1%
        "SAT_MIN": 0.0,
        "SAT_MAX": 255.0
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


def check_saturation(img_path, min_sat, max_sat):
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


def detect_holes_natural(img_path, threshold):
    img = cv2.imread(img_path)
    if img is None: return False, ""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return False, ""
    largest_contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(largest_contour)
    hull_mask = np.zeros_like(gray)
    cv2.drawContours(hull_mask, [hull], -1, 255, thickness=cv2.FILLED)
    holes = cv2.bitwise_xor(hull_mask, binary)
    holes = cv2.bitwise_and(holes, hull_mask)
    hull_area = np.sum(hull_mask > 0)
    if hull_area == 0: return False, ""
    ratio = np.sum(holes > 0) / hull_area
    return ratio > threshold, f"Ratio: {ratio:.1%}"


def detect_holes_synthetic_debug(img_path, min_area_ratio, debug_dir=None, save_name=None):
    """
    v14 Logic with FULL Visual Debugging (Green=Found, Red=Rejected).
    """
    img = cv2.imread(img_path)
    if img is None: return False, "None"

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)

    # Denoise
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return False, "Empty"
    largest_contour = max(contours, key=cv2.contourArea)
    obj_area = cv2.contourArea(largest_contour)
    if obj_area < 100: return False, "Too Small"

    debug_img = img.copy() if debug_dir else None
    detected_corruption = False
    reason = "Natural"

    # --- HULL CALCULATION ---
    hull = cv2.convexHull(largest_contour)
    hull_mask = np.zeros_like(gray)
    cv2.drawContours(hull_mask, [hull], -1, 255, thickness=cv2.FILLED)

    # Draw Hull in BLUE (for context)
    if debug_img is not None:
        cv2.drawContours(debug_img, [hull], -1, (255, 0, 0), 1)

    # --- A. CHECK INTERNAL HOLES ---
    holes_mask = cv2.bitwise_xor(hull_mask, binary)
    holes_mask = cv2.bitwise_and(holes_mask, hull_mask)
    hole_cnts, _ = cv2.findContours(holes_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in hole_cnts:
        area = cv2.contourArea(c)
        if area / obj_area < min_area_ratio: continue
        x, y, w, h = cv2.boundingRect(c)
        solidity = area / (w * h)

        if solidity > 0.85:
            detected_corruption = True
            reason = f"Internal Block (Solidity: {solidity:.2f})"
            if debug_img is not None:
                cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # GREEN
        else:
            if debug_img is not None:
                cv2.drawContours(debug_img, [c], -1, (0, 0, 255), 2)  # RED (Rejected)

    # --- B. CHECK EDGE HOLES ---
    defects_mask = cv2.bitwise_xor(hull_mask, binary)
    defects_mask = cv2.bitwise_and(defects_mask, hull_mask)
    defect_cnts, _ = cv2.findContours(defects_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in defect_cnts:
        area = cv2.contourArea(c)
        if area / obj_area < min_area_ratio: continue
        x, y, w, h = cv2.boundingRect(c)
        solidity = area / (w * h)

        if solidity > 0.90:
            detected_corruption = True
            reason = f"Edge Cut (Solidity: {solidity:.2f})"
            if debug_img is not None:
                cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # GREEN
        else:
            if debug_img is not None:
                cv2.drawContours(debug_img, [c], -1, (0, 0, 255), 2)  # RED (Rejected)

    if debug_dir:
        fname = save_name if save_name else os.path.basename(img_path)
        cv2.imwrite(os.path.join(debug_dir, fname), debug_img)

    return detected_corruption, reason


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
    processed_dir = os.path.join(args.out_dir, "train")
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
            # 1. Holes
            if settings['CHECK_HOLES']:
                is_artificial = False
                reason = ""
                if mode == "synthetic":
                    is_artificial, reason = detect_holes_synthetic_debug(img_path, settings['HOLE_THRESHOLD'],
                                                                         debug_dir)
                else:
                    is_artificial, reason = detect_holes_natural(img_path, settings['HOLE_THRESHOLD'])

                if is_artificial:
                    decision = "REPAIR"
                    note = reason

            # 2. Saturation
            if decision == "NONE":
                sat_status, sat_val = check_saturation(img_path, settings['SAT_MIN'], settings['SAT_MAX'])
                if sat_status == "HIGH":
                    decision = "REPAIR"
                    note = f"Oversaturated ({sat_val:.1f})"

            # 3. Quality
            if decision == "NONE":
                score = get_quality_score(img_path, settings['USE_CROP'])
                score = round(score, 2)
                if score < settings['BAD_LIMIT']:
                    decision = "REPAIR"
                    note = f"Blurry/Noisy (Score {score})"

            # Copy processed image
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

    if settings['CHECK_HOLES']:
        is_artificial = False
        reason = ""
        if mode == "synthetic":
            # Save single debug image
            is_artificial, reason = detect_holes_synthetic_debug(image_path, settings['HOLE_THRESHOLD'], debug_dir,
                                                                 "debug_single.png")
        else:
            is_artificial, reason = detect_holes_natural(image_path, settings['HOLE_THRESHOLD'])

        print(f"⚫ Geometry Check: {reason}")
        if is_artificial:
            print("🚩 Result: REPAIR (Geometry)")
            return

    sat_status, sat_val = check_saturation(image_path, settings['SAT_MIN'], settings['SAT_MAX'])
    print(f"🎨 Saturation: {sat_val:.1f}")
    if sat_status == "HIGH":
        print("🚩 Result: REPAIR (Oversaturated)")
        return

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