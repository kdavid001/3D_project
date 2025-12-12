#!/usr/bin/env python3
"""
load_images.py

Usage:
    python load_images.py --input_dir ./raw_images --out_dir ./preprocessed --target_size 1024 --resize_mode pad

What it does:
- loads all images in input_dir
- optionally resizes them to target_size (preserve aspect ratio by 'pad' or 'crop' or 'none')
- computes basic stats per image (shape, mean, std, brightness)
- if pose_file provided, tries to read and attach poses to manifest
- writes manifest.json with metadata for each image
- saves processed images to out_dir/images/

Dependencies:
    pip install pillow numpy opencv-python tqdm
"""

import os
import json
import argparse
from PIL import Image, ImageOps
import numpy as np
from tqdm import tqdm

# supported file types
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png"}


def is_image_file(filename):
    return os.path.splitext(filename.lower())[1] in SUPPORTED_EXTS


def load_images_list(input_dir):
    files = sorted(
        [
            os.path.join(input_dir, f)
            for f in os.listdir(input_dir)
            if is_image_file(f)
        ]
    )
    return files


def pil_to_np(img):
    arr = np.array(img)  # H W C or H W for grayscale
    if arr.ndim == 2:
        arr = np.expand_dims(arr, axis=-1)
    return arr


def compute_image_stats(np_img):
    # np_img expected H W C, dtype uint8
    arr = np_img.astype(np.float32) / 255.0
    mean = float(arr.mean())
    std = float(arr.std())
    # brightness as mean of luminance (simple)
    if arr.shape[2] == 1:
        lum = arr[..., 0]
    else:
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    brightness = float(lum.mean())
    return {"mean": mean, "std": std, "brightness": brightness}


def resize_image(img: Image.Image, target_size, mode="pad"):
    """
    mode: 'pad' (preserve aspect, pad to square), 'crop' (center-crop to square then resize),
          'stretch' (ignore aspect), 'none' (return original)
    target_size: int or (w,h)
    """
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
        # resize to fit inside target, then pad
        img.thumbnail((target_w, target_h), Image.LANCZOS)
        pad_w = target_w - img.width
        pad_h = target_h - img.height
        left = pad_w // 2
        top = pad_h // 2
        right = pad_w - left
        bottom = pad_h - top
        return ImageOps.expand(img, border=(left, top, right, bottom), fill=(0, 0, 0))
    elif mode == "crop":
        # center-crop to square then resize
        min_side = min(w, h)
        left = (w - min_side) // 2
        top = (h - min_side) // 2
        cropped = img.crop((left, top, left + min_side, top + min_side))
        return cropped.resize((target_w, target_h), Image.LANCZOS)
    else:
        raise ValueError("Unknown resize mode: " + str(mode))


def read_pose_file(pose_path):
    """
    Try to load a simple pose file:
    - If it's a COLMAP-style text file or an Nx7 (qw,qx,qy,qz,tx,ty,tz) format, attempt to parse.
    - Otherwise, return None.
    This function is intentionally permissive: it returns a dict mapping basename->pose or None.
    """
    if not os.path.exists(pose_path):
        return None
    poses = {}
    try:
        with open(pose_path, "r") as fh:
            lines = [l.strip() for l in fh if l.strip()]
        for l in lines:
            parts = l.split()
            # common simple format: filename tx ty tz qx qy qz qw  (or similar)
            # try to detect filename first
            if len(parts) >= 7 and any(parts[0].endswith(ext) for ext in SUPPORTED_EXTS):
                fname = parts[0]
                vals = list(map(float, parts[1:]))
                poses[os.path.basename(fname)] = vals
            else:
                # try "image_id qw qx qy qz tx ty tz camera_id"
                if len(parts) >= 8:
                    # skip image id and camera id heuristics - best-effort
                    # search for a token that looks like filename in later entries
                    for token in parts:
                        if any(token.endswith(ext) for ext in SUPPORTED_EXTS):
                            idx = parts.index(token)
                            fname = parts[idx]
                            vals = list(map(float, parts[:idx] + parts[idx+1:]))
                            poses[os.path.basename(fname)] = vals
                            break
    except Exception as e:
        print("Warning: could not parse pose file:", e)
        return None

    return poses if poses else None


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def main(args):
    input_dir = args.input_dir
    out_dir = args.out_dir
    target_size = args.target_size
    resize_mode = args.resize_mode

    ensure_dir(out_dir)
    images_out_dir = os.path.join(out_dir, "images")
    ensure_dir(images_out_dir)

    files = load_images_list(input_dir)
    if not files:
        print(f"No images found in {input_dir}. Supported: {SUPPORTED_EXTS}")
        return

    # try to read poses if provided
    poses = None
    if args.pose_file:
        poses = read_pose_file(args.pose_file)
        if poses:
            print(f"Loaded poses for {len(poses)} entries from {args.pose_file}")
        else:
            print("No usable poses parsed from pose file (continuing without poses).")

    manifest = {"images": [], "total_images": len(files)}
    print(f"Found {len(files)} images. Processing...")

    for path in tqdm(files):
        base = os.path.basename(path)
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"Could not open {path}: {e}")
            continue

        orig_w, orig_h = img.size
        proc_img = resize_image(img, target_size, resize_mode) if target_size else img
        proc_w, proc_h = proc_img.size

        np_img = pil_to_np(proc_img)
        stats = compute_image_stats(np_img)

        out_path = os.path.join(images_out_dir, base)
        proc_img.save(out_path, format="PNG")

        entry = {
            "filename": base,
            "input_path": path,
            "saved_path": out_path,
            "orig_width": orig_w,
            "orig_height": orig_h,
            "proc_width": proc_w,
            "proc_height": proc_h,
            "stats": stats,
        }
        if poses and base in poses:
            entry["pose"] = poses[base]
        manifest["images"].append(entry)

    # global stats
    all_means = [img["stats"]["mean"] for img in manifest["images"]]
    all_brightness = [img["stats"]["brightness"] for img in manifest["images"]]
    if manifest["images"]:
        manifest["global"] = {
            "mean_mean": float(np.mean(all_means)),
            "mean_brightness": float(np.mean(all_brightness)),
        }
    # write manifest
    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"Saved {len(manifest['images'])} processed images to {images_out_dir}")
    print(f"Manifest written to {manifest_path}")
    print("Done.")


if __name__ == "__process_file__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Folder with raw images")
    parser.add_argument("--out_dir", required=True, help="Folder to write preprocessed images and manifest")
    parser.add_argument("--target_size", type=int, default=None,
                        help="If set, resize images to this square size (e.g. 1024). Use resize_mode to control strategy.")
    parser.add_argument("--resize_mode", choices=["pad", "crop", "stretch", "none"], default="pad",
                        help="How to resize while preserving aspect.")
    parser.add_argument("--pose_file", default=None, help="Optional pose file (simple text).")
    args = parser.parse_args()
    main(args)