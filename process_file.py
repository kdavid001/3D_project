#!/usr/bin/env python3
"""
load_images.py

Usage:
    python load_images.py --input_dir ./raw_images --out_dir ./preprocessed --target_size 1024 --resize_mode pad

What it does:
- loads all images in input_dir
- optionally resizes them to target_size (preserve aspect ratio by 'pad' or 'crop' or 'none')
- computes basic stats per image (shape, mean, std, brightness)
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
import struct
import cv2

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


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_colmap_images_bin(path):
    images = {}
    with open(path, "rb") as fid:
        num_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_images):
            image_id = read_next_bytes(fid, 4, "I")[0]
            qw, qx, qy, qz = read_next_bytes(fid, 32, "dddd")
            tx, ty, tz = read_next_bytes(fid, 24, "ddd")
            camera_id = read_next_bytes(fid, 4, "I")[0]

            name = b""
            while True:
                c = fid.read(1)
                if c == b"\x00":
                    break
                name += c
            name = name.decode("utf-8")

            num_points2D = read_next_bytes(fid, 8, "Q")[0]
            fid.read(num_points2D * 24)  # skip points2D

            images[name] = {
                "qvec": [qw, qx, qy, qz],
                "tvec": [tx, ty, tz],
                "camera_id": camera_id,
            }
    return images


def read_colmap_cameras_bin(path):
    cameras = {}
    with open(path, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            cam_id = read_next_bytes(fid, 4, "I")[0]
            model_id = read_next_bytes(fid, 4, "i")[0]
            width = read_next_bytes(fid, 8, "Q")[0]
            height = read_next_bytes(fid, 8, "Q")[0]
            num_params = read_next_bytes(fid, 8, "Q")[0]
            params = read_next_bytes(fid, 8 * num_params, "d" * num_params)
            cameras[cam_id] = {
                "model_id": model_id,
                "width": width,
                "height": height,
                "params": list(params),
            }
    return cameras


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


# quality metric functions
"""
This section of code functions are to create a quality metric check
for each images so that the diffusion model would only be triggered for this images

"""


def laplacian_variance(gray):
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def image_entropy(gray):
    hist = np.histogram(gray, bins=256, range=(0, 1), density=True)[0]
    hist = hist[hist > 0]
    return -np.sum(hist * np.log2(hist))


def edge_density(gray):
    edges = cv2.Canny((gray * 255).astype(np.uint8), 50, 150)
    return edges.mean() / 255.0


# End of Quality metric functions

def compute_quality_metrics(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    return {
        "blur": laplacian_variance(gray),
        "entropy": image_entropy(gray),
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "edge_density": edge_density(gray),
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


def screen_image(metrics, dataset_stats, z_thresh=2.0, min_flags=2):
    flags = []

    for k, v in metrics.items():
        z = abs(v - dataset_stats[k]["mean"]) / dataset_stats[k]["std"]
        if z > z_thresh:
            flags.append(k)

    return {
        "needs_diffusion": len(flags) >= min_flags,
        "flags": flags
    }


def main_function(args, np_img):
    input_dir = args.input_dir
    out_dir = args.out_dir
    colmap_metadata = args.use_colmap_metadata
    target_size = args.target_size
    resize_mode = args.resize_mode

    potential_dir = os.path.join(input_dir, "images")
    if os.path.exists(potential_dir):
        input_images = os.listdir(potential_dir)
    else:
        input_images = args.input_images

    ensure_dir(out_dir) # just for the output directory
    images_out_dir = os.path.join(out_dir, "images")
    ensure_dir(images_out_dir)

    image_files = [
        f for f in os.listdir(input_images)
        if f.lower().endswith(SUPPORTED_EXTS)
    ]

    files = load_images_list(input_dir)
    if not files:
        print(f"No images found in {input_dir}. Supported: {SUPPORTED_EXTS}")
        return

    quality = compute_quality_metrics(np_img)
    entry["quality"] = quality
    quality_list.append(quality)

    dataset_stats = compute_dataset_stats(quality_list)

    for entry in manifest["images"]:
        decision = screen_image(entry["quality"], dataset_stats)
        entry["screening"] = decision

    for image in dataset:
        if image.screening.needs_diffusion:
            run diffusion
        else:
            skip


def main(args):
    input_dir = args.input_dir
    out_dir = args.out_dir
    colmap_metadata = args.use_colmap_metadata
    target_size = args.target_size
    resize_mode = args.resize_mode

    ensure_dir(out_dir)
    images_out_dir = os.path.join(out_dir, "images")
    ensure_dir(images_out_dir)

    files = load_images_list(input_dir)
    if not files:
        print(f"No images found in {input_dir}. Supported: {SUPPORTED_EXTS}")
        return

    if colmap_metadata == 1:
        colmap_dir = os.path.join(args.input_dir, "sparse", "0")
        colmap_images = None
        cameras = None
        images_bin_path = os.path.join(colmap_dir, "images.bin")
        cameras_bin_path = os.path.join(colmap_dir, "cameras.bin")
        if os.path.exists(images_bin_path) and os.path.exists(cameras_bin_path):
            try:
                colmap_images = read_colmap_images_bin(images_bin_path)
                cameras = read_colmap_cameras_bin(cameras_bin_path)
                print(f"Loaded COLMAP data: {len(colmap_images)} images, {len(cameras)} cameras")
            except Exception as e:
                print(f"Failed to load COLMAP data: {e}")
        else:
            print("COLMAP binary files not found; continuing without COLMAP data.")

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
        if colmap_images and base in colmap_images:
            entry["pose"] = colmap_images[base]
            entry["camera"] = cameras[colmap_images[base]["camera_id"]]
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Folder with raw images")
    parser.add_argument("--out_dir", required=True, help="Folder to write preprocessed images and manifest")
    parser.add_argument("----use_colmap_metadata", default=1, type=int, help="Whether to use colmap metadata"
                                                                             "use '0'->False, '1'->True this accepts "
                                                                             "integers only")
    parser.add_argument("--target_size", type=int, default=None,
                        help="If set, resize images to this square size (e.g. 1024). Use resize_mode to control"
                             " strategy.")
    parser.add_argument("--resize_mode", choices=["pad", "crop", "stretch", "none"], default="pad",
                        help="How to resize while preserving aspect.")
    args = parser.parse_args()
    main(args)
