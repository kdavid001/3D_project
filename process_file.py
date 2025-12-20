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
import math

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


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


# quality metric functions
"""
This section of code functions are to create a quality metric check
for each images so that the diffusion model would only be triggered for this images

"""


def laplacian_variance(gray):
    gray_u8 = (gray * 255).astype(np.uint8)
    """ 
    Note: gray is float32 in range [0, 1] OpenCV’s optimized Laplacian 
    path does not support this specific source → destination 
    combination on macOS builds
    """
    return cv2.Laplacian(gray_u8, cv2.CV_64F).var()


def image_entropy(gray):
    hist = np.histogram(gray, bins=256, range=(0, 1))[0]
    prob = hist / (hist.sum() + 1e-8)
    prob = prob[prob > 0]
    return float(-np.sum(prob * np.log2(prob)))


def edge_density(gray):
    edges = cv2.Canny((gray * 255).astype(np.uint8), 50, 150)
    return edges.mean() / 255.0


def saturation_ratio(gray, low=0.02, high=0.98):
    return np.mean((gray < low) | (gray > high))


def motion_blur_score(gray):
    sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(sobelx ** 2 + sobely ** 2)
    return float(mag.mean())


# For black holes, this part doesn't matter for real-life sinerios
def low_texture_ratio(gray, thresh=0.01):
    return np.mean(gray.std(axis=0) < thresh)


# End of Quality metric functions


def compute_dataset_stats(metrics_list):
    stats = {}
    for key in metrics_list[0]:
        values = np.array([m[key] for m in metrics_list])
        stats[key] = {
            "mean": float(values.mean()),
            "std": float(values.std() + 1e-6),
        }
    return stats


def compute_quality_metrics(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    return {
        "blur": laplacian_variance(gray),
        "motion_blur": motion_blur_score(gray),
        "entropy": image_entropy(gray),
        "brightness": float(gray.mean()),  # works on dark images
        "contrast": float(gray.std()),
        "edge_density": edge_density(gray),
        "saturation_ratio": saturation_ratio(gray),  # This is more of high saturation checker
        "low_texture_ratio": low_texture_ratio(gray),
    }


def screen_image(metrics, dataset_stats, z_thresh, min_flags):
    flags = []
    STRUCTURAL_KEYS = {
        "blur", "motion_blur", "edge_density",
        "entropy", "saturation_ratio", "low_texture_ratio"
    }

    for k, v in metrics.items():
        z = abs(v - dataset_stats[k]["mean"]) / dataset_stats[k]["std"]
        if z > z_thresh:
            flags.append(k)

    structural_flags = [f for f in flags if f in STRUCTURAL_KEYS]

    decision = "NONE"

    # --- REPAIR conditions ---
    if (
            "blur" in flags or
            "motion_blur" in flags or
            "low_texture_ratio" in flags or
            "saturation_ratio" in flags
    ):
        decision = "REPAIR"

    # --- NOVEL VIEW conditions ---
    elif (
            metrics["entropy"] > dataset_stats["entropy"]["mean"] and
            metrics["edge_density"] > dataset_stats["edge_density"]["mean"]
    ):
        decision = "NOVEL_VIEW"

    needs_diffusion = decision != "NONE"

    return {
        "needs_diffusion": needs_diffusion,
        "flags": flags,
        "decision": decision
    }


manifest = {
    "images": []
}
quality_list = []


def process_func(args):
    input_dir = args.input_dir
    out_dir = args.out_dir
    # colmap_metadata = args.use_colmap_metadata
    target_size = args.target_size
    resize_mode = args.resize_mode

    potential_dir = os.path.join(input_dir, "images")
    if os.path.exists(potential_dir):
        input_images_dir = potential_dir
    else:
        input_images_dir = input_dir

    ensure_dir(out_dir)  # just for the output directory
    images_out_dir = os.path.join(out_dir, "images")
    ensure_dir(images_out_dir)

    files = load_images_list(input_images_dir)
    if not files:
        print(f"No images found in {input_images_dir}. Supported: {SUPPORTED_EXTS}")
        return

    for path in tqdm(files):
        base = os.path.basename(path)
        try:
            img = Image.open(path).convert("RGB")
            proc_img = resize_image(img, target_size, resize_mode) if target_size else img
            np_img = pil_to_np(proc_img)

            out_path = os.path.join(images_out_dir, base)
            proc_img.save(out_path)
            quality = compute_quality_metrics(np_img)

            entry = {
                "filename": base,
                "quality": quality,
            }

            manifest["images"].append(entry)
            quality_list.append(quality)

        except Exception as e:
            print(f"Error processing {base}: {e}")
            continue

    dataset_stats = compute_dataset_stats(quality_list)

    for entry in manifest["images"]:
        # TODO: Edit this for loop if you aren't getting
        #  good results for the selection process

        decision = screen_image(
            entry["quality"],
            dataset_stats,
            z_thresh=1.5,  # 2 is recommended, but detection works better with 1.5.
            min_flags=2
        )
        entry["screening"] = decision
        # print(entry["filename"], decision["needs_diffusion"], decision["flags"])

        # --- mask generation ---
        mask_dir = os.path.join(out_dir, "masks")
        ensure_dir(mask_dir)

        if decision["needs_diffusion"]:
            # reload processed image to generate mask
            img_path = os.path.join(images_out_dir, entry["filename"])
            img_rgb = np.array(Image.open(img_path).convert("RGB"))

            mask = generate_repair_mask(
                img_rgb,
                decision["flags"]
            )

            mask_name = entry["filename"].rsplit(".", 1)[0] + ".png"
            mask_path = os.path.join(mask_dir, mask_name)
            Image.fromarray(mask).save(mask_path)

            entry["mask_path"] = f"masks/{mask_name}"
        else:
            entry["mask_path"] = None

    num_flagged = sum(
        1 for e in manifest["images"]
        if e["screening"]["needs_diffusion"]
    )

    print(f"{num_flagged} / {len(manifest['images'])} images flagged for diffusion")
    # Save the manifest with updated screening and mask info
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest


# For mask generation
def blur_mask(gray, thresh=30):
    lap = cv2.Laplacian((gray * 255).astype(np.uint8), cv2.CV_32F)
    mag = np.abs(lap)
    mask = mag < thresh  # low detail = needs repair
    return (mask.astype(np.uint8)) * 255


def saturation_mask(gray, low=0.02, high=0.98):
    mask = (gray < low) | (gray > high)
    return (mask.astype(np.uint8)) * 255


def low_texture_mask(gray, std_thresh=0.005):
    mean = cv2.GaussianBlur(gray, (15, 15), 0)
    sq_mean = cv2.GaussianBlur(gray ** 2, (15, 15), 0)
    local_var = sq_mean - mean ** 2
    mask = local_var < std_thresh
    return (mask.astype(np.uint8)) * 255


def generate_repair_mask(img_rgb, flags):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    masks = []

    if "blur" in flags or "confirmed_blur" in flags:
        masks.append(blur_mask(gray))

    if "low_texture_ratio" in flags:
        masks.append(low_texture_mask(gray))

    if masks:
        final_mask = np.maximum.reduce(masks)
        # small dilation to give context but not too big
        kernel = np.ones((3, 3), np.uint8)
        final_mask = cv2.dilate(final_mask, kernel, iterations=1)
    else:
        final_mask = np.zeros_like(gray, dtype=np.uint8)

    return final_mask.astype(np.uint8)

    # if "missing_region" in flags:
    #     masks.append(saturation_mask(gray))
    #     masks.append(low_texture_mask(gray))


def skip(entry):
    pass


def main(args):
    manifest = process_func(args)
    for entry in manifest["images"]:
        # print(entry["filename"], entry["screening"], entry.get("mask_path"))
        for entry in manifest["images"]:
            if entry["screening"]["decision"] == "REPAIR":
                print(f"Will run diffusion inpainting for {entry['filename']}")

            elif entry["screening"]["decision"] == "NOVEL_VIEW":
                print(f"will run Diffusion Img-Img on {entry['filename']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Folder with raw images")
    parser.add_argument("--out_dir", required=True, help="Folder to write preprocessed images and manifest")
    # parser.add_argument("----use_colmap_metadata", default=1, type=int, help="Whether to use colmap metadata"
    #                                                                          "use '0'->False, '1'->True this accepts "
    #                                                                          "integers only")

    parser.add_argument("--target_size", type=int, default=None,
                        help="If set, resize images to this square size (e.g. 1024). Use resize_mode to control"
                             " strategy.")
    parser.add_argument("--resize_mode", choices=["pad", "crop", "stretch", "none"], default="pad",
                        help="How to resize while preserving aspect.")
    args = parser.parse_args()
    main(args)
