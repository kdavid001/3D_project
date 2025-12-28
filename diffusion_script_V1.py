"""
Step 3: run_diffusion.py - Final Processing with Stable Diffusion

Usage:
    python run_diffusion.py --step1_dir ./preprocessed --step2_dir ./inpainting_ready

Changes:
- Copies "NONE" (Good) images to output folder.
- Replaces "REPAIR" (Bad) images with the fixed AI version.
- Creates NEW extra files for "NOVEL_VIEW" (Original + Variation).
"""

import json
import os
import argparse
import shutil
import torch
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline, StableDiffusionImg2ImgPipeline
from tqdm import tqdm

# ----------------------------
# Config
# ----------------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def resize_to_multiple_of_8(img):
    """Resizes image to be divisible by 8 (required for VAE)"""
    w, h = img.size
    w = w - (w % 8)
    h = h - (h % 8)
    if w != img.size[0] or h != img.size[1]:
        return img.resize((w, h), Image.Resampling.LANCZOS)
    return img

def main(args):
    print(f"🚀 Initializing Diffusion Models on {DEVICE}...")

    # 1. Load Pipelines
    #    We load both because we have a hybrid dataset
    model_inpaint = "runwayml/stable-diffusion-inpainting"
    model_img2img = "runwayml/stable-diffusion-v1-5"

    pipe_inpaint = StableDiffusionInpaintPipeline.from_pretrained(
        model_inpaint, torch_dtype=torch.float16, safety_checker=None
    ).to(DEVICE)

    pipe_img2img = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_img2img, torch_dtype=torch.float16, safety_checker=None
    ).to(DEVICE)

    # Enable optimizations for Colab (T4 GPU)
    pipe_inpaint.enable_xformers_memory_efficient_attention()
    pipe_img2img.enable_xformers_memory_efficient_attention()

    # 2. Load Step 1 Manifest
    manifest_path = os.path.join(args.step1_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Cannot find Step 1 manifest at {manifest_path}")

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    # 3. Create Output Directory
    final_output_dir = os.path.join(args.step1_dir, "final_results")
    os.makedirs(final_output_dir, exist_ok=True)

    print(f"📂 Processing {len(manifest['images'])} images...")

    for entry in tqdm(manifest["images"]):
        filename = entry["filename"]
        decision = entry["screening"]["decision"]

        original_img_path = os.path.join(args.step1_dir, "images", filename)

        # Determine output path for the MAIN file
        # (This preserves the original filename for NONE and REPAIR)
        main_save_path = os.path.join(final_output_dir, filename)

        if not os.path.exists(original_img_path):
            print(f"⚠️ Missing original image: {original_img_path}")
            continue

        try:
            # ---------------------------------------------------------
            # CASE 1: NONE (Good Image)
            # ---------------------------------------------------------
            if decision == "NONE":
                # Just copy the file to the final folder
                shutil.copy2(original_img_path, main_save_path)
                continue

            # Load Original Image (Clean) for AI processing
            image = Image.open(original_img_path).convert("RGB")
            image = resize_to_multiple_of_8(image)
            generator = torch.Generator(device=DEVICE).manual_seed(42)

            # ---------------------------------------------------------
            # CASE 2: REPAIR (Inpainting)
            # ---------------------------------------------------------
            if decision == "REPAIR":
                # Find the mask in Step 2 folder (ensure png extension)
                mask_filename = os.path.splitext(filename)[0] + ".png"
                mask_path = os.path.join(args.step2_dir, "masks", mask_filename)

                if not os.path.exists(mask_path):
                    print(f"❌ Mask missing for {filename}, copying original instead.")
                    shutil.copy2(original_img_path, main_save_path)
                    continue

                mask = Image.open(mask_path).convert("L")
                mask = resize_to_multiple_of_8(mask)

                # Run Inpainting
                result = pipe_inpaint(
                    prompt="high quality, sharp focus, realistic texture, seamless blend",
                    negative_prompt="blur, noise, artifacts, distorted, seams, gray patches",
                    image=image,
                    mask_image=mask,
                    guidance_scale=7.5,
                    num_inference_steps=30,
                    strength=0.99,
                    generator=generator
                ).images[0]

                # Save FIXED image (Overwrite bad original in output folder)
                result.save(main_save_path)

            # ---------------------------------------------------------
            # CASE 3: NOVEL VIEW (Img2Img)
            # ---------------------------------------------------------
            elif decision == "NOVEL_VIEW":
                # 1. Copy the Original Perfect Image first
                shutil.copy2(original_img_path, main_save_path)

                # 2. Generate the Variation
                result = pipe_img2img(
                    prompt="same scene, highly detailed, photorealistic, 8k, slightly different lighting",
                    image=image,
                    strength=0.35,  # Low strength to keep geometry similar
                    guidance_scale=7.5,
                    generator=generator
                ).images[0]

                # 3. Save as a NEW file (e.g., image_novel.jpg)
                base_name, ext = os.path.splitext(filename)
                novel_filename = f"{base_name}_novel{ext}"
                novel_save_path = os.path.join(final_output_dir, novel_filename)

                result.save(novel_save_path)

        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")
            continue

    print(f"\n✅ All done! Results saved to: {final_output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step1_dir", type=str, required=True, help="Folder from Step 1 (contains manifest.json and /images)")
    parser.add_argument("--step2_dir", type=str, required=True, help="Folder from Step 2 (contains /masks)")
    args = parser.parse_args()
    main(args)