""" pip install diffusers transformers accelerate safetensors pillow torch torchvision """
import json
import os
from PIL import Image
import torch
from diffusers import (
    StableDiffusionInpaintPipeline,
    StableDiffusionImg2ImgPipeline
)
from tqdm import tqdm
import argparse

# ----------------------------
# Config
# ----------------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def resize_to_multiple_of_8(img):
    w, h = img.size
    w = w - (w % 8)
    h = h - (h % 8)
    return img.crop((0, 0, w, h))


def main_prog(DATASET_DIR):
    global DEVICE
    generator = torch.Generator(device=DEVICE).manual_seed(42)
    # DATASET_DIR =      # upload out_dir here
    OUT_IMAGES_DIR = os.path.join(DATASET_DIR, "images")
    MASK_DIR = os.path.join(DATASET_DIR, "masks")
    MANIFEST_PATH = os.path.join(DATASET_DIR, "manifest.json")

    MODEL_INPAINT = "runwayml/stable-diffusion-inpainting"
    MODEL_IMG2IMG = "runwayml/stable-diffusion-v1-5"

    pipe_inpaint = StableDiffusionInpaintPipeline.from_pretrained(
        MODEL_INPAINT,
        torch_dtype=torch.float16,
        safety_checker=None
    ).to(DEVICE)

    pipe_img2img = StableDiffusionImg2ImgPipeline.from_pretrained(
        MODEL_IMG2IMG,
        torch_dtype=torch.float16,
        safety_checker=None
    ).to(DEVICE)

    for pipe in [pipe_inpaint, pipe_img2img]:
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
    # ----------------------------
    # Load manifest
    # ----------------------------
    with open(MANIFEST_PATH, "r") as f:
        manifest = json.load(f)

        for entry in tqdm(manifest["images"]):

            decision = entry["screening"]["decision"]

            if decision == "NONE":
                continue

            image_path = os.path.join(OUT_IMAGES_DIR, entry["filename"])
            image = Image.open(image_path).convert("RGB")
            image = resize_to_multiple_of_8(image)

            with torch.no_grad():

                # -----------------------
                # REPAIR (INPAINT)
                # -----------------------
                if decision == "REPAIR":
                    mask_path = os.path.join(DATASET_DIR, entry["diffusion"]["mask_path"])
                    mask = Image.open(mask_path).convert("L")
                    mask = resize_to_multiple_of_8(mask)

                    result = pipe_inpaint(
                        prompt="photorealistic repair, same scene, same geometry, same lighting",
                        image=image,
                        mask_image=mask,
                        guidance_scale=7.5,
                        num_inference_steps=30,
                        generator=generator
                    ).images[0]

                # -----------------------
                # NOVEL VIEW (IMG2IMG)
                # -----------------------
                elif decision == "NOVEL_VIEW":
                    result = pipe_img2img(
                        prompt="same scene, slightly different camera angle, photorealistic",
                        image=image,
                        strength=0.35,  # IMPORTANT: low strength preserves geometry
                        guidance_scale=7.5,
                        num_inference_steps=30,
                        generator=generator
                    ).images[0]

                else:
                    continue

            result.save(image_path)  # 🔁 Overwrite or save copy
        print("✔ Diffusion repair completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--DATASET_DIR", type=str, default="./dataset")
    args = parser.parse_args()
    main_prog(args.DATASET_DIR)
