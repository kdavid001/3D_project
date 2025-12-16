""" pip install diffusers transformers accelerate safetensors pillow torch torchvision """
import json
import os
from PIL import Image
import torch
from diffusers import StableDiffusionInpaintPipeline
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

    MODEL_ID = "runwayml/stable-diffusion-inpainting"
    DEVICE = "cuda"

    # ----------------------------
    # Load pipeline
    # ----------------------------
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        safety_checker=None
    ).to(DEVICE)

    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass
    # ----------------------------
    # Load manifest
    # ----------------------------
    with open(MANIFEST_PATH, "r") as f:
        manifest = json.load(f)

        # ----------------------------
        # Diffusion loop
        # ----------------------------
        for entry in tqdm(manifest["images"]):
            if not entry["screening"]["needs_diffusion"]:
                continue

            image_path = os.path.join(OUT_IMAGES_DIR, entry["filename"])
            mask_path = os.path.join(DATASET_DIR, entry["mask_path"])

            image = Image.open(image_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")

            # Resize to 512 for diffusion (optional but recommended)
            image = resize_to_multiple_of_8(image)
            mask = resize_to_multiple_of_8(mask)

            with torch.no_grad():
                result = pipe(
                    prompt="photorealistic, same scene, same lighting, no new objects",
                    image=image,
                    mask_image=mask,
                    guidance_scale=7.5,
                    num_inference_steps=30,
                    generator=generator
                ).images[0]

            # 🔁 Overwrite original image
            result.save(image_path)

        print("✔ Diffusion repair completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--DATASET_DIR", type=str, default="./dataset")
    args = parser.parse_args()
    main_prog(args.DATASET_DIR)
