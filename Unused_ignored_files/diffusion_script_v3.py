#!/usr/bin/env python3
"""
COMBINED PIPELINE V2.0 (SERVER EDITION: SV3D + ControlNet)
HYBRID DATA ARCHITECTURE + STRICT GEOMETRY PATCH APPLIED

MODES:
1. --mode synthesis (DEFAULT): Runs Stable Video 3D (21 Orbital Frames, 576x576).
2. --mode restoration: Runs ControlNet logic (Fixes natural images, Keeps BG).
"""

import sys
import os
import shutil
import gc
import argparse
import json
import torch
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

# Diffusers imports (Standard models)
from diffusers import (
    DiffusionPipeline,
    StableDiffusionControlNetImg2ImgPipeline,
    ControlNetModel,
    UniPCMultistepScheduler
)

# ==========================================
# 🚨 CRITICAL PATCH 🚨
# ==========================================
try:
    import torchvision.transforms.functional_tensor
except ImportError:
    try:
        import torchvision.transforms.functional as functional

        sys.modules["torchvision.transforms.functional_tensor"] = functional
    except ImportError:
        pass


# ==========================================

def get_name_from_path(path):
    return os.path.basename(os.path.normpath(path))


def flush_memory():
    gc.collect()
    torch.cuda.empty_cache()


# ==========================================
# PHASE 0: PRE-PROCESSING (Synthesis Only)
# ==========================================
def process_for_sv3d(pil_image):
    """Applies black background and centering to 576x576 natively."""
    canvas_size = 576

    # Changed from Gray (127, 127, 127) to Pitch Black (0, 0, 0)
    canvas = Image.new("RGB", (canvas_size, canvas_size), (0, 0, 0))

    w, h = pil_image.size
    scale = (canvas_size * 0.85) / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized_obj = pil_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    x = (canvas_size - new_w) // 2
    y = (canvas_size - new_h) // 2

    canvas.paste(resized_obj, (x, y))
    return canvas


# ==========================================
# PHASE 1A: SYNTHESIS (LOCAL SV3D LOGIC)
# ==========================================
def run_synthesis_phase(input_dir, temp_dir, candidates):
    print(f"\n🔹 PHASE 1: Synthesizing (Stable Video 3D Mode)...")

    import sys
    import torch
    import numpy as np
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
    from diffusers import AutoencoderKL, EulerDiscreteScheduler

    # 1. Point Python to the downloaded folder
    repo_path = "/content/drive/MyDrive/pythonprojects_2/final_year_project/3D_project/sv3d-diffusers"
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)

    try:
        from diffusers_sv3d import SV3DUNetSpatioTemporalConditionModel, StableVideo3DDiffusionPipeline

        # --- THE "RESEARCHER TYPO" FIX ---
        from diffusers_sv3d.pipelines.stable_video_diffusion import pipeline_stable_video_3d_diffusion
        if not hasattr(pipeline_stable_video_3d_diffusion, "retrieve_timesteps"):
            def custom_retrieve_timesteps(scheduler, num_inference_steps=None, device=None, timesteps=None, sigmas=None,
                                          **kwargs):
                scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
                return scheduler.timesteps, num_inference_steps

            pipeline_stable_video_3d_diffusion.retrieve_timesteps = custom_retrieve_timesteps
        # ---------------------------------

    except ImportError as e:
        print(f"❌ ERROR: Could not import from sv3d-diffusers: {e}")
        return

    print("⏳ Loading Components (Piecemeal approach from infer.py)...")
    SV3D_DIFFUSERS = "chenguolin/sv3d-diffusers"

    # Load each part individually, forcing fp16 to save A100 memory
    unet = SV3DUNetSpatioTemporalConditionModel.from_pretrained(SV3D_DIFFUSERS, subfolder="unet",
                                                                torch_dtype=torch.float16)
    vae = AutoencoderKL.from_pretrained(SV3D_DIFFUSERS, subfolder="vae", torch_dtype=torch.float16)
    scheduler = EulerDiscreteScheduler.from_pretrained(SV3D_DIFFUSERS, subfolder="scheduler")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(SV3D_DIFFUSERS, subfolder="image_encoder",
                                                                  torch_dtype=torch.float16)
    feature_extractor = CLIPImageProcessor.from_pretrained(SV3D_DIFFUSERS, subfolder="feature_extractor")

    # Manually assemble the pipeline
    pipeline = StableVideo3DDiffusionPipeline(
        image_encoder=image_encoder, feature_extractor=feature_extractor,
        unet=unet, vae=vae, scheduler=scheduler
    ).to("cuda")

    # --- THE VERSION BRIDGE FIX (UPGRADED) ---
    pipeline.video_processor = pipeline.image_processor

    def custom_postprocess_video(video, output_type="pil"):
        if video.dim() == 5:
            video = video.permute(0, 2, 1, 3, 4)
            video = video.squeeze(0)
        flat_images = pipeline.image_processor.postprocess(video, output_type=output_type)
        return [flat_images]

    pipeline.video_processor.postprocess_video = custom_postprocess_video
    # -----------------------------------------------------

    # 2. Setup the Camera Orbit Math
    num_frames = 21
    elevation = 10.0
    elevations_deg = [elevation] * num_frames
    polars_rad = [np.deg2rad(90 - e) for e in elevations_deg]
    azimuths_deg = np.linspace(0, 360, num_frames + 1)[1:] % 360
    azimuths_rad = [np.deg2rad((a - azimuths_deg[-1]) % 360) for a in azimuths_deg]
    azimuths_rad[:-1].sort()

    for entry in tqdm(candidates, desc="Generating SV3D Swarm"):
        filename = entry["filename"]
        raw_fallback = input_dir.replace("output_processed", "output_train")
        paths = [
            os.path.join(input_dir, "processed_train", filename),
            os.path.join(raw_fallback, "train", filename),
            os.path.join(input_dir, "images", filename),
            os.path.join(input_dir, "input", filename),
            os.path.join(input_dir, filename)
        ]
        img_path = next((p for p in paths if os.path.exists(p)), None)
        if not img_path: continue

        input_img = Image.open(img_path).convert("RGB")

        clean_input = process_for_sv3d(input_img)
        clean_input.save(os.path.join(temp_dir, f"anchor_{filename}"))

        # 🟢 STRICT CONSISTENCY PATCH: Lock the random seed
        strict_generator = torch.manual_seed(42)

        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.float16):
                result_frames = pipeline(
                    clean_input,
                    height=576,
                    width=576,
                    num_frames=num_frames,

                    # 🟢 STRICT CONSISTENCY PATCH: Force high-fidelity geometry
                    decode_chunk_size=21,  # Decode all frames simultaneously
                    num_inference_steps=50,  # Double the refinement steps
                    generator=strict_generator,  # Lock the noise seed

                    polars_rad=polars_rad,
                    azimuths_rad=azimuths_rad,
                ).frames[0]

        for idx, frame in enumerate(result_frames):
            save_name = f"synth_{os.path.splitext(filename)[0]}_v{idx}.png"
            frame.save(os.path.join(temp_dir, save_name))

        del result_frames
        flush_memory()

    del pipeline
    flush_memory()


# ==========================================
# PHASE 1B: RESTORATION (ControlNet Logic)
# ==========================================
def run_restoration_phase(input_dir, temp_dir, candidates, prompt):
    print(f"\n🔹 PHASE 1: Restoring (ControlNet Tile Mode)...")
    print(f"   Prompt: '{prompt}'")

    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11f1e_sd15_tile",
        torch_dtype=torch.float16
    )

    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None
    ).to("cuda")

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    success_count = 0

    for i, entry in enumerate(tqdm(candidates, desc="Restoring Images")):
        filename = entry["filename"]

        raw_fallback = input_dir.replace("output_processed", "output_train")
        possible_paths = [
            os.path.join(input_dir, "processed_train", filename),
            os.path.join(raw_fallback, "train", filename),
            os.path.join(input_dir, "images", filename),
            os.path.join(input_dir, "input", filename),
            os.path.join(input_dir, filename)
        ]

        img_path = next((p for p in possible_paths if os.path.exists(p)), None)

        if not img_path:
            if i == 0:
                print(f"\n❌ CRITICAL ERROR: Could not find image '{filename}'")
            continue

        original_image = Image.open(img_path).convert("RGB")

        w, h = original_image.size
        new_w = (w // 8) * 8
        new_h = (h // 8) * 8
        if new_w != w or new_h != h:
            original_image = original_image.resize((new_w, new_h))

        clean_image = pipe(
            prompt,
            image=original_image,
            control_image=original_image,
            negative_prompt="blur, noise, grain, low resolution, distorted, plastic, cartoon",
            num_inference_steps=30,
            strength=0.35,
            guidance_scale=7.0
        ).images[0]

        save_name = f"restored_{filename}"
        if not save_name.lower().endswith(".png"):
            save_name = os.path.splitext(save_name)[0] + ".png"

        clean_image.save(os.path.join(temp_dir, save_name))
        success_count += 1

    del pipe
    del controlnet
    flush_memory()


# ==========================================
# PHASE 2: UPSCALE
# ==========================================
def run_upscale_phase(temp_dir, final_dir):
    print(f"\n🔹 PHASE 2: Upscaling & Standardizing to 1024x1024...")

    if not os.path.exists('weights/RealESRGAN_x4plus.pth'):
        os.makedirs('weights', exist_ok=True)
        torch.hub.download_url_to_file(
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
            'weights/RealESRGAN_x4plus.pth'
        )

    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    upsampler = RealESRGANer(scale=4, model_path='weights/RealESRGAN_x4plus.pth', model=model, tile=400, half=True,
                             gpu_id=0)

    valid_exts = ('.png', '.jpg', '.jpeg')
    files = [f for f in os.listdir(temp_dir) if f.lower().endswith(valid_exts)]

    for filename in tqdm(files, desc="Standardizing"):
        if "FULL_GRID" in filename: continue

        # 🟢 ARCHITECTURE PATCH: Anchor files (original reference photos) are safely passed through
        img_path = os.path.join(temp_dir, filename)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None: continue

        h, w = img.shape[:2]

        if w < 1024 or h < 1024:
            output, _ = upsampler.enhance(img, outscale=4)
        else:
            output = img

        output = cv2.resize(output, (1024, 1024), interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(os.path.join(final_dir, filename), output)


def main(args):
    print(f"🔍 Hardware: {torch.cuda.get_device_name(0)}")
    print(f"⚙️  MODE: {args.mode.upper()}")

    dataset_name = get_name_from_path(args.input_dir)