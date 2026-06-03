# Unified 3DGS Pipeline — `unified_pipeline.ipynb`

> **STATUS: ✅ IN USE** — Single Colab notebook orchestrating all three augmentation modes via a Gradio web interface. Located in `Pipeline_files/`.

---

## Overview

`unified_pipeline.ipynb` consolidates three previously separate pipeline modes into one notebook, controlled entirely through a browser-based Gradio UI. The user sets a scene name and mode, launches the interface, and the notebook handles all augmentation, staging, SfM, training, and evaluation steps.

| Mode | Augmentation Strategy | Input Type |
|---|---|---|
| A — Synthesis | Zero123++ novel view generation | Object-centric (black background, NeRF-style) |
| B — Restoration | ControlNet Tile image restoration | Degraded real-world images (blur, noise, exposure failure) |
| C — ViewCrafter | Video diffusion novel view synthesis | Natural scenes (real backgrounds, sparse multi-view) |

---

## Notebook Cell Structure

| Cell | ID | Purpose | Mode |
|---|---|---|---|
| — | `md-title` | Run order instructions | All |
| 1 | `cell-1-mount` | Mount Google Drive | All |
| 2 | `cell-2-config` | Set `SCENE_NAME` and `PIPELINE_MODE` | All |
| 3 | `cell-3-common-deps` | Filtered requirements.txt + safe version upgrades + accelerate patch | All |
| 4 | `cell-4-ab-setup` | HuggingFace login + basicsr compatibility patch | A, B only |
| 5a | `cell-5a-vc-clone` | Clone ViewCrafter repo + install condacolab → **runtime restarts** | C only |
| 5a-post | `990e66f0` | Post-restart: re-clone ViewCrafter + create conda env with torch 2.1.0 | C only |
| 5c | `cell-5c-vc-models` | Download DUSt3R + ViewCrafter checkpoints (~25 GB) | C only |
| 6 | `cell-6-cuda-colmap` | Compile CUDA submodules + install COLMAP + ffmpeg | All |
| 7 | `cell-7-gradio` | Launch Gradio UI (3 tabs) | All |
| 8 | `cell-8-convert-ai` | SfM — SuperPoint + LightGlue + COLMAP pose estimation | All (manual) |
| 9 | `cell-9-train` | 3DGS training — 30,000 iterations | All (manual) |
| 10 | `cell-10-metrics` | Render held-out views + compute PSNR / SSIM / LPIPS | All (manual) |

---

## Dependency Management (Cell 3)

`requirements.txt` pins several packages to old versions incompatible with Colab 2025 defaults.  
Cell 3 resolves this by:

1. Parsing `requirements.txt` line-by-line and filtering out version-sensitive packages
2. Installing the filtered requirements (retains pyiqa, basicsr, rembg, etc.)
3. Installing managed packages at safe minimum versions:

| Package | requirements.txt pin | Safe version | Reason |
|---|---|---|---|
| numpy | 1.26.4 | ≥ 2.0 | Colab C extensions compiled against numpy 2.x ABI |
| huggingface_hub | 0.22.2 | ≥ 0.33.5 | Required by diffusers 0.27+ and gradio 5.x |
| diffusers | 0.26.3 | ≥ 0.27.2 | 0.26.x imports `cached_download` removed in hf_hub 0.24 |
| transformers | 4.38.2 | ≥ 4.41.0 | Required by peft 0.19+ |
| peft | 0.10.0 | ≥ 0.17.0 | Required by diffusers 0.27+ |
| accelerate | 0.28.0 | ≥ 0.31.0, < 1.0 | 1.x has circular import bug in big_modeling |

4. Patching `accelerate.utils.memory` to add `clear_device_cache` if the resolved version lacks it (peft 0.19+ requires it)

---

## Gradio UI — Mode A (Synthesis)

```
User enters scene name → clicks "Run Synthesis Pipeline"
        ↓
[1/3] process_file.py  (MUSIQ quality screener)
        → NOVEL_VIEW images: route to Zero123++
        → REPAIR images: route to ControlNet
        ↓
[2/3] diffusion_script_v0.py --mode synthesis
        → anchor_*.png  (real photos, black BG, rembg-cleaned)
        → synth_*_v[0-5].png  (6 Zero123++ views per anchor, rembg applied)
        → restored_*.png  (ControlNet repaired images)
        → RealESRGAN upscale → 1024×1024
        ↓
[3/3] Stage PNGs → gaussian-splatting/input_dataset/{scene}/input/
        → Gallery preview shown in Gradio
        → builtins._gs_filename set for Cells 8–10
```

---

## Gradio UI — Mode B (Restoration)

```
User enters scene name + ControlNet prompt → clicks "Run Restoration Pipeline"
        ↓
[1/3] process_file.py  (MUSIQ quality screener)
        ↓
[2/3] diffusion_script_v0.py --mode restoration --prompt "{prompt}"
        → ControlNet Tile applied to all images (strength 0.35)
        → RealESRGAN upscale → 1024×1024
        ↓
[3/3] Stage PNGs → gaussian-splatting/input_dataset/{scene}/input/
        → Gallery preview shown in Gradio
```

---

## Gradio UI — Mode C (ViewCrafter)

```
User enters scene name + video_length + ddim_steps → clicks "Run ViewCrafter Pipeline"
        ↓
[1/3] ViewCrafter inference (via /usr/local/envs/viewcrafter_env/bin/python)
        → DUSt3R builds coarse point cloud from sparse input images
        → Video diffusion model generates interpolated novel-view video
        → Output: render.mp4 (shown in Gradio video widget)
        ↓
[2/3] ffmpeg frame extraction
        → select=not(mod(n,4))  — every 4th frame sampled
        → e.g. 25-frame video → ~6 extracted PNGs
        → frames saved to: final_{scene}_run/extracted_frames/
        ↓
[3/3] Merge + Stage
        → original photos (output_train/{scene}/train/) +
          extracted frames (extracted_frames/)
        → copied to gaussian-splatting/input_dataset/{scene}/input/
        → Gallery preview of merged set shown in Gradio
        → render.mp4 shown in Gradio video widget
```

---

## Manual Cells (8–10) — Run After Gradio Completes

### Cell 8 — Pose Estimation (convert_ai.py)

```
Copies staged images from Drive → Colab SSD /content/local_workspace/{scene}/
        ↓
convert_ai.py --source_path /content/local_workspace/{scene}/
        → NetVLAD: top-50 visual neighbour pairing
        → SuperPoint: neural keypoint extraction (max 4096 keypoints/image)
        → LightGlue: attention-based geometric matching
        → PyCOLMAP: bundle adjustment → cameras.bin, images.bin, points3D.bin
        → COLMAP image_undistorter: lens correction → sparse/0/ + images/
        ↓
Syncs sparse/0/ back to Drive
```

### Cell 9 — 3DGS Training (train.py)

```
Stages COLMAP output to /content/local_workspace/{scene}/
        ↓
train.py -s {scene} -m {scene}_final_run --eval --opacity_reset_interval 9000
        → 30,000 iteration photometric optimisation
        → Adaptive Density Control: clone/split/prune Gaussians
        → opacity_reset_interval=9000 prevents "Pruning Massacre"
        ↓
Copies trained model → Drive/gaussian-splatting/output/{scene}_final_run/
```

### Cell 10 — Evaluation (render.py + metrics.py)

```
render.py -m {scene}_final_run --skip_train
        → renders held-out test views
        ↓
metrics.py -m {scene}_final_run
        → computes PSNR, SSIM, LPIPS vs ground truth
        → writes results.json
```

---

## Mode C — Runtime Restart Sequence

Cell 5a installs `condacolab`, which triggers an automatic Colab runtime restart that wipes `/content/` and all pip installs. After restart, run in order:

1. **Cell 1** — Re-mount Drive
2. **Cell 2** — Re-run config (`PIPELINE_MODE = "viewcrafter"`)
3. **Cell 3** — Re-install all deps (gradio is wiped by restart)
4. **Cell 5a-post** — Re-clone ViewCrafter + create conda env
5. **Cell 5c** — Download model checkpoints (cached on Drive if previously downloaded)
6. **Cell 6** → **Cell 7** → continue normally

---

## ViewCrafter Dependency Isolation

ViewCrafter requires `torch==2.1.0` (incompatible with Colab's base torch). This is resolved by running it in an isolated conda environment:

```
/usr/local/envs/viewcrafter_env/bin/python inference.py ...
```

The conda env has its own torch. The base Colab Python (used for Modes A and B) never interacts with it. No version conflict exists.

---

## Output Structure

```
Google Drive:
gaussian-splatting/
├── input_dataset/{scene}/
│   └── input/          ← staged images (all modes)
├── sparse/0/
│   ├── cameras.bin
│   ├── images.bin
│   └── points3D.bin
└── output/{scene}_final_run/
    ├── point_cloud/    ← trained Gaussian model
    └── results.json    ← PSNR / SSIM / LPIPS
```

---

## Known Issues and Fixes Applied

| Issue | Root Cause | Fix |
|---|---|---|
| `cached_download` ImportError | diffusers 0.26.x + hf_hub ≥ 0.24 | Upgrade diffusers ≥ 0.27.2 |
| numpy ABI crash | requirements.txt downgrades numpy 2.x → 1.x | Filter numpy from requirements.txt, pin ≥ 2.0 |
| accelerate circular import | accelerate 1.x big_modeling circular dep | Cap accelerate < 1.0 |
| `clear_device_cache` missing | peft 0.19+ needs it, accelerate 0.28 lacks it | File patch + sys.modules flush |
| `trust_remote_code` error | diffusers 0.27+ enforces remote code opt-in | Add `trust_remote_code=True` to Zero123++ load call |
