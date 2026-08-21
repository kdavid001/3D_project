# User Guide — Generative Augmented 3D Reconstruction Pipeline

This guide covers two things:
1. **Standalone usage** — how to download and use each component independently (Sections 1–7)
2. **Project folder usage** — how to use this pipeline exactly as it was developed, with the full project folder structure and Colab notebooks (Section 8 onwards)

The full end-to-end reconstruction (augmentation → pose estimation → 3DGS training) is orchestrated by `Pipeline_files/unified_pipeline.ipynb` on Google Colab.

---

## Model Reference — When to Use Each

### Zero123++ (Mode A)

| | |
|---|---|
| **Use for** | Object-centric scenes with black background (NeRF synthetic: lego, hotdog, etc.) |
| **Don't use for** | Unbounded real-world scenes, scenes without black background |
| **Input requirement** | Black background — run `remove_bg.py` first if needed |
| **VRAM** | ~16 GB minimum |

### ControlNet Tile (Mode B)

| | |
|---|---|
| **Use for** | Images flagged as REPAIR by the quality screener — restores blur, exposure, colour cast while preserving structure |
| **Don't use for** | Generating new viewpoints — this only restores, it does not synthesise |
| **Key parameters** | Strength: 0.35, Guidance scale: 7.0, Steps: 30 |
| **VRAM** | ~12 GB minimum |

### ViewCrafter (Mode C)

| | |
|---|---|
| **Use for** | Real-world outdoor/indoor scenes with sparse wide-baseline coverage |
| **Don't use for** | Object-centric black-background scenes (use Zero123++), fewer than 6 input images |
| **Key parameters** | video_length: 25 (reduce to 20 for OOM), ddim_steps: 50, resolution: 576×1024 |
| **VRAM** | ~38–40 GB — requires A100 on Colab. T4/V100 will OOM |

### SuperPoint + LightGlue (Stage 3)

| | |
|---|---|
| **Use for** | Always — replaces SIFT in COLMAP for all pipeline runs |
| **Known limitation** | Exhaustive matching scales O(N²). Manageable at 60 images, slow at 300+ |

### RealESRGAN (Post-processing, Modes A and B)

| | |
|---|---|
| **Use for** | 4× upscaling of Zero123++ (256×256) and ControlNet (512×512) outputs to 1024×1024 |
| **Weight file** | `RealESRGAN_x4plus.pth` — downloaded automatically on first run |

---

## Dependency Warnings

### requirements.txt — Version Conflicts

The `requirements.txt` contains version pins written for an earlier Colab environment. Several packages conflict with Colab 2025 defaults and must not be installed at their pinned versions. Cell 3 of the unified pipeline handles this automatically by filtering them out and reinstalling at safe versions.

**Packages managed separately (do not install from requirements.txt directly):**

| Package | Issue | Safe action |
|---|---|---|
| `numpy` | Pin predates numpy 2.0; ABI breaks PyTorch | Cell 3 installs at latest compatible version |
| `huggingface-hub` | Old pin incompatible with new diffusers | Cell 3 installs at safe version |
| `diffusers` | Version pin breaks with transformers ≥4.40 | Cell 3 installs at safe version |
| `transformers` | ViewCrafter requires <4.40; main pipeline requires ≥4.40 | Handled by separate conda env (`viewcrafter_env`) |
| `peft` | Circular import bug with accelerate 1.x | Cell 3 installs at safe version |
| `accelerate` | Version 1.x introduces circular import with peft | Cell 3 manages version |

> **Warning:** Do not run `pip install -r requirements.txt` directly in a Colab notebook. Always use Cell 3, which applies the version filter before installing. Running requirements.txt directly will likely break diffusers, transformers, or numpy.

### ViewCrafter Conda Environment

ViewCrafter cannot share an environment with the main pipeline due to a hard `transformers < 4.40` dependency. The notebook creates a separate conda environment (`viewcrafter_env`) with:
- Python 3.10
- PyTorch 2.1.0 + CUDA 12.1
- numpy < 2.0 (PyTorch 2.1.0 is incompatible with numpy 2.0+)
- open-clip-torch pinned to 2.20.0 (newer versions break ViewCrafter's CLIP encoder)

> **Warning:** Do not install packages into `viewcrafter_env` from the base Colab environment. Always use `/usr/local/envs/viewcrafter_env/bin/python -m pip install` or the `conda install -n viewcrafter_env` prefix.

### Automatic Compatibility Patches

The Gradio UI (Cell 7) automatically applies the following patches before running ViewCrafter. These are applied every run — they are idempotent:

| File patched | Issue fixed |
|---|---|
| `ViewCrafter/lvdm/models/utils_diffusion.py` | `betas.numpy()` fails with newer PyTorch — replaced with `.tolist()` |
| `ViewCrafter/extern/dust3r/dust3r/utils/image.py` | `Image.ANTIALIAS` removed in Pillow 10+ — replaced with `Image.LANCZOS` |
| `torchvision/transforms/functional.py` | `torch.from_numpy()` ABI break — replaced with `torch.frombuffer()` |
| `torchvision/io/video.py` | `frame.pict_type = "NONE"` invalid in newer PyAV — replaced with `0` |
| `ViewCrafter/lvdm/modules/encoders/condition.py` | `input_patchnorm` attribute missing in newer open-clip — replaced with `getattr` fallback |

If any of these patches fail, the error is logged in the Live Log panel of the Gradio UI and the ViewCrafter run will abort before inference begins.

### basicsr Patch (Modes A and B)

Cell 4 applies this patch:
```bash
sed -i 's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/
        from torchvision.transforms.functional import rgb_to_grayscale/' \
    /usr/local/lib/python3.12/dist-packages/basicsr/data/degradations.py
```
`functional_tensor` was removed in torchvision 0.16. Without this patch, importing `diffusers` in Python 3.12 raises an `ImportError` that aborts the entire augmentation run. This patch is only needed for Modes A and B — Mode C uses a separate conda environment that has a different torchvision version.

---

---

## 1. Environment Setup

All local scripts require the `nsenv` virtual environment. Activate it before running any Python command:

```bash
source nsenv/bin/activate
```

To deactivate when finished:

```bash
deactivate
```

The key dependencies are: `torch`, `pyiqa`, `opencv-python`, `numpy`, `matplotlib`, `diffusers`, `pycolmap`, `hloc`.

---

## 2. Input Preparation: Converting HEIC Images (iPhone)

iPhone cameras produce HEIC files by default. COLMAP and the quality screener require JPEG or PNG inputs.

### On macOS (no extra install needed)

Convert a single file:

```bash
sips -s format jpeg -s formatOptions 95 input.HEIC --out output.jpg
```

Convert an entire folder:

```bash
for f in /path/to/heic_folder/*.HEIC; do
    sips -s format jpeg -s formatOptions 95 "$f" --out "${f%.HEIC}.jpg"
done
```

### On Google Colab

```python
!pip install pillow-heif
from pillow_heif import register_heif_opener
from PIL import Image
register_heif_opener()
img = Image.open("photo.HEIC")
img.save("photo.jpg", quality=95)
```

> **Note:** `sips` uses the Display P3 colour profile embedded in iPhone HEICs. The output JPEG retains the correct colours without any manual profile conversion.

---

## 3. Quality Screening — `process_file.py`

The quality screener evaluates each image across five pillars (MUSIQ sharpness, saturation, exposure, contrast, colour cast) and routes it to either `NOVEL_VIEW` (good enough to use as-is) or `REPAIR` (needs generative restoration).

### Modes

| Mode | Scene type | MUSIQ threshold | Notes |
|---|---|---|---|
| `natural` | Outdoor real-world photos | BAD_LIMIT=40, HARD_LIMIT=20 | No centre-crop; tuned for natural daylight |
| `indoor` | Indoor real-world photos | BAD_LIMIT=30, HARD_LIMIT=15 | Lower MUSIQ limits (artificial lighting depresses scores); relaxed contrast and colour cast limits |
| `synthetic` | NeRF synthetic renders | BAD_LIMIT=65, HARD_LIMIT=30 | Centre-crops black background before scoring |

Use `--mode indoor` for classroom, office, or any artificially lit scene. Use `--mode natural` for outdoor scenes. The default is `synthetic`.

### Single image test

```bash
python3 process_file.py \
  --mode natural \
  --test_image output_train/train/images/00025.jpg \
  --out_dir ./debug_out \
  --debug
```

This writes a three-panel diagnostic chart to `./debug_out/debug_test.png` showing the image, its RGB histogram, and per-pillar pass/fail results.

### Batch processing — natural scenes

```bash
python3 process_file.py \
  --mode natural \
  --input_dir ./output_train/train \
  --out_dir ./output_processed/train \
  --debug
```

### Batch processing — NeRF synthetic

```bash
python3 process_file.py \
  --mode synthetic \
  --input_dir ./output_train/lego \
  --out_dir ./output_processed/lego \
  --debug
```

### Outputs

| Path | Contents |
|---|---|
| `<out_dir>/processed_train/` | Images that passed (NOVEL_VIEW) — copied here |
| `<out_dir>/manifest.json` | Per-image decision log with score, decision, and reason |
| `<out_dir>/debug_visuals/` | Diagnostic charts (only when `--debug` is set) |

---

## 4. Dataset Corruption Simulator — `corrupt_data.py`

Used to create the sparse/degraded dataset conditions used in ablation studies. Applies random blur, exposure shifts, noise, and optional frame deletion.

### Natural scenes (e.g. train dataset)

```bash
python3 corrupt_data.py \
  --clean_dir ./others/tandt_db/tandt/train \
  --out_dir ./output_train/ \
  --corrupt_prob 0.8 \
  --delete_prob 0.65 \
  --seed 42
```

### NeRF synthetic scenes

```bash
python3 corrupt_data.py \
  --clean_dir ./nerf_synthetic/lego \
  --out_dir ./output_train \
  --use_geometry \
  --corrupt_prob 0.8 \
  --delete_prob 0.65 \
  --seed 42
```

`--use_geometry` reads the scene JSON and removes the negative-X camera sector to simulate sparse viewpoint coverage.

### Key parameters

| Flag | Default | Effect |
|---|---|---|
| `--corrupt_prob` | 0.5 | Probability of applying degradation to each image |
| `--delete_prob` | 0.0 | Probability of deleting each image (simulates sparse capture) |
| `--use_geometry` | off | NeRF synthetic only — removes one hemisphere of views |
| `--seed` | 42 | Random seed for reproducibility |

---

## 5. Generative Augmentation — `diffusion_script_v0.py`

Runs locally when GPU VRAM permits. For large scenes, use the Colab notebook instead.

Two modes are available:

| Mode | Model | Use case |
|---|---|---|
| `synthesis` | Zero123++ | Object-centric scenes with black background — generates 6 novel views |
| `restoration` | ControlNet Tile | Natural images flagged as REPAIR — restores structure while fixing degradation |

### Synthesis (Zero123++)

```bash
python3 diffusion_script_v0.py \
  --mode synthesis \
  --input_dir ./output_processed/lego/processed_train \
  --out_dir ./augmented_lego
```

Input images must have a black background. The script selects the top-20 quality candidates from the manifest and generates a 2×3 multi-view grid per image.

### Restoration (ControlNet Tile)

```bash
python3 diffusion_script_v0.py \
  --mode restoration \
  --input_dir ./output_processed/train/processed_train \
  --out_dir ./restored_train
```

Reads the manifest and processes only images marked `REPAIR`. Background is preserved.

> **VRAM requirement:** ~16 GB for synthesis, ~12 GB for restoration. For real-world scenes, ViewCrafter (Colab only) produces better novel views than Zero123++.

---

## 6. Pose Estimation — `convert_ai.py`

Replaces classical COLMAP SIFT with SuperPoint + LightGlue exhaustive feature matching, enabling camera registration from as few as 12 sparse images.

`convert_ai.py` lives inside the `gaussian-splatting/` folder and must be run from there:

```bash
cd gaussian-splatting
python3 convert_ai.py \
  --source_path /path/to/scene_folder \
  --images input
```

The `--images` argument specifies the subfolder within `source_path` that contains the images (default: `input`).

### What it does

1. Runs SuperPoint keypoint extraction on all images
2. Builds all-pairs exhaustive match list
3. Runs LightGlue to match every image pair
4. Runs COLMAP `incremental_mapping` using the AI matches (no SIFT)
5. Runs COLMAP `image_undistorter` to produce the `images/` and `sparse/` directories needed by 3DGS `train.py`

### Outputs

```
<source_path>/
├── distorted/sparse/0/    # COLMAP sparse model (cameras.bin, images.bin, points3D.bin)
├── images/                # Undistorted images
└── sparse/0/              # Final sparse model ready for 3DGS
```

> **Note:** The script auto-installs `hloc` and clones `SuperGluePretrainedNetwork` on first run if they are not present. `SuperGluePretrainedNetwork` must be in the same directory as `convert_ai.py` (i.e. inside `gaussian-splatting/`).

---

## 7. Full Pipeline — `Pipeline_files/unified_pipeline.ipynb`

The main end-to-end pipeline runs on Google Colab (GPU runtime). Full notebook documentation is in [`Pipeline_files/README.md`](Pipeline_files/README.md). It supports three modes:

| Mode | Description |
|---|---|
| A — Zero123++ | Object-centric synthetic scenes; black background required; generates 6 novel views per anchor |
| B — ControlNet Tile | Natural images flagged for restoration; structure-preserving repair (strength 0.35) |
| C — ViewCrafter | Real-world scenes; video diffusion from sparse views via DUSt3R point cloud conditioning |

Open the notebook in Colab, select a GPU runtime (A100 recommended for Mode C), and follow the cell-by-cell instructions. Key parameters are set in the configuration cell at the top.

### ViewCrafter-specific notes (Mode C)

- Install `pyav` in the notebook before running: `!pip install av`
- Rename input images to numeric filenames (`000.jpg`, `001.jpg`, …) — ViewCrafter sorts by `int(stem)` and will silently mis-order non-numeric names
- Set `video_length ≤ 20` to avoid OOM on A100 (40 GB); `video_length=25` at `ddim_steps=50` requires ~40 GB VRAM
- ViewCrafter requires a separate conda environment (`viewcrafter_env`) due to a hard `transformers < 4.40` dependency conflict with the main pipeline

---

## 8. Using the Full Project Folder (as developed)

This section documents how the pipeline was actually run during development. If you have a copy of the full project folder (`pythonprojects_2/`), this is the exact setup and workflow used.

### 8.1 Google Drive Folder Structure

The entire project must be uploaded to Google Drive at the following path. The Colab notebooks hard-code this location:

```
MyDrive/
└── pythonprojects_2/
    └── final_year_project/
        ├── 3D_project/          ← pipeline scripts, data, weights
        └── gaussian-splatting/  ← modified 3DGS repo (training, pose estimation)
```

The notebooks use two base path constants defined in Cell 2 (the config cell):

```python
DRIVE_BASE = "/content/drive/MyDrive/pythonprojects_2/final_year_project/3D_project"
GS_BASE    = "/content/drive/MyDrive/pythonprojects_2/final_year_project/gaussian-splatting"
```

If you rename or move either folder, update these two variables in Cell 2 before running anything else.

---

### 8.2 `3D_project/` — Scripts and Data

```
3D_project/
├── process_file.py           # Quality screener (main, use this one)
├── process_file_strict.py    # Stricter threshold variant (used for comparison)
├── diffusion_script_v0.py    # Generative augmentation — Modes A and B (main version)
├── corrupt_data.py           # Dataset corruption simulator
├── remove_bg.py              # Background removal (rembg wrapper for Mode A prep)
├── rename.py                 # Renames images to numeric filenames (required for ViewCrafter)
├── Converter.py              # HEIC → JPEG batch converter
├── requirements.txt          # pip dependencies for all local scripts
├── checkpoints/
│   ├── DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth   # DUSt3R backbone (used by ViewCrafter)
│   └── model_sparse.ckpt                          # ViewCrafter model weights
├── weights/
│   └── RealESRGAN_x4plus.pth                      # RealESRGAN upscaler weights
├── output_train/             # Corrupted/sparse input scenes (input to pipeline)
│   ├── train/                # Train scene — sparse + degraded
│   ├── lego/                 # NeRF synthetic lego
│   ├── hotdog/               # NeRF synthetic hotdog
│   ├── human_heart_diff/     # Human heart (object-centric)
│   ├── classroom/            # Indoor classroom scene
│   └── ...                   # Other tested scenes
├── output_processed/         # After quality screening — manifest.json + passing images
│   ├── train/
│   ├── Lego/
│   ├── eie_building/
│   └── ...
├── output_baseline/          # Full clean datasets (upper bound baselines)
│   ├── train_dense/          # Train scene — all 301 images, clean
│   ├── eie_building_dense/
│   └── ...
├── output_baseline_d_s/      # Sparse+degraded baselines (classical SIFT comparison)
│   ├── train/
│   ├── hotdog/
│   └── ...
└── ViewCrafter/              # ViewCrafter repo (cloned locally for reference)
    ├── checkpoints/          # Symlinked or copied from 3D_project/checkpoints/
    └── ...
```

> **Storage note:** `output_train/`, `output_baseline/`, and `output_processed/` contain image sets and are large. `ViewCrafter/` contains model code and is ~500 MB without checkpoints. The checkpoints themselves (`model_sparse.ckpt`) are ~25 GB. Do not re-download if already present.

---

### 8.3 `gaussian-splatting/` — Modified Repo

This is the official [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) repo with the following files **added**:

| File | Purpose |
|---|---|
| `convert_ai.py` | AI pose estimation — SuperPoint + LightGlue replacing SIFT **(main addition, use this)** |
| `convert_ai_v2.py` | Updated variant of convert_ai.py (experimental) |
| `boost_contrast.py` | Contrast enhancement utility for preprocessing |
| `clean_splats.py` | Post-training Gaussian pruning utility |
| `dust3r_bridge.py` | DUSt3R point cloud integration for ViewCrafter conditioning |
| `inject_poses.py` | Injects camera poses from external source into COLMAP format |
| `mast3r_bridge.py` | MUSt3R integration (experimental, not used in main pipeline) |
| `train_2.py` | Modified training script variant |
| `pipeline_renderscript.ipynb` | Standalone render + metrics notebook |
| `results.md` | Experimental results log |

All original files (`train.py`, `render.py`, `metrics.py`, `convert.py`, etc.) are unmodified from the upstream repo. Use `convert_ai.py` **instead of** `convert.py` for AI-based pose estimation.

#### Input dataset structure (per scene)

Each scene lives inside `gaussian-splatting/input_dataset/<scene_name>/` and must have this structure before training:

```
input_dataset/<scene_name>/
├── input/          ← all images (original anchors + augmented views merged here)
├── distorted/
│   └── sparse/0/  ← raw COLMAP output (cameras.bin, images.bin, points3D.bin)
├── images/         ← undistorted images (produced by convert_ai.py)
└── sparse/
    └── 0/          ← final sparse model ready for train.py
```

The `input/` folder is the key handoff point: the Gradio UI in the Colab notebook writes all augmented frames plus the original anchor images into this folder before pose estimation runs.

#### Output structure (per scene)

Trained models are saved to `gaussian-splatting/output/<scene_name>_final_run/`. The naming convention used was:

| Output folder | What it represents |
|---|---|
| `train_final_run_convert_ai` | Train scene — proposed pipeline (ViewCrafter + LightGlue) |
| `train_final_run_convert_py` | Train scene — classical SIFT baseline |
| `baseline_train_dense_final_run` | Train scene — full 301 images, dense upper bound |
| `baseline_train_d_s_final_run` | Train scene — sparse+degraded, classical SIFT |
| `baseline_hotdog_final_run` | Hotdog — full clean dataset baseline |
| `baseline_hotdog_d_s_final_run` | Hotdog — sparse+degraded baseline |

---

### 8.4 Running the Unified Pipeline Notebook

Open `Pipeline_files/unified_pipeline.ipynb` in Google Colab. Select a GPU runtime (A100 recommended for Mode C).

#### Cell-by-cell run order

| Cell | What it does | Notes |
|---|---|---|
| Cell 1 | Mount Google Drive | Run first every session |
| Cell 2 | Set `SCENE_NAME` and `PIPELINE_MODE` | **Only cell you need to edit** |
| Cell 3 | Install common dependencies | Run after every Colab restart |
| Cell 4 | HuggingFace login + basicsr patch | Mode A/B only — skipped automatically for Mode C |
| Cell 5a | Clone ViewCrafter + install condacolab | **Mode C only** — runtime restarts after this |
| Cell 5a-post | Re-clone ViewCrafter + create conda env | Run immediately after the restart (Mode C only) |
| Cell 5c | Download ViewCrafter checkpoints (~25 GB) | Mode C only — skip if already downloaded |
| Cell 6 | Compile CUDA submodules + install COLMAP | Always run regardless of mode |
| Cell 7 | Launch Gradio UI | Run augmentation here — choose mode, upload images, click Run |
| Cell 8 | Pose estimation (convert_ai.py) | Run after Gradio reports Done |
| Cell 9 | 3DGS training — 30,000 iterations | Run after Cell 8 completes |
| Cell 10 | Render test views + compute PSNR/SSIM/LPIPS | Run after Cell 9 completes |

#### Cell 2 — Config (the only cell you edit)

```python
SCENE_NAME     = "train"          # must match folder name in input_dataset/
PIPELINE_MODE  = "viewcrafter"    # "synthesis" | "restoration" | "viewcrafter"
RESTORE_PROMPT = "high quality photo, detailed, sharp focus, 8k"
```

`SCENE_NAME` must exactly match the folder name inside `gaussian-splatting/input_dataset/`. The notebook uses this name to locate input images and write output files.

#### Mode C restart sequence

Mode C (ViewCrafter) requires condacolab, which forces a Colab runtime restart. After the restart, run cells in this exact order:

1. Cell 1 — Re-mount Drive
2. Cell 2 — Re-run config (keep `PIPELINE_MODE = "viewcrafter"`)
3. Cell 3 — Re-install common deps (restart wipes pip)
4. Cell 5a-post — Re-clone ViewCrafter + set up conda env
5. Cell 5c — Check/download checkpoints
6. Cell 6 onwards — continue normally

---

### 8.5 Running the Baseline Pipelines

Two additional notebooks handle the baseline conditions:

| Notebook | Purpose | Scene path variable |
|---|---|---|
| `Pipeline_files/baseline_pipeline.ipynb` | Full clean dataset baseline (Baseline A) | `rawdata_path` → `output_baseline/<scene>/` |
| `Pipeline_files/baseline_pipeline_Sparse_Degraded.ipynb` | Sparse+degraded, classical SIFT (Baseline B) | `rawdata_path` → `output_baseline_d_s/<scene>/` |

In both notebooks, change `rawdata_path` in Cell 2 to point to the correct scene folder. The notebook derives the scene name automatically from the folder name and prefixes the output accordingly (e.g. `baseline_train`, `baseline_train_d_s`).

---

### 8.6 Model Checkpoints and Weights

The following model files must be present before running the pipeline. They are large and not included in the GitHub repository.

| File | Location in project | Size | Source |
|---|---|---|---|
| `DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth` | `3D_project/checkpoints/` | ~1.1 GB | [NAVER DUSt3R releases](https://github.com/naver/dust3r) |
| `model_sparse.ckpt` | `3D_project/checkpoints/` | ~25 GB | [ViewCrafter releases](https://github.com/Drexubery/ViewCrafter) |
| `RealESRGAN_x4plus.pth` | `3D_project/weights/` | ~67 MB | [Real-ESRGAN releases](https://github.com/xinntao/Real-ESRGAN) |

The Colab notebook (Cell 5c) downloads the ViewCrafter and DUSt3R checkpoints automatically if they are not already present at `3D_project/checkpoints/`. RealESRGAN weights are downloaded automatically by the `diffusion_script_v0.py` script on first run.

---

### 8.7 Preparing a New Scene

To run the pipeline on a new scene:

1. **Create the scene folder** inside `gaussian-splatting/input_dataset/`:
   ```
   gaussian-splatting/input_dataset/<your_scene_name>/
   └── input/     ← place your images here (JPEG or PNG, numerically named)
   ```

2. **Rename images to numeric filenames** if using ViewCrafter (Mode C):
   ```bash
   # From 3D_project/ with nsenv active
   python3 rename.py --input_dir ../gaussian-splatting/input_dataset/<scene>/input
   ```
   This renames images to `000.jpg`, `001.jpg`, etc. ViewCrafter requires this.

3. **Run quality screening** (optional but recommended):
   ```bash
   python3 process_file.py \
     --mode natural \
     --input_dir ../gaussian-splatting/input_dataset/<scene>/input \
     --out_dir ./output_processed/<scene> \
     --debug
   ```

4. **Set `SCENE_NAME`** in Cell 2 of the unified pipeline notebook and run.

---

### 8.8 Utility Scripts

These scripts were used during development and may be useful for dataset preparation:

| Script | What it does | Example use |
|---|---|---|
| `remove_bg.py` | Removes background using rembg — required for Mode A (Zero123++) | `python3 remove_bg.py --input_dir ./input --out_dir ./input_nobg` |
| `rename.py` | Renames all images in a folder to numeric sequence (`000.jpg`, `001.jpg`, …) | `python3 rename.py --input_dir ./input` |
| `Converter.py` | Batch HEIC → JPEG conversion | Run directly: `python3 Converter.py` |
| `boost_contrast.py` | Applies CLAHE contrast enhancement | Located in `gaussian-splatting/` |
| `clean_splats.py` | Prunes small/transparent Gaussians from a trained .ply file | Located in `gaussian-splatting/` |

---

## 9. Gradio UI — Parameter Reference

The Gradio interface (Cell 7 of the unified pipeline) exposes the following parameters. These are the values used during development.

### Mode A — Synthesis (Zero123++)

| Field | Default | What it controls |
|---|---|---|
| Scene Name | `hotdog` | Must match folder in `output_train/` |

Internally fixed parameters (set in `diffusion_script_v0.py`):
- Top-20 quality-ranked images selected as anchors
- 6 novel views generated per anchor image
- RealESRGAN ×4 upscaling to 1024×1024

### Mode B — Restoration (ControlNet Tile)

| Field | Default | What it controls |
|---|---|---|
| Scene Name | `train` | Must match folder in `output_train/` |
| ControlNet Prompt | `high quality photo, detailed, sharp focus, 8k` | Guides restoration — keep general and quality-focused |

Internally fixed parameters:
- ControlNet strength: 0.35 (low enough to preserve structure)
- Guidance scale: 7.0, Steps: 30
- RealESRGAN ×4 upscaling to 1024×1024

### Mode C — ViewCrafter

| Field | Range | Default | What it controls |
|---|---|---|---|
| Scene Name | — | `train` | Must match folder in `output_train/` |
| Video Length | 10–50 | 25 | Frames per generated clip. Reduce to 20 if OOM on A100. |
| DDIM Steps | 20–80 | 50 | Diffusion sampling steps. Higher = better quality, slower. |

ViewCrafter runs at resolution 576×1024 using sparse-view interpolation mode with the `model_sparse.ckpt` checkpoint.

---

## 10. Training Flags Reference

When Cell 9 runs `train.py`, it uses these flags:

```bash
python train.py \
    -s "{scene_folder}" \
    -m "{output_folder}" \
    --eval
```

| Flag | Value | Effect |
|---|---|---|
| `-s` | scene folder | Source: must contain `images/` and `sparse/0/` |
| `-m` | output folder | Where the trained model and renders are saved |
| `--eval` | — | Reserves a held-out test split for PSNR/SSIM/LPIPS evaluation |

Cell 9 copies data to the Colab local SSD (`/content/local_workspace/`) before training — this significantly speeds up I/O compared to training directly on Drive. The trained model is copied back to Drive after training completes.

---

## 11. What Gets Staged Into `input/` — Mode by Mode

Understanding what ends up in `gaussian-splatting/input_dataset/{scene}/input/` before COLMAP runs is important for debugging.

**Mode A (Synthesis):**
- Original anchor images that passed quality screening (PNG, after RealESRGAN upscale)
- 6 novel views per anchor generated by Zero123++
- Total: original count + (6 × anchor count)

**Mode B (Restoration):**
- ControlNet-restored versions of REPAIR-tagged images
- NOVEL_VIEW images copied as-is
- All upscaled to 1024×1024 via RealESRGAN
- Total: same count as input (no new viewpoints added)

**Mode C (ViewCrafter):**
- All original photos from `output_train/{scene}/train/` (normalised, not quality-filtered out)
- All frames extracted from ViewCrafter's output video via FFmpeg
- Total: original count + all extracted frames (e.g. 12 originals + ~275 frames = ~287; after COLMAP registration, effective unique views ≈ 60)

---

## 12. Troubleshooting

### Quality screener passes everything / fails everything

Check which mode you used. `synthetic` (BAD_LIMIT=65) is strict — most real-world photos will fail. `natural` (BAD_LIMIT=40) is appropriate for outdoor scenes. Run with `--debug` to generate per-image diagnostic charts showing which pillar failed.

### COLMAP registers 0 cameras

The `input/` folder likely has too few images or insufficient visual overlap. Check:
- At least 12 images in `input/` with overlapping viewpoints
- Images are not all from nearly identical positions
- For Mode B: if all images were REPAIR-tagged, the augmented set adds no new viewpoints — this limits what COLMAP can register

### ViewCrafter produces hallucinated frames

A small number of hallucinated frames (typically 3–5 out of 40+) is normal. COLMAP's robust estimation treats outlier views as noise. If hallucination is severe, reduce `video_length` or increase `ddim_steps`.

### Out of memory during ViewCrafter

Reduce `video_length` from 25 to 20 in the Gradio slider. Cell 6 removes TensorFlow automatically to free GPU memory — ensure Cell 6 ran before launching the Gradio UI.

### Metrics show N/A or results.json is missing

The `--eval` flag splits images into train/test. If `input/` has fewer than ~15 images, the test split may be too small to evaluate. Ensure enough images are in `input/` before running Cell 8.

### Drive sync is slow during Cells 8 and 9

Data is copied between Drive and the local SSD before processing. On large image sets (200+ images) this can take several minutes. This is normal — Drive → SSD transfer is the bottleneck, not the computation itself.

---

## Notes

- Always activate `nsenv` before running local scripts.
- Use `--debug` with `process_file.py` when tuning thresholds — the three-panel charts show exactly which pillar failed and by how much.
- For GPU-heavy runs (diffusion augmentation, 3DGS training), use the Colab notebook rather than local scripts.
- All experimental results are logged in `Code_Reports/experimental_results.md`.
- The `input_old/` folder inside each `input_dataset/<scene>/` contains archived inputs from previous runs — safe to ignore.
- The live ViewCrafter log is written to Drive at `output_processed/{scene}/final_{scene}_run/viewcrafter_run.log` — useful for debugging if Colab disconnects mid-run.

---

**Author:** David Ogunmola (kdavid001)
**Type:** B.Eng Final Year Project
**Repository:** https://github.com/kdavid001/3D_project
