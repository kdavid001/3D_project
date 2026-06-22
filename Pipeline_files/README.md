# Pipeline Files — Unified 3DGS Augmentation Pipeline

This folder contains all Colab notebooks used to run the pipeline end-to-end.

---

## Notebooks

| Notebook | Purpose |
|---|---|
| `unified_pipeline.ipynb` | **Main pipeline** — all three augmentation modes (A/B/C) + pose estimation + training |
| `baseline_pipeline.ipynb` | Baseline A — full clean dataset, classical SIFT |
| `baseline_pipeline_Sparse_Degraded.ipynb` | Baseline B — sparse + degraded, classical SIFT |
| `Full_pipeline_v1.ipynb` | Archived earlier version (reference only) |
| `Full_pipeline_v2.ipynb` | Archived earlier version (reference only) |
| `Full_pipeline_Using_Video models.ipynb` | Archived ViewCrafter prototype (reference only) |

For all new runs, use `unified_pipeline.ipynb` only.

---

## unified_pipeline.ipynb — Full Reference

### What It Does

Takes a sparse or degraded image set from Google Drive, routes it through one of three generative augmentation strategies, estimates camera poses using SuperPoint + LightGlue, trains a 3D Gaussian Splatting model, and computes PSNR / SSIM / LPIPS on held-out test views — all from a single notebook with a browser-based Gradio interface.

---

### Google Drive Folder Requirement

The notebook expects this exact folder layout in your Google Drive:

```
MyDrive/
└── pythonprojects_2/
    └── final_year_project/
        ├── 3D_project/
        │   ├── output_train/          ← your input scenes go here
        │   ├── output_processed/      ← quality screener writes here
        │   ├── requirements.txt
        │   ├── process_file.py
        │   └── diffusion_script_v0.py
        └── gaussian-splatting/
            ├── convert_ai.py          ← AI pose estimator
            ├── train.py
            ├── render.py
            ├── metrics.py
            ├── input_dataset/         ← staged images land here
            └── output/                ← trained models saved here
```

If you rename or move either `3D_project/` or `gaussian-splatting/`, update `DRIVE_BASE` and `GS_BASE` in Cell 2.

---

### Cell-by-Cell Reference

#### Cell 1 — Mount Google Drive
```python
drive.mount('/content/drive', force_remount=True)
```
Run this first every Colab session. Re-run after any runtime restart.

---

#### Cell 2 — Configuration (only cell you need to edit)
```python
SCENE_NAME     = "train"          # folder inside output_train/ on Drive
PIPELINE_MODE  = "viewcrafter"    # "synthesis" | "restoration" | "viewcrafter"
RESTORE_PROMPT = "high quality photo, detailed, sharp focus, 8k"
```

`SCENE_NAME` must exactly match the folder name inside `output_train/` and `input_dataset/`. The notebook uses this name throughout all subsequent cells.

`RESTORE_PROMPT` is only used by Mode B (ControlNet restoration). It guides the diffusion model — keep it general and quality-focused.

---

#### Cell 3 — Common Dependencies
Installs all pip requirements from `requirements.txt`, with special handling for packages that have version pins incompatible with Colab 2025 (`numpy`, `diffusers`, `transformers`, `huggingface-hub`, `peft`, `accelerate`). These are filtered out and reinstalled at safe versions.

Must be re-run after every condacolab restart (Mode C).

If this cell crashes with a numpy or ABI error, restart the runtime and re-run from Cell 1 — it is safe to re-run.

---

#### Cell 4 — Mode A/B Setup (skipped automatically for Mode C)
- Logs in to HuggingFace using the `HF_TOKEN` secret stored in Colab's user secrets
- Applies a one-line compatibility patch to `basicsr` (torchvision API change in Python 3.12)

To set your HuggingFace token: In Colab, go to the key icon (Secrets) in the left sidebar → add a secret named `HF_TOKEN` → paste your token.

---

#### Cells 5a / 5a-post / 5c — ViewCrafter Setup (Mode C only)

**Cell 5a** — Clones ViewCrafter from GitHub and installs `condacolab`. The runtime **restarts automatically** after this cell. This is expected behaviour — do not try to prevent it.

**After the restart**, re-run in this exact order:
1. Cell 1 — Re-mount Drive
2. Cell 2 — Re-set config (keep `PIPELINE_MODE = "viewcrafter"`)
3. Cell 3 — Reinstall common deps (restart wipes all pip installs)
4. Cell 5a-post — Re-clone ViewCrafter + create conda env

**Cell 5a-post** — Creates the `viewcrafter_env` conda environment with:
- Python 3.10
- PyTorch 2.1.0 + CUDA 12.1
- numpy < 2.0 (required for PyTorch 2.1.0 compatibility)
- ViewCrafter's specific dependencies

This takes 5–10 minutes. Run it once — the env persists for the session.

**Cell 5c** — Downloads model checkpoints to `/content/ViewCrafter/checkpoints/`:
- `DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth` (~1.1 GB) — DUSt3R backbone
- `model_sparse.ckpt` (~23 GB) — ViewCrafter model weights

Downloads are skipped if the files already exist. Total download time on a Colab A100: approximately 20–40 minutes depending on network.

---

#### Cell 6 — CUDA Submodules + COLMAP (always run)
- Compiles `diff-gaussian-rasterization` and `simple-knn` CUDA extensions on the local NVMe SSD (faster than Drive)
- Installs `ffmpeg`, `colmap`, `xvfb`, `pycolmap`
- Removes TensorFlow to free GPU memory

Always run this cell regardless of mode, even after a restart.

---

#### Cell 7 — Gradio UI

Launches the augmentation interface at a public URL (valid for 72 hours). The notebook prints the URL — open it in any browser.

**Mode A — Synthesis tab:**
- Field: `Scene Name` — enter the scene name (must match folder in `output_train/`)
- Input expected at: `3D_project/output_train/{scene}/` (flat folder of images)
- What runs: MUSIQ screening → Zero123++ (6 views/anchor) → RealESRGAN ×4 → staged to `input_dataset/{scene}/input/`

**Mode B — Restoration tab:**
- Fields: `Scene Name`, `ControlNet Prompt`
- Input expected at: `3D_project/output_train/{scene}/` (flat folder of images)
- What runs: MUSIQ screening → ControlNet Tile (strength 0.35) → RealESRGAN ×4 → staged to `input_dataset/{scene}/input/`

**Mode C — ViewCrafter tab:**
- Fields: `Scene Name`, `Video Length` (slider 10–50, default 25), `DDIM Steps` (slider 20–80, default 50)
- Input expected at: `3D_project/output_train/{scene}/train/` — note the `/train/` subdirectory
- What runs: image normalisation → MUSIQ screening → ViewCrafter sparse-view interpolation → FFmpeg frame extraction → originals + extracted frames staged to `input_dataset/{scene}/input/`

**ViewCrafter parameters:**
- `Video Length` — number of frames per generated video clip. Default 25. Reduce to 20 if you get OOM errors on A100 (40 GB).
- `DDIM Steps` — diffusion sampling steps. Default 50. Higher = better quality, slower. 50 is the practical limit on A100.

**What the Gradio UI does automatically:**
1. Copies raw images from `output_train/` to `output_processed/` (originals are never modified)
2. Converts HEIC/BMP/TIFF to JPEG, resizes images above 1920px
3. Runs `process_file.py` for quality screening
4. Renames images to numeric filenames (required for ViewCrafter sort order)
5. Applies compatibility patches to ViewCrafter's Python files automatically
6. Runs inference and extracts frames
7. Merges original photos + synthesised frames into `input_dataset/{scene}/input/`

Progress is shown in the Live Log panel inside the UI. For Mode C, a live log file is also written to Drive at `output_processed/{scene}/final_{scene}_run/viewcrafter_run.log` — check this file if you get disconnected.

When the log shows `Run Cell 8 (convert_ai) next` — the Gradio step is complete.

---

#### Cell 8 — Pose Estimation
```python
python convert_ai.py --source_path "{local_path}"
```
Copies `input_dataset/{scene}/input/` to the Colab local SSD, runs SuperPoint + LightGlue exhaustive matching, runs COLMAP `incremental_mapping`, runs `image_undistorter`, then syncs the results (`sparse/`, `images/`, `distorted/`) back to Drive.

Expected output on Drive after Cell 8:
```
gaussian-splatting/input_dataset/{scene}/
├── input/         ← all augmented images (input to convert_ai.py)
├── images/        ← undistorted images (output of image_undistorter)
├── distorted/
│   └── sparse/0/ ← raw COLMAP model
└── sparse/
    └── 0/
        ├── cameras.bin
        ├── images.bin
        └── points3D.bin
```

---

#### Cell 9 — 3DGS Training
```python
python train.py \
    -s "{LOCAL_INPUT}" \
    -m "{LOCAL_OUTPUT}" \
    --eval \
    --opacity_reset_interval 9000
```
Runs 30,000 iterations of 3DGS training on the local SSD (faster I/O), then copies the trained model to Drive at `gaussian-splatting/output/{scene}_final_run/`.

Key flags:
- `--eval` — reserves a held-out test split for evaluation
- `--opacity_reset_interval 9000` — resets Gaussian opacity every 9,000 iterations to prevent floaters

Training time: approximately 45–60 minutes on an A100.

---

#### Cell 10 — Render + Metrics
```python
python render.py -m "{DRIVE_OUT}" -s "{LOCAL_INPUT}" --skip_train
python metrics.py -m "{DRIVE_OUT}"
```
Renders novel test views using the trained model, then computes PSNR, SSIM, and LPIPS against the ground-truth held-out images. Results saved to `output/{scene}_final_run/results.json`.

---

### Complete Run Order Summary

**Modes A and B:**
```
Cell 1 → Cell 2 → Cell 3 → Cell 4 → Cell 6 → Cell 7 (Gradio) → Cell 8 → Cell 9 → Cell 10
```

**Mode C (ViewCrafter):**
```
Cell 1 → Cell 2 → Cell 3 → Cell 5a → [RESTART] →
Cell 1 → Cell 2 → Cell 3 → Cell 5a-post → Cell 5c → Cell 6 → Cell 7 (Gradio) → Cell 8 → Cell 9 → Cell 10
```

---

### Pipeline Data Flow

```
3D_project/output_train/{scene}/          ← raw input (corrupted/sparse images)
        │
        │  [Cell 7 — Gradio UI]
        ▼
3D_project/output_processed/{scene}/      ← after screening + augmentation
        │   ├── Mode A: Zero123++ grid PNGs + RealESRGAN upscales
        │   ├── Mode B: ControlNet restored JPEGs + RealESRGAN upscales
        │   └── Mode C: ViewCrafter frames (extracted_frames/) + originals
        │
        │  [Cell 7 — staging step]
        ▼
gaussian-splatting/input_dataset/{scene}/input/   ← merged originals + augmented views
        │
        │  [Cell 8 — convert_ai.py]
        ▼
gaussian-splatting/input_dataset/{scene}/sparse/0/  ← COLMAP sparse model + undistorted images
        │
        │  [Cell 9 — train.py, 30,000 iterations]
        ▼
gaussian-splatting/output/{scene}_final_run/         ← trained 3DGS model (.ply)
        │
        │  [Cell 10 — render.py + metrics.py]
        ▼
gaussian-splatting/output/{scene}_final_run/results.json  ← PSNR / SSIM / LPIPS
```

---

### Input Folder Requirements by Mode

| Mode | Scene input path | Image format | Notes |
|---|---|---|---|
| A — Synthesis | `output_train/{scene}/` | JPG/PNG/HEIC | Flat folder, any filenames |
| B — Restoration | `output_train/{scene}/` | JPG/PNG/HEIC | Flat folder, any filenames |
| C — ViewCrafter | `output_train/{scene}/train/` | JPG/PNG/HEIC | Must be in `/train/` subfolder |

Mode C expects images in a `/train/` subdirectory because the Train scene dataset from Tanks and Temples has this structure. When using a custom scene, create the subfolder manually: `output_train/{your_scene}/train/`.

---

### Common Issues and Fixes

| Problem | Cause | Fix |
|---|---|---|
| `DRIVE_BASE not found` | Drive not mounted | Re-run Cell 1 |
| `HF_TOKEN not found` | Secret not set in Colab | Add `HF_TOKEN` in Colab Secrets (key icon, left sidebar) |
| Gradio URL not working | Share link expired (72h limit) | Re-run Cell 7 to get a new URL |
| Mode C OOM error | video_length too high | Reduce video_length to 20 in the Gradio slider |
| `No cameras registered` | Insufficient feature matches | Check that `input/` has enough images and they have visual overlap |
| `results.json` missing after Cell 10 | `--eval` split too small | Ensure at least 10–15 images in `input/` before running Cell 8 |
| Cell 3 crashes on restart | numpy ABI mismatch | Restart runtime, re-run from Cell 1 |
| ViewCrafter produces black frames | Wrong .mp4 used | Ensure `diffusion.mp4` is used for object-centric scenes, not `render.mp4` |

---

### Output Structure (Drive)

```
gaussian-splatting/
├── input_dataset/{scene}/
│   ├── input/              ← all images fed to COLMAP
│   ├── images/             ← undistorted images
│   ├── distorted/sparse/0/ ← raw COLMAP output
│   └── sparse/0/           ← final model for train.py
└── output/{scene}_final_run/
    ├── point_cloud/
    │   └── iteration_30000/
    │       └── point_cloud.ply   ← trained 3DGS model
    ├── train/                    ← rendered training views
    ├── test/                     ← rendered test views
    └── results.json              ← PSNR / SSIM / LPIPS
```

---

### Related Files

| File | Location | Purpose |
|---|---|---|
| `user_guide.md` | project root | Full standalone and project-folder usage guide |
| `algorithm_for_codes/algo_unified_pipeline.md` | project root | Algorithm 6 pseudocode |
| `Code_Reports/flowdiagram/unified_pipeline.txt` | project root | Mermaid pipeline flowchart |
| `diffusion_script_v0.py` | `3D_project/` | Mode A/B augmentation engine |
| `convert_ai.py` | `gaussian-splatting/` | AI pose estimator (SuperPoint + LightGlue) |
| `process_file.py` | `3D_project/` | Five-pillar quality screener |
