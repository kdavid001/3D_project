# unified_pipeline.ipynb

**Unified 3DGS Augmentation Pipeline** — single Colab notebook for all three pipeline modes.

---

## What This Does

Takes a sparse or degraded image set, augments it using one of three generative strategies, estimates camera poses, and trains a 3D Gaussian Splatting model — all from one notebook with a browser-based Gradio interface.

---

## Quick Start

### Step 1 — Open in Colab
Upload `unified_pipeline.ipynb` from this folder to Google Colab (or open directly from Drive).

### Step 2 — Edit Cell 2 only
```python
SCENE_NAME    = "hotdog"       # folder name inside output_train/ on Drive
PIPELINE_MODE = "synthesis"    # "synthesis" | "restoration" | "viewcrafter"
```

### Step 3 — Run cells in order
| Cells | Action |
|---|---|
| 1 → 4 | Setup (Modes A and B) |
| 1 → 5a → restart → 1 → 2 → 3 → 5a-post → 5c | Setup (Mode C — see below) |
| 6 | CUDA + COLMAP install (always) |
| 7 | Launch Gradio UI |
| 8 → 9 → 10 | Run manually after Gradio completes |

### Step 4 — Use the Gradio interface
The notebook prints a public URL (valid 72 hours). Open it, enter your scene name, and click Run. Progress logs update in real time.

### Step 5 — Run Cells 8–10 manually
After Gradio reports Done, run Cell 8 (pose estimation), Cell 9 (training), Cell 10 (metrics) in sequence.

---

## Modes

### Mode A — Synthesis (Zero123++)
**Use when:** Object-centric NeRF-style dataset (e.g. hotdog, lego, chair). Images have or can be composited onto black backgrounds.

**What it does:**
- MUSIQ quality screener classifies each image as Novel View candidate or Repair candidate
- Good images → Zero123++ generates 6 novel views at fixed relative camera offsets
- Degraded images → ControlNet Tile restoration
- All outputs upscaled to 1024×1024 via RealESRGAN

### Mode B — Restoration (ControlNet Tile)
**Use when:** Real-world images degraded by blur, noise, or exposure failure. Natural backgrounds should be preserved.

**What it does:**
- ControlNet Tile (strength 0.35) applied to correct degradation while preserving structure
- No novel view generation — focuses purely on image quality improvement
- RealESRGAN upscale to 1024×1024

### Mode C — ViewCrafter (Natural Scenes)
**Use when:** Natural scene with real backgrounds and sparse camera coverage. No black background compositing required.

**What it does:**
- DUSt3R builds a coarse point cloud from your sparse photos
- ViewCrafter video diffusion generates an interpolated novel-view video (render.mp4)
- ffmpeg extracts every 4th frame from the video
- Extracted frames + original photos merged into a single image set for COLMAP

> **Note:** ViewCrafter's point cloud is a coarse internal scaffold — its actual output is the rendered video (2D frames). These frames are what feed into COLMAP and 3DGS. ViewCrafter performs view synthesis; 3DGS performs scene reconstruction. They serve different purposes.

---

## Mode C — Runtime Restart

Cell 5a installs `condacolab`, which automatically restarts the Colab runtime. After restart:

1. Re-run **Cell 1** (Mount Drive)
2. Re-run **Cell 2** (set `PIPELINE_MODE = "viewcrafter"`)
3. Re-run **Cell 3** (reinstall deps — gradio is wiped by restart)
4. Run **Cell 5a-post** (re-clone ViewCrafter + create conda env)
5. Run **Cell 5c** (download checkpoints — cached if previously downloaded)
6. Continue from **Cell 6**

---

## Pipeline Flow (All Modes)

```
raw images (output_train/{scene}/)
        ↓
[Cell 7 — Gradio]
        ├── Mode A: Zero123++ + ControlNet + RealESRGAN
        ├── Mode B: ControlNet + RealESRGAN  
        └── Mode C: ViewCrafter → render.mp4 → ffmpeg frames
        ↓
gaussian-splatting/input_dataset/{scene}/input/
        ↓
[Cell 8] convert_ai.py
        NetVLAD → SuperPoint → LightGlue → PyCOLMAP → undistortion
        → sparse/0/cameras.bin + images.bin + points3D.bin
        ↓
[Cell 9] train.py --eval --opacity_reset_interval 9000
        → 30,000 iteration 3DGS training
        ↓
[Cell 10] render.py + metrics.py
        → PSNR / SSIM / LPIPS (results.json)
```

---

## Input Requirements

| Mode | Expected input folder structure |
|---|---|
| A, B | `output_train/{scene}/` with images (jpg/png) |
| C | `output_train/{scene}/train/` with images |

Scene name must match the folder name exactly. Set in Cell 2.

---

## Output Structure (Google Drive)

```
gaussian-splatting/
├── input_dataset/{scene}/
│   ├── input/          ← augmented images staged for COLMAP
│   ├── images/         ← undistorted images (after Cell 8)
│   └── sparse/0/
│       ├── cameras.bin
│       ├── images.bin
│       └── points3D.bin
└── output/{scene}_final_run/
    ├── point_cloud/    ← trained 3DGS model (.ply)
    ├── train/          ← rendered training views
    ├── test/           ← rendered test views
    └── results.json    ← PSNR / SSIM / LPIPS
```

---

## Dependency Notes

`requirements.txt` contains old version pins incompatible with Colab 2025. Cell 3 handles this automatically by filtering out version-sensitive packages and reinstalling them at safe versions. You do not need to edit `requirements.txt`.

If Cell 3 crashes with a numpy or ABI error, restart the runtime and re-run — it is safe to re-run from Cell 1.

---

## Related Files

| File | Location | Purpose |
|---|---|---|
| `Code_Reports/unified_pipeline.md` | project root | Full technical reference |
| `algorithm_for_codes/algo_unified_pipeline.md` | project root | Pseudocode algorithm |
| `diffusion_script_v0.py` | project root | Mode A/B augmentation engine |
| `convert_ai.py` | gaussian-splatting/ | SfM pose estimator |
| `thesis_notes_ch3_ch4.md` | project root | Dissertation methodology notes |
