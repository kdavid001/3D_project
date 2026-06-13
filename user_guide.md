# User Guide — Generative Augmented 3D Reconstruction Pipeline

This guide covers local environment setup and how to run each script in the pipeline. The full end-to-end reconstruction (augmentation → pose estimation → 3DGS training) is orchestrated by `Pipeline_files/unified_pipeline.ipynb` on Google Colab. The local scripts documented here are used for dataset preparation and inspection.

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

### MUSIQ soft/hard logic

MUSIQ scores can be miscalibrated for indoor scenes shot under artificial lighting. The screener uses a two-level gate:

- `score < HARD_LIMIT` — always fail regardless of the other four pillars (genuinely unusable image)
- `HARD_LIMIT ≤ score < BAD_LIMIT` — only fail if at least one other pillar also fails (borderline, corroborated failure)
- `score ≥ BAD_LIMIT` — MUSIQ passes; other pillars decide independently

This prevents sharp indoor images from being incorrectly flagged while still catching genuinely blurry outdoor images that coincidentally pass all radiometric checks.

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

```bash
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

> **Note:** The script auto-installs `hloc` and clones `SuperGluePretrainedNetwork` on first run if they are not present.

---

## 7. Full Pipeline — `Pipeline_files/unified_pipeline.ipynb`

The main end-to-end pipeline runs on Google Colab (GPU runtime). It supports three modes:

| Mode | Description |
|---|---|
| A — Zero123++ | Object-centric synthetic scenes; black background required |
| B — ViewCrafter | Real-world scenes; video diffusion from sparse views via DUSt3R point cloud |
| C — ControlNet | Natural images flagged for restoration |

Open the notebook in Colab, select a GPU runtime (A100 recommended for Mode B), and follow the cell-by-cell instructions. Key parameters are set in the configuration cell at the top.

### ViewCrafter-specific notes (Mode B)

- Install `pyav` in the notebook before running: `!pip install av`
- Rename input images to numeric filenames (`000.jpg`, `001.jpg`, …) — ViewCrafter sorts by `int(stem)` and will silently mis-order non-numeric names
- Set `video_length ≤ 20` to avoid OOM on A100 (40 GB); `video_length=25` at `ddim_steps=50` requires ~40 GB VRAM
- ViewCrafter requires a separate conda environment (`viewcrafter_env`) due to a hard `transformers < 4.40` dependency conflict with the main pipeline

---

## Notes

- Always activate `nsenv` before running local scripts.
- Use `--debug` with `process_file.py` when tuning thresholds — the three-panel charts show exactly which pillar failed and by how much.
- For GPU-heavy runs (diffusion augmentation, 3DGS training), use the Colab notebook rather than local scripts.
- All experimental results are logged in `Code_Reports/experimental_results.md`.

---

**Author:** David Ogunmola (kdavid001)
**Type:** B.Eng Final Year Project
