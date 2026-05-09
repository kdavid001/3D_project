# Analytic Pose Injector — `inject_poses.py`

> **STATUS: ✅ IN USE** — Replaces `convert_ai.py` in the pipeline. Should be used in Full_pipeline_v1.ipynb Cell 9 (currently still on convert_ai.py — needs updating).

---

## Overview

Bypasses COLMAP Structure-from-Motion entirely. Instead of feature matching, it writes camera poses directly into COLMAP binary format using:
- **Known fixed geometry** for Zero123++ synthesised views (analytically derived from the model's training config)
- **Equatorial distribution** (or user-specified azimuths) for real anchor photos

This solves the SfM collapse problem where COLMAP only registers 2/28 cameras because black-background images starve SIFT of gradient information.

---

## Why This Replaces convert_ai.py

COLMAP feature matching (SIFT / SuperPoint) requires texture gradients distributed across the image. Black-background object images concentrate all features on the object silhouette only. Combined with Zero123++'s micro-hallucinated views, COLMAP collapses to 2 registered cameras out of 28+ — unusable for 3DGS.

`inject_poses.py` sidesteps this completely: Zero123++ v1.2 generates views at **fixed, analytically known relative poses**. These can be written directly without any feature matching.

---

## Zero123++ Fixed Offsets

| View | Azimuth (relative to anchor) | Elevation |
|---|---|---|
| v0 | +30° | +20° |
| v1 | +90° | −10° |
| v2 | +150° | +20° |
| v3 | +210° | −10° |
| v4 | +270° | +20° |
| v5 | +330° | −10° |

Source: sudo-ai/zero123plus-v1.2 training config / Appendix B.

---

## Camera Models Written

Two separate PINHOLE camera models (critical — real and synthetic cameras have genuinely different optical properties):

| Model ID | Used For | Focal Length | FOV |
|---|---|---|---|
| 1 | Real anchor photos | Width in pixels (~53°) | ~53° |
| 2 | Zero123++ synth views | Derived from Objaverse 49.13° | 49.13° |

---

## Image Classification

Reads `anchor_` and `synth_` prefixes directly from the images directory — no manifest required. This is why rename.py should be skipped: it destroys the prefix information.

- `anchor_*.jpg/png` → real camera, camera_id=1
- `synth_{base}_v{n}.png` → synthetic view, camera_id=2, offset applied from corresponding anchor's azimuth

---

## Point Cloud Initialisation

Generates a Fibonacci sphere of N points (default 500) as the initial point cloud. This is far superior to random sampling — Fibonacci lattice guarantees near-uniform coverage so 3DGS densification has starting Gaussians near every surface, not just the visible front.

Radius 0.5 (half the scene extent) places Gaussians on the object surface rather than at the origin (which causes immediate pruning).

Also writes `pose_manifest.json` — human-readable JSON of all registered cameras, useful for debugging and the thesis appendix.

---

## Usage

```bash
# Standard (anchor_/synth_ naming preserved — rename.py was skipped)
python inject_poses.py --source_path /content/local_workspace/hotdog

# If rename.py already ran (sequential names, anchor_ prefix destroyed)
python inject_poses.py --source_path /content/local_workspace/hotdog \
    --images_dir images --num_real 4

# If you know the actual azimuths of your real photos
python inject_poses.py --source_path /content/local_workspace/hotdog \
    --real_azimuths 0 90 180 270

# More init points for richer Gaussian densification
python inject_poses.py --source_path /content/local_workspace/hotdog \
    --num_init_points 5000
```

---

## Arguments

| Argument | Default | Description |
|---|---|---|
| `--source_path` | required | Root dataset folder |
| `--images_dir` | `input` | Subfolder with images |
| `--num_real` | auto | Required only if rename.py already ran |
| `--real_azimuths` | evenly spaced | Override with measured angles for accuracy |
| `--real_elevation` | 0.0° | Elevation for real photos |
| `--radius` | 2.0 | Camera orbital radius |
| `--image_size` | 1024 | Square image resolution |
| `--num_init_points` | 500 | Fibonacci sphere point count |

---

## Output

Writes directly to `{source_path}/sparse/0/`:
- `cameras.bin` — 2 camera models (real + synth PINHOLE)
- `images.bin` — all registered images with poses
- `points3D.bin` — Fibonacci sphere init cloud
- `pose_manifest.json` — human-readable debug summary

After running, go straight to `train.py` — no undistortion step needed.

```bash
python train.py -s "{source_path}" -m "{output_path}" \
    --eval --opacity_reset_interval 9000
```
