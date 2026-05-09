# AI-Powered COLMAP Pipeline — `convert_ai.py`

> **STATUS: ❌ NOT IN USE** — Replaced by `inject_poses.py`. Still present in Full_pipeline_v1.ipynb Cell 9 but should be swapped out. Kept in codebase as a reference implementation.

---

## Why It Was Replaced

`convert_ai.py` uses SuperPoint + LightGlue feature matching for Structure-from-Motion. This works well on natural images with rich textures and gradients, but **collapses on black-background object images**:

- Black backgrounds concentrate all SIFT/SuperPoint keypoints on the object silhouette only
- Zero123++ synthesised views have micro-hallucinated textures that don't match real photos
- Result: COLMAP registers only 2 of 28 cameras — unusable for 3DGS

`inject_poses.py` bypasses feature matching entirely by writing analytically known Zero123++ poses directly to COLMAP binary format. All cameras registered by construction.

---

## When It Is Still Valid

`convert_ai.py` remains the correct tool for **natural scene images with original backgrounds** (no black compositing), where the images are not Zero123++ outputs and have sufficient texture for feature matching. If you are ever running 3DGS on real-world photos without any synthetic augmentation, this is the right pipeline.

---

## Overview

Automates the full Structure-from-Motion pipeline using the **Hierarchical Localization (hloc)** library:

1. **Cleanup** — deletes old `distorted/` and `sparse/` folders
2. **Feature Extraction (SuperPoint)** — neural keypoints on every image
3. **Feature Matching (LightGlue)** — exhaustive all-pairs geometric matching
4. **Reconstruction (PyCOLMAP)** — solves camera positions in 3D space
5. **Undistortion (COLMAP CLI)** — corrects lens distortion, outputs to `images/` + `sparse/0/`

---

## Usage

```bash
python convert_ai.py --source_path "/content/local_workspace/hotdog"
```

| Argument | Description |
|---|---|
| `--source_path` | Root folder containing the `input/` subfolder |
| `--images` | Subfolder name (default: `input`) |

---

## Common Issues

- **SfM collapse on black-background images** — use `inject_poses.py` instead
- **`Xvfb` Error** — use `xvfb-run -a python convert_ai.py ...` on headless servers
- **Memory OOM on >200 images** — switch to `pairs_from_retrieval` (sequential) instead of exhaustive matching

---

## Output Structure

```
/hotdog/
├── input/              # Original images
├── distorted/          # AI features.h5, matches.h5
├── images/             # Undistorted images
└── sparse/
    └── 0/
        ├── cameras.bin
        ├── images.bin
        └── points3D.bin
```
