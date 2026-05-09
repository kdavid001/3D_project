# Full Pipeline Notebook — `Full_pipeline_v1.ipynb`

> **STATUS: ✅ IN USE** — Primary Colab notebook for running the complete augmented pipeline (Condition C). Located in `Pipeline_files/`.

---

## Notebook Structure

| Cell | Purpose | Status |
|---|---|---|
| 0 | Mount Google Drive | ✅ No changes needed |
| 1 | pip install from requirements.txt + check CUDA | ✅ No changes needed |
| 2 | Set paths — `rawdata_path`, `processed_path`, `filename` | ✅ No changes needed |
| 3 | Run `process_file.py` — MUSIQ quality screening, produces `manifest.json` | ✅ No changes needed |
| 4 | HuggingFace login + basicsr patch + set `mode` variable | ✅ No changes needed |
| 5 | Run `diffusion_script_v0.py` | ✅ No changes needed |
| 6 | Copy diffusion output → `gaussian-splatting/input_dataset/{name}/input/` | ✅ No changes needed |
| 7 | Install 3DGS submodules (diff-gaussian-rasterization, simple-knn) + COLMAP + ffmpeg | ✅ No changes needed |
| 8 | Pose estimation + COLMAP binary generation | ⚠️ NEEDS UPDATE — still runs `convert_ai.py`, should run `inject_poses.py` |
| 9 | Stage data locally + run `train.py` + copy to Drive | ⚠️ NEEDS UPDATE — missing `--opacity_reset_interval 9000` |
| 10 | Re-stage + run `render.py` + run `metrics.py` → SSIM/LPIPS/PSNR | ✅ Correct |

---

## Cell 8 — Required Update

**Current (broken for black-background images):**
```python
!python convert_ai.py --source_path "{local_path}"
```

**Replace with:**
```python
%cd /content/drive/MyDrive/pythonprojects_2/final_year_project/3D_project
!python inject_poses.py \
    --source_path "{local_path}" \
    --num_init_points 5000
```

Note: `inject_poses.py` writes directly to `{local_path}/sparse/0/` — no undistortion step needed. Remove the undistortion and file-moving logic that follows `convert_ai.py` if present.

---

## Cell 9 — Required Update

**Current (missing opacity fix):**
```python
!python train.py -s "{LOCAL_INPUT}" -m "{LOCAL_OUTPUT}" --eval
```

**Replace with:**
```python
!python train.py -s "{LOCAL_INPUT}" -m "{LOCAL_OUTPUT}" \
    --eval \
    --opacity_reset_interval 9000
```

**Why**: Default `--opacity_reset_interval 3000` causes the "Pruning Massacre" — Adaptive Density Control resets opacities 10 times during a 30k run, repeatedly destroying valid Gaussians in sparse-camera conditions. Setting 9000 reduces this to 3 resets and stabilises training. Without this fix, Condition A (98 cameras) produces PSNR ~4 dB instead of the expected ~32 dB.

---

## Data Flow (Full Pipeline)

```
Google Drive: output_train/{name}/          ← raw NeRF dataset
        ↓
Cell 3: process_file.py
        ↓
Google Drive: output_processed/{name}/manifest.json
        ↓
Cell 5: diffusion_script_v0.py
        ↓
Google Drive: output_processed/{name}/final_{name}_run/
        anchor_*.png  (real photos, black BG)
        synth_*_v*.png (Zero123++ views, rembg applied)
        ↓
Cell 6: copy to gaussian-splatting/input_dataset/{name}/input/
        ↓
Colab SSD: /content/local_workspace/{name}/input/
        ↓
Cell 8: inject_poses.py   [NEEDS UPDATE from convert_ai.py]
        → /content/local_workspace/{name}/sparse/0/
          cameras.bin, images.bin, points3D.bin, pose_manifest.json
        ↓
Cell 9: train.py --eval --opacity_reset_interval 9000   [NEEDS FLAG]
        → /content/local_workspace/{name}_final_run/
        ↓
Cell 10: render.py + metrics.py
        → results.json  (PSNR, SSIM, LPIPS on test split)
        ↓
Google Drive: gaussian-splatting/output/{name}_final_run/
```

---

## Baseline Notebooks (Pipeline_files/)

| Notebook | Purpose | Status |
|---|---|---|
| `baseline_pipeline.ipynb` | Condition A — full/clean dataset, original 3DGS pipeline | ✅ Used for Condition A results |
| `baseline_pipeline_Sparse_Degraded.ipynb` | Condition B — sparse/degraded, no augmentation | ✅ Used for Condition B results |
| `Full_pipeline_v1.ipynb` | Condition C — proposed augmented pipeline | ✅ Used for Condition C results |

Both baseline notebooks also need `--opacity_reset_interval 9000` added to their `train.py` call before Condition A results are valid.

---

## Current Evaluation Results (from Cell 10 output)

| Metric | Value |
|---|---|
| PSNR (test) | 13.18 dB |
| SSIM (test) | 0.7807 |
| LPIPS (test) | 0.1901 |

SSIM and LPIPS show improvement over Condition B (0.747 / 0.245). PSNR drop is explained by hemispheric coverage gap — see `thesis_notes_ch3_ch4.md` Section 4.2.3.

---

## Key Known Issues

1. **Condition A PSNR invalid** — baseline notebook needs `--opacity_reset_interval 9000` rerun
2. **Cell 8 still on convert_ai.py** — will fail / produce bad results on black-background synthesised images
3. **remove_bg.py cell** — no longer needed as rembg is now integrated directly into `diffusion_script_v0.py`
