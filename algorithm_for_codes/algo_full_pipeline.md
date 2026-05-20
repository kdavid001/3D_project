# Algorithm: Full Augmented 3DGS Pipeline Orchestrator
# File: Full_pipeline_v1.ipynb
# Status: IN USE — primary Colab notebook (Condition C)

---

**Algorithm 11: FULL_AUGMENTED_3DGS_PIPELINE**

**Input:**
- rawdata_path  : Google Drive path to raw NeRF dataset
- processed_path: Google Drive path for processed/augmented output
- filename      : scene/object name (e.g., "hotdog", "train")
- mode          : "synthesis", "restoration", or "natural"

**Output:**
- Trained 3DGS model checkpoint at gaussian-splatting/output/{filename}_final_run/
- Evaluation metrics: PSNR, SSIM, LPIPS on held-out test split

---

```
BEGIN FULL_AUGMENTED_3DGS_PIPELINE(rawdata_path, processed_path, filename, mode)

  // ============================================================
  // CELL 0: Environment Setup
  // ============================================================
  MOUNT Google Drive
  INSTALL dependencies from requirements.txt
  VERIFY CUDA availability

  // ============================================================
  // CELL 2: Path Configuration
  // ============================================================
  raw_input    ← rawdata_path / filename / "train"
  out_dir      ← processed_path / filename
  final_output ← gaussian-splatting / input_dataset / filename / input

  // ============================================================
  // CELL 3: Quality Screening (Algorithm 4 — process_files.py)
  // ============================================================
  // Score every image against Five Pillars (MUSIQ, saturation,
  // exposure, contrast, colour cast)
  RUN process_files.py(
    input_dir = raw_input,
    mode      = mode,       // "synthetic" for blender datasets
    debug     = False
  )
  // Produces: manifest.json with per-image routing tags

  // ============================================================
  // CELL 4: Model Authentication & Patch
  // ============================================================
  LOGIN to HuggingFace (for Zero123++ and ControlNet model access)
  APPLY basicsr functional_tensor compatibility patch
  // Required for RealESRGAN on PyTorch >= 2.0

  SET mode variable for diffusion_script

  // ============================================================
  // CELL 5: Generative Augmentation (Algorithm 5 — diffusion_script_v0.py)
  // ============================================================
  // Routes images by manifest.json tag:
  //   NOVEL_VIEW → Zero123++ (6 novel views per anchor)
  //   REPAIR     → ControlNet Tile (image restoration)
  // All outputs upscaled to 1024×1024 via RealESRGAN
  RUN diffusion_script_v0.py(
    input_dir = out_dir,
    out_dir   = out_dir,
    mode      = mode
  )
  // Produces: out_dir/final_{filename}_run/
  //   anchor_*.png, synth_*_v[0-5].png, restored_*.png

  // ============================================================
  // CELL 6: Stage Data for Training
  // ============================================================
  COPY out_dir / final_{filename}_run / *.png
       → gaussian-splatting / input_dataset / {filename} / input /

  local_path ← /content/local_workspace / {filename}
  COPY from Drive to Colab SSD (faster I/O for training)

  // ============================================================
  // CELL 7: Build 3DGS Dependencies
  // ============================================================
  INSTALL diff-gaussian-rasterization (CUDA submodule)
  INSTALL simple-knn (CUDA submodule)
  INSTALL COLMAP and ffmpeg

  // ============================================================
  // CELL 8: Pose Estimation (Algorithm 6 — convert_ai.py)
  // ============================================================
  // SuperPoint + LightGlue feature matching → PyCOLMAP SfM
  // → COLMAP undistortion
  RUN convert_ai.py(
    source_path = local_path
  )
  // Produces: local_path/sparse/0/ {cameras.bin, images.bin, points3D.bin}
  //           local_path/images/   {undistorted images}

  // ============================================================
  // CELL 9: 3DGS Training
  // ============================================================
  RUN train.py(
    source_path              = local_path,
    model_path               = local_path + "_final_run",
    eval                     = True,
    opacity_reset_interval   = 9000   // reduces Adaptive Density Control pruning
  )
  // 30,000 iteration training loop
  // Adaptive Density Control: clone/split dense regions, prune transparent Gaussians
  // opacity_reset_interval=9000 prevents "Pruning Massacre" in sparse-camera conditions

  COPY trained model → Drive / gaussian-splatting / output / {filename}_final_run /

  // ============================================================
  // CELL 10: Evaluation
  // ============================================================
  RUN render.py(
    model_path = local_path + "_final_run",
    source_path= local_path
  )
  // Renders novel views from held-out test camera positions

  RUN metrics.py(
    model_path = local_path + "_final_run"
  )
  // Computes: PSNR, SSIM, LPIPS vs ground truth test images

  LOAD results.json
  PRINT PSNR, SSIM, LPIPS

END FULL_AUGMENTED_3DGS_PIPELINE
```

---

**Complete Data Flow:**

```
Google Drive: raw dataset (NeRF Synthetic format)
        ↓  Cell 3
manifest.json  (MUSIQ scores + NOVEL_VIEW / REPAIR tags)
        ↓  Cell 5
final_{name}_run/  (anchor_*.png + synth_*_v*.png + restored_*.png @ 1024×1024)
        ↓  Cell 6
gaussian-splatting/input_dataset/{name}/input/
        ↓  Cell 8
sparse/0/  (cameras.bin, images.bin, points3D.bin)
        ↓  Cell 9
train.py → {name}_final_run/  (trained .ply + checkpoints)
        ↓  Cell 10
render.py + metrics.py → results.json (PSNR, SSIM, LPIPS)
```

---

**Experimental Conditions:**

| Condition | Notebook                              | Description                                  |
|-----------|---------------------------------------|----------------------------------------------|
| A         | baseline_pipeline.ipynb              | Full clean dataset, standard 3DGS            |
| B         | baseline_pipeline_Sparse_Degraded.ipynb | Sparse + degraded inputs, no augmentation |
| C         | Full_pipeline_v1.ipynb (this file)   | Proposed augmented pipeline                  |

---

**Critical Training Parameter:**
- `--opacity_reset_interval 9000` is mandatory for sparse-camera conditions.
- Default value (3000) triggers 10 Adaptive Density Control resets during a 30k-iteration run, repeatedly pruning valid Gaussians before they densify.
- Setting 9000 reduces resets to 3, stabilising training when camera coverage is limited.
