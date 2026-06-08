# Algorithm Summary: Full Augmented 3DGS Pipeline Orchestrator
# Source: Full_pipeline_v1.ipynb | Full algorithm: algorithm_for_codes/algo_full_pipeline.md
# Status: IN USE — primary Colab notebook (Condition C experiment)

---

**Algorithm 11 (Summary): FULL_AUGMENTED_3DGS_PIPELINE**

**Input:**
- rawdata_path   : Drive path to raw NeRF dataset
- processed_path : Drive path for augmented output
- filename       : scene name (e.g., "hotdog", "lego")
- mode           : "synthesis" | "restoration" | "natural"

**Output:**
- Trained 3DGS model: gaussian-splatting/output/{filename}_final_run/
- results.json: PSNR, SSIM, LPIPS on held-out test split

---

```
BEGIN FULL_AUGMENTED_3DGS_PIPELINE(rawdata_path, processed_path, filename, mode)

  // Cell 0: Environment setup
  MOUNT Google Drive
  INSTALL dependencies from requirements.txt

  // Cell 2: Paths
  raw_input ← rawdata_path / filename / "train"
  out_dir   ← processed_path / filename

  // Cell 3: Quality screening (Algorithm 4 — process_files.py)
  RUN process_files.py(input_dir=raw_input, mode=mode)
  // → manifest.json: NOVEL_VIEW | REPAIR per image

  // Cell 4: Auth + compatibility patch
  LOGIN HuggingFace
  APPLY basicsr functional_tensor patch   // RealESRGAN compatibility

  // Cell 5: Generative augmentation (Algorithm 5 — diffusion_script_v0.py)
  RUN diffusion_script_v0.py(input_dir=out_dir, out_dir=out_dir, mode=mode)
  // NOVEL_VIEW → Zero123++ (6 views each) + RealESRGAN → 1024×1024
  // REPAIR     → ControlNet Tile (strength 0.35) + RealESRGAN → 1024×1024

  // Cell 6: Stage data for training
  COPY out_dir/final_{filename}_run/*.png
       → gaussian-splatting/input_dataset/{filename}/input/
  COPY Drive → /content/local_workspace/{filename}   // Drive→SSD for faster I/O

  // Cell 7: Build CUDA submodules
  INSTALL diff-gaussian-rasterization, simple-knn, COLMAP, ffmpeg

  // Cell 8: Pose estimation (Algorithm 6 — convert_ai.py)
  RUN convert_ai.py(source_path=local_path)
  // → sparse/0/{cameras,images,points3D}.bin + images/ (undistorted)

  // Cell 9: 3DGS training
  RUN train.py(
    source_path            = local_path,
    model_path             = local_path + "_final_run",
    eval                   = True,
    opacity_reset_interval = 9000      // prevents Pruning Massacre in sparse runs
  )
  // 30,000 iteration photometric optimisation + Adaptive Density Control

  COPY trained model → Drive/gaussian-splatting/output/{filename}_final_run/

  // Cell 10: Evaluation
  RUN render.py(model_path, source_path, skip_train=True)
  RUN metrics.py(model_path)
  // → results.json: PSNR, SSIM, LPIPS

END FULL_AUGMENTED_3DGS_PIPELINE
```

---

**Experimental Conditions:**

| Condition | Notebook                               | Description                         |
|-----------|----------------------------------------|-------------------------------------|
| A         | baseline_pipeline.ipynb               | Full clean dataset, standard 3DGS   |
| B         | baseline_pipeline_Sparse_Degraded.ipynb | Sparse + degraded, no augmentation|
| C         | Full_pipeline_v1.ipynb (this)         | Proposed augmented pipeline         |

---
