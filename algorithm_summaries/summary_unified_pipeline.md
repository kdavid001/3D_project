# Algorithm Summary: Unified 3DGS Augmentation Pipeline
# Source: unified_pipeline.ipynb | Full algorithm: algorithm_for_codes/algo_unified_pipeline.md
# Status: IN USE — single notebook for all three pipeline modes (supersedes Full_pipeline_v1.ipynb)

---

**Algorithm 12 (Summary): UNIFIED_3DGS_PIPELINE**

**Input:**
- SCENE_NAME    : folder name inside output_train/ (e.g., "hotdog", "lego")
- PIPELINE_MODE : "synthesis" | "restoration" | "viewcrafter"
- RESTORE_PROMPT: ControlNet prompt (Mode B only)
- video_length  : frames for ViewCrafter (Mode C only, default 25)
- ddim_steps    : diffusion sampling steps (Mode C only, default 50)

**Output:**
- Trained 3DGS model: gaussian-splatting/output/{SCENE_NAME}_final_run/
- results.json: PSNR, SSIM, LPIPS

---

```
BEGIN UNIFIED_3DGS_PIPELINE(SCENE_NAME, PIPELINE_MODE, ...)

  // Cell 1-3: Mount Drive, configure paths, install dependencies
  MOUNT Google Drive
  rawdata_path ← DRIVE_BASE / output_train / SCENE_NAME
  processed    ← DRIVE_BASE / output_processed / SCENE_NAME

  // Cell 4: HuggingFace auth + basicsr patch (Modes A and B only)
  IF PIPELINE_MODE IN ["synthesis", "restoration"] THEN
    LOGIN HuggingFace
    APPLY basicsr functional_tensor compatibility patch
  END IF

  // Cell 5: ViewCrafter setup (Mode C only — triggers runtime restart)
  IF PIPELINE_MODE == "viewcrafter" THEN
    CLONE ViewCrafter; INSTALL condacolab; condacolab.install()
    // → Runtime restarts here; re-run Cells 1-3 then continue
    CREATE conda env "viewcrafter_env" (Python 3.10)
    INSTALL pytorch, pytorch3d, einops, kornia, open-clip, transformers<4.40, etc.
    DOWNLOAD DUSt3R_ViTLarge + model_sparse.ckpt (~23 GB) → checkpoints/
  END IF

  // Cell 6: 3DGS CUDA submodules (always)
  INSTALL diff-gaussian-rasterization, simple-knn, COLMAP, ffmpeg, pycolmap

  // ================================================================
  // Cell 7: Gradio UI — runs augmentation when user clicks "Run"
  // ================================================================

  // --- MODE A: Synthesis (Zero123++) ---
  BEGIN run_synthesis(SCENE_NAME):

    RUN process_file.py(mode="synthetic", input_dir=rawdata, out_dir=processed)
    // MUSIQ → NOVEL_VIEW | REPAIR routing

    RUN diffusion_script_v0.py(input_dir=processed, out_dir=processed, mode="synthesis")
    // anchor_*.png + synth_*_v[0-5].png + restored_*.png → 1024×1024

    COPY processed/final_{SCENE_NAME}_run/*.png
         → GS_BASE/input_dataset/{SCENE_NAME}/input/

  END run_synthesis

  // --- MODE B: Restoration (ControlNet Tile) ---
  BEGIN run_restoration(SCENE_NAME, prompt):

    RUN process_file.py(mode="natural", ...)
    RUN diffusion_script_v0.py(mode="restoration", prompt=prompt)
    // ControlNet: strength=0.35, guidance=7.0, steps=30 → 1024×1024

    COPY to input_dataset/{SCENE_NAME}/input/

  END run_restoration

  // --- MODE C: ViewCrafter (natural scenes) ---
  BEGIN run_viewcrafter(SCENE_NAME, video_length, ddim_steps):

    RUN process_file.py(mode="natural", ...)  // quality screen
    CLEAR processed_train/ from previous runs
    RENAME images to numeric stems (0001.jpg, 0002.jpg, ...)

    INSTALL av (PyAV) in viewcrafter_env
    RUN inference.py via viewcrafter_env:
      --image_dir screened_dir  --mode sparse_view_interp
      --ckpt_path model_sparse.ckpt
      --model_path DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth
      --video_length video_length  --ddim_steps ddim_steps
      --height 576  --width 1024
    // DUSt3R builds point cloud; video diffusion generates interpolated frames
    // Output: render.mp4 → diffusion.mp4

    RUN ffmpeg: diffusion.mp4 → extracted_frames/frame_%04d.png

    MERGE original photos + extracted frames
         → GS_BASE/input_dataset/{SCENE_NAME}/input/

  END run_viewcrafter

  // ============================================================
  // Cell 8: Pose estimation (manual — all modes)
  // ============================================================
  COPY Drive input_dataset/{SCENE_NAME} → /content/local_workspace/{SCENE_NAME}

  RUN convert_ai.py(source_path=local_path)
  // SuperPoint keypoints → LightGlue exhaustive matching → PyCOLMAP SfM
  // → sparse/0/ {cameras, images, points3D}.bin + undistorted images/

  COPY local_path/sparse/ → Drive input_dataset/{SCENE_NAME}/

  // ============================================================
  // Cell 9: 3DGS training (manual — all modes)
  // ============================================================
  RUN train.py(
    source = local_path,
    model  = local_path + "_final_run",
    eval   = True,
    opacity_reset_interval = 9000   // 3 ADC resets over 30k iters
  )
  COPY trained model → Drive/gaussian-splatting/output/{SCENE_NAME}_final_run/

  // ============================================================
  // Cell 10: Evaluation (manual — all modes)
  // ============================================================
  RUN render.py → renders held-out test cameras
  RUN metrics.py → results.json: PSNR, SSIM, LPIPS

END UNIFIED_3DGS_PIPELINE
```

---

**Mode Routing Summary:**

| Mode | Augmentation                          | Pose Estimation | Training |
|------|---------------------------------------|-----------------|----------|
| A    | Zero123++ (6 views/anchor) + ESRGAN   | convert_ai.py   | train.py |
| B    | ControlNet Tile (strength 0.35) + ESRGAN | convert_ai.py | train.py |
| C    | ViewCrafter → ffmpeg frame extraction | convert_ai.py   | train.py |

---

**Critical Parameters:**

| Parameter                 | Value | Reason                                        |
|---------------------------|-------|-----------------------------------------------|
| opacity_reset_interval    | 9000  | Prevents Pruning Massacre in sparse-camera runs|
| eval                      | True  | Required for valid train/test split metrics   |
| ViewCrafter video_length  | ≤20   | 25+ causes OOM on 80 GB GPU at ddim_steps=50  |
| Numeric filename rename   | Yes   | ViewCrafter sorts by int(stem); IMG_XXXX fails|

---
