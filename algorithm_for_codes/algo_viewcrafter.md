# Algorithm: ViewCrafter Video Diffusion Augmentation
# File: Pipeline_files/unified_pipeline.ipynb (Cell 7 — run_viewcrafter)
# Status: IN USE — Mode C of the Unified Pipeline

---

> **Note on Architecture:** ViewCrafter cannot be packaged as a standalone script alongside
> diffusion_script_v0.py because its dependencies conflict with Zero123++ and ControlNet.
> Specifically, it requires transformers < 4.40.0, PyTorch3D, and a separate conda environment
> (viewcrafter_env, Python 3.10). It is therefore executed as a subprocess from within the
> unified pipeline notebook, with its own isolated conda environment.

---

**Algorithm 13: VIEWCRAFTER_AUGMENTATION**

**Input:**
- image_dir    : folder containing quality-screened source images (≥ 2, portrait OK)
- output_dir   : destination for generated video and extracted frames
- video_length : number of interpolated frames to generate (default 25; use ≤ 20 to avoid OOM)
- ddim_steps   : DDIM diffusion sampling steps (default 50)

**Output:**
- output_dir/render.mp4      : coarse 3D point cloud render (DUSt3R pass)
- output_dir/diffusion.mp4   : final video-diffusion-interpolated output
- output_dir/extracted_frames/frame_%04d.png : individual frames extracted from diffusion.mp4
- GS_BASE/input_dataset/{scene}/input/ : merged dataset (original photos + extracted frames)

---

```
BEGIN VIEWCRAFTER_AUGMENTATION(image_dir, output_dir, video_length, ddim_steps)

  // ============================================================
  // PRE-PROCESSING — Quality Screening (process_file.py, mode="natural")
  // ============================================================
  // Five-Pillar screener (Algorithm 4) with LENIENT natural thresholds:
  //   MUSIQ ≥ 40.0, saturation clipped < 5%, brightness [45,250],
  //   contrast std ≥ 10.0, colour cast distance < 60.0
  //
  // ViewCrafter operates on real-world photos — use "natural" mode, NOT "synthetic".
  // "synthetic" mode applies stricter thresholds calibrated for blender-rendered datasets.

  IF screened_dir EXISTS THEN
    DELETE screened_dir   // remove stale files from previous failed runs
  END IF

  RUN process_file.py(
    mode      = "natural",
    input_dir = raw_image_dir,
    out_dir   = processed_dir,
    debug     = True
  )
  // Passing images written to processed_dir/processed_train/
  // Failing images labelled REPAIR — not used by ViewCrafter
  // debug_visuals/ written if debug=True

  screened_dir ← processed_dir / "processed_train"

  // ============================================================
  // PRE-PROCESSING — Numeric Filename Renaming
  // ============================================================
  // ViewCrafter's load_initial_dir() sorts images by int(stem).
  // Files named IMG_XXXX.jpg or similar will raise:
  //   ValueError: invalid literal for int() with base 10: 'IMG_5594'
  // Solution: rename all files to zero-padded numeric stems before passing to ViewCrafter.

  images ← LIST all .jpg/.jpeg/.png in screened_dir, SORT alphanumerically
  counter ← 1
  FOR each file F in images DO
    new_name ← ZERO_PAD(counter, 4) + FILE_EXT(F)   // e.g., "0001.jpg"
    RENAME F to screened_dir / new_name
    counter ← counter + 1
  END FOR
  // Result: 0001.jpg, 0002.jpg, ... (sequential, parseable as int)

  // ============================================================
  // PRE-PROCESSING — Install PyAV in viewcrafter_env
  // ============================================================
  // torchvision.io.write_video (called by ViewCrafter's save_video()) requires av (PyAV).
  // PyAV is not included in the conda environment spec and must be installed at runtime.

  RUN pip install av inside viewcrafter_env

  // ============================================================
  // PHASE 1 — ViewCrafter Inference (sparse_view_interp mode)
  // ============================================================
  // ViewCrafter operates in two stages internally:
  //
  //   Stage 1 — DUSt3R Reconstruction:
  //     Processes all C(N,2) image pairs simultaneously.
  //     Outputs a coarse dense 3D point cloud in a canonical frame.
  //     Renders a preliminary point cloud video → render.mp4
  //
  //   Stage 2 — Video Diffusion Refinement:
  //     Conditioned on the DUSt3R point cloud geometry.
  //     Generates video_length interpolated frames between sparse viewpoints.
  //     Uses DDIM sampling with ddim_steps steps.
  //     Output → diffusion.mp4
  //
  // Memory note: video_length=25 at ddim_steps=50 requires ~40 GB VRAM.
  // On an 80 GB GPU with model already loaded (~43 GB), this causes OOM.
  // Use video_length ≤ 20 for safe operation on an 80 GB GPU.

  CREATE output_dir

  OPEN Drive log file at output_dir/viewcrafter_run.log  // persist output across disconnections

  RUN inference.py via viewcrafter_env with:
    --image_dir    screened_dir
    --out_dir      output_dir
    --mode         sparse_view_interp
    --bg_trd       0.2               // background confidence threshold for DUSt3R masking
    --seed         123
    --ckpt_path    ./checkpoints/model_sparse.ckpt
    --config       ./configs/inference_pvd_1024.yaml
    --model_path   ./checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth
    --ddim_steps   ddim_steps
    --video_length video_length
    --device       cuda:0
    --height       576
    --width        1024
  REDIRECT stdout + stderr → viewcrafter_run.log (line-buffered)

  IF inference.py returns non-zero exit code THEN
    PRINT last 5000 chars of viewcrafter_run.log
    RAISE "ViewCrafter inference failed — see log"
  END IF

  // ============================================================
  // PHASE 2 — Frame Extraction (ffmpeg)
  // ============================================================
  // Extract every frame from diffusion.mp4 as individual PNG images.
  // All frames are extracted (not subsampled) because:
  //   - Input is sparse (12 images typical); COLMAP needs maximum coverage.
  //   - ViewCrafter interpolates geometry between viewpoints — every frame adds a new camera.

  frames_dir ← output_dir / "extracted_frames"
  CREATE frames_dir

  RUN ffmpeg:
    -i  output_dir/diffusion.mp4
    -q:v 2                          // near-lossless JPEG quality factor
    frames_dir/frame_%04d.png

  extracted_frames ← LIST all .png in frames_dir

  // ============================================================
  // PHASE 3 — Dataset Merge and Staging
  // ============================================================
  // Combine the original real photos with the ViewCrafter-generated frames.
  // The originals anchor the COLMAP reconstruction; the generated frames
  // fill the angular gaps between sparse capture positions.

  staging_dir ← GS_BASE / input_dataset / SCENE_NAME / input
  CREATE staging_dir

  // Copy original screened photos
  FOR each photo P in screened_dir DO
    COPY P to staging_dir
  END FOR

  // Copy generated frames
  FOR each frame F in extracted_frames DO
    COPY F to staging_dir
  END FOR

  // staging_dir now contains:
  //   - N real photos (renamed 0001.jpg … 000N.jpg)
  //   - M extracted frames (frame_0001.png … frame_000M.png)
  // Total image count ≈ N + video_length fed into convert_ai.py

END VIEWCRAFTER_AUGMENTATION
```

---

**Data Flow:**

```
Raw photos (output_train/{scene}/train/)
        ↓  process_file.py (mode="natural")
processed_train/  (quality-screened, numeric renamed)
        ↓  viewcrafter_env / inference.py
render.mp4  (DUSt3R point cloud render)
diffusion.mp4  (video diffusion output)
        ↓  ffmpeg
extracted_frames/frame_%04d.png
        ↓  merge with original photos
GS_BASE/input_dataset/{scene}/input/
        ↓  convert_ai.py  (Algorithm 6)
sparse/0/ + images/
        ↓  train.py → 3DGS model
```

---

**Dependency Isolation:**

| Component         | Environment      | Key Constraint                          |
|-------------------|------------------|-----------------------------------------|
| Zero123++ (Mode A)| Main Colab env   | transformers ≥ 4.41, diffusers ≥ 0.27  |
| ControlNet (Mode B)| Main Colab env  | Same as above                           |
| ViewCrafter (Mode C)| viewcrafter_env (conda, Python 3.10) | transformers < 4.40, pytorch3d |
| RealESRGAN        | Main Colab env   | basicsr functional_tensor patch needed  |

---

**Key Parameters:**

| Parameter      | Value            | Reason                                                         |
|----------------|------------------|----------------------------------------------------------------|
| mode           | sparse_view_interp | Uses DUSt3R point cloud as geometry prior for interpolation  |
| video_length   | ≤ 20             | 25 at ddim_steps=50 exceeds free VRAM on 80 GB GPU            |
| ddim_steps     | 50               | Quality/speed trade-off (reduce to 30 if still OOM)           |
| height × width | 576 × 1024       | Native ViewCrafter inference resolution                        |
| bg_trd         | 0.2              | DUSt3R background confidence threshold for masking             |
| seed           | 123              | Reproducible diffusion output                                  |
| Frame extract  | All frames       | Sparse input (12 imgs) requires maximum ViewCrafter coverage   |

---

**Known Limitations:**
- Portrait-mode photos and telephoto shots reduce 3D geometry quality in the DUSt3R stage
- Landscape orientation at wide angle provides better angular parallax for point cloud building
- CUDA state persists in Colab — restart runtime between failed ViewCrafter attempts to recover VRAM
- render.mp4 appearing without diffusion.mp4 indicates diffusion stage is still running (not a failure)
