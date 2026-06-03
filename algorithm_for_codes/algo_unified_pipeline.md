# Algorithm: Unified 3DGS Augmentation Pipeline
# File: Pipeline_files/unified_pipeline.ipynb
# Status: IN USE — single notebook for all three pipeline modes

---

**Algorithm 12: UNIFIED_3DGS_PIPELINE**

**Input:**
- SCENE_NAME     : folder name inside `output_train/` on Drive (e.g. "hotdog", "lego")
- PIPELINE_MODE  : one of `"synthesis"` | `"restoration"` | `"viewcrafter"`
- RESTORE_PROMPT : ControlNet conditioning prompt (Mode B only)
- video_length   : number of frames ViewCrafter generates (Mode C only, default 25)
- ddim_steps     : diffusion sampling steps (Mode C only, default 50)

**Output:**
- Trained 3DGS model at `gaussian-splatting/output/{SCENE_NAME}_final_run/`
- Evaluation metrics: PSNR, SSIM, LPIPS (results.json)

---

```
BEGIN UNIFIED_3DGS_PIPELINE(SCENE_NAME, PIPELINE_MODE, ...)

  // ============================================================
  // CELL 1: Environment Setup
  // ============================================================
  MOUNT Google Drive at /content/drive/

  // ============================================================
  // CELL 2: Path Configuration
  // ============================================================
  DRIVE_BASE   ← /content/drive/MyDrive/.../3D_project
  GS_BASE      ← /content/drive/MyDrive/.../gaussian-splatting
  rawdata_path ← DRIVE_BASE / output_train / SCENE_NAME
  processed    ← DRIVE_BASE / output_processed / SCENE_NAME

  // ============================================================
  // CELL 3: Dependency Management
  // ============================================================
  MANAGED ← {numpy, huggingface_hub, diffusers, transformers, peft, accelerate}

  READ requirements.txt line by line
  FOR EACH line IN requirements.txt:
    pkg ← extract package name (strip version specifiers)
    IF pkg NOT IN MANAGED:
      ADD line to filtered_requirements
    ELSE:
      SKIP (will be installed at safe version below)
  END FOR

  INSTALL filtered_requirements  // pyiqa, basicsr, rembg, etc.

  INSTALL:
    numpy >= 2.0                 // prevents ABI crash from requirements.txt downgrade
    huggingface_hub >= 0.33.5    // required by diffusers 0.27+ and gradio 5.x
    diffusers >= 0.27.2          // drops cached_download, adds trust_remote_code
    transformers >= 4.41.0       // required by peft 0.19+
    peft >= 0.17.0               // required by diffusers 0.27+
    accelerate >= 0.31, < 1.0    // avoids circular import in accelerate 1.x
    gradio

  // Patch accelerate if clear_device_cache missing (peft 0.19+ requires it)
  FLUSH accelerate from sys.modules  // avoid stale module cache
  IMPORT accelerate.utils.memory
  IF 'clear_device_cache' NOT IN file contents:
    APPEND clear_device_cache() shim to file on disk
    RELOAD module
  END IF

  // ============================================================
  // CELL 4: Mode A/B Authentication (skipped for Mode C)
  // ============================================================
  IF PIPELINE_MODE IN ["synthesis", "restoration"]:
    LOGIN to HuggingFace using HF_TOKEN secret
    APPLY basicsr functional_tensor compatibility patch
      // sed: replace functional_tensor import with functional import
  END IF

  // ============================================================
  // CELLS 5a / 5a-post / 5c: ViewCrafter Setup (Mode C only)
  // ============================================================
  IF PIPELINE_MODE == "viewcrafter":

    // Cell 5a (triggers runtime restart)
    CLONE ViewCrafter → /content/ViewCrafter
    INSTALL condacolab
    condacolab.install()  // ← RUNTIME RESTARTS HERE
    // After restart: re-run Cells 1, 2, 3, then continue

    // Cell 5a-post (run after restart)
    IF /content/ViewCrafter NOT EXISTS:
      RE-CLONE ViewCrafter
    END IF
    CREATE conda env "viewcrafter_env" (python 3.10)
    INSTALL in viewcrafter_env:
      pytorch=2.1.0  torchvision  pytorch-cuda=12.1  pytorch3d
      einops  imageio  kornia  moviepy  open-clip-torch
      opencv-python  pytorch-lightning  roma  timm
      transformers<4.40.0  trimesh  omegaconf

    // Cell 5c (model downloads)
    DOWNLOAD DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth → /content/ViewCrafter/checkpoints/
    DOWNLOAD model_sparse.ckpt (~23 GB)              → /content/ViewCrafter/checkpoints/
  END IF

  // ============================================================
  // CELL 6: 3DGS Infrastructure (always run)
  // ============================================================
  REMOVE tensorflow  // free GPU memory
  COPY GS_BASE/submodules → /content/submodules_local  // compile on SSD
  INSTALL diff-gaussian-rasterization (CUDA submodule)
  INSTALL simple-knn (CUDA submodule)
  INSTALL COLMAP, ffmpeg, xvfb, pycolmap

  // ============================================================
  // CELL 7: Gradio UI Launch
  // ============================================================
  LAUNCH Gradio interface with 3 tabs (Mode A / B / C)
  // User fills in scene name and clicks Run button
  // Gradio generator functions yield log updates + gallery previews
  // When Gradio reports Done → proceed to Cells 8–10

  // ================================================================
  // GRADIO GENERATOR — MODE A: Synthesis (Zero123++)
  // ================================================================
  BEGIN run_synthesis(scene_name):

    rawdata   ← DRIVE_BASE / output_train / scene_name
    processed ← DRIVE_BASE / output_processed / scene_name

    [1/3] RUN process_file.py(
            mode      = "synthetic",
            input_dir = rawdata,
            out_dir   = processed
          )
          // MUSIQ scores each image
          // NOVEL_VIEW → route to Zero123++
          // REPAIR     → route to ControlNet

    [2/3] RUN diffusion_script_v0.py(
            input_dir = processed,
            out_dir   = processed,
            mode      = "synthesis"
          )
          // anchor_*.png  : real images composited on black background
          // synth_*_v[0-5].png : 6 Zero123++ views per anchor (rembg applied)
          // restored_*.png : ControlNet-repaired degraded images
          // All outputs upscaled to 1024×1024 via RealESRGAN

    [3/3] COPY processed / final_{scene_name}_run / *.png
               → GS_BASE / input_dataset / scene_name / input /
          SET builtins._gs_filename ← scene_name
          YIELD gallery preview (up to 16 images)

  END run_synthesis

  // ================================================================
  // GRADIO GENERATOR — MODE B: Restoration (ControlNet Tile)
  // ================================================================
  BEGIN run_restoration(scene_name, prompt):

    [1/3] RUN process_file.py(mode="synthetic", ...)

    [2/3] RUN diffusion_script_v0.py(
            mode    = "restoration",
            prompt  = prompt
          )
          // ControlNet Tile: strength=0.35, guidance=7.0, steps=30
          // Corrects blur/noise/exposure failure while preserving structure
          // RealESRGAN upscale → 1024×1024

    [3/3] COPY to input_dataset/
          SET builtins._gs_filename
          YIELD gallery preview

  END run_restoration

  // ================================================================
  // GRADIO GENERATOR — MODE C: ViewCrafter (Natural Scenes)
  // ================================================================
  BEGIN run_viewcrafter(scene_name, video_length, ddim_steps):

    img_dir    ← output_processed / scene_name / processed_{scene_name}
                 (fallback: output_train / scene_name / train)
    output_dir ← output_processed / scene_name / final_{scene_name}_run

    [1/3] RUN via viewcrafter_env python:
            inference.py
              --image_dir    img_dir
              --out_dir      output_dir
              --mode         sparse_view_interp
              --ckpt_path    checkpoints/model_sparse.ckpt
              --model_path   checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth
              --ddim_steps   ddim_steps
              --video_length video_length
              --device       cuda:0
              --height 576  --width 1024

          // DUSt3R builds coarse point cloud from sparse input images
          // Video diffusion generates interpolated frames → render.mp4
          // render.mp4 shown in Gradio video widget

    [2/3] RUN ffmpeg:
            -i render.mp4
            → extracted_frames/frame_%04d.png
            // All frames extracted — sparse input (12 images) means
            // maximum ViewCrafter coverage is needed for 3DGS quality

    [3/3] MERGE and STAGE:
          COPY original photos (output_train/{scene}/train/)
               + extracted frames (extracted_frames/)
               → GS_BASE / input_dataset / scene_name / input /

          SET builtins._gs_filename
          YIELD gallery preview + render.mp4 in video widget

  END run_viewcrafter

  // ============================================================
  // CELL 8: Pose Estimation — convert_ai.py (manual, all modes)
  // ============================================================
  filename  ← builtins._gs_filename  (set by Gradio; fallback: SCENE_NAME)
  drive_path← GS_BASE / input_dataset / filename
  local_path← /content/local_workspace / filename

  COPY drive_path/input/ → local_path/input/   // Drive → Colab SSD

  RUN convert_ai.py --source_path local_path
  // Stage 1: NetVLAD global descriptors → top-50 neighbour pairs per image
  // Stage 2: SuperPoint keypoint extraction (max 4096 per image)
  // Stage 3: LightGlue GNN matching on NetVLAD pairs
  // Stage 4: PyCOLMAP bundle adjustment → cameras.bin, images.bin, points3D.bin
  // Stage 5: COLMAP image_undistorter (headless via xvfb-run)
  //          → local_path/sparse/0/ + local_path/images/

  COPY local_path/* → drive_path/   // sync sparse/0/ back to Drive

  // ============================================================
  // CELL 9: 3DGS Training — train.py (manual, all modes)
  // ============================================================
  LOCAL_INPUT  ← /content/local_workspace / filename
  LOCAL_OUTPUT ← /content/local_workspace / filename_final_run
  DRIVE_OUT    ← GS_BASE / output / filename_final_run

  COPY drive input_dataset → LOCAL_INPUT/images/
  COPY drive sparse/0/     → LOCAL_INPUT/sparse/

  RUN train.py
    -s LOCAL_INPUT
    -m LOCAL_OUTPUT
    --eval                         // activates train/test split for valid metrics
    --opacity_reset_interval 9000  // 3 ADC resets over 30k iters (not 10)

  // 30,000 iteration photometric optimisation
  // Adaptive Density Control: clone dense regions, prune transparent Gaussians
  // opacity_reset_interval=9000 prevents "Pruning Massacre" in sparse-camera runs

  COPY LOCAL_OUTPUT → DRIVE_OUT

  // ============================================================
  // CELL 10: Evaluation (manual, all modes)
  // ============================================================
  RUN render.py -m DRIVE_OUT -s LOCAL_INPUT --skip_train
  // Renders held-out test cameras

  RUN metrics.py -m DRIVE_OUT
  // Computes PSNR, SSIM, LPIPS vs ground truth → results.json

  PRINT PSNR, SSIM, LPIPS from results.json

END UNIFIED_3DGS_PIPELINE
```

---

**Full Data Flow:**

```
Google Drive: output_train/{scene}/          ← raw input images
        ↓  Cell 7 (Gradio)
        │
        ├─ MODE A/B: diffusion_script_v0.py
        │       ↓
        │   output_processed/{scene}/final_{scene}_run/
        │       anchor_*.png  +  synth_*_v*.png  +  restored_*.png
        │
        └─ MODE C: ViewCrafter → render.mp4 → ffmpeg → frame_*.png
                + original photos from output_train/{scene}/train/
        │
        ↓  (all modes merge here)
GS_BASE/input_dataset/{scene}/input/         ← staged augmented images
        ↓  Cell 8
convert_ai.py → sparse/0/ (cameras.bin, images.bin, points3D.bin)
        ↓  Cell 9
train.py → {scene}_final_run/ (trained .ply + checkpoints)
        ↓  Cell 10
render.py + metrics.py → results.json (PSNR, SSIM, LPIPS)
        ↓
Google Drive: gaussian-splatting/output/{scene}_final_run/
```

---

**Mode Routing Summary:**

| Mode | Augmentation Tool | Pose Estimation | Training |
|---|---|---|---|
| A — Synthesis | Zero123++ (6 views/anchor) + RealESRGAN | convert_ai.py | train.py |
| B — Restoration | ControlNet Tile (strength 0.35) + RealESRGAN | convert_ai.py | train.py |
| C — ViewCrafter | Video diffusion → ffmpeg frame extraction | convert_ai.py | train.py |

All modes converge at the same SfM → 3DGS → metrics pipeline.

---

**Critical Parameters:**

| Parameter | Value | Reason |
|---|---|---|
| `--opacity_reset_interval` | 9000 | Prevents "Pruning Massacre" in sparse-camera conditions |
| `--eval` | True | Required for valid train/test split metrics |
| ffmpeg frame extraction | All frames | Sparse input (12 images) requires maximum coverage; COLMAP overhead is acceptable |
| SuperPoint max keypoints | 4096 | Suppresses low-confidence peripheral detections |
| NetVLAD top-K | 50 | O(N·K) matching instead of O(N²) exhaustive |
| ViewCrafter video_length | 25 (default) | 25 interpolated frames between sparse viewpoints |
| ViewCrafter ddim_steps | 50 (default) | Diffusion sampling quality/speed trade-off |
