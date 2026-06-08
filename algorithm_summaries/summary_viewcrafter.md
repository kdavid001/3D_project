# Algorithm Summary: ViewCrafter Video Diffusion Augmentation
# Source: unified_pipeline.ipynb (Cell 7 — run_viewcrafter) | Full algorithm: algorithm_for_codes/algo_viewcrafter.md

---

**Algorithm 13 (Summary): VIEWCRAFTER_AUGMENTATION**

**Input:**
- image_dir    : quality-screened source images (real-world photos)
- output_dir   : destination for video output and extracted frames
- video_length : interpolated frames to generate (use ≤ 20 to avoid OOM)
- ddim_steps   : DDIM sampling steps (default 50)

**Output:**
- diffusion.mp4               : video-diffusion-interpolated novel views
- extracted_frames/*.png      : individual frames from diffusion.mp4
- input_dataset/{scene}/input/: merged dataset (originals + frames) for 3DGS

---

```
BEGIN VIEWCRAFTER_AUGMENTATION(image_dir, output_dir, video_length, ddim_steps)

  // Pre-processing 1: Quality screening (lenient natural thresholds)
  IF screened_dir EXISTS → DELETE  // prevent stale file accumulation
  RUN process_file.py(mode="natural", input_dir=raw_dir, out_dir=processed_dir)
  screened_dir ← processed_dir / "processed_train"

  // Pre-processing 2: Rename images to numeric stems
  // ViewCrafter sorts by int(stem) → IMG_XXXX.jpg raises ValueError
  images ← LIST + SORT all images in screened_dir
  FOR each image_i DO
    RENAME to ZERO_PAD(i+1, 4) + ext   // 0001.jpg, 0002.jpg, ...
  END FOR

  // Pre-processing 3: Install PyAV in viewcrafter_env
  // torchvision.io.write_video requires av — not in conda spec
  RUN pip install av inside viewcrafter_env

  // Phase 1: ViewCrafter inference (sparse_view_interp mode)
  //   Stage A — DUSt3R: processes all C(N,2) pairs → coarse point cloud → render.mp4
  //   Stage B — Video diffusion: conditioned on point cloud → video_length frames → diffusion.mp4
  OPEN Drive log at output_dir/viewcrafter_run.log  // survives browser disconnect
  RUN inference.py via viewcrafter_env {
    --image_dir    screened_dir
    --out_dir      output_dir
    --mode         sparse_view_interp
    --ckpt_path    model_sparse.ckpt
    --model_path   DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth
    --ddim_steps   ddim_steps
    --video_length video_length
    --height 576   --width 1024
    --bg_trd 0.2   --seed 123
  } REDIRECT stdout+stderr → viewcrafter_run.log

  // Phase 2: Frame extraction (ffmpeg)
  // All frames extracted — sparse input needs maximum angular coverage
  RUN ffmpeg -i diffusion.mp4 → extracted_frames/frame_%04d.png

  // Phase 3: Dataset merge and staging
  COPY original photos (screened_dir)  → input_dataset/{scene}/input/
  COPY extracted frames (frames_dir)   → input_dataset/{scene}/input/
  // Combined: N real photos + M generated frames → convert_ai.py input

END VIEWCRAFTER_AUGMENTATION
```

---

**Pipeline Position:**

```
Real photos (output_train/{scene}/train/)
        ↓  [this algorithm]
input_dataset/{scene}/input/   ← N originals + M frames
        ↓  convert_ai.py (Algorithm 6)
sparse/0/ + images/
        ↓  train.py → 3DGS model
```

---

**Why Separate from diffusion_script_v0.py:**

| Aspect           | diffusion_script_v0.py     | ViewCrafter                         |
|------------------|----------------------------|-------------------------------------|
| Dependencies     | Main Colab env             | Separate conda env (Python 3.10)    |
| transformers ver | ≥ 4.41                     | < 4.40 (hard conflict)              |
| Runtime restart  | Not required               | Required (condacolab installs conda)|
| Output type      | Individual images (PNG)    | Video → extracted frames            |

---
