# CoDiffusion Engine — `diffusion_script_v0.py`

> **STATUS: ✅ IN USE** — Core augmentation engine. Called in Full_pipeline_v1.ipynb Cell 5.

---

## Overview

Dual-mode generative pipeline that routes images based on MUSIQ quality scores from `manifest.json`:
- **Synthesis mode** — takes good images, generates 6 novel views via Zero123++
- **Restoration mode** — takes degraded images, repairs them via ControlNet Tile (SD 1.5)

Both modes feed into a shared RealESRGAN upscaling phase that standardises all output to 1024×1024.

---

## Architecture

```
manifest.json
     │
     ├─ NOVEL_VIEW / NONE  →  Synthesis Engine (Zero123++)
     │                              ↓
     │                        6 novel views per anchor
     │                              ↓
     │                        rembg (synth views only)
     │                              ↓
     │                        temp_dir  synth_*.png
     │
     ├─ REPAIR / BAD / DISCARD / blur  →  Restoration Engine (ControlNet)
     │                                          ↓
     │                                    restored_*.png → temp_dir
     │
     └─ Everything else (good images)  →  Pass-through
                                               ↓
                                         anchor_*.png → temp_dir (no modification)

temp_dir  →  RealESRGAN ×4  →  resize 1024×1024  →  final_{name}_run/
```

---

## Modes

### Synthesis Mode (`--mode synthesis`)

1. Reads `NOVEL_VIEW` / `NONE` candidates from `manifest.json` (top 20 by score)
2. For each anchor image:
   - Composites onto 512×512 black canvas (object scaled to 85%)
   - Saves as `anchor_{filename}` in temp_dir — **NOT processed by rembg**
   - Passes to Zero123++ v1.2 (75 inference steps)
   - Saves full 2×3 grid as `FULL_GRID_{filename}` (debug only)
   - Crops 6 tiles → saves as `synth_{stem}_v0.png` … `synth_{stem}_v5.png`
   - **Applies rembg to every synth tile** — strips grey Zero123++ background, pastes onto pure black canvas
3. All files in temp_dir → RealESRGAN upscale → final_dir

**Why rembg on synth only**: Zero123++ outputs a characteristic grey/off-white background. If passed to 3DGS, the silhouette mask registers the background as foreground, destroying depth supervision. Anchor images already have correct black backgrounds from step 2 — rembg on them would be redundant.

### Restoration Mode (`--mode restoration`)

1. Reads `REPAIR` / `BAD` / `DISCARD` / `blur` candidates from manifest → ControlNet pipeline
2. Reads all other manifest entries → copied directly to temp_dir as `anchor_{stem}.png` (pass-through)
3. ControlNet settings: strength=0.35, guidance=7.0, steps=30
4. Restored images saved as `restored_{stem}.png` → temp_dir
5. All files → RealESRGAN → final_dir

**Known limitation**: ControlNet cannot recover information destroyed by severe blur/noise. It hallucinates plausible-looking but geometrically incorrect texture on heavily degraded images. This is an inherent constraint, documented in thesis Chapter 3.

---

## Output File Naming

| Prefix | Source | Downstream |
|---|---|---|
| `anchor_` | Real photo (black BG, centred) | inject_poses.py recognises as real camera |
| `synth_*_v[0-5].png` | Zero123++ tiles, rembg applied | inject_poses.py recognises as synthetic view |
| `restored_` | ControlNet output | Treated as anchor by inject_poses.py |
| `FULL_GRID_` | Raw Zero123++ grid | Skipped by upscaler, debug only |

---

## Key Bug Fixes (vs original version)

| Bug | Fix |
|---|---|
| Good images saved as `good_{filename}` — invisible to pipeline | Now saved as `anchor_{stem}.png` — picked up by upscaler and inject_poses |
| Zero123++ grey backgrounds caused silhouette mask to be all-1s | rembg integrated directly into synthesis loop, runs on synth tiles only |
| `restored_` prefix not recognised by inject_poses.py | Documented limitation — inject_poses handles by treating unrecognised prefix as real image |

---

## Usage

```bash
# Synthesis (NeRF / object datasets)
python diffusion_script_v0.py \
  --input_dir "/path/to/output_processed/hotdog" \
  --out_dir   "/path/to/output_processed/hotdog" \
  --mode synthesis

# Restoration (natural scene images)
python diffusion_script_v0.py \
  --input_dir "/path/to/output_processed/scene" \
  --out_dir   "/path/to/output_processed/scene" \
  --mode restoration \
  --prompt "high quality photo, detailed, sharp focus, 8k"
```

---

## Dependencies

```
diffusers, transformers, accelerate
basicsr, realesrgan
rembg, onnxruntime-gpu
opencv-python, pillow, tqdm, torch
```

Note: `basicsr` requires the `functional_tensor` compatibility patch applied in Cell 4 of the notebook before this script runs.
