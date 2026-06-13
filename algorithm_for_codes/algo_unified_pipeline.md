# Algorithm: Unified Augmented 3DGS Pipeline
# File: Pipeline_files/unified_pipeline.ipynb
# Status: IN USE — single notebook for all three pipeline modes

---

**Algorithm 6: UNIFIED_AUGMENTED_PIPELINE**

**Input:**
- S : sparse image set (JPEG / PNG / HEIC)
- m : screening mode ∈ {natural, indoor, synthetic}

**Output:**
- Trained 3DGS scene (.ply + rendered views)
- Evaluation metrics: PSNR, SSIM, LPIPS

---

```
Convert all HEIC images in S to JPEG
FOR each image i in S DO
    Apply Algorithm 2 (PROCESS_FILES):
        Compute 5-pillar quality score
        Tag i as NOVEL_VIEW or REPAIR
        Log result to manifest.json
END FOR
FOR each tagged image i DO
    IF tag(i) = REPAIR THEN
        Apply Algorithm 3 (GENERATIVE_AUGMENTATION, Mode B)
        Apply RealESRGAN x4 upscaling
    ELSE IF tag(i) = NOVEL_VIEW AND m = synthetic THEN
        Apply Algorithm 3 (GENERATIVE_AUGMENTATION, Mode A)
        Apply RealESRGAN x4 upscaling
    ELSE IF tag(i) = NOVEL_VIEW AND m ∈ {natural, indoor} THEN
        Apply Algorithm 4 (VIEWCRAFTER_AUGMENTATION, Mode C)
        Extract frames from diffusion.mp4 via FFmpeg
    END IF
END FOR
Merge augmented outputs with original anchor images → augmented set A
Apply Algorithm 5 (AI_SFM_POSE_ESTIMATION) on A
IF no cameras registered THEN
    EXIT — insufficient feature matches
END IF
Initialise 3DGS from sparse point cloud
Optimise for 30,000 iterations with Adaptive Density Control
Compute PSNR, SSIM, LPIPS on held-out test views
RETURN trained scene + metrics
```

---

**Mode Routing Summary:**

| Mode | Trigger | Augmentation Tool | Post-process |
|---|---|---|---|
| A — Novel View Synthesis | NOVEL_VIEW + synthetic | Zero123++ (6 views/anchor) | RealESRGAN x4 |
| B — Image Restoration | REPAIR (any mode) | ControlNet Tile (strength 0.35) | RealESRGAN x4 |
| C — Scene Interpolation | NOVEL_VIEW + natural/indoor | ViewCrafter (DUSt3R conditioning) | FFmpeg frame extraction |

All modes converge at Algorithm 5 (SfM) → 3DGS training → metrics evaluation.

---

**Critical Parameters:**

| Parameter | Value |
|---|---|
| 3DGS training iterations | 30,000 |
| ViewCrafter DDIM steps | 50 |
| ViewCrafter frames per clip | 25 |
| ControlNet strength | 0.35 |
| RealESRGAN scale | x4 (→ 1024×1024) |
| MUSIQ BAD_LIMIT (natural) | 40.0 |
| MUSIQ BAD_LIMIT (indoor) | 30.0 |
| MUSIQ BAD_LIMIT (synthetic) | 65.0 |
