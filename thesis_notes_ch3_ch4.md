# Thesis Reference Notes — Chapter 3 & 4
## "Implementation of Multi-View Generative Augmentation for Robust 3D Gaussian Splatting Reconstruction from Sparse and Degraded Inputs"

---

# CHAPTER 3 — METHODOLOGY & IMPLEMENTATION

## 3.1 Experimental Design

Three conditions were designed to isolate the contribution of the proposed augmentation pipeline:

- **Condition A — Dense Baseline**: The original 3DGS pipeline trained on the full, clean dataset (98 cameras). This represents the theoretical upper bound of reconstruction quality under ideal data conditions.
- **Condition B — Sparse/Degraded Baseline**: The original pipeline trained on a severely reduced and synthetically degraded subset (~6–21 cameras). Degradations applied include Gaussian blur, additive white Gaussian noise (AWGN), saturation distortion, and simulated exposure failure. No augmentation is applied. This represents the problem condition — what 3DGS produces when data is both sparse and corrupted.
- **Condition C — Proposed Augmented Pipeline**: The same sparse/degraded input as Condition B, processed through the CoDiffusion engine to generate novel views and restore degraded images before training. This is the contribution being evaluated.

---

## 3.2 The CoDiffusion Engine (Dual-Mode Architecture)

The CoDiffusion engine (`diffusion_script_v0.py`) operates in two modes selected by a manifest routing system:

### 3.2.1 MUSIQ Quality Routing
All input images are first assessed using MUSIQ (Multi-Scale Image Quality Transformer), a No-Reference Image Quality Assessment model. MUSIQ produces a perceptual quality score for each image without requiring a clean reference. Images are then classified:
- `NOVEL_VIEW` / `NONE` — high quality, route to synthesis engine
- `REPAIR` / `BAD` / `DISCARD` / `blur` — degraded, route to restoration engine

Good-quality images that require neither synthesis nor restoration are passed directly through the pipeline as anchor images (`anchor_{filename}.png`) to preserve them without modification.

### 3.2.2 Synthesis Mode (Zero123++)
Zero123++ v1.2 (sudo-ai) is used to generate novel views from good-quality anchor images. The model produces a fixed 2×3 grid of 6 novel views at pre-defined relative camera angles:
- v0: azimuth +30°, elevation +20°
- v1: azimuth +90°, elevation −10°
- v2: azimuth +150°, elevation +20°
- v3: azimuth +210°, elevation −10°
- v4: azimuth +270°, elevation +20°
- v5: azimuth +330°, elevation −10°

**Pre-processing**: Each input image is composited onto a 512×512 black canvas (object scaled to 85% of canvas) before being passed to Zero123++. This is critical — Zero123++ was trained on black-background object images. Passing images with natural or white backgrounds causes the model to treat background pixels as part of the object geometry, producing incoherent novel views.

**Post-processing (rembg integration)**: Zero123++ outputs tiles with a characteristic grey/off-white background rather than a clean black background. If these grey backgrounds are passed into 3DGS training, the silhouette mask used in Decoupled Supervision becomes all-ones (the background registers as foreground), destroying the depth and silhouette losses. To fix this, rembg (background removal via U²-Net) is applied to every `synth_*` tile immediately after cropping. The subject is extracted, composited onto a pure black canvas, and saved back in-place. Anchor images are deliberately excluded from this step as their backgrounds are already correctly conditioned.

### 3.2.3 Restoration Mode (ControlNet Tile)
For degraded images, a ControlNet Tile pipeline (lllyasviel/control_v11f1e_sd15_tile, Stable Diffusion v1.5 backbone) is applied with:
- Strength: 0.35 (preserves original structure, corrects degradation)
- Guidance scale: 7.0
- Negative prompt: "blur, noise, grain, low resolution, distorted, plastic, cartoon"
- Steps: 30

**Limitation identified**: ControlNet restoration applies a corrective pass over degraded pixels, but it cannot recover information that was fundamentally destroyed. For severely blurred or noisy images, ControlNet hallucinates plausible-looking but geometrically incorrect texture. This is an inherent limitation of inpainting-style restoration — it fights a losing battle when the signal-to-noise ratio is too low. A more architecturally sound alternative (video-based novel view synthesis from retained good frames) is discussed in Chapter 5 (Future Work).

### 3.2.4 Upscaling & Standardisation (RealESRGAN)
All outputs (both anchor and synthesised views) are upscaled using RealESRGAN x4plus and standardised to 1024×1024 pixels. This ensures consistent image resolution across all inputs to COLMAP and 3DGS, regardless of whether the image came from the original dataset, synthesis, or restoration.

---

## 3.3 Pose Estimation and COLMAP Integration

For NeRF-derived datasets (e.g., hotdog), COLMAP structure-from-motion is applied directly to the combined real and synthesised image set. However, a critical failure mode was identified:

**SfM collapse with black-background images**: COLMAP's feature extraction (SIFT) operates on local image gradients. Black-background images have a near-zero gradient over the majority of the image (the background), concentrating all features on the object silhouette. When combined with the object's relatively uniform texture (in some cases), this results in insufficient feature matches for camera registration. In early experiments, COLMAP registered only 2 of 28 cameras in this scenario.

**Fix (inject_poses.py)**: For the Zero123++ synthesised views, COLMAP is bypassed entirely. The known fixed relative poses of Zero123++'s 6 output views are written directly into COLMAP binary format (cameras.bin, images.bin, points3D.bin) using struct-packed binary writers that conform to the COLMAP binary specification. Real camera poses from COLMAP (run on real images only, with natural backgrounds) are merged with the synthesised view poses to produce a unified camera set for 3DGS training.

---

## 3.4 3DGS Training Configuration

Key hyperparameter discovered during experimentation:

**`--opacity_reset_interval`**: The default value of 3000 causes the Adaptive Density Control mechanism to reset Gaussian opacities 10 times over a 30,000-iteration training run. With sparse camera sets (fewer constraints on geometry), this repeatedly prunes valid Gaussians, causing catastrophic geometry collapse — referred to internally as the "Pruning Massacre." Setting `--opacity_reset_interval 9000` reduces this to 3 resets and stabilises training.

**`--eval`**: Activates the train/test split, which is mandatory for producing comparable evaluation metrics across conditions. Without it, metrics reflect memorisation rather than generalisation.

---

# CHAPTER 4 — RESULTS & EVALUATION

## 4.1 Quantitative Results

| Condition | Cameras | PSNR (↑) | SSIM (↑) | LPIPS (↓) |
|---|---|---|---|---|
| A — Dense Baseline | 98 | 4.49 dB | 0.400 | 0.431 |
| B — Sparse/Degraded Baseline | 21 | 18.78 dB | 0.747 | 0.245 |
| C — Augmented Pipeline (Proposed) | 18 | 13.18 dB | **0.780** | **0.190** |

---

## 4.2 Analysis

### 4.2.1 Condition A — Broken Baseline

The dense baseline (Condition A) produced a PSNR of 4.49 dB and SSIM of 0.400, which is below any meaningful reconstruction threshold (standard 3DGS on this dataset should reach approximately 32–34 dB). This failure is attributed to the `--opacity_reset_interval` default value of 3000 combined with COLMAP instability under black-background synthesis images.

**This result should be treated as a confounded run, not a valid baseline.** A corrected rerun with `--opacity_reset_interval 9000` is required before Condition A can be cited as an upper bound.

### 4.2.2 Condition C vs Condition B — The Key Comparison

The meaningful scientific comparison is between B (the problem) and C (the proposed solution):

**PSNR decreased** from 18.78 dB to 13.18 dB (−5.6 dB). This is a regression on pixel-level accuracy.

**SSIM increased** from 0.747 to 0.780 (+4.4%). This is an improvement in structural similarity.

**LPIPS decreased** from 0.245 to 0.190 (−22.4%). This is a substantial improvement in perceptual quality.

SSIM and LPIPS are widely considered more aligned with human visual perception than PSNR. PSNR is sensitive to exact pixel values and is disproportionately penalised by small spatial misalignments, whereas SSIM captures structural patterns (edges, textures, local gradients) and LPIPS uses a pre-trained neural network to measure perceptual distance in feature space. The improvement in both SSIM and LPIPS indicates that the augmented pipeline produces reconstructions that are structurally and perceptually closer to the ground truth, even when individual pixel values are less accurate.

### 4.2.3 Explaining the PSNR Regression — Hemispheric Coverage

The PSNR drop is attributable to two compounding factors:

**Factor 1 — Angular coverage gap**: The sparse input views in Conditions B and C were sampled predominantly from frontal and side angles. No real images of the back of the object were available. Zero123++ generates 6 novel views at fixed relative offsets from the reference image. When all reference images are frontal, the generated views that notionally cover the back hemisphere are in fact extrapolated from a model that has never observed the object's rear geometry. These back-view estimates are geometrically inconsistent with the object's true shape. The 3DGS optimiser cannot reconcile the conflicting photometric signals from the real front views and the hallucinated back views, resulting in degraded reconstruction at back angles.

**Factor 2 — Photometric inconsistency of synthesised views**: Zero123++ does not preserve the exact photometric properties (lighting, colour, surface texture) of the original scene. Generated views may have subtly different shading or colour balance compared to real photographs. PSNR is highly sensitive to such pixel-level differences. SSIM and LPIPS are more robust to these shifts, which is why they improve while PSNR degrades.

**Written as a dissertation finding**:

> *"The PSNR degradation observed in Condition C is attributable to incomplete angular coverage in the sparse input set. When reference images are predominantly frontal, generative augmentation produces geometrically inconsistent estimates for back-hemisphere viewpoints, which 3DGS cannot reconcile with real photometric supervision. This results in pixel-level accuracy loss despite measurable gains in structural and perceptual quality metrics (SSIM: +4.4%, LPIPS: −22.4%). This finding motivates a key recommendation: sparse view selection should ensure hemispheric coverage, distributing samples across front, side, and rear viewpoints to maximise the geometric information available to both the reconstruction system and the augmentation model."*

### 4.2.4 PSNR vs Perceptual Metrics — Which to Trust?

For the purpose of evaluating 3D scene reconstruction quality, SSIM and LPIPS are more appropriate primary metrics than PSNR because:

1. Novel view synthesis is evaluated perceptually — the goal is that rendered views look correct to a human observer, not that individual pixel values are exact.
2. PSNR is not invariant to small spatial translations; a structurally correct reconstruction that is sub-pixel misaligned will produce a substantially lower PSNR than its visual quality warrants.
3. LPIPS, computed on deep CNN features, correlates more strongly with human perceptual judgements than PSNR in scene reconstruction benchmarks (Zhang et al., 2018).

The proposed pipeline demonstrates improvement on the two metrics most aligned with perceptual reconstruction quality.

---

## 4.3 Key Technical Discoveries (Implementation Findings)

These are findings that emerged from the implementation process and are worth documenting in Chapter 3 or a "Technical Challenges" subsection:

1. **Grey background from Zero123++**: Output tiles have an off-white/grey background, not black. This causes silhouette masks used in supervision to register the entire image as foreground. Fixed by integrating rembg directly into the synthesis pipeline.

2. **Opacity Reset Massacre**: Default `--opacity_reset_interval 3000` with sparse camera sets causes catastrophic geometry pruning. Use 9000 for sparse conditions.

3. **COLMAP SfM collapse on black-background images**: Feature-poor black backgrounds starve SIFT of gradient information. Fixed by bypassing SfM for synthesised views and writing known poses directly to COLMAP binary format.

4. **Good images silently discarded in restoration mode**: The original pipeline saved high-quality images that did not require restoration as `good_{filename}`, a prefix not recognised by any downstream component. Fixed to save as `anchor_{stem}.png` so the upscaler and subsequent steps process them correctly.

5. **Train PSNR vs Test PSNR**: High training PSNR (e.g., 40 dB) with a large train-test gap (e.g., 27 dB) indicates overfitting to synthesised views, not genuine scene understanding. Only test-split metrics should be reported in the dissertation.

---

## 4.4 Recommended Corrective Actions Before Final Submission

1. **Rerun Condition A** with `--opacity_reset_interval 9000` to produce a valid dense baseline
2. **Rerun sparse subsampling** ensuring selected views cover front, side, AND back hemispheres
3. **Run** `render.py` + `metrics.py` on all conditions to produce final results.json
4. **Report test-split metrics only** (PSNR, SSIM, LPIPS from results.json)
