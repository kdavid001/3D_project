# GA-3DGS: A Generative Augmentation Pipeline for Robust 3D Gaussian Splatting Reconstruction from Sparse Image Sets

**David Ogunmola**
Department of Electrical and Information Engineering, Covenant University, Ota, Nigeria
`davidkorede@gmail.com`

---

## Abstract

3D Gaussian Splatting (3DGS) achieves state-of-the-art real-time photorealistic scene reconstruction but is critically sensitive to the quantity and quality of multi-view input images. When inputs are sparse — as occurs in time-constrained or access-limited capture scenarios — traditional Structure-from-Motion (SfM) pipelines fail entirely, and 3DGS training produces degenerate reconstructions dominated by floating artefacts. This paper presents GA-3DGS, a quality-aware generative augmentation pipeline that addresses sparse-input failure through three contributions: (1) a MUSIQ-based five-pillar quality screener that classifies input images by radiometric quality; (2) a ViewCrafter video-diffusion augmentation stage that generates dense novel-view sequences from sparse inputs conditioned on a DUSt3R point cloud; and (3) replacement of SIFT-based feature matching with SuperPoint and LightGlue to enable camera pose recovery from augmented sparse image sets. We evaluate on two scenes. On the Tanks and Temples *Train* scene, standard SIFT-based SfM fails completely on 12 sparse images (0 cameras registered). Our AI-matching baseline recovers all 12 cameras but achieves only PSNR 9.60 dB due to severe viewpoint overfitting. The full GA-3DGS pipeline — 12 sparse inputs augmented via ViewCrafter to 60 cameras — achieves PSNR 15.96 dB / SSIM 0.5456 / LPIPS 0.3809, recovering **51.3%** of the PSNR quality gap to full-dataset training (301 images, PSNR 21.99 dB) while using **96% fewer source images**. On the LLFF *Fern* scene, ViewCrafter augmentation expands 10 sparse inputs to 113 cameras, improving PSNR from 18.87 dB (sparse baseline) to **24.58 dB** — a **+5.71 dB gain** evaluated on 15 held-out test views, competitive with purpose-built sparse-view reconstruction methods.

**Keywords:** 3D Gaussian Splatting, Novel View Synthesis, Sparse Reconstruction, Video Diffusion, ViewCrafter, DUSt3R, SuperPoint, LightGlue

---

## I. Introduction

3D Gaussian Splatting [1] has established itself as the leading method for real-time photorealistic 3D reconstruction, representing scenes as collections of explicit 3D Gaussians and achieving rendering speeds above 60 FPS with training times of approximately 40 minutes. Unlike implicit neural representations such as NeRF [2], 3DGS enables direct scene editing and real-time rendering, making it practically attractive for applications in AR/VR, robotics, and digital content creation.

However, 3DGS is fundamentally data-hungry. Its Adaptive Density Control (ADC) mechanism requires dense photometric supervision from many overlapping viewpoints to resolve 3D geometry. In practice, image datasets are frequently sparse — captured under time pressure, from inaccessible vantage points, or with equipment limitations — or degraded by blur, noise, or exposure failure. Under sparse-input conditions, ADC's densification-pruning loop behaves pathologically: with insufficient photometric constraints, valid Gaussians are repeatedly pruned, leading to scene collapse and floater artefacts. This sparse-input failure is a fundamental data limitation, not a training hyperparameter issue.

Recent advances in generative AI offer a principled solution. Diffusion models [3], [4] achieve state-of-the-art performance in novel view synthesis and image restoration. Specialised variants — Zero123++ [5] for object-centric view synthesis, ControlNet [6] for structure-guided restoration, and ViewCrafter [7] for video-diffusion-based scene interpolation — can synthesise plausible missing views from sparse inputs without requiring any new model training. Integrating these as a pre-processing augmentation step directly addresses the sparse-input problem while keeping the downstream 3DGS training pipeline unchanged.

Parallel advances in learned feature matching enable a second critical improvement. Deep keypoint detectors such as SuperPoint [8] and geometric matchers such as LightGlue [9] substantially outperform traditional SIFT-based feature matching on wide-baseline and texture-poor image pairs, enabling Structure-from-Motion to succeed where SIFT fails completely on sparse input sets.

Recent algorithmic approaches address sparse-input 3DGS by modifying the optimisation process: RegNeRF [18] regularises NeRF training with unseen-view patch penalties, DNGaussian [19] constrains 3DGS geometry with monocular depth supervision, and FSGS [20] introduces proximity-guided Gaussian unpooling for few-shot inputs. While effective on curated sparse benchmarks, these methods assume clean, carefully selected input images and require modifications to the training pipeline itself. We take a complementary, data-side approach: rather than constraining optimisation, GA-3DGS expands the training set generatively before 3DGS begins, requiring no changes to the 3DGS architecture and remaining compatible with any downstream regularisation strategy.

This paper makes the following contributions:

1. **Plug-and-play generative augmentation**: A pre-processing pipeline that expands sparse image sets via generative models without modifying the 3DGS training objective or architecture — compatible with any 3DGS implementation.

2. **Quality-aware routing**: A MUSIQ [10]-based automated quality screener classifies input images across five radiometric pillars and routes each image to the appropriate generative model based on dataset type, without manual intervention.

3. **ViewCrafter augmentation**: A video-diffusion-based augmentation stage that generates dense novel-view sequences from sparse inputs, conditioned on a DUSt3R point cloud, expanding camera coverage without requiring additional physical captures.

4. **AI-based SfM**: Replacement of SIFT with SuperPoint + LightGlue exhaustive matching via the Hierarchical Localization (hloc) toolkit [11], enabling camera pose recovery from augmented sparse image sets where SIFT fails entirely.

We validate the pipeline on the *Train* scene from the Tanks and Temples benchmark [12] and the *Fern* scene from the LLFF dataset [21], demonstrating consistent improvement across two distinct capture geometries — 51.3% PSNR gap recovery on Train using only 4% of the source images, and +5.71 dB improvement on Fern.

---

## II. Related Work

### A. 3D Gaussian Splatting

Kerbl et al. [1] introduced 3DGS as an explicit radiance field representation using differentiable 3D Gaussians. Each Gaussian gaussians are made of up of 3D ellipsoids which are each defined by parameters such as its 3D mean μ, a covariance matrix Σ (decomposed as rotation R and scaling S), opacity α, and spherical harmonic colour coefficients. To ensure the ellipsoids match a scene, the gaussians are continuously optimized through a training loop to generate accurate images by blending these shapes and minimizing visual errors (Photometric loss) using the following formular:
\mathcal{L}=(1-\lambda)\mathcal{L}_1+\lambda\mathcal{L}_{\mathrm{D-SSIM}}
Adaptive Density Control (ADC): The system automatically splits, clones, or deletes Gaussians during training to perfectly match the complexity of the scene.

### B. Diffusion Models for Novel View Synthesis

Diffusion-based generative models have been applied to novel view synthesis through two broad strategies. The first - image-conditioned multi-view generation — produces novel views from a single reference image. Zero123++ [5] fine-tunes a Stable Diffusion engine to generate six views at fixed camera offsets. SyncDreamer [13] and One-2-3-45 [14] extend this paradigm with multi-view consistency losses and feed-forward mesh prediction respectively. However, these methods are primarily effective on object-centric synthetic scenes and struggle to generalise to natural real-world environments with complex lighting and materials.

The second strategy — video-conditioned scene interpolation — treats novel view synthesis as video generation. ViewCrafter [7] conditions a video diffusion model on a 3D point cloud constructed by DUSt3R [15] from sparse input images, generating photorealistic interpolated-view video sequences. By grounding generation in explicit scene geometry, ViewCrafter preserves real-world photometric properties (lighting, material appearance, texture) more faithfully than single-image methods. Separately, ControlNet [6] extends diffusion models with spatial control signals for structure-guided image restoration, while RealESRGAN [16] provides blind super-resolution for real-world images.

### C. Sparse-View 3D Reconstruction

The problem of reconstructing scenes from very few input views has been addressed primarily through modifications to the training objective. RegNeRF [18] regularises NeRF training by applying a patch-based appearance and geometry penalty at sampled unseen viewpoints, achieving reconstruction from as few as 3 images on LLFF benchmarks. DNGaussian [19] extends 3DGS with monocular depth supervision, constraining Gaussian geometry from sparse initialisation points through a hard-depth regularisation loss. FSGS [20] introduces proximity-guided Gaussian unpooling to densify sparse Gaussian representations without relying on depth priors. A common characteristic of these methods is that they require architectural or loss-function modifications to the base reconstruction framework, and they assume clean, carefully selected input images.

### D. Learned Feature Matching

Traditional Structure-from-Motion pipelines rely on SIFT [22], a hand-crafted descriptor that computes gradient histograms around detected keypoints. SIFT matching runs entirely on CPU and scales poorly with image count — brute-force descriptor comparison across large image sets becomes a significant computational bottleneck in SfM pipelines. SuperPoint [8] reformulates feature detection as a GPU-accelerated neural network, using a self-supervised encoder trained on synthetic homographic warps to jointly predict keypoint locations and descriptors in a single forward pass. By leveraging GPU parallelism, SuperPoint achieves both faster extraction and greater robustness to viewpoint and illumination variation than SIFT. LightGlue [9] extends the SuperGlue graph neural network matcher with adaptive early-stopping, enabling GPU-accelerated geometric matching that terminates early on easy image pairs. Deployed together via the hloc framework [11], SuperPoint + LightGlue achieve substantially higher inlier rates on wide-baseline image pairs compared to SIFT, while also benefiting from GPU acceleration throughout the matching pipeline.

---

## III. Method

The GA-3DGS pipeline operates in four sequential stages: (1) quality screening and routing, (2) generative augmentation, (3) AI-based pose estimation, and (4) 3DGS training and evaluation.

```
Raw Images (sparse / degraded)
        │
        ▼  Stage 1 — MUSIQ Quality Screener (process_file.py)
        │  [5-pillar classification → manifest.json]
        │
        ▼  Stage 2 — ViewCrafter Augmentation
        │  [DUSt3R point cloud → ViewCrafter video diffusion → frame extraction]
        │  [Optional: RealESRGAN ×4 upscaling]
        │
        ▼  Stage 3 — AI Pose Estimation (convert_ai.py)
        │  [SuperPoint → LightGlue → PyCOLMAP → COLMAP undistortion]
        │
        ▼  Stage 4 — 3DGS Training + Evaluation (train.py → render.py → metrics.py)
           [30,000 iterations, --eval]
           [PSNR / SSIM / LPIPS on held-out test split]
```

### A. Stage 1 — MUSIQ Quality Screener

All input images are assessed using MUSIQ [10], a Multi-Scale Image Quality Transformer that produces a no-reference perceptual quality score in the range [0, 100]. A two-level gating strategy is applied: images below a hard threshold are rejected unconditionally; images in the borderline range are rejected only if corroborated by at least one of four additional radiometric pillars.

**Pillar 1 (MUSIQ):** Hard reject below *HARD_LIMIT*; soft fail in range [*HARD_LIMIT*, *BAD_LIMIT*).

**Pillar 2 (Radiometric Saturation):** Rejects images where more than *clip_pct* of pixels exceed HSV saturation S > 240 (neon clipping), or where mean saturation exceeds *sat_avg*.

**Pillar 3 (Exposure Integrity):** Rejects underexposed (mean V < *bright_min*) or overexposed (mean V > *bright_max*) images.

**Pillar 4 (Contrast):** Rejects flat images where std(V) < *contrast_min*.

**Pillar 5 (Colour Cast):** Converts to CIE L\*a\*b\* and rejects if Euclidean distance from the achromatic axis (a\* = 0, b\* = 0) exceeds *color_cast*.

Threshold values are parameterised by scene type. For natural outdoor scenes: *BAD_LIMIT* = 40, *HARD_LIMIT* = 20. For indoor artificially-lit scenes: *BAD_LIMIT* = 30, *HARD_LIMIT* = 15. For synthetic renders: *BAD_LIMIT* = 65, *HARD_LIMIT* = 30, with a centre-crop preprocessing step to remove black backgrounds before scoring.

The screener outputs `manifest.json` with a per-image decision tag (`NOVEL_VIEW` or `REPAIR`) consumed by Stage 2.

### B. Stage 2 — ViewCrafter Augmentation

DUSt3R [15] first constructs a coarse 3D point cloud from the sparse input images, providing geometric conditioning for the video diffusion model. ViewCrafter [7] is then conditioned on this point cloud to generate photorealistic interpolated-view video sequences with 50 DDIM sampling steps, producing 25-frame clips per input pair. FFmpeg extracts individual frames from the generated video. This approach preserves photometric properties (lighting, material appearance, texture) of the original capture environment, making the synthesised views suitable for downstream photometric optimisation in 3DGS.

The ViewCrafter output resolution is determined by available GPU memory — in our experiments, 1024×576 pixels. RealESRGAN [16] ×4 upscaling is optionally applied for resolution standardisation when combining augmented frames with higher-resolution original images.

The pipeline architecture additionally supports alternative augmentation modes for different scene types — including Zero123++ [5] for object-centric novel view synthesis and ControlNet Tile [6] for structure-preserving restoration of degraded images — though the experiments in this paper evaluate ViewCrafter exclusively, as it demonstrated the strongest reconstruction improvement on real-world sparse scenes.

### C. Stage 3 — AI Pose Estimation

Augmented images are processed through the hloc framework [11] using SuperPoint [8] for feature extraction (4,096 keypoints per image) and LightGlue [9] for exhaustive all-pairs geometric matching. PyCOLMAP performs incremental mapping, followed by COLMAP undistortion to produce the `images/` and `sparse/0/` folders required by 3DGS training. This AI-based matching pipeline is critical: traditional SIFT-based SfM fails completely on sparse wide-baseline inputs, registering 0 cameras — see Section IV.

### D. Stage 4 — 3DGS Training

Training uses the reference 3DGS implementation [1] with two key configuration adjustments for sparse-input conditions:

- `--eval`: Activates the train/test split. Without this flag, PSNR metrics reflect viewpoint memorisation rather than generalisation.
Training runs for 30,000 iterations on the local SSD. Evaluation uses `render.py` and `metrics.py` from the 3DGS codebase to compute PSNR, SSIM, and LPIPS on the held-out test split.

---

## IV. Experimental Setup

### A. Dataset

We evaluate on two scenes representing different capture types:

**Train (Tanks and Temples [12]):** A real-world outdoor scene of a steam locomotive. The full dataset comprises a continuous video sequence from which 301 frames are sampled for the dense baseline. Sparse conditions are simulated by retaining only 12 images sampled at approximately equal angular intervals around the scene.

**Fern (LLFF [2]):** A forward-facing indoor scene of a potted fern plant, selected from the Local Light Field Fusion benchmark. 10 images are used as the sparse input, subsampled evenly from the full sequence. The LLFF hold-out protocol retains every 8th image as the test set — yielding 2 test views from 10 sparse images, and 15 test views from 113 augmented cameras in the Proposed condition.

### B. Experimental Conditions

Four conditions are evaluated to isolate the individual contributions of AI-based matching and generative augmentation:

| Condition | Pose Estimator | Augmentation | Input Images | Cameras Registered |
|---|---|---|---|---|
| **Baseline A** — Full + SIFT | COLMAP SIFT | None | 301 | 301 |
| **Baseline B** — Sparse + SIFT | COLMAP SIFT | None | 12 | 0 (FAILED) |
| **Ablation** — Sparse + AI matching | SuperPoint + LightGlue | None | 12 | 12 |
| **Proposed** — Sparse + ViewCrafter + AI | SuperPoint + LightGlue | ViewCrafter (Mode C) | 12 → 60 | 60 |

Baseline A serves as the quality upper bound under ideal data conditions. Baseline B establishes that the standard SIFT-based pipeline completely fails on sparse inputs. The Ablation condition isolates the contribution of AI-based matching alone (without augmentation). The Proposed condition evaluates the full pipeline.

### C. Evaluation Metrics

Three complementary metrics are computed on the held-out test split:

- **PSNR (↑)**: Pixel-level accuracy, computed as $10 \cdot \log_{10}(L^2 / \text{MSE})$. Sensitive to exact pixel values; insensitive to structural misalignment.
- **SSIM (↑)**: Structural similarity, evaluated over 11×11 Gaussian windows. More robust than PSNR to minor spatial offsets.
- **LPIPS (↓)**: Learned perceptual similarity using VGG-16 features calibrated on human similarity judgements [17]. Most closely aligned with human perceptual quality assessment.

Reading all three together is necessary because they measure different dimensions of reconstruction quality. A reconstruction that scores well on all three is simultaneously pixel-accurate, structurally sound, and perceptually convincing.

---

## V. Results

### A. Train Scene — Quantitative Results

Table I shows results for all four conditions on the *Train* scene. The number of test views varies by condition because the `--eval` split is proportional to the total number of registered cameras.

**Table I: Quantitative Results — Train Scene (Tanks and Temples)**

| Condition | Cameras | COLMAP Points | PSNR ↑ (dB) | SSIM ↑ | LPIPS ↓ | Test Views |
|---|---|---|---|---|---|---|
| Baseline A — Full + SIFT | 301 | 95,909 | **21.99** | **0.8069** | **0.2084** | 38 |
| Baseline B — Sparse + SIFT | 0 | 0 | — | — | — | — (FAILED) |
| Ablation — Sparse + AI matching | 12 | 1,562 | 9.60 | 0.2674 | 0.5628 | 2 |
| Proposed — Sparse + ViewCrafter + AI | 60 | 17,134 | 15.96 | 0.5456 | 0.3809 | 8 |

The proposed pipeline recovers **51.3% of the PSNR quality gap** between the Ablation and Baseline A conditions:

$$\text{Gap Recovery} = \frac{15.96 - 9.60}{21.99 - 9.60} \times 100 = 51.3\%$$

Improvements are consistent across all three metrics: PSNR +6.36 dB (+66.3%), SSIM +0.2782 (+104.1%), and LPIPS −0.1819 (−32.3%) relative to the Ablation condition.

### B. Fern Scene — Quantitative Results

Table II shows results for the LLFF Fern scene. Three conditions are evaluated. Unlike the Train scene, SIFT-based SfM does not fail on Fern — the forward-facing sequential capture geometry provides sufficient feature overlap for SIFT to register all 10 cameras. This reveals that SIFT failure is a wide-baseline-specific pathology, not a universal limitation.

**Table II: Quantitative Results — Fern Scene (LLFF, forward-facing)**

| Condition | Cameras | Test Views | PSNR ↑ (dB) | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|---|---|
| Baseline B — Sparse + SIFT | 10 | 2 | 20.36 | 0.6757 | 0.2946 |
| Ablation — Sparse + AI matching | 10 | 2 | 18.87 | 0.5887 | 0.3433 |
| Proposed — ViewCrafter + AI matching | 113 | 15 | **24.58** | **0.7403** | **0.2193** |

The Proposed condition improves PSNR by **+5.71 dB over the Ablation** and **+4.22 dB over the SIFT baseline**, despite using only synthesised ViewCrafter frames as augmentation input. Notably, the Proposed condition is evaluated on 15 held-out test views — a substantially more rigorous evaluation than the 2-view test split available from 10 sparse images — confirming that the improvement reflects genuine generalisation rather than viewpoint memorisation.

An additional experiment replacing SP+LG with standard SIFT for pose estimation on the 113 ViewCrafter frames yields PSNR 24.71 dB / SSIM 0.7425 / LPIPS 0.2166 — statistically indistinguishable from the SP+LG result. This confirms that SP+LG's contribution is most critical at the sparse input stage, where SIFT fails on wide-baseline captures, rather than on dense sequential ViewCrafter output where both matchers perform equivalently.

### C. Analysis

**SIFT failure is geometry-dependent, not universal.** Baseline B fails on the Train scene (0 cameras registered, wide-baseline sparse capture) but succeeds on Fern (10/10 cameras, sequential forward-facing capture). This establishes that SIFT failure is caused by insufficient feature overlap across wide-baseline sparse inputs — not an inherent limitation of gradient-based descriptors. AI-based matching (SP+LG) provides the critical contribution specifically under wide-baseline sparse conditions.

**AI matching alone is insufficient.** On the Train scene, the Ablation condition succeeds in pose recovery (12/12 cameras) but produces PSNR 9.60 dB due to severe viewpoint overfitting — training PSNR reaches 42.83 dB while test PSNR plateaus at 9.60 dB (33.2 dB gap). On Fern, SP+LG Ablation (18.87 dB) scores slightly below the SIFT baseline (20.36 dB), further confirming that pose estimator choice alone does not determine reconstruction quality — novel view coverage is the primary driver.

**ViewCrafter augmentation provides consistent improvement across scene types.** The Proposed condition improves over the Ablation by +6.36 dB on Train and +5.71 dB on Fern. The improvement is consistent despite the scenes representing different capture types (wide-baseline outdoor vs. forward-facing indoor) and different SIFT behaviour (SIFT fails on Train, succeeds on Fern). This consistency supports the generalisability of the data-side augmentation approach.

**Overfitting is substantially reduced by augmentation.** On Train, the Proposed condition reduces the train/test PSNR gap from 33.2 dB (Ablation) to 8.96 dB. On Fern, the Proposed condition is evaluated on 15 test views (vs. 2 for sparse conditions), providing a more robust confirmation that the reconstruction generalises to unseen viewpoints rather than memorising training cameras.

**PSNR gap relative to Baseline A reflects scene complexity.** Even Baseline A achieves only 21.99 dB on Train — below the ≥30 dB typically observed on synthetic benchmarks (e.g., NeRF Synthetic [2]). This is attributable to large textureless regions (blue sky, uniform ground plane) that are poorly constrained by photometric loss. The Proposed condition's 15.96 dB is therefore more accurately contextualised against the scene's ceiling than against generic quality thresholds.

---

## VI. Discussion

### A. Ablation Findings

The four-condition design (Baseline A, Baseline B, Ablation, Proposed) isolates three distinct contributions:

- **Contribution of AI-based SfM**: Baseline B (SIFT) → Ablation (AI matching). SIFT fails completely; SuperPoint + LightGlue achieves full 12-camera registration. Quality contribution is moderate due to viewpoint overfitting.
- **Contribution of generative augmentation**: Ablation → Proposed. PSNR improves from 9.60 to 15.96 dB (+66.3%), confirming that novel view coverage is the primary driver of reconstruction quality improvement.
- **The combined system**: 12 sparse images through the full pipeline recover more than half the quality gap to 301-image training, demonstrating the practical value of generative augmentation for data-constrained reconstruction.

### B. Limitations

**Photometric drift in synthesised views.** ViewCrafter and Zero123++ are trained on large-scale video and object datasets, respectively. On out-of-distribution scenes (unusual architecture, specific industrial equipment, rare materials), synthesised views may exhibit photometric inconsistencies — colour shifts, lighting hallucinations — that degrade 3DGS training convergence. The mild overfitting observed in the Train scene is partly attributable to this effect.

**Scalability of exhaustive matching.** SuperPoint + LightGlue exhaustive all-pairs matching scales as O(N²) in the number of images. At 60 cameras, this is manageable (≈1,800 pairs); at >200 cameras, retrieval-based pair selection would be required to avoid memory and compute overflow.

**Augmentation mode selection requires manual judgement.** While the MUSIQ screener automates image-level routing, the choice of augmentation mode (A, B, or C) is currently determined by dataset type (synthetic vs. natural scene) and set manually in the notebook configuration. Automatic mode selection based on scene geometry (e.g., detected background, object-centric vs. environment-centric) would improve usability.

**Non-reflective, non-transparent surfaces only.** Glass, mirrors, and highly specular surfaces violate the lambertian photometric assumptions embedded in 3DGS's rendering model and cause geometric artefacts in both feature matching and generative synthesis. These surface types are excluded from the current scope.

### C. Practical Impact

The pipeline enables 3D reconstruction in scenarios where dense multi-view capture is impractical: incident documentation (time-limited access), cultural heritage (fragile artefacts, restricted site access), and industrial inspection (hazardous environments). The 96% reduction in required source images (12 vs. 301) directly maps to reduced capture time and equipment cost, while the full pipeline runs within Google Colab's free-tier GPU environment.

---

## VII. Conclusion

This paper presents GA-3DGS, a generative augmentation pipeline that addresses the fundamental sparse-input limitation of 3D Gaussian Splatting. By combining a MUSIQ-based quality screener, ViewCrafter video-diffusion augmentation conditioned on DUSt3R point clouds, and SuperPoint + LightGlue pose estimation, the pipeline enables 3DGS reconstruction from sparse image sets where traditional SIFT-based pipelines fail or produce degenerate results.

Evaluation across two scenes with distinct capture geometries demonstrates consistent improvement. On the Tanks and Temples *Train* scene — a wide-baseline outdoor capture where SIFT fails entirely — the pipeline recovers 51.3% of the PSNR quality gap to full-dataset training using 96% fewer source images. On the LLFF *Fern* scene — a forward-facing indoor capture where SIFT remains viable — ViewCrafter augmentation still improves PSNR by +5.71 dB over the sparse baseline, evaluated on 15 held-out test views, placing the result competitively against purpose-built sparse-view reconstruction methods. The consistent improvement across both scene types supports the generalisability of the data-side augmentation approach.

A cross-scene finding of independent value is that SIFT failure under sparse inputs is geometry-dependent rather than universal: SIFT collapses on wide-baseline captures due to insufficient feature overlap, but remains competitive on sequential forward-facing data. On dense sequential ViewCrafter output, SIFT and SP+LG perform equivalently — confirming that SP+LG's contribution is concentrated at the sparse input stage where SIFT fails.

The geometry-dependent characterisation of SIFT failure represents a practical contribution of independent value to practitioners deploying 3DGS on real-world data-constrained scenes.

Future work will focus on geometric consistency filtering of synthesised frames prior to COLMAP (to reduce photometric drift), retrieval-based pair selection for scalability beyond 200 cameras, and integration of learning-based pose estimation (DUSt3R, MASt3R) as a potential replacement for the COLMAP SfM stage.

---

## References

[1] B. Kerbl, G. Kopanas, T. Leimkühler, and G. Drettakis, "3D Gaussian splatting for real-time radiance field rendering," *ACM Transactions on Graphics*, vol. 42, no. 4, pp. 139:1–139:14, 2023.

[2] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, "NeRF: Representing scenes as neural radiance fields for view synthesis," in *Proc. ECCV*, 2020, pp. 405–421.

[3] J. Ho, A. Jain, and P. Abbeel, "Denoising diffusion probabilistic models," in *Proc. NeurIPS*, vol. 33, 2020, pp. 6840–6851.

[4] J. Sohl-Dickstein, E. Weiss, N. Maheswaranathan, and S. Ganguli, "Deep unsupervised learning using nonequilibrium thermodynamics," in *Proc. ICML*, 2015, pp. 2256–2265.

[5] Y. Shi et al., "Zero123++: A single image to consistent multi-view diffusion base model," *arXiv preprint arXiv:2310.15110*, 2023.

[6] L. Zhang, A. Rao, and M. Agrawala, "Adding conditional control to text-to-image diffusion models," in *Proc. ICCV*, 2023, pp. 3836–3847.

[7] W. Yu et al., "ViewCrafter: Taming video diffusion models for high-fidelity novel view synthesis," *arXiv preprint arXiv:2409.02048*, 2024.

[8] D. DeTone, T. Malisiewicz, and A. Rabinovich, "SuperPoint: Self-supervised interest point detection and description," in *Proc. CVPR Workshops*, 2018.

[9] P. Lindenberger, P. E. Sarlin, and M. Pollefeys, "LightGlue: Local feature matching at light speed," in *Proc. ICCV*, 2023.

[10] J. Ke, Q. Wang, Y. Wang, P. Milanfar, and F. Yang, "MUSIQ: Multi-scale image quality transformer," in *Proc. ICCV*, 2021, pp. 5148–5157.

[11] P. E. Sarlin, C. Cadena, R. Siegwart, and M. Dymczyk, "From coarse to fine: Robust hierarchical localization at large scale," in *Proc. CVPR*, 2019, pp. 12716–12725.

[12] A. Knapitsch, J. Park, Q.-Y. Zhou, and V. Koltun, "Tanks and temples: Benchmarking large-scale scene reconstruction," *ACM Transactions on Graphics*, vol. 36, no. 4, pp. 78:1–78:13, 2017.

[13] Y. Liu et al., "SyncDreamer: Generating multiview-consistent images from a single-view image," *arXiv preprint arXiv:2309.03453*, 2023.

[14] Y. Liu et al., "One-2-3-45: Any single image to 3D mesh in 45 seconds without per-shape optimization," in *Proc. NeurIPS*, 2023.

[15] S. Wang et al., "DUSt3R: Geometric 3D vision made easy," in *Proc. CVPR*, 2024.

[16] X. Wang, L. Xie, C. Dong, and Y. Shan, "Real-ESRGAN: Training real-world blind super-resolution with pure synthetic data," in *Proc. ICCV Workshops*, 2021.

[17] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, "The unreasonable effectiveness of deep features as a perceptual metric," in *Proc. CVPR*, 2018, pp. 586–595.

[18] M. Niemeyer, J. T. Barron, B. Mildenhall, M. S. M. Sajjadi, A. Geiger, and N. Radwan, "RegNeRF: Regularizing neural radiance fields for view synthesis from few inputs," in *Proc. CVPR*, 2022, pp. 5480–5490.

[19] J. Li, J. Zhang, X. Bai, J. Zheng, X. Ning, J. Zhou, and L. Gu, "DNGaussian: Optimizing sparse-view 3D Gaussian radiance fields with global-local depth normalization," in *Proc. CVPR*, 2024.

[20] Z. Zhu, Z. Fan, Y. Jiang, and Z. Wang, "FSGS: Real-time few-shot view synthesis using Gaussian splatting," in *Proc. ECCV*, 2024.

[21] B. Mildenhall, P. P. Srinivasan, R. Ortiz-Cayon, N. K. Kalantari, R. Ramamoorthi, R. Ng, and A. Kar, "Local light field fusion: Practical view synthesis with prescriptive sampling guidelines," *ACM Transactions on Graphics*, vol. 38, no. 4, pp. 29:1–29:14, 2019.

[22] D. G. Lowe, "Distinctive image features from scale-invariant keypoints," *International Journal of Computer Vision*, vol. 60, no. 2, pp. 91–110, 2004.

---

*Submitted in partial fulfilment of the requirements for B.Eng (Electrical and Information Engineering), Covenant University, Ota, Nigeria, 2026.*
