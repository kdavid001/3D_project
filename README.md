# Multi-view Generative Augmented 3D Reconstruction Pipeline
**B.Eng Final Year Project — David Ogunmola (kdavid001)**

---

## About This Research

This project investigates the reconstruction of photorealistic 3D scenes from sparse image inputs using **3D Gaussian Splatting (3DGS)** as the core rendering framework. The central problem addressed is that real-world 3D capture is rarely ideal, photographers cannot always provide the dense, uniformly distributed, high-quality imagery that classical Structure-from-Motion (SfM) and neural rendering pipelines assume. This work proposes an augmentation pipeline that recovers reconstruction quality from as few as 12 sparse input images, evaluated with ViewCrafter on two scenes. A quality screener flags images with blur, poor exposure or colour cast before reconstruction.

The pipeline operates in four stages:
1. **Quality Screening** — A five-pillar radiometric screener (MUSIQ + saturation, exposure, contrast, colour cast) evaluates each input image and accepts or rejects it using scene-adaptive thresholds (natural, indoor, synthetic).
2. **Generative Augmentation** — Three diffusion-based modes handle different scene types: **Zero123++** (Mode A) synthesises geometrically consistent novel views for object-centric scenes; **ControlNet Tile** (Mode B) restores degraded images; **ViewCrafter** (Mode C) interpolates between sparse viewpoints for real-world scenes using DUSt3R point cloud conditioning. Only Mode C (ViewCrafter) is evaluated in the results below.
3. **AI-Driven Pose Estimation** — **SuperPoint + LightGlue** exhaustive feature matching replaces classical SIFT for COLMAP-based camera pose estimation, enabling registration in cases where SIFT fails entirely.
4. **3DGS Training and Evaluation** — Standard 3DGS training for 30,000 iterations with the `--eval` flag. PSNR, SSIM and LPIPS are computed on held-out test views.

### Key Result
The proposed pipeline recovers **51.3% of the quality gap** between a sparse no-augmentation baseline (PSNR 9.60 dB) and the full 301-image upper bound (PSNR 21.99 dB), using only **12 sparse input images** — a **96% reduction** in acquisition effort.

| Condition | Images | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|---|
| Baseline B — Sparse + SIFT | 12 (FAILED) | N/A | N/A | N/A |
| Ablation — Sparse + AI matching | 12 | 9.60 dB | 0.2674 | 0.5628 |
| **Proposed — ViewCrafter + AI matching** | **12** | **15.96 dB** | **0.5456** | **0.3809** |
| Baseline A — Full dataset + SIFT | 301 | 21.99 dB | 0.8069 | 0.2084 |

On the indoor Fern scene, ViewCrafter expanded 10 sparse images to 113 views and raised PSNR from 18.87 to 24.58 dB, evaluated on 15 held-out views.

---

## Repository Structure

```
3D_project/
├── process_file.py          # Five-pillar quality screener (MUSIQ-based)
├── convert_ai.py            # SuperPoint + LightGlue → COLMAP pose estimation
│                            #   (replaces convert.py from gaussian-splatting)
├── diffusion_script_v0.py   # Generative augmentation (Modes A / B)
├── corrupt_data.py          # Dataset corruption simulator
├── My_relaxed_convert.py    # Modified gaussian-splatting convert.py (COLMAP)
├── rename.py                # Dataset organising helper
├── requirements.txt
├── user_guide.md            # Setup and usage
├── DEVNOTES.md              # Development notes
├── Pipeline_files/
│   └── unified_pipeline.ipynb
├── images_for/              # Rendered results
└── output_images/           # Screener output examples
```

---

## Visual Results — Train Scene

Rendered novel views comparing reconstruction quality across conditions (same scene, held-out test viewpoints):

| Ablation — 12 images, AI matching, no augmentation | Proposed — 12 images + ViewCrafter | Baseline A — 301 images (upper bound) |
|:---:|:---:|:---:|
| ![Ablation](images_for/d_s_renders/00001.png) | ![Proposed](images_for/mine_renders/00004.png) | ![Dense](images_for/dense_renders/00018.png) |
| PSNR 9.60 dB · SSIM 0.267 · LPIPS 0.563 | **PSNR 15.96 dB · SSIM 0.546 · LPIPS 0.381** | PSNR 21.99 dB · SSIM 0.807 · LPIPS 0.208 |

The proposed pipeline recovers visible scene geometry and texture with only 12 input images. The ablation (same 12 images, no ViewCrafter augmentation) produces a heavily artefacted reconstruction due to insufficient camera coverage for 3DGS densification.

---

## Dependencies

### 1. Clone this repository
```bash
git clone https://github.com/kdavid001/3D_project.git
cd 3D_project
```

### 2. Clone gaussian-splatting (required for training)
This project builds on the official 3D Gaussian Splatting implementation. Clone it separately:
```bash
git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive
```
Use `convert_ai.py` from this repo **instead of** `convert.py` from gaussian-splatting for AI-based pose estimation. All other training scripts (`train.py`, `render.py`, `metrics.py`) come from gaussian-splatting.

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. ViewCrafter (Mode C — real-world scenes)
ViewCrafter requires a separate Conda environment. See [user_guide.md](user_guide.md) for full setup instructions including DUSt3R and model checkpoint downloads.

---

## Quickstart

See [user_guide.md](user_guide.md) for full setup and usage instructions, including:
- Running the quality screener (`process_file.py`)
- Running generative augmentation (Modes A / B / C)
- Running AI-based pose estimation (`convert_ai.py`)
- Training and evaluating the 3DGS scene

---

## Research Directions

Several components of this pipeline have potential as standalone research contributions and are being explored for future publication:

- **Domain-Adaptive Radiometric Quality Screening for 3D Reconstruction** — The five-pillar MUSIQ-based screener (`process_file.py`) with scene-adaptive thresholding (natural / indoor / synthetic modes) addresses a gap in existing pipelines, which either skip input quality assessment entirely or rely on a single metric. A focused study on its false-positive/false-negative rate across diverse scene types could form a standalone contribution.

- **Learned Feature Matching as a SIFT Replacement for Sparse-Input 3DGS** — `convert_ai.py` demonstrates that SuperPoint + LightGlue exhaustive matching enables camera registration from as few as 12 wide-baseline images where classical SIFT fails completely. A systematic benchmarking study across scene types and sparsity levels would establish this as a replicable finding.

- **Generative Augmentation Routing for 3DGS Reconstruction** — The augmentation mode is currently chosen manually from the scene type. Automating that choice from the quality screen and scene type, and extending the evaluation to more scenes and sparsity levels, is the natural next step toward a full research paper.

---

## Citation

If you use this work, please also cite the following:

- Kerbl et al. (2023) — [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- Shi et al. (2023) — [Zero123++](https://github.com/sudo-ai-3d/zero123plus)
- Yu et al. (2024) — [ViewCrafter](https://github.com/Drexubery/ViewCrafter)
- Lindenberger et al. (2023) — [LightGlue](https://github.com/cvg/LightGlue)
- Wang et al. (2023) — [DUSt3R](https://github.com/naver/dust3r)
- Ke et al. (2021) — [MUSIQ](https://github.com/google-research/google-research/tree/master/musiq)
- DeTone et al. (2018) — [SuperPoint](https://github.com/magicleap/SuperPointPretrainedNetwork)
- Sarlin et al. (2019) — [hloc](https://github.com/cvg/Hierarchical-Localization)
- Schönberger and Frahm (2016) — [COLMAP](https://colmap.github.io)
