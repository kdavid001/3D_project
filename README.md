# 3D_project
*"The Beginning of an Unseen End"*

---

## About This Research

This project investigates the reconstruction of photorealistic 3D scenes from sparse, degraded image inputs using **3D Gaussian Splatting (3DGS)** as the core rendering framework. The central problem addressed is that real-world 3D capture is rarely ideal  photographers cannot always provide the dense, uniformly distributed, high-quality imagery that classical Structure-from-Motion (SfM) and neural rendering pipelines assume. This work proposes and evaluates an augmentation pipeline that recovers reconstruction quality from as few as 12 input images, including images affected by blur, poor exposure, colour cast, and missing viewpoints.

The pipeline operates in three stages. First, a five-pillar radiometric quality screener (MUSIQ + saturation, exposure, contrast, and colour cast checks) evaluates each input image and routes it to the appropriate augmentation path. Second, generative augmentation is applied: object-centric scenes use **Zero123++** to synthesise geometrically consistent novel views, degraded images use **ControlNet Tile** for structure-preserving restoration, and real-world scenes use **ViewCrafter** (video diffusion conditioned on a DUSt3R point cloud) to interpolate between sparse viewpoints. Third, **SuperPoint + LightGlue** exhaustive feature matching replaces classical SIFT for pose estimation via COLMAP, enabling camera registration in cases where SIFT fails entirely.

Experiments are conducted across two scene types NeRF Synthetic objects (train, hotdog, lego) and real-world architectural/indoor scenes under four conditions: full-dataset baseline, sparse-input baseline (SIFT), sparse-input ablation (AI matching, no augmentation), and the full proposed pipeline. Results demonstrate that the proposed approach recovers over 50% of the quality gap between a sparse no-augmentation baseline and the full-dataset upper bound, as measured by PSNR, SSIM, and LPIPS.

---

## Repository Structure

```
3D_project/
├── process_file.py           # Five-pillar quality screener (MUSIQ-based)
├── diffusion_script_v0.py    # Generative augmentation (Zero123++ / ControlNet)
├── corrupt_data.py           # Dataset corruption simulator (blur, exposure, noise)
├── boost_contrast.py         # CLAHE contrast enhancement preprocessing
├── convert_ai.py             # SuperPoint + LightGlue → COLMAP pose estimation
├── Pipeline_files/
│   └── unified_pipeline.ipynb  # Main Colab notebook (Modes A / B / C)
├── output_train/             # Raw input datasets
├── output_processed/         # Screened and augmented outputs
├── algorithm_for_codes/      # Detailed pseudocode for all algorithms
├── algorithm_summaries/      # Condensed pseudocode summaries
└── Code_Reports/             # Experimental results and code documentation
```

## Quickstart

See [user_guide.md](user_guide.md) for full setup and usage instructions.

---

**Author:** David Ogunmola (kdavid001)
**Type:** B.Eng Final Year Project