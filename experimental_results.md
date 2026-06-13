# Experimental Results Log
# Project: Generative Augmented 3D Reconstruction Pipeline
# Last updated: 2026-06-07

---

## Conditions Reference

| Label | Pose Estimation | Augmentation | Input |
|---|---|---|---|
| Baseline A | SIFT (`convert.py`) | None | Full dataset |
| Baseline B | SIFT (`convert.py`) | None | Sparse (~12 images) |
| Ablation | SuperPoint + LightGlue (`convert_ai.py`) | None | Sparse (~12 images) |
| Proposed | SuperPoint + LightGlue (`convert_ai.py`) | ViewCrafter / Zero123++ / ControlNet | Sparse (~12 images) |

---

## Master Results Table

> Metric direction: **PSNR ↑ higher is better** | **SSIM ↑ higher is better** | **LPIPS ↓ lower is better**

| Scene | Type | Condition | Cameras | COLMAP Points | Init Points | PSNR ↑ (dB) | SSIM ↑ | LPIPS ↓ | Status |
|---|---|---|---|---|---|---|---|---|---|
| train | Real-world | Baseline A — Full + SIFT | 301 | 95,909 | 95,909 | 21.99 | 0.8069 | 0.2084 | Complete |
| train | Real-world | Baseline B — Sparse + SIFT | 0 | 0 | 0 | N/A | N/A | N/A | FAILED |
| train | Real-world | Ablation — Sparse + AI (no aug) | 12 | 1,562 | 1,562 | 9.60 | 0.2674 | 0.5628 | Complete |
| train | Real-world | Proposed — Sparse + ViewCrafter + AI | 60 | 17,134 | 1,835 | **15.96** | **0.5456** | **0.3809** | Complete |
| classroom | Real-world | Baseline A — Full + SIFT | — | — | — | — | — | — | Pending |
| classroom | Real-world | Baseline B — Sparse + SIFT | 0 | 0 | 0 | N/A | N/A | N/A | FAILED |
| classroom | Real-world | Ablation — Sparse + AI (no aug) | 12 | 1,150 | 1,150 | 11.72 | 0.3810 | 0.6180 | Complete |
| classroom | Real-world | Proposed — Sparse + ViewCrafter + AI | — | — | — | — | — | — | Pending |

---

## Detailed Run Logs

---

### Scene: train | Condition: Baseline B — Sparse + SIFT | FAILED

**Date:** 2026-05-28
**Dataset:** `baseline_train_d_s` — ~12 sparse images, no augmentation
**Pose estimation:** `convert.py` (standard COLMAP SIFT exhaustive)

```
Bundle adjustment:
    Residuals : 368  |  Initial cost : 0.438076 px  |  Final cost : 0.184285 px
    Termination : No convergence
  => No good initial image pair found.
ERROR: Mapper failed with code 256
```

**Outcome:** Complete reconstruction failure. SIFT cannot find sufficient matches across 12 sparse images.

---

### Scene: train | Condition: Ablation — Sparse + AI (no augmentation) | COMPLETE

**Date:** 2026-05-29
**Dataset:** `baseline_train_d_s` — 12 sparse images, no augmentation
**Pose estimation:** `convert_ai.py` (SuperPoint + LightGlue exhaustive)

```
SfM: 12/12 cameras registered | 1,562 init points

Training (30,000 iterations):
    [7000]  Test PSNR:  9.558 dB  |  Train PSNR: 33.455 dB
    [30000] Test PSNR:  9.606 dB  |  Train PSNR: 42.834 dB

metrics.py (2 test views):
    PSNR : 9.5977 dB
    SSIM : 0.2674
    LPIPS: 0.5628
```

**Notes:** Severe overfitting — train/test PSNR gap of 33 dB at iter 30000. With only 12 cameras (2 held-out test views), 3DGS memorises training viewpoints and fails completely on unseen angles. Demonstrates that AI matching alone is insufficient — novel view coverage from augmentation is required.

---

### Scene: train | Condition: Proposed — Sparse + ViewCrafter + AI | COMPLETE

**Date:** 2026-05-28
**Dataset:** 12 sparse images → ViewCrafter augmented → 60 cameras registered
**Pose estimation:** `convert_ai.py` (SuperPoint + LightGlue exhaustive)

```
SfM (convert_ai.py):
    Bundle adjustment — Residuals: 158,986 | Convergence
    Reconstruction: 60 images | 17,134 COLMAP points
    Undistortion: 60/60 complete
    Note: "No good initial image pair" is a secondary model attempt after all 60
          images were already registered — not a failure.

Training (train.py — 30,000 iterations):
    Init points : 1,835  (undistortion-filtered subset of 17,134 COLMAP points)
    [7000]  Test PSNR: 16.115 dB  |  Train PSNR: 21.679 dB
    [30000] Test PSNR: 16.004 dB  |  Train PSNR: 24.960 dB

metrics.py (8 test views):
    PSNR : 15.9594 dB
    SSIM : 0.5456
    LPIPS: 0.3809
```

**Notes:** Mild overfitting — test PSNR slightly degraded from iter 7000→30000 while train PSNR kept rising. Scene-level factors (blue sky, textureless regions) contribute to low absolute PSNR.

---

### Scene: train | Condition: Baseline A — Full + SIFT | COMPLETE

**Date:** 2026-05-29
**Dataset:** Full video sequence — 301 images, no augmentation
**Pose estimation:** `convert.py` (standard COLMAP SIFT exhaustive)

```
SfM:
    Residuals : 1,341,926  |  Convergence
    Reconstruction: 301 images | 95,909 COLMAP points
    Undistortion: 301/301 complete

Training (30,000 iterations):
    Init points : 95,909
    [7000]  Test PSNR: 19.738 dB  |  Train PSNR: 21.564 dB
    [30000] Test PSNR: 22.014 dB  |  Train PSNR: 26.841 dB

metrics.py (38 test views):
    PSNR : 21.9942 dB
    SSIM : 0.8069
    LPIPS: 0.2084
```

**Notes:** Healthy training — train/test gap only 4.8 dB at iter 30000, indicating good generalisation with 301 cameras. Upper-bound reference for this scene. The Proposed pipeline (12 sparse images + ViewCrafter) recovers ~51.3% of the quality gap between the Ablation (9.60 dB) and this upper bound (21.99 dB). Calculation: (15.96 − 9.60) / (21.99 − 9.60) × 100 = 51.3%.

---

### Scene: classroom | Condition: Baseline B — Sparse + SIFT | FAILED

**Date:** 2026-06-07
**Dataset:** `baseline_classroom_d_s` — 12 sparse images (converted from HEIC, no augmentation)
**Pose estimation:** `convert.py` (standard COLMAP SIFT exhaustive)
**Images:** 12 real-world indoor classroom photographs (iPhone, HEIC → JPEG)

```
Bundle adjustment:
    Residuals : 204  |  Parameters : 164  |  Iterations : 101
    Initial cost : 2.8255 px  |  Final cost : 0.325353 px
    Termination : No convergence
  => Filtered observations: 0
  => Filtered images: 0
  => No good initial image pair found.
Elapsed time: 0.772 [minutes]
ERROR: failed to create sparse model
ERROR: Mapper failed with code 256
```

**Outcome:** Complete reconstruction failure — identical pattern to train Baseline B. SIFT cannot establish sufficient matches across 12 sparse indoor images. Confirms that the sparse SIFT failure is not scene-specific but a systematic limitation of feature-based matching under sparse, wide-baseline capture conditions.

---

### Scene: classroom | Condition: Ablation — Sparse + AI (no augmentation) | SfM COMPLETE — Training Pending

**Date:** 2026-06-07
**Dataset:** `baseline_classroom_d_s` — 12 sparse images, no augmentation
**Pose estimation:** `convert_ai.py` (SuperPoint + LightGlue exhaustive)

```
SfM (convert_ai.py):
    Reconstruction: 12/12 cameras registered | 1,150 COLMAP points
    num_observations         : 2,898
    mean_track_length        : 2.52
    mean_observations/image  : 241.5
    mean_reprojection_error  : 1.183 px
    Undistortion: 12/12 complete

Init points: — (pending train.py)
```

**Notes:** All 12 cameras registered successfully — SuperPoint + LightGlue succeeds where SIFT failed completely. Point cloud (1,150 points) is lower than train ablation (1,562 points), consistent with the indoor scene having textureless surfaces (walls, ceiling, uniform floor) that yield fewer repeatable keypoints. Mean reprojection error of 1.183 px is acceptable.

```
Training (30,000 iterations):
    Init points : 1,150
    [7000]  Test PSNR: 12.085 dB  |  Train PSNR: 30.558 dB
    [30000] Test PSNR: 11.730 dB  |  Train PSNR: 39.405 dB

metrics.py (2 test views):
    PSNR : 11.7219 dB
    SSIM : 0.3810
    LPIPS: 0.6180  (VGG network)
```

**Notes:** Severe overfitting confirmed — train/test gap of 27.7 dB at iter 30000, and test PSNR *degraded* from iter 7000 (12.085) to iter 30000 (11.730) as the model memorised training viewpoints. Mirrors the train ablation pattern (33.2 dB gap). Slightly better absolute PSNR than train ablation (11.72 vs 9.60) but worse LPIPS (0.618 vs 0.563), reflecting the classroom's textureless surfaces (walls, ceiling) which are perceptually harder to reconstruct. Establishes the lower bound for the classroom scene — ViewCrafter augmentation (Proposed) should improve significantly.

---
