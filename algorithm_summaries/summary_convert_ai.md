# Algorithm Summary: AI-Powered Structure-from-Motion Pose Estimator
# Source: convert_ai.py | Full algorithm: algorithm_for_codes/algo_convert_ai.md

---

**Algorithm 6 (Summary): AI_SFM_POSE_ESTIMATION**
**Status: IN USE — primary pose estimator for all three pipeline modes**

**Input:**
- source_path : root dataset folder (must contain input/ subfolder)

**Output:**
- sparse/0/cameras.bin  : camera intrinsics
- sparse/0/images.bin   : camera extrinsics (R, t per image)
- sparse/0/points3D.bin : sparse 3D point cloud
- images/               : undistorted images for 3DGS training

---

```
BEGIN AI_SFM_POSE_ESTIMATION(source_path)

  images_dir ← source_path / "input"

  // Step 1: Cleanup stale artifacts
  DELETE distorted_dir, sparse_dir if they exist
  CREATE distorted_dir / "sparse"

  // Step 2: Neural feature extraction (SuperPoint)
  // Detects repeatable keypoints + 256-dim descriptors
  extract_features(
    conf      = "superpoint_aachen",
    image_dir = images_dir,
    output    = features.h5
  )

  // Step 3: Exhaustive pair matching (LightGlue)
  // All-pairs matching: each image matched against every other image
  // Attention GNN filters geometric outliers
  pairs ← GENERATE all_pairs from images_dir   // O(N²) — viable for small N
  match_features(conf="lightglue", pairs, features → matches.h5)

  // Step 4: SfM bundle adjustment (PyCOLMAP)
  reconstruction ← pycolmap.incremental_mapping(
    database = distorted_dir / "database.db",
    images   = images_dir,
    output   = distorted_dir / "sparse",
    features = features.h5,
    matches  = matches.h5
  )
  IF reconstruction is EMPTY → RAISE "SfM failed — no cameras registered"

  // Step 5: Lens undistortion (COLMAP CLI, headless via xvfb-run)
  // Writes final sparse/0/ + images/ for 3DGS training
  RUN COLMAP image_undistorter {
    input  : distorted_dir / "sparse" / "0",
    output : source_path,
    type   : "COLMAP"
  }

END AI_SFM_POSE_ESTIMATION
```

---

**Pipeline Position:**

```
diffusion_script_v0.py → anchor_*.png + synth_*_v*.png
        ↓
convert_ai.py → sparse/0/ + images/
        ↓
train.py (3D Gaussian Splatting)
```

---
