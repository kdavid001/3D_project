# Algorithm: AI-Powered Structure-from-Motion (SfM) Pose Estimator
# File: convert_ai.py
# Status: IN USE — primary pose estimator for all three pipeline modes

---

**Algorithm 6: AI_SFM_POSE_ESTIMATION**

**Input:**
- source_path : root dataset folder (must contain input/ subfolder with images)
- images      : name of images subfolder (default: "input")

**Output:**
- sparse/0/cameras.bin  : COLMAP camera intrinsics
- sparse/0/images.bin   : camera extrinsics (rotation + translation per image)
- sparse/0/points3D.bin : sparse 3D point cloud
- images/               : undistorted images for 3DGS training

---

```
BEGIN AI_SFM_POSE_ESTIMATION(source_path, images="input")

  images_dir   ← source_path / images
  distorted_dir← source_path / "distorted"
  sparse_dir   ← source_path / "sparse"
  output_dir   ← source_path / "images"

  // -------------------------------------------------------
  // Step 1: Cleanup — remove stale reconstruction artifacts
  // -------------------------------------------------------
  DELETE distorted_dir if it exists
  DELETE sparse_dir    if it exists
  CREATE distorted_dir / "sparse"

  // -------------------------------------------------------
  // Step 2: Feature Extraction (SuperPoint neural keypoints)
  // -------------------------------------------------------
  // SuperPoint detects repeatable interest points using a
  // fully convolutional neural network trained on homographic
  // correspondences. Outputs keypoints + 256-dim descriptors.
  features_path ← distorted_dir / "features.h5"

  extract_features(
    conf        = "superpoint_aachen",   // SuperPoint configuration
    image_dir   = images_dir,
    feature_path= features_path
  )

  // -------------------------------------------------------
  // Step 3: Feature Matching (LightGlue exhaustive matching)
  // -------------------------------------------------------
  // LightGlue performs all-pairs geometric matching:
  // matches each image against every other image.
  // Uses attention-based graph neural network to filter outliers.
  pairs_path  ← distorted_dir / "pairs.txt"
  matches_path← distorted_dir / "matches.h5"

  pairs ← GENERATE exhaustive_pairs from all images in images_dir
  WRITE pairs to pairs_path

  match_features(
    conf         = "lightglue",
    pairs        = pairs_path,
    features     = features_path,
    matches      = matches_path
  )

  // -------------------------------------------------------
  // Step 4: SfM Reconstruction (PyCOLMAP)
  // -------------------------------------------------------
  // COLMAP solves for camera positions and 3D structure
  // using the matched correspondences from Step 3.
  reconstruction ← pycolmap.incremental_mapping(
    database_path  = distorted_dir / "database.db",
    image_path     = images_dir,
    output_path    = distorted_dir / "sparse",
    features       = features_path,
    matches        = matches_path
  )

  IF reconstruction is EMPTY THEN
    RAISE ERROR "SfM failed — no cameras registered"
    // Likely cause: black-background images starving SuperPoint of gradients
  END IF

  // -------------------------------------------------------
  // Step 5: Image Undistortion (COLMAP CLI)
  // -------------------------------------------------------
  // Corrects radial/tangential lens distortion.
  // Outputs undistorted images + COLMAP binary format used by 3DGS.
  RUN COLMAP image_undistorter with {
    image_path    : images_dir,
    input_path    : distorted_dir / "sparse" / "0",
    output_path   : source_path,
    output_type   : "COLMAP"
  }
  // Writes to source_path/images/ and source_path/sparse/0/

END AI_SFM_POSE_ESTIMATION
```

---

**Pipeline Position:**

```
diffusion_script_v0.py
        ↓  produces anchor_*.png + synth_*_v*.png
convert_ai.py
        ↓  produces cameras.bin, images.bin, points3D.bin
train.py (3D Gaussian Splatting)
```

---

**Known Limitation:**
- SuperPoint/LightGlue requires textured images with distributed gradient information.
- Black-background images (composited objects) concentrate keypoints on the silhouette edge only, causing COLMAP to register as few as 2 out of 28 cameras — insufficient for 3DGS.
- For black-background datasets, the equatorial pose injection approach (inject_poses.py) provides a deterministic fallback that bypasses feature matching entirely.
