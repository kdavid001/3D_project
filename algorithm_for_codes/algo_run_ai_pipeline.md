# Algorithm: Auto Splats Cleaner — Early SfM Pipeline (Superseded)
# File: run_ai_pipeline.py
# Status: NOT IN USE — superseded by convert_ai.py

---

> **Note on Status:** run_ai_pipeline.py is an earlier iteration of the AI-powered COLMAP pipeline.
> It is functionally identical to convert_ai.py (Algorithm 6) and was superseded by it.
> The same SfM collapse limitation applies. Refer to Algorithm 6 for the current implementation.

---

**Algorithm 10: AUTO_SPLATS_SFM (Early Version)**

**Input:**
- source_path : root dataset folder (must contain input/ subfolder)
- images      : name of images subfolder (default: "input")

**Output:**
- sparse/0/cameras.bin  : COLMAP camera intrinsics
- sparse/0/images.bin   : camera poses
- sparse/0/points3D.bin : sparse 3D point cloud
- images/               : undistorted output images

---

```
BEGIN AUTO_SPLATS_SFM(source_path, images="input")

  images_dir    ← source_path / images
  distorted_dir ← source_path / "distorted"

  // Step 1: Cleanup
  DELETE distorted_dir if it exists
  DELETE source_path / "sparse" if it exists
  CREATE distorted_dir / "sparse"

  // Step 2: Neural Feature Extraction (SuperPoint)
  features_h5 ← distorted_dir / "features.h5"
  extract_features(
    conf       = "superpoint_aachen",
    image_dir  = images_dir,
    output     = features_h5
  )

  // Step 3: Exhaustive Pair Matching (LightGlue)
  pairs_txt   ← distorted_dir / "pairs.txt"
  matches_h5  ← distorted_dir / "matches.h5"

  pairs ← GENERATE all image pairs (exhaustive)
  WRITE pairs to pairs_txt

  match_features(
    conf     = "lightglue",
    pairs    = pairs_txt,
    features = features_h5,
    output   = matches_h5
  )

  // Step 4: SfM Reconstruction (PyCOLMAP)
  pycolmap.incremental_mapping(
    database_path = distorted_dir / "database.db",
    image_path    = images_dir,
    output_path   = distorted_dir / "sparse"
  )

  // Step 5: Image Undistortion
  RUN COLMAP image_undistorter {
    image_path  : images_dir,
    input_path  : distorted_dir / "sparse" / "0",
    output_path : source_path
  }

  // Step 6: Move .bin files to sparse/0/
  MOVE cameras.bin, images.bin, points3D.bin → source_path / "sparse" / "0"

END AUTO_SPLATS_SFM
```

---

**Replacement Chain:**

```
run_ai_pipeline.py   (this file — early version)
        ↓  replaced by
convert_ai.py        (cleaner implementation — see Algorithm 6)
```

See Algorithm 6 (convert_ai.py) for the active implementation and full technical notes.
The SfM collapse problem on black-background images applies equally to both versions.
