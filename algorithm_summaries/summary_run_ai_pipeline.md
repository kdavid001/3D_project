# Algorithm Summary: Auto Splats SfM (Early Version)
# Source: run_ai_pipeline.py | Full algorithm: algorithm_for_codes/algo_run_ai_pipeline.md
# Status: SUPERSEDED — replaced by convert_ai.py (Algorithm 6)

---

**Algorithm 10 (Summary): AUTO_SPLATS_SFM**

> Functionally identical to Algorithm 6 (convert_ai.py). Same SuperPoint + LightGlue
> exhaustive matching → PyCOLMAP SfM → COLMAP undistortion pipeline.
> Refer to summary_convert_ai.md for the active implementation.

**Input / Output:** Same as Algorithm 6 (AI_SFM_POSE_ESTIMATION)

---

```
BEGIN AUTO_SPLATS_SFM(source_path, images="input")

  // Step 1: Cleanup stale artifacts
  DELETE distorted_dir, sparse_dir if they exist
  CREATE distorted_dir / "sparse"

  // Step 2: Neural feature extraction (SuperPoint)
  extract_features(conf="superpoint_aachen", image_dir, output=features.h5)

  // Step 3: Exhaustive pair matching (LightGlue)
  pairs ← GENERATE all_pairs from images
  match_features(conf="lightglue", pairs, features → matches.h5)

  // Step 4: SfM bundle adjustment (PyCOLMAP)
  pycolmap.incremental_mapping(database.db, images, output=distorted/sparse)

  // Step 5: Lens undistortion (COLMAP CLI)
  RUN COLMAP image_undistorter → source_path/images/ + source_path/sparse/0/

  // Step 6: Move .bin files to sparse/0/
  MOVE cameras.bin, images.bin, points3D.bin → sparse/0/

END AUTO_SPLATS_SFM
```

---

**Superseded by:** Algorithm 6 (convert_ai.py) — same logic, cleaner implementation.
Same black-background SfM collapse limitation applies to both versions.

---
