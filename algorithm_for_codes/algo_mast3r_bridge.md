# Algorithm: MASt3R Neural Pose Bridge (Deferred — Future Work)
# File: mast3r_bridge.py
# Status: DEFERRED — architecturally complete, not integrated in current experiment

---

> **Note on Status:** This algorithm is the full architectural solution for natural real-world scenes
> where the equatorial pose assumption (inject_poses.py) does not hold.
> Deferred due to insufficient results on the current test dataset; documented as Future Work.

---

**Algorithm 8: MAST3R_POSE_BRIDGE**

**Input:**
- source_path     : root dataset folder
- images_dir      : subfolder containing anchor_*.png and synth_*.png
- num_init_points : number of points for point cloud initialisation (default: 5000)
- conf_threshold  : MASt3R confidence filter threshold (default: 1.5)

**Output:**
- sparse/0/cameras.bin      : COLMAP camera intrinsics (real + synthetic)
- sparse/0/images.bin       : all registered images with poses
- sparse/0/points3D.bin     : MASt3R dense point cloud (confidence-filtered)
- sparse/0/pose_manifest.json : human-readable camera pose summary

---

```
BEGIN MAST3R_POSE_BRIDGE(source_path, images_dir, num_init_points, conf_threshold)

  images_path ← source_path / images_dir
  real_paths  ← FILTER images WHERE filename STARTS WITH "anchor_"
  synth_paths ← FILTER images WHERE filename MATCHES "synth_*_v*.png"

  // =======================================================
  // PHASE 1 — MASt3R Inference (Neural Pose Estimation)
  // =======================================================
  // MASt3R performs global alignment over all C(N,2) image pairs simultaneously.
  // Outputs camera intrinsics + extrinsics in a shared arbitrary world frame.
  // No feature matching or SfM required — all cameras registered by construction.

  cameras, pts3D, colours ← run_mast3r(
    image_paths     = real_paths,
    device          = "cuda",
    niter           = 300,                // global alignment iterations
    conf_threshold  = conf_threshold      // filter low-confidence 3D points
  )
  // cameras: list of { R_c2w, C_world, focal, cx, cy } per real image
  // pts3D:   [M, 3] array of 3D point positions
  // colours: [M, 3] RGB colours for each point

  // =======================================================
  // PHASE 2 — Scene Normalisation (Y-up alignment)
  // =======================================================
  // MASt3R's world frame is arbitrary (scale, rotation, translation).
  // We normalise to a canonical frame: origin at point cloud centroid, Y-up.
  // This is required so Zero123++ fixed offsets are geometrically valid.

  // Step 2a: Translate centroid to origin
  centroid ← MEAN(pts3D, axis=0)
  pts3D    ← pts3D - centroid
  FOR each cam in cameras DO
    cam.C_world ← cam.C_world - centroid
  END FOR

  // Step 2b: Estimate world "up" direction from camera orientations
  up_estimated ← estimate_world_up(cameras)
  // Computes mean of (-camera_Y_axis) across all cameras in world frame

  // Step 2c: Rotate scene so estimated up aligns with canonical Y = [0,1,0]
  R_align ← align_to_y_up(up_estimated)
  // Uses Rodrigues' rotation formula: R = I + sin(θ)K + (1−cos(θ))K²

  pts3D ← (R_align × pts3D^T)^T
  FOR each cam in cameras DO
    cam.C_world  ← R_align × cam.C_world
    cam.R_c2w   ← R_align × cam.R_c2w
  END FOR

  mean_radius ← MEAN(NORM(cam.C_world) for cam in cameras)

  // =======================================================
  // PHASE 3 — Synthetic View Registration
  // =======================================================
  // Each anchor camera now has a well-defined (azimuth, elevation, radius)
  // in the normalised Y-up frame.
  // Zero123++ v1.2 fixed offsets are applied relative to each anchor's azimuth.

  Zero123_offsets ← [
    {az:+30°, el:+20°}, {az:+90°, el:−10°}, {az:+150°, el:+20°},
    {az:+210°,el:−10°}, {az:+270°, el:+20°}, {az:+330°,el:−10°}
  ]

  synth_records ← EMPTY LIST

  FOR each anchor_cam in cameras DO
    az_deg, el_deg, radius ← camera_to_spherical(anchor_cam.C_world)

    FOR v in {0,...,5} DO
      synth_az ← az_deg + Zero123_offsets[v].az
      synth_el ← el_deg + Zero123_offsets[v].el

      C_synth ← SPHERICAL_TO_CARTESIAN(synth_az, synth_el, mean_radius)
      R_c2w_synth ← LOOK_AT(origin=[0,0,0], camera_pos=C_synth, up=[0,1,0])

      // Convert c2w → COLMAP w2c convention
      R_w2c ← TRANSPOSE(R_c2w_synth)
      t_w2c ← -R_w2c × C_synth

      APPEND { filename:"synth_..._v{v}", R:R_w2c, t:t_w2c } to synth_records
    END FOR
  END FOR

  // =======================================================
  // PHASE 4 — COLMAP Binary Output
  // =======================================================
  // Convert real camera poses from MASt3R c2w to COLMAP w2c convention
  real_records ← EMPTY LIST
  FOR each cam in cameras DO
    R_w2c ← TRANSPOSE(cam.R_c2w)
    t_w2c ← -R_w2c × cam.C_world
    APPEND { filename:cam.name, camera_id:1, R:R_w2c, t:t_w2c } to real_records
  END FOR

  // Define camera models
  cam_real  ← { id:1, model:PINHOLE, focal:cam.focal, cx:cam.cx, cy:cam.cy }
  cam_synth ← { id:2, model:PINHOLE, focal:derived from 49.13° FOV, cx:..., cy:... }

  WRITE cam_real, cam_synth to cameras.bin
  WRITE CONCAT(real_records, synth_records) to images.bin
  WRITE pts3D, colours to points3D.bin
  WRITE all camera data to pose_manifest.json

END MAST3R_POSE_BRIDGE
```

---

**Comparison: inject_poses.py vs mast3r_bridge.py**

| Feature              | inject_poses.py             | mast3r_bridge.py              |
|----------------------|-----------------------------|-------------------------------|
| Real photo poses     | Equatorial assumption       | MASt3R neural estimation      |
| SfM required         | No                          | No                            |
| Point cloud          | Fibonacci sphere (synthetic)| MASt3R dense, filtered        |
| Accuracy             | Approximate                 | Near-SfM quality              |
| Natural scenes       | Limited                     | Yes                           |
| Status               | Discarded                   | Deferred (Future Work)        |

---

**Key Functions:**
- `run_mast3r()` — MASt3R global alignment inference
- `estimate_world_up()` — mean camera-Y to estimate gravity direction
- `align_to_y_up()` — Rodrigues rotation to canonical Y-up frame
- `normalise_scene()` — centroid translation + up alignment
- `camera_to_spherical()` — Cartesian camera centre → (az, el, radius)
- `c2w_to_colmap()` — MASt3R c2w format → COLMAP w2c format
