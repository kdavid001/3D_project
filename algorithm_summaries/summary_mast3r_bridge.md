# Algorithm Summary: MASt3R Neural Pose Bridge
# Source: mast3r_bridge.py | Full algorithm: algorithm_for_codes/algo_mast3r_bridge.md
# Status: DEFERRED — architecturally complete; documented as Future Work

---

**Algorithm 8 (Summary): MAST3R_POSE_BRIDGE**

**Input:**
- source_path    : root dataset folder (input/ contains anchor_* and synth_*)
- conf_threshold : MASt3R 3D point confidence filter (default: 1.5)
- num_init_points: not used — MASt3R provides real point cloud

**Output:**
- sparse/0/cameras.bin  : PINHOLE models (real + synthetic)
- sparse/0/images.bin   : all images with MASt3R-derived poses
- sparse/0/points3D.bin : MASt3R dense point cloud (confidence-filtered)

---

```
BEGIN MAST3R_POSE_BRIDGE(source_path, conf_threshold)

  real_paths  ← FILTER input/ WHERE filename STARTS WITH "anchor_"
  synth_paths ← FILTER input/ WHERE filename MATCHES "synth_*_v*.png"

  // Phase 1: Neural pose estimation (MASt3R global alignment)
  // All C(N,2) image pairs processed simultaneously — no SfM required
  cameras, pts3D, colours ← run_mast3r(
    images         = real_paths,
    niter          = 300,
    conf_threshold = conf_threshold
  )

  // Phase 2: Scene normalisation (arbitrary MASt3R frame → canonical Y-up)
  centroid ← MEAN(pts3D)
  pts3D    ← pts3D - centroid
  cameras  ← cameras - centroid      // translate to origin

  up_est ← MEAN of (-camera_Y_axis) across all cameras
  R_align ← RODRIGUES_ROTATION(up_est → [0,1,0])

  pts3D   ← R_align × pts3D         // rotate scene
  cameras ← R_align × cameras       // rotate all camera poses

  // Phase 3: Register synthetic views using Zero123++ fixed offsets
  Zero123_offsets ← [
    v0:(+30°,+20°), v1:(+90°,−10°), v2:(+150°,+20°),
    v3:(+210°,−10°),v4:(+270°,+20°), v5:(+330°,−10°)
  ]

  FOR each anchor_cam in cameras DO
    az, el, r ← CARTESIAN_TO_SPHERICAL(anchor_cam.C_world)
    FOR v in {0..5} DO
      C_synth ← SPHERICAL_TO_CARTESIAN(az + offset.az, el + offset.el, mean_r)
      R_w2c, t_w2c ← INVERT(LOOK_AT(C_synth))
      APPEND { synth_v, camera_id:2, R_w2c, t_w2c }
    END FOR
  END FOR

  // Phase 4: Write COLMAP binary output
  WRITE cameras.bin  { cam_real (id:1), cam_synth (id:2) }
  WRITE images.bin   { real_records + synth_records }
  WRITE points3D.bin { pts3D, colours }

END MAST3R_POSE_BRIDGE
```

---

**Comparison vs inject_poses.py:**

| Feature          | Algorithm 7 (inject)  | Algorithm 8 (MASt3R)      |
|------------------|-----------------------|---------------------------|
| Real photo poses | Equatorial assumption | MASt3R neural estimation  |
| Point cloud      | Fibonacci sphere      | MASt3R dense, filtered    |
| Natural scenes   | Limited               | Yes                       |
| Status           | Discarded             | Deferred (Future Work)    |

---
