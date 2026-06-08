# Algorithm Summary: Analytic Pose Injector
# Source: inject_poses.py | Full algorithm: algorithm_for_codes/algo_inject_poses.md
# Status: DISCARDED — convert_ai.py is used for all pipeline modes

---

**Algorithm 7 (Summary): ANALYTIC_POSE_INJECTION**

**Input:**
- source_path    : root dataset folder
- real_azimuths  : azimuth angles for real anchor photos
- real_elevation : elevation for real photos (default: 0.0°)
- radius         : orbital camera radius (default: 2.0)
- num_init_points: Fibonacci sphere point count (default: 500)

**Output:**
- sparse/0/cameras.bin     : two PINHOLE camera models (real + synthetic)
- sparse/0/images.bin      : all images with injected camera poses
- sparse/0/points3D.bin    : Fibonacci sphere point cloud

---

```
BEGIN ANALYTIC_POSE_INJECTION(source_path, real_azimuths, real_elevation, radius, ...)

  real_images  ← FILTER input/ WHERE filename STARTS WITH "anchor_"
  synth_images ← FILTER input/ WHERE filename MATCHES "synth_*_v*.png"

  // Step 1: Define two PINHOLE cameras (real FOV ≠ Zero123++ FOV)
  cam_real  ← PINHOLE { focal: image_size,        id: 1 }
  cam_synth ← PINHOLE { focal: 49.13° Objaverse,  id: 2 }
  WRITE cameras.bin

  // Step 2: Assign equatorial poses to real anchors
  IF real_azimuths not provided → LINSPACE(0°, 360°, N_anchors)

  FOR each anchor A_i DO
    C_world ← SPHERICAL_TO_CARTESIAN(real_azimuths[i], real_elevation, radius)
    R_c2w   ← LOOK_AT(origin=[0,0,0], pos=C_world, up=[0,1,0])
    R_w2c, t_w2c ← INVERT(R_c2w, C_world)   // COLMAP w2c convention
    APPEND { A_i, camera_id:1, R_w2c, t_w2c }
  END FOR

  // Step 3: Assign fixed-offset poses to Zero123++ synthetic views
  Zero123_offsets ← [
    v0:(+30°,+20°), v1:(+90°,−10°), v2:(+150°,+20°),
    v3:(+210°,−10°),v4:(+270°,+20°), v5:(+330°,−10°)
  ]

  FOR each synth_file DO
    base, v_idx ← PARSE "synth_{base}_v{v_idx}"
    anchor_az   ← real_azimuths[ INDEX of base ]
    offset      ← Zero123_offsets[v_idx]
    C_world ← SPHERICAL_TO_CARTESIAN(anchor_az + offset.az, offset.el, radius)
    R_w2c, t_w2c ← INVERT(LOOK_AT(C_world))
    APPEND { synth_file, camera_id:2, R_w2c, t_w2c }
  END FOR

  WRITE images.bin

  // Step 4: Generate Fibonacci sphere initial point cloud
  FOR i in RANGE(num_init_points) DO
    theta ← ARCCOS(1 - 2×(i+0.5)/N)
    phi   ← 2π×i / golden_ratio
    APPEND point (0.5×SIN(θ)COS(φ), 0.5×COS(θ), 0.5×SIN(θ)SIN(φ))
  END FOR
  WRITE points3D.bin

END ANALYTIC_POSE_INJECTION
```

---

**Why Discarded:**
- Equatorial anchor pose assumption is an approximation of real photo positions
- convert_ai.py recovers true camera geometry from image content via SfM
- Superseded by Algorithm 6 for all three pipeline modes

---
