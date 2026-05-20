# Algorithm: Analytic Pose Injector (DISCARDED)
# File: inject_poses.py
# Status: DISCARDED — convert_ai.py is now used for all pipeline modes

---

> **Note on Status:** This algorithm is documented for completeness and thesis reference.
> inject_poses.py was removed from the active pipeline because convert_ai.py (SuperPoint + LightGlue SfM)
> was found to produce better results across all three modes. The analytic pose injection approach
> is preserved here as a methodological reference.

---

**Algorithm 7: ANALYTIC_POSE_INJECTION**

**Background — Zero123++ Fixed View Offsets:**
Zero123++ v1.2 generates 6 synthetic views at analytically known relative poses from each anchor:

| View | Azimuth Offset | Elevation |
|------|---------------|-----------|
|  v0  |     +30°      |   +20°    |
|  v1  |     +90°      |   −10°    |
|  v2  |     +150°     |   +20°    |
|  v3  |     +210°     |   −10°    |
|  v4  |     +270°     |   +20°    |
|  v5  |     +330°     |   −10°    |

**Input:**
- source_path     : root dataset folder
- images_dir      : subfolder containing anchor_*.png + synth_*.png (default: "input")
- real_azimuths   : list of azimuth angles for real anchor photos (default: evenly spaced)
- real_elevation  : elevation for real photos in degrees (default: 0.0)
- radius          : orbital camera radius (default: 2.0)
- image_size      : square resolution in pixels (default: 1024)
- num_init_points : number of Fibonacci sphere points for initial point cloud (default: 500)

**Output:**
- sparse/0/cameras.bin     : two PINHOLE camera models (real + synthetic)
- sparse/0/images.bin      : all images with injected poses
- sparse/0/points3D.bin    : Fibonacci sphere point cloud
- sparse/0/pose_manifest.json : human-readable camera pose summary

---

```
BEGIN ANALYTIC_POSE_INJECTION(source_path, images_dir, real_azimuths,
                               real_elevation, radius, image_size, num_init_points)

  images_path ← source_path / images_dir
  output_path ← source_path / "sparse" / "0"
  CREATE output_path

  // -------------------------------------------------------
  // Step 1: Classify images by filename prefix
  // -------------------------------------------------------
  all_images ← LIST all files in images_path

  real_images  ← FILTER all_images WHERE filename STARTS WITH "anchor_"
  synth_images ← FILTER all_images WHERE filename MATCHES "synth_{base}_v{n}.png"

  // -------------------------------------------------------
  // Step 2: Define two PINHOLE camera models
  // -------------------------------------------------------
  // Real anchor photos have a different FOV to Zero123++ synthetic outputs
  cam_real ← {
    id: 1, model: PINHOLE,
    focal_length: image_size,        // ≈ 53° FOV
    cx: image_size/2, cy: image_size/2
  }
  cam_synth ← {
    id: 2, model: PINHOLE,
    focal_length: derived from 49.13° Objaverse FOV,
    cx: image_size/2, cy: image_size/2
  }

  WRITE cam_real, cam_synth to cameras.bin

  // -------------------------------------------------------
  // Step 3: Assign poses to real anchor images (equatorial)
  // -------------------------------------------------------
  IF real_azimuths is NOT PROVIDED THEN
    real_azimuths ← LINSPACE(0°, 360°, COUNT(real_images), endpoint=False)
  END IF

  image_records ← EMPTY LIST

  FOR i, anchor in ENUMERATE(real_images) DO
    az_rad ← real_azimuths[i] in RADIANS
    el_rad ← real_elevation in RADIANS

    // Convert spherical (az, el, r) → Cartesian camera centre
    C_world ← [
      radius × COS(el_rad) × COS(az_rad),
      radius × SIN(el_rad),
      radius × COS(el_rad) × SIN(az_rad)
    ]

    // Construct camera rotation: look-at origin from C_world, Y-up convention
    R_c2w ← LOOK_AT(origin=[0,0,0], camera_pos=C_world, up=[0,1,0])

    // Convert camera-to-world (c2w) → COLMAP world-to-camera (w2c)
    R_w2c ← TRANSPOSE(R_c2w)
    t_w2c ← -R_w2c × C_world

    APPEND { filename:anchor, camera_id:1, R:R_w2c, t:t_w2c } to image_records

  END FOR

  // -------------------------------------------------------
  // Step 4: Assign poses to synthetic views (fixed offsets)
  // -------------------------------------------------------
  Zero123_offsets ← [
    {az_offset:+30°, el:+20°},   // v0
    {az_offset:+90°, el:−10°},   // v1
    {az_offset:+150°,el:+20°},   // v2
    {az_offset:+210°,el:−10°},   // v3
    {az_offset:+270°,el:+20°},   // v4
    {az_offset:+330°,el:−10°}    // v5
  ]

  FOR each synth_file in synth_images DO
    base, v_idx ← PARSE "synth_{base}_v{v_idx}" from filename

    // Look up the azimuth of the corresponding anchor
    anchor_az ← real_azimuths[ INDEX of base in real_images ]

    offset ← Zero123_offsets[v_idx]
    az_rad ← (anchor_az + offset.az_offset) in RADIANS
    el_rad ← offset.el in RADIANS

    C_world ← [
      radius × COS(el_rad) × COS(az_rad),
      radius × SIN(el_rad),
      radius × COS(el_rad) × SIN(az_rad)
    ]
    R_c2w ← LOOK_AT(origin=[0,0,0], camera_pos=C_world, up=[0,1,0])
    R_w2c ← TRANSPOSE(R_c2w)
    t_w2c ← -R_w2c × C_world

    APPEND { filename:synth_file, camera_id:2, R:R_w2c, t:t_w2c } to image_records

  END FOR

  WRITE image_records to images.bin

  // -------------------------------------------------------
  // Step 5: Generate initial point cloud (Fibonacci sphere)
  // -------------------------------------------------------
  // Fibonacci lattice guarantees near-uniform surface coverage
  // for 3DGS Adaptive Density Control initialisation
  points ← EMPTY LIST
  golden_ratio ← (1 + SQRT(5)) / 2

  FOR i in RANGE(0, num_init_points) DO
    theta ← ARCCOS(1 - 2×(i + 0.5) / num_init_points)
    phi   ← 2π × i / golden_ratio
    x ← 0.5 × SIN(theta) × COS(phi)   // radius 0.5 = half scene extent
    y ← 0.5 × COS(theta)
    z ← 0.5 × SIN(theta) × SIN(phi)
    APPEND (x, y, z) to points
  END FOR

  WRITE points to points3D.bin

  // -------------------------------------------------------
  // Step 6: Write human-readable debug manifest
  // -------------------------------------------------------
  WRITE { all camera poses + filenames } to pose_manifest.json

END ANALYTIC_POSE_INJECTION
```

---

**Why This Was Discarded:**
- The equatorial assumption for real anchor photo poses is an approximation; actual photo azimuths may differ
- convert_ai.py uses data-driven SfM (SuperPoint + LightGlue) which recovers true camera geometry from the images themselves
- For all three pipeline modes (synthesis, restoration, natural), convert_ai.py was found to produce better-registered camera models than the analytic injection approach
