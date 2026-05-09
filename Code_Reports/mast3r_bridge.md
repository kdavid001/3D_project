# MASt3R Pose Bridge — `mast3r_bridge.py`

> **STATUS: ⏸ NOT IN USE (DEFERRED)** — Architecturally complete but not integrated into the current experiment. Deferred to Future Work. Use `inject_poses.py` for the current pipeline.

---

## What This Does (vs inject_poses.py)

`inject_poses.py` assumes real photos are distributed on an equator — a reasonable approximation for object-centric NeRF datasets but inaccurate for real-world photos taken at arbitrary positions.

`mast3r_bridge.py` replaces that equatorial assumption with **MASt3R** (Matching And Stereo 3D Reconstruction) — a neural network that estimates true camera poses from real photos without any feature matching or SfM. It is the more accurate and more general replacement.

| Feature | inject_poses.py | mast3r_bridge.py |
|---|---|---|
| Real photo poses | Equatorial assumption | MASt3R neural estimation |
| Requires SfM | No | No |
| Point cloud | Fibonacci sphere (synthetic) | MASt3R dense, confidence-filtered |
| Accuracy | Approximate | Near-SfM quality |
| Natural scenes | Limited (assumes equator) | Yes |
| Installation | None | MASt3R + DUST3R required |
| Status | IN USE | Deferred |

---

## Architecture (4 Phases)

### Phase 1 — MASt3R Inference
All N real anchor photos fed simultaneously through MASt3R global alignment (all C(N,2) pairs). Outputs per-camera (R, t, focal, cx, cy) in a shared world frame plus a dense confidence-filtered point cloud. All N cameras registered by construction — no SfM collapse possible.

### Phase 2 — Scene Normalisation
MASt3R's world frame is arbitrary (scale, rotation, translation). The script:
1. Translates the point-cloud centroid to the origin
2. Estimates world "up" from the mean of camera Y-axes
3. Applies Rodrigues rotation to align estimated up → canonical Y-up
4. Transforms all camera poses and points consistently

This is critical: `spherical_pose()` (used for synth views) assumes Y-up convention. Without this normalisation, synth view placement would be geometrically wrong.

### Phase 3 — Synthetic View Registration
Each anchor camera now has a well-defined (azimuth, elevation, radius) in the normalised frame. Synthetic views are placed at Zero123++ v1.2 fixed offsets using the same `spherical_pose()` math as `inject_poses.py`.

### Phase 4 — COLMAP Binary Output
Writes `cameras.bin`, `images.bin`, `points3D.bin`, `pose_manifest.json` — identical format to `inject_poses.py`, consumed directly by `train.py`.

---

## Key Functions

```python
run_mast3r(real_paths, device, ckpt, niter, conf_threshold, num_init_points)
# → cameras (list of dicts), pts [M,3], cols [M,3]

estimate_world_up(cameras)
# → unit vector: estimated world up direction (−mean camera-Y in world space)

align_to_y_up(up_estimated)
# → 3×3 rotation: maps estimated_up → [0,1,0]  (Rodrigues formula)

normalise_scene(cameras, pts)
# → cameras_norm, pts_norm, mean_radius

camera_to_spherical(C_world)
# → (az_deg, el_deg, radius)

c2w_to_colmap(R_c2w, C_world)
# → R_w2c, t_w2c  (MASt3R c2w → COLMAP w2c)
```

---

## Installation (Colab)

```python
!git clone --recursive https://github.com/naver/mast3r /content/mast3r
!pip install -e "/content/mast3r[demo]" -q
!pip install roma -q

import sys
sys.path.insert(0, '/content/mast3r')
sys.path.insert(0, '/content/mast3r/dust3r')
```

---

## Usage

```bash
python mast3r_bridge.py \
    --source_path /content/local_workspace/human_heart \
    --images_dir  input \
    --num_init_points 5000
```

---

## Why It's Deferred

MASt3R integration was tested in earlier runs but the outputs were not good enough on the actual test dataset at that stage of the project. Decision was made to step back to the original pipeline (inject_poses.py + equatorial assumption) for the thesis experiment, which works correctly on the hotdog NeRF dataset. MASt3R remains the correct architectural choice for **natural real-world scenes** where the equatorial assumption does not hold, and is documented as Future Work in Chapter 5.

---

## Thesis Reference

Described in Chapter 3 under "Pose Estimation and COLMAP Integration". The failure of COLMAP SfM on black-background images (2/28 cameras registered) is the motivation. `mast3r_bridge.py` is the full solution; `inject_poses.py` is the working approximation used for the current experiment.
