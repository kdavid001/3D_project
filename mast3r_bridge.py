#!/usr/bin/env python3
"""
mast3r_bridge.py — MASt3R Pose Estimation + Zero123++ Registration → COLMAP Binary
=====================================================================================
Replaces BOTH convert_ai.py and inject_poses.py in one script.

Architecture
------------
Phase 1 — MASt3R (real photos only):
  All N real anchor photos fed simultaneously through MASt3R global alignment.
  MASt3R predicts per-camera (R, t, focal, cx, cy) in a shared world frame
  plus a dense, confidence-filtered point cloud.  No feature matching.
  No SfM collapse.  All N cameras registered by construction.

Phase 2 — Scene normalisation:
  The MASt3R world frame is arbitrary (scale, rotation, translation).
  We translate so the point-cloud centroid is at the origin, then rotate so
  the estimated world-up (−mean camera-Y) aligns with the COLMAP/3DGS Y-up
  convention.  All camera poses and points are transformed consistently.

Phase 3 — Zero123++ synthetic view registration:
  Each anchor camera now has a well-defined (azimuth, elevation, radius) in the
  normalised world frame.  Synthetic views are placed at the known Zero123++ v1.2
  offsets from each anchor using the same spherical_pose() math as inject_poses.py.
  Because the world frame is now Y-up, all downstream code (train.py, viewer) is
  geometrically consistent.

Phase 4 — COLMAP binary output:
  Writes cameras.bin, images.bin, points3D.bin + pose_manifest.json.
  Consumed directly by train.py (original or Decoupled-Supervision version).

MASt3R installation (run once in Colab):
  !git clone --recursive https://github.com/naver/mast3r /content/mast3r
  !pip install -e "/content/mast3r[demo]" -q
  !pip install roma -q

  Then add to sys.path before calling this script:
    import sys
    sys.path.insert(0, '/content/mast3r')
    sys.path.insert(0, '/content/mast3r/dust3r')

Usage:
  python mast3r_bridge.py \\
      --source_path /content/local_workspace/human_heart \\
      --images_dir  input \\
      [--mast3r_ckpt  naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric] \\
      [--conf_threshold  1.5] \\
      [--num_init_points 5000] \\
      [--niter 300] \\
      [--device cuda]
"""

import os
import sys
import struct
import json
import math
import argparse
import numpy as np
from pathlib import Path
from itertools import combinations

# ── Zero123++ v1.2 relative-pose offsets ──────────────────────────────────────
# Azimuth offset (degrees) relative to anchor's azimuth.
# Elevation (degrees) absolute (relative to equatorial plane).
ZERO123_OFFSETS = [
    ( 30.0, +20.0),   # v0
    ( 90.0, -10.0),   # v1
    (150.0, +20.0),   # v2
    (210.0, -10.0),   # v3
    (270.0, +20.0),   # v4
    (330.0, -10.0),   # v5
]

OBJAVERSE_FOV_DEG = 49.13   # Zero123++ training FOV → synthetic camera intrinsics


# ══════════════════════════════════════════════════════════════════════════════
# 1.  MASt3R INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

def run_mast3r(real_paths: list,
               device: str = "cuda",
               ckpt: str = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric",
               niter: int = 300,
               conf_threshold: float = 1.5,
               num_init_points: int = 5000):
    """
    Run MASt3R global alignment on all real anchor photos simultaneously.

    Returns
    -------
    cameras : list of dicts, length = len(real_paths)
        {
          'path'   : str           absolute image path,
          'R_c2w'  : np.ndarray [3,3]  camera-to-world rotation (MASt3R frame),
          'C_world': np.ndarray [3]    camera centre in world space,
          'f'      : float         focal length in ORIGINAL pixel units,
          'cx'     : float         principal point x (original pixels),
          'cy'     : float         principal point y (original pixels),
          'W'      : int,  'H': int    original image resolution,
        }
    pts   : np.ndarray [M, 3]  float64  point cloud in MASt3R world space
    cols  : np.ndarray [M, 3]  uint8    RGB colours for each point
    """
    try:
        from mast3r.model import AsymmetricMASt3R
        from dust3r.inference import inference
        from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
        from dust3r.utils.image import load_images
    except ImportError as e:
        raise ImportError(
            f"MASt3R import failed: {e}\n"
            "Install with:\n"
            "  !git clone --recursive https://github.com/naver/mast3r /content/mast3r\n"
            "  !pip install -e '/content/mast3r[demo]' -q && pip install roma -q\n"
            "Then add to sys.path:\n"
            "  sys.path.insert(0, '/content/mast3r')\n"
            "  sys.path.insert(0, '/content/mast3r/dust3r')"
        )

    from PIL import Image as PILImage

    print(f"[MASt3R] Loading checkpoint: {ckpt}")
    model = AsymmetricMASt3R.from_pretrained(ckpt).to(device).eval()

    print(f"[MASt3R] Loading {len(real_paths)} images (resized to 512 for inference) ...")
    mast3r_imgs = load_images(real_paths, size=512)

    # Cache original resolutions for focal-length rescaling
    orig_dims = {}
    for p in real_paths:
        w, h = PILImage.open(p).size
        orig_dims[p] = (w, h)

    # All-pairs inference: C(N, 2) pairs
    n = len(mast3r_imgs)
    if n < 2:
        raise ValueError("Need at least 2 real anchor photos for MASt3R.")
    pairs = [(mast3r_imgs[i], mast3r_imgs[j])
             for i, j in combinations(range(n), 2)]
    print(f"[MASt3R] {len(pairs)} image pairs → running inference ...")

    import torch
    with torch.no_grad():
        output = inference(pairs, model, device, batch_size=1, verbose=False)

    print(f"[MASt3R] Global alignment (PointCloudOptimizer, niter={niter}) ...")
    scene = global_aligner(
        output, device=device,
        mode=GlobalAlignerMode.PointCloudOptimizer,
        verbose=False,
    )
    scene.compute_global_alignment(
        init="mst", niter=niter, schedule="cosine", lr=0.01
    )

    # ── Extract per-camera poses and intrinsics ────────────────────────────────
    poses_c2w = scene.get_im_poses().detach().cpu().numpy()   # [N, 4, 4]
    focals    = scene.get_focals().detach().cpu().numpy()     # [N] at MASt3R scale
    pps       = scene.get_principal_points().detach().cpu().numpy()  # [N, 2]

    cameras = []
    for i, path in enumerate(real_paths):
        W_orig, H_orig = orig_dims[path]
        # true_shape = (H_resized, W_resized) at MASt3R's 512-max scale
        H_mast3r = int(mast3r_imgs[i]['true_shape'][0])
        W_mast3r = int(mast3r_imgs[i]['true_shape'][1])

        # Scale focal + pp from MASt3R resolution → original resolution
        sx = W_orig / float(W_mast3r)
        sy = H_orig / float(H_mast3r)
        s  = (sx + sy) / 2.0   # symmetric scale (images are usually square)

        M       = poses_c2w[i]          # [4, 4] camera-to-world
        R_c2w   = M[:3, :3].copy()      # camera axes in world space
        C_world = M[:3,  3].copy()      # camera centre in world space

        cameras.append({
            'path':    path,
            'R_c2w':   R_c2w,
            'C_world': C_world,
            'f':  float(focals[i]) * s,
            'cx': float(pps[i, 0]) * sx,
            'cy': float(pps[i, 1]) * sy,
            'W':  W_orig,
            'H':  H_orig,
        })

    # ── Extract confidence-filtered point cloud ────────────────────────────────
    import torch as _torch
    pts3d_maps = scene.get_pts3d()          # list of [H, W, 3] tensors
    try:
        conf_maps = scene.get_conf()        # list of [H, W] tensors
    except AttributeError:
        # Older DUST3R API fallback: accept all points
        conf_maps = [_torch.ones(p.shape[:2]) for p in pts3d_maps]

    scene_imgs = scene.imgs                 # list of np.ndarray [H, W, 3] ∈ [0,1]

    all_pts, all_cols = [], []
    for pts, conf, img in zip(pts3d_maps, conf_maps, scene_imgs):
        pts_np  = pts.detach().cpu().numpy().reshape(-1, 3)
        conf_np = conf.detach().cpu().numpy().reshape(-1)
        col_np  = (img.reshape(-1, 3) * 255).clip(0, 255).astype(np.uint8)

        mask = conf_np > conf_threshold
        if mask.sum() == 0:
            # Lower threshold gracefully if no points survive
            mask = conf_np > (conf_threshold * 0.5)
        all_pts.append(pts_np[mask])
        all_cols.append(col_np[mask])

    pts  = np.concatenate(all_pts,  axis=0).astype(np.float64)
    cols = np.concatenate(all_cols, axis=0)
    print(f"[MASt3R] {len(pts):,} high-confidence points before subsampling.")

    if len(pts) > num_init_points:
        idx  = np.random.choice(len(pts), num_init_points, replace=False)
        pts  = pts[idx]
        cols = cols[idx]
    print(f"[MASt3R] Using {len(pts):,} init points for 3DGS.")

    return cameras, pts, cols


# ══════════════════════════════════════════════════════════════════════════════
# 2.  WORLD-FRAME NORMALISATION
# ══════════════════════════════════════════════════════════════════════════════

def estimate_world_up(cameras: list) -> np.ndarray:
    """
    Estimate world 'up' direction as − mean(camera-Y in world space).

    In OpenCV convention, camera Y points DOWN. The world 'up' vector is
    therefore the negative of each camera's Y axis expressed in world space.
    For a typical set of object-centric photos this converges to a stable
    estimate of the gravity direction.
    """
    ups = []
    for cam in cameras:
        cam_y_world = cam['R_c2w'][:, 1]   # second column = camera-Y in world
        ups.append(-cam_y_world)            # negate → world up
    up = np.mean(ups, axis=0)
    norm = np.linalg.norm(up)
    if norm < 1e-6:
        return np.array([0.0, 1.0, 0.0])
    return up / norm


def align_to_y_up(up_estimated: np.ndarray) -> np.ndarray:
    """
    Return a 3×3 rotation matrix R_align such that
        R_align @ up_estimated  ≈  [0, 1, 0]

    Rodrigues' rotation formula.  Used to rotate the entire MASt3R world frame
    so the estimated up axis maps to the canonical Y-up convention that
    inject_poses.py / spherical_pose() assume.
    """
    target = np.array([0.0, 1.0, 0.0])
    c = float(np.dot(up_estimated, target))

    if c > 1.0 - 1e-6:
        return np.eye(3)                              # already aligned
    if c < -1.0 + 1e-6:
        return np.diag([1.0, -1.0, -1.0])            # anti-aligned: flip Y

    v   = np.cross(up_estimated, target)              # rotation axis (not normalised)
    s   = np.linalg.norm(v)
    kx  = np.array([[    0, -v[2],  v[1]],
                    [ v[2],     0, -v[0]],
                    [-v[1],  v[0],     0]])
    R   = np.eye(3) + kx + kx @ kx * ((1.0 - c) / (s * s))
    return R


def normalise_scene(cameras: list, pts: np.ndarray):
    """
    1. Translate so the point-cloud centroid is at the world origin.
    2. Rotate the world frame so the estimated 'up' aligns with Y-up.
    3. Compute mean orbital radius (used for synthetic-view placement).

    Returns modified cameras + points IN PLACE (copies returned).
    """
    centroid   = pts.mean(axis=0)
    up_raw     = estimate_world_up(cameras)
    R_align    = align_to_y_up(up_raw)

    # Shift + rotate points
    pts_norm   = (R_align @ (pts - centroid).T).T

    # Shift + rotate camera poses
    cameras_norm = []
    for cam in cameras:
        C_new   = R_align @ (cam['C_world'] - centroid)
        R_c2w_new = R_align @ cam['R_c2w']
        cam_n   = dict(cam)
        cam_n['C_world'] = C_new
        cam_n['R_c2w']   = R_c2w_new
        cameras_norm.append(cam_n)

    # Compute orbital radius = mean distance from normalised cameras to origin
    radii      = [np.linalg.norm(c['C_world']) for c in cameras_norm]
    mean_radius = float(np.mean(radii))
    print(f"[Normalise] Centroid translated to origin.")
    print(f"[Normalise] Up-vector aligned to Y-axis.")
    print(f"[Normalise] Mean orbital radius: {mean_radius:.4f}")

    return cameras_norm, pts_norm, mean_radius


# ══════════════════════════════════════════════════════════════════════════════
# 3.  COORDINATE MATHEMATICS  (same convention as inject_poses.py)
# ══════════════════════════════════════════════════════════════════════════════

def camera_to_spherical(C_world: np.ndarray) -> tuple:
    """
    Convert a camera-centre position (Y-up, origin = object centre) to
    (azimuth_deg, elevation_deg, radius).

    Convention matches inject_poses.py / spherical_pose():
      azimuth  : 0° = +Z axis, increases counter-clockwise when viewed from above.
      elevation: positive = above equator.
    """
    r  = np.linalg.norm(C_world)
    if r < 1e-8:
        return 0.0, 0.0, 0.0
    d  = C_world / r
    el = math.degrees(math.asin(float(np.clip(d[1], -1.0, 1.0))))   # Y-up elevation
    az = math.degrees(math.atan2(float(d[0]), float(d[2])))          # atan2(X, Z)
    return az, el, r


def spherical_pose(az_deg: float, el_deg: float, radius: float) -> tuple:
    """
    World-to-camera (R, t) for a camera on a sphere of `radius`, looking inward.

    Copied verbatim from inject_poses.py so the two scripts stay in sync.
    """
    az  = math.radians(az_deg)
    el  = math.radians(el_deg)

    pos = radius * np.array([
        math.cos(el) * math.sin(az),
        math.sin(el),
        math.cos(el) * math.cos(az),
    ])

    cam_z = -pos / np.linalg.norm(pos)

    world_up = np.array([0.0, 1.0, 0.0])
    if abs(float(np.dot(cam_z, world_up))) > 0.99:
        world_up = np.array([0.0, 0.0, 1.0])

    cam_x = np.cross(cam_z, world_up)
    cam_x /= np.linalg.norm(cam_x)
    cam_y = np.cross(cam_x, cam_z)

    R = np.stack([cam_x, cam_y, cam_z], axis=0)   # world-to-camera
    t = -R @ pos
    return R, t


def c2w_to_colmap(R_c2w: np.ndarray, C_world: np.ndarray) -> tuple:
    """
    Convert MASt3R camera-to-world (R_c2w, C_world) →
    COLMAP world-to-camera (R_w2c, t_w2c).
    """
    R_w2c = R_c2w.T
    t_w2c = -R_w2c @ C_world
    return R_w2c, t_w2c


def rot_to_quat(R: np.ndarray) -> np.ndarray:
    """
    3×3 rotation matrix → [qw, qx, qy, qz].  Shepperd's method.
    (Identical to inject_poses.py.)
    """
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s  = 0.5 / math.sqrt(trace + 1.0)
        qw, qx = 0.25 / s, (R[2, 1] - R[1, 2]) * s
        qy, qz = (R[0, 2] - R[2, 0]) * s, (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s  = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        qw, qx = (R[2, 1] - R[1, 2]) / s, 0.25 * s
        qy, qz = (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s  = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        qw, qx = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s
        qy, qz = 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s  = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        qw, qx = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s
        qy, qz = (R[1, 2] + R[2, 1]) / s, 0.25 * s
    return np.array([qw, qx, qy, qz])


# ══════════════════════════════════════════════════════════════════════════════
# 4.  COLMAP BINARY WRITERS  (format-identical to inject_poses.py)
# ══════════════════════════════════════════════════════════════════════════════

def write_cameras_bin(path: Path, cam_list: list):
    with open(path, 'wb') as f:
        f.write(struct.pack('<Q', len(cam_list)))
        for cam in cam_list:
            f.write(struct.pack('<I', cam['camera_id']))
            f.write(struct.pack('<i', 1))                      # PINHOLE model
            f.write(struct.pack('<QQ', cam['width'], cam['height']))
            for p in [cam['f'], cam['f'], cam['cx'], cam['cy']]:
                f.write(struct.pack('<d', float(p)))


def write_images_bin(path: Path, img_list: list):
    with open(path, 'wb') as f:
        f.write(struct.pack('<Q', len(img_list)))
        for img in img_list:
            f.write(struct.pack('<I', img['image_id']))
            for q in img['qvec']:
                f.write(struct.pack('<d', float(q)))
            for t in img['tvec']:
                f.write(struct.pack('<d', float(t)))
            f.write(struct.pack('<I', img['camera_id']))
            f.write(img['name'].encode('utf-8') + b'\x00')
            f.write(struct.pack('<Q', 0))          # num_points2d = 0


def write_points3d_bin(path: Path, pts: np.ndarray, cols: np.ndarray):
    """
    pts  : [M, 3] float64   world-space XYZ (already normalised)
    cols : [M, 3] uint8     RGB
    """
    M = len(pts)
    with open(path, 'wb') as f:
        f.write(struct.pack('<Q', M))
        for i in range(M):
            x, y, z = float(pts[i, 0]), float(pts[i, 1]), float(pts[i, 2])
            r, g, b = int(cols[i, 0]), int(cols[i, 1]), int(cols[i, 2])
            f.write(struct.pack('<Q',   i + 1))       # point3d_id
            f.write(struct.pack('<ddd', x, y, z))
            f.write(struct.pack('<BBB', r, g, b))
            f.write(struct.pack('<d',   1.0))          # reprojection error
            f.write(struct.pack('<Q',   0))            # track_length = 0


# ══════════════════════════════════════════════════════════════════════════════
# 5.  IMAGE CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════

def classify_images(image_dir: Path):
    """
    Classify images in image_dir using anchor_/synth_ prefixes.

    Returns
    -------
    real_paths  : list[str]   absolute paths to real anchor photos
    synth_meta  : list[dict]
        Each dict: { 'path': str, 'anchor_base': str, 'v_idx': int }
    """
    valid = {'.jpg', '.jpeg', '.png'}
    all_files = sorted(
        f for f in image_dir.iterdir() if f.suffix.lower() in valid
    )

    real_paths = []
    synth_meta = []

    for f in all_files:
        stem = f.stem
        if stem.lower().startswith('anchor_'):
            real_paths.append(str(f))
        elif stem.lower().startswith('synth_') and '_v' in stem:
            split = stem.rfind('_v')
            try:
                v_idx = int(stem[split + 2:])
                anchor_base = stem[len('synth_'):split]
                synth_meta.append({
                    'path':        str(f),
                    'anchor_base': anchor_base,
                    'v_idx':       v_idx,
                })
            except ValueError:
                pass   # ignore files like synth_foo_v_weird

    synth_meta.sort(key=lambda x: (x['anchor_base'], x['v_idx']))
    return real_paths, synth_meta


# ══════════════════════════════════════════════════════════════════════════════
# 6.  MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def build_colmap_scene(source_path: str,
                       images_dir:   str   = "input",
                       mast3r_ckpt:  str   = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric",
                       conf_threshold: float = 1.5,
                       num_init_points: int  = 5000,
                       niter:         int   = 300,
                       device:        str   = "cuda"):

    source   = Path(source_path)
    img_dir  = source / images_dir
    out_dir  = source / "sparse" / "0"
    out_dir.mkdir(parents=True, exist_ok=True)

    SEP = "=" * 70
    print(SEP)
    print("  mast3r_bridge.py — MASt3R Pose Estimation + Zero123++ Registration")
    print(SEP)
    print(f"  Source : {source}")
    print(f"  Images : {img_dir}")
    print(f"  Output : {out_dir}")

    # ── Classify images ───────────────────────────────────────────────────────
    real_paths, synth_meta = classify_images(img_dir)

    if not real_paths:
        raise FileNotFoundError(
            f"No anchor_ prefixed images found in {img_dir}.\n"
            "Ensure your real photos are named anchor_*.jpg/png."
        )

    print(f"\n[Classification]")
    print(f"  Real  (anchor_*) : {len(real_paths)}")
    print(f"  Synth (synth_*)  : {len(synth_meta)}")

    # ── Phase 1: MASt3R ───────────────────────────────────────────────────────
    print(f"\n[Phase 1] MASt3R inference on {len(real_paths)} real photos ...")
    cameras, pts, cols = run_mast3r(
        real_paths,
        device         = device,
        ckpt           = mast3r_ckpt,
        niter          = niter,
        conf_threshold = conf_threshold,
        num_init_points= num_init_points,
    )

    # ── Phase 2: Normalise scene ──────────────────────────────────────────────
    print(f"\n[Phase 2] Normalising world frame (centred + Y-up) ...")
    cameras, pts, mean_radius = normalise_scene(cameras, pts)

    # Compute spherical coords for each anchor (now in normalised world frame)
    for cam in cameras:
        az, el, r = camera_to_spherical(cam['C_world'])
        cam['az'], cam['el'], cam['r'] = az, el, r

    # ── Phase 3: Build anchor → synth lookup ──────────────────────────────────
    anchor_name_to_cam = {}
    for cam in cameras:
        base = Path(cam['path']).stem.replace('anchor_', '')
        anchor_name_to_cam[base] = cam

    # ── Phase 4: Construct COLMAP entries ─────────────────────────────────────
    print(f"\n[Phase 3] Constructing COLMAP binary entries ...")

    # Camera models
    # Real: mean MASt3R intrinsics (shared model; close enough for 3DGS init)
    mean_f  = float(np.mean([c['f']  for c in cameras]))
    mean_cx = float(np.mean([c['cx'] for c in cameras]))
    mean_cy = float(np.mean([c['cy'] for c in cameras]))
    W_real  = cameras[0]['W']
    H_real  = cameras[0]['H']

    cam_entries = [
        {'camera_id': 1, 'width': W_real, 'height': H_real,
         'f': mean_f, 'cx': mean_cx, 'cy': mean_cy},
    ]

    # Synthetic: Objaverse 49.13° FOV
    if synth_meta:
        from PIL import Image as PILImage
        sw, sh = PILImage.open(synth_meta[0]['path']).size
        f_synth = sw / (2.0 * math.tan(math.radians(OBJAVERSE_FOV_DEG / 2.0)))
        cam_entries.append(
            {'camera_id': 2, 'width': sw, 'height': sh,
             'f': f_synth, 'cx': sw / 2.0, 'cy': sh / 2.0}
        )

    img_entries = []
    manifest    = []
    img_id      = 1

    # ── Real photos (world-to-camera from MASt3R) ──────────────────────────
    print("\n[Real Photo Poses]  (from MASt3R global alignment)")
    for cam in cameras:
        R_w2c, t_w2c = c2w_to_colmap(cam['R_c2w'], cam['C_world'])
        qvec = rot_to_quat(R_w2c)

        name = Path(cam['path']).name
        img_entries.append({
            'image_id':  img_id,
            'qvec':      qvec.tolist(),
            'tvec':      t_w2c.tolist(),
            'camera_id': 1,
            'name':      name,
        })
        manifest.append({
            'image_id': img_id, 'type': 'real', 'name': name,
            'az': cam['az'], 'el': cam['el'], 'r': float(cam['r']),
        })
        print(f"  [Real  #{img_id:03d}] {name:<60s} "
              f"az={cam['az']:+7.2f}°  el={cam['el']:+6.2f}°  r={cam['r']:.3f}")
        img_id += 1

    # ── Synthetic views (analytically derived) ────────────────────────────
    print("\n[Synthetic View Poses]  (Zero123++ offsets from MASt3R anchor poses)")
    for sm in synth_meta:
        anchor_cam = anchor_name_to_cam.get(sm['anchor_base'])
        if anchor_cam is None:
            print(f"  [WARN] No anchor found for '{sm['anchor_base']}', skipping.")
            continue

        d_az, d_el = ZERO123_OFFSETS[sm['v_idx']]
        az_s  = anchor_cam['az'] + d_az
        el_s  = anchor_cam['el'] + d_el
        r_s   = mean_radius

        R_w2c, t_w2c = spherical_pose(az_s, el_s, r_s)
        qvec = rot_to_quat(R_w2c)

        name = Path(sm['path']).name
        img_entries.append({
            'image_id':  img_id,
            'qvec':      qvec.tolist(),
            'tvec':      t_w2c.tolist(),
            'camera_id': 2,
            'name':      name,
        })
        manifest.append({
            'image_id': img_id, 'type': 'synth', 'name': name,
            'az': az_s, 'el': el_s, 'r': r_s,
            'anchor': sm['anchor_base'], 'v_idx': sm['v_idx'],
        })
        print(f"  [Synth #{img_id:03d}] {name:<60s} "
              f"az={az_s:+7.2f}°  el={el_s:+6.2f}°  "
              f"← {sm['anchor_base']}+v{sm['v_idx']}")
        img_id += 1

    # ── Phase 5: Write binary files ───────────────────────────────────────────
    print(f"\n[Phase 4] Writing COLMAP sparse/0/ ...")
    write_cameras_bin (out_dir / "cameras.bin",  cam_entries)
    write_images_bin  (out_dir / "images.bin",   img_entries)
    write_points3d_bin(out_dir / "points3D.bin", pts, cols)

    with open(out_dir / "pose_manifest.json", 'w') as f:
        json.dump(manifest, f, indent=2)

    # ── Summary ───────────────────────────────────────────────────────────────
    n_real  = len(real_paths)
    n_synth = img_id - 1 - n_real
    print()
    print(SEP)
    print("  COLMAP Sparse Reconstruction Written")
    print(SEP)
    print(f"  cameras.bin   → {len(cam_entries)} camera models  "
          f"(real PINHOLE + synth PINHOLE)")
    print(f"  images.bin    → {img_id - 1} registered images  "
          f"({n_real} real  +  {n_synth} synthetic)")
    print(f"  points3D.bin  → {len(pts):,} init points  "
          f"(MASt3R dense, conf > {conf_threshold})")
    print(f"  pose_manifest.json → human-readable debug summary")
    print(f"\n  Output: {out_dir}")
    print(f"\n  Next step:")
    print(f'    python train.py -s "{source_path}" -m <output_path> \\')
    print(f"        --opacity_reset_interval 9000")
    print(f"    (Do NOT pass --eval)")
    print(SEP)


# ══════════════════════════════════════════════════════════════════════════════
# 7.  CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="MASt3R-driven COLMAP binary generation for sparse 3DGS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--source_path",     required=True,
                   help="Root dataset folder (must contain images_dir subfolder).")
    p.add_argument("--images_dir",      default="input",
                   help="Subfolder with all images (default: input).")
    p.add_argument("--mast3r_ckpt",
                   default="naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric",
                   help="HuggingFace checkpoint for MASt3R.")
    p.add_argument("--conf_threshold",  type=float, default=1.5,
                   help="MASt3R confidence threshold for point-cloud filtering (default 1.5).")
    p.add_argument("--num_init_points", type=int,   default=5000,
                   help="Max init points in points3D.bin (default 5000).")
    p.add_argument("--niter",           type=int,   default=300,
                   help="Global alignment iterations (default 300).")
    p.add_argument("--device",          default="cuda",
                   help="Torch device (default: cuda).")
    args = p.parse_args()

    build_colmap_scene(
        source_path     = args.source_path,
        images_dir      = args.images_dir,
        mast3r_ckpt     = args.mast3r_ckpt,
        conf_threshold  = args.conf_threshold,
        num_init_points = args.num_init_points,
        niter           = args.niter,
        device          = args.device,
    )
