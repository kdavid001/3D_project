#!/usr/bin/env python3
"""
inject_poses.py — Analytic Camera Pose Injection for Zero123++ Generative Views
================================================================================
REPLACES convert_ai.py in the pipeline.

Problem solved:
  convert_ai.py collapses to 2/28 registered cameras because COLMAP feature
  matching cannot handle Zero123++'s black-background, micro-hallucinated views.

Solution:
  Zero123++ v1.2 generates views at FIXED, ANALYTICALLY KNOWN relative poses.
  This script writes a valid COLMAP sparse/0/ reconstruction directly,
  registering ALL images without any feature matching.

Camera assignment:
  - Real (anchor) photos  → distributed evenly on equator (or user-specified)
  - Zero123++ synth views → offset from their source photo by known azimuths/elevations

Pipeline position (replaces Step 3 in notebook cell-8):
  diffusion_script_v0.py  →  [SKIP rename.py]  →  inject_poses.py  →  train.py

Usage (preferred — original synth_/anchor_ names preserved):
    python inject_poses.py --source_path /content/local_workspace/human_heart

Usage (if rename.py already ran — sequential names):
    python inject_poses.py --source_path /content/local_workspace/human_heart \\
        --images_dir images --num_real 4

Optional — if you know the actual azimuth of each real photo:
    python inject_poses.py --source_path /path/to/scene \\
        --real_azimuths 0 90 180 270
"""

import os
import struct
import json
import argparse
import numpy as np
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Zero123++ v1.2: 6 fixed relative poses per input image
# Azimuth is RELATIVE to source real photo. Elevation is absolute (degrees).
# Source: sudo-ai/zero123plus-v1.2 training config / Appendix B of paper.
#
# Grid layout (2 cols × 3 rows, row-major crop order):
#   Row 0 Col 0 → v0,  Row 0 Col 1 → v1
#   Row 1 Col 0 → v2,  Row 1 Col 1 → v3
#   Row 2 Col 0 → v4,  Row 2 Col 1 → v5
# ─────────────────────────────────────────────────────────────────────────────
ZERO123_OFFSETS = [
    ( 30, +20),   # v0: front-right,  elevated
    ( 90, -10),   # v1: right side,   depressed
    (150, +20),   # v2: back-right,   elevated
    (210, -10),   # v3: back-left,    depressed
    (270, +20),   # v4: left side,    elevated
    (330, -10),   # v5: front-left,   depressed
]

# Objaverse render FOV used in Zero123++ training — gives correct intrinsics
ZERO123_FOV_DEG = 49.13

# Camera orbital radius (object sits at origin, scale normalised to 1.0)
CAMERA_RADIUS = 2.0


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def classify_by_prefix(image_dir: Path):
    """
    Classify images using the anchor_/synth_ naming convention from
    diffusion_script_v0.py (preferred — skip rename.py to preserve these).

    Returns
    -------
    real_photos : list[str]
        Filenames of real (anchor) photos, sorted alphabetically.
    synth_views : list[tuple[str, str, int]]
        (filename, source_base, view_idx) sorted by (source_base, view_idx).
    """
    valid = {'.jpg', '.jpeg', '.png'}
    files = sorted(f.name for f in image_dir.iterdir() if f.suffix.lower() in valid)

    real_photos = []
    synth_views = []

    for fname in files:
        stem = Path(fname).stem
        if stem.startswith('anchor_'):
            real_photos.append(fname)
        elif stem.startswith('synth_') and '_v' in stem:
            split_pos   = stem.rfind('_v')
            view_idx    = int(stem[split_pos + 2:])
            source_base = stem[len('synth_'):split_pos]
            synth_views.append((fname, source_base, view_idx))
        # Skip FULL_GRID debug images and other artefacts silently

    synth_views.sort(key=lambda x: (x[1], x[2]))
    return real_photos, synth_views


def classify_sequential(image_dir: Path, num_real: int):
    """
    Fallback when rename.py has already converted to sequential names (00001.jpg …).

    Assumption (enforced by rename.py's alphabetical sort):
      - First num_real files (alphabetically) = real anchor photos
      - Remaining files = synthetic views, ordered as all v0..v5 for photo 0,
        then v0..v5 for photo 1, etc.

    Returns same format as classify_by_prefix.
    """
    valid = {'.jpg', '.jpeg', '.png'}
    files = sorted(f.name for f in image_dir.iterdir() if f.suffix.lower() in valid)

    real_photos = files[:num_real]
    synth_files = files[num_real:]
    synth_views = []
    for i, fname in enumerate(synth_files):
        source_idx = i // 6
        view_idx   = i % 6
        synth_views.append((fname, f"photo_{source_idx:02d}", view_idx))

    return real_photos, synth_views


# ─────────────────────────────────────────────────────────────────────────────
# COORDINATE MATHEMATICS
# ─────────────────────────────────────────────────────────────────────────────

def spherical_pose(az_deg: float, el_deg: float, radius: float = CAMERA_RADIUS):
    """
    Compute world-to-camera (R, t) for a camera on a sphere of given radius,
    looking at the world origin.

    Convention: COLMAP / OpenCV
      Camera X = right,  Y = down,  Z = into scene (forward)
      Y-up world, right-handed coordinate system.

    Parameters
    ----------
    az_deg  : azimuth in degrees. 0° = +Z world axis. Increases counter-clockwise
              when viewed from above (i.e., az=90° → camera at +X world).
    el_deg  : elevation in degrees. Positive = above equator.
    radius  : distance from world origin to camera centre.

    Returns
    -------
    R : np.ndarray shape [3, 3]  world-to-camera rotation
    t : np.ndarray shape [3]     world-to-camera translation  (= -R @ cam_pos)
    """
    az  = np.radians(az_deg)
    el  = np.radians(el_deg)

    # Camera position on sphere (Y-up parameterisation)
    pos = radius * np.array([
        np.cos(el) * np.sin(az),   # X component
        np.sin(el),                 # Y component (height)
        np.cos(el) * np.cos(az),   # Z component
    ])

    # Camera Z axis: points FROM camera TOWARD origin (into scene)
    cam_z = -pos / np.linalg.norm(pos)

    # Gimbal-lock guard at poles (|cam_z · world_up| ≈ 1)
    world_up = np.array([0.0, 1.0, 0.0])
    if abs(float(np.dot(cam_z, world_up))) > 0.99:
        world_up = np.array([0.0, 0.0, 1.0])

    # Camera X (right) and Y (down) — derived from cam_z and world_up
    cam_x = np.cross(cam_z, world_up)
    cam_x /= np.linalg.norm(cam_x)
    cam_y = np.cross(cam_x, cam_z)        # right-handed: X × Z = down

    # World-to-camera rotation matrix: rows = camera basis vectors in world
    R = np.stack([cam_x, cam_y, cam_z], axis=0)   # shape [3, 3]
    t = -R @ pos
    return R, t


def rot_to_quat(R: np.ndarray) -> np.ndarray:
    """
    Convert 3×3 rotation matrix to unit quaternion [qw, qx, qy, qz].
    Uses Shepperd's numerically stable method.
    """
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s  = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (R[2, 1] - R[1, 2]) * s
        qy = (R[0, 2] - R[2, 0]) * s
        qz = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s  = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s  = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s  = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    return np.array([qw, qx, qy, qz])


# ─────────────────────────────────────────────────────────────────────────────
# POINT CLOUD INITIALISATION
# ─────────────────────────────────────────────────────────────────────────────

def fibonacci_sphere(n: int, radius: float = 0.5) -> np.ndarray:
    """
    Generate n near-uniformly distributed points on a sphere of given radius
    using the Fibonacci lattice (golden-angle) method.

    This is far superior to random sampling for 3DGS initialisation: it
    guarantees even coverage of the object volume so densification has
    starting Gaussians near every surface, not just the visible front face.

    Returns np.ndarray shape [n, 3].
    """
    golden = (1.0 + np.sqrt(5.0)) / 2.0
    i      = np.arange(n, dtype=float)
    theta  = np.arccos(1.0 - 2.0 * (i + 0.5) / n)
    phi    = 2.0 * np.pi * i / golden

    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)
    return np.stack([x, y, z], axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# COLMAP BINARY FORMAT WRITERS
# ─────────────────────────────────────────────────────────────────────────────

def write_cameras_bin(path: Path, cameras: list):
    """
    Write cameras.bin in COLMAP binary format.

    cameras : list of dicts with keys:
        camera_id : int
        model_id  : int   1 = PINHOLE  →  params = [fx, fy, cx, cy]
        width     : int
        height    : int
        params    : list[float]
    """
    with open(path, 'wb') as f:
        f.write(struct.pack('<Q', len(cameras)))
        for cam in cameras:
            f.write(struct.pack('<I', cam['camera_id']))
            f.write(struct.pack('<i', cam['model_id']))
            f.write(struct.pack('<Q', cam['width']))
            f.write(struct.pack('<Q', cam['height']))
            for p in cam['params']:
                f.write(struct.pack('<d', float(p)))


def write_images_bin(path: Path, images: list):
    """
    Write images.bin in COLMAP binary format.

    images : list of dicts with keys:
        image_id  : int
        qvec      : [qw, qx, qy, qz]   world-to-camera quaternion
        tvec      : [tx, ty, tz]        world-to-camera translation
        camera_id : int
        name      : str                 filename exactly as it appears in images/ dir
    """
    with open(path, 'wb') as f:
        f.write(struct.pack('<Q', len(images)))
        for img in images:
            f.write(struct.pack('<I', img['image_id']))
            for q in img['qvec']:
                f.write(struct.pack('<d', float(q)))
            for t in img['tvec']:
                f.write(struct.pack('<d', float(t)))
            f.write(struct.pack('<I', img['camera_id']))
            f.write(img['name'].encode('utf-8') + b'\x00')
            f.write(struct.pack('<Q', 0))   # num_points2D = 0 (no 2D-3D links)


def write_points3d_bin(path: Path, points: list):
    """
    Write points3D.bin in COLMAP binary format.

    points : list of dicts with keys:
        point3d_id : int
        xyz        : [x, y, z]
        rgb        : [r, g, b]   uint8
        error      : float
    """
    with open(path, 'wb') as f:
        f.write(struct.pack('<Q', len(points)))
        for pt in points:
            f.write(struct.pack('<Q', pt['point3d_id']))
            for v in pt['xyz']:
                f.write(struct.pack('<d', float(v)))
            for c in pt['rgb']:
                f.write(struct.pack('<B', int(c)))
            f.write(struct.pack('<d', float(pt['error'])))
            f.write(struct.pack('<Q', 0))   # track_length = 0


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Inject analytic camera poses for Zero123++ views into COLMAP format.\n"
            "Replaces convert_ai.py — no feature matching required."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--source_path', type=Path, required=True,
        help='Root dataset folder (must contain an images/ or input/ subfolder).'
    )
    parser.add_argument(
        '--images_dir', type=str, default='input',
        help='Subfolder containing all images (default: input). '
             'Use "images" if you have already run the 3DGS undistorter.'
    )
    parser.add_argument(
        '--num_real', type=int, default=None,
        help='Number of real photos. REQUIRED if rename.py already ran and destroyed '
             'the anchor_/synth_ prefixes. First --num_real files (alphabetically) '
             'are treated as real photos.'
    )
    parser.add_argument(
        '--real_azimuths', type=float, nargs='+', default=None,
        help='Explicit azimuth angles (degrees) for each real photo, in the same '
             'alphabetical order as the files. If omitted, photos are distributed '
             'evenly around the equator. Override this with measured or estimated '
             'angles for better geometric accuracy.'
    )
    parser.add_argument(
        '--real_elevation', type=float, default=0.0,
        help='Elevation angle for real photos in degrees (default 0 = equator). '
             'Increase if photos were taken from slightly above the object.'
    )
    parser.add_argument(
        '--radius', type=float, default=CAMERA_RADIUS,
        help=f'Camera orbital radius in scene units (default {CAMERA_RADIUS}). '
             'Object is assumed to sit at the origin.'
    )
    parser.add_argument(
        '--image_size', type=int, default=1024,
        help='Image width AND height in pixels (default 1024 after RealESRGAN upscale). '
             'Assumed square. Both camera models use this resolution.'
    )
    parser.add_argument(
        '--num_init_points', type=int, default=500,
        help='Number of initialisation points in the synthetic point cloud (default 500). '
             'More points → richer Gaussian initialisation → better densification.'
    )
    args = parser.parse_args()

    # ── Resolve paths ─────────────────────────────────────────────────────────
    images_dir = args.source_path / args.images_dir
    sparse_dir = args.source_path / 'sparse' / '0'

    if not images_dir.exists():
        raise FileNotFoundError(
            f"Images directory not found: {images_dir}\n"
            f"Check --images_dir (currently '{args.images_dir}'). "
            f"Common values: 'input' (before undistortion) or 'images' (after)."
        )

    sparse_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*70}")
    print(f"  inject_poses.py — Analytic Zero123++ Pose Injection")
    print(f"{'='*70}")
    print(f"  Source: {args.source_path}")
    print(f"  Images: {images_dir}")
    print(f"  Output: {sparse_dir}\n")

    # ── 1. CLASSIFY IMAGES ────────────────────────────────────────────────────
    if args.num_real is not None:
        print(f"[Mode] Sequential naming detected (--num_real={args.num_real})")
        real_photos, synth_views = classify_sequential(images_dir, args.num_real)
    else:
        print("[Mode] Prefix naming (anchor_/synth_) — rename.py was skipped ✓")
        real_photos, synth_views = classify_by_prefix(images_dir)

    n_real  = len(real_photos)
    n_synth = len(synth_views)

    if n_real == 0:
        raise RuntimeError(
            "No real photos found.\n"
            "If rename.py already ran, pass --num_real N.\n"
            "If using prefix naming, ensure files start with 'anchor_'."
        )

    print(f"\n[Classification]")
    print(f"  Real photos  : {n_real}")
    print(f"  Synth views  : {n_synth}  ({n_synth // 6} batches × 6 views)")
    print(f"  Total        : {n_real + n_synth}")

    # ── 2. ASSIGN REAL PHOTO AZIMUTHS ─────────────────────────────────────────
    if args.real_azimuths is not None:
        if len(args.real_azimuths) != n_real:
            raise ValueError(
                f"--real_azimuths has {len(args.real_azimuths)} values "
                f"but there are {n_real} real photos."
            )
        real_azimuths = list(args.real_azimuths)
        print(f"\n[Real Photo Poses]  Using user-specified azimuths.")
    else:
        real_azimuths = [360.0 * i / n_real for i in range(n_real)]
        print(f"\n[Real Photo Poses]  Distributing evenly around equator.")
        print(f"  (Pass --real_azimuths A1 A2 ... if you know actual angles.)")

    # Map source_base → azimuth for synthetic view lookup
    base_to_azimuth = {}
    for i, fname in enumerate(real_photos):
        stem = Path(fname).stem
        base = stem[len('anchor_'):] if stem.startswith('anchor_') else f"photo_{i:02d}"
        base_to_azimuth[base] = real_azimuths[i]

    # ── 3. CAMERA INTRINSIC MODELS ────────────────────────────────────────────
    # Two shared PINHOLE camera models:
    #   camera_id=1  →  real photos        (focal length = image width, ~53° FOV)
    #   camera_id=2  →  Zero123++ synthetics (focal length from known 49.13° FOV)
    #
    # Using two separate models is critical: real photos and Zero123++ frames
    # have genuinely different optical properties and must not share intrinsics.

    W  = args.image_size
    H  = args.image_size
    cx = W / 2.0
    cy = H / 2.0

    # Zero123++ intrinsic — derived from Objaverse training FOV
    f_zero123 = W / (2.0 * np.tan(np.radians(ZERO123_FOV_DEG / 2.0)))

    # Real photo intrinsic — conservative default (width ≈ focal length ≈ 53° FOV)
    # Override by passing --real_azimuths or by running DUSt3R on the real photos first
    f_real = float(W)

    camera_real  = dict(camera_id=1, model_id=1, width=W, height=H,
                        params=[f_real, f_real, cx, cy])
    camera_synth = dict(camera_id=2, model_id=1, width=W, height=H,
                        params=[f_zero123, f_zero123, cx, cy])

    print(f"\n[Camera Models]")
    print(f"  Real   (id=1): PINHOLE  f={f_real:.1f}px   ({2*np.degrees(np.arctan(W/(2*f_real))):.1f}° FOV)")
    print(f"  Synth  (id=2): PINHOLE  f={f_zero123:.1f}px  ({ZERO123_FOV_DEG:.2f}° FOV, Objaverse)")

    # ── 4. BUILD IMAGE ENTRIES ────────────────────────────────────────────────
    images   = []
    image_id = 1

    print(f"\n[Registering Images]")

    # Real photos
    for i, fname in enumerate(real_photos):
        az = real_azimuths[i]
        el = args.real_elevation
        R, t  = spherical_pose(az, el, args.radius)
        qvec  = rot_to_quat(R)
        images.append(dict(image_id=image_id, qvec=qvec.tolist(),
                           tvec=t.tolist(), camera_id=1, name=fname))
        print(f"  [Real  #{image_id:03d}] {fname:<45s}  az={az:6.1f}°  el={el:+.1f}°")
        image_id += 1

    # Synthetic views
    for fname, source_base, view_idx in synth_views:
        # Resolve source azimuth
        if source_base in base_to_azimuth:
            source_az = base_to_azimuth[source_base]
        else:
            try:
                photo_idx = int(source_base.split('_')[-1])
                source_az = real_azimuths[photo_idx % n_real]
            except (ValueError, IndexError):
                print(f"  [WARN] Cannot resolve source azimuth for {fname} — defaulting to 0°")
                source_az = 0.0

        # Apply Zero123++ offset: azimuth wraps mod 360°, elevation is absolute
        rel_az, abs_el = ZERO123_OFFSETS[view_idx]
        abs_az = (source_az + rel_az) % 360.0

        R, t  = spherical_pose(abs_az, abs_el, args.radius)
        qvec  = rot_to_quat(R)
        images.append(dict(image_id=image_id, qvec=qvec.tolist(),
                           tvec=t.tolist(), camera_id=2, name=fname))
        print(f"  [Synth #{image_id:03d}] {fname:<45s}  az={abs_az:6.1f}°  el={abs_el:+d}°"
              f"  ← {source_base} + v{view_idx}")
        image_id += 1

    # ── 5. POINT CLOUD INITIALISATION ────────────────────────────────────────
    # Fibonacci sphere: near-uniform distribution over the object's bounding volume.
    # 3DGS initialises one Gaussian per point — more points → richer starting density
    # → more effective densification under the depth prior loss.
    #
    # Using radius=0.5 (half the scene extent) so Gaussians initialise ON the object
    # surface rather than at the origin (which would cause immediate pruning).
    np.random.seed(42)
    pts      = fibonacci_sphere(args.num_init_points, radius=0.5)
    points3d = []
    for i, xyz in enumerate(pts):
        rgb = np.clip(np.random.randint(100, 200, size=3), 0, 255).tolist()
        points3d.append(dict(point3d_id=i+1, xyz=xyz.tolist(),
                             rgb=rgb, error=1.0))

    # ── 6. WRITE COLMAP BINARY FILES ─────────────────────────────────────────
    cameras_path   = sparse_dir / 'cameras.bin'
    images_path    = sparse_dir / 'images.bin'
    points3d_path  = sparse_dir / 'points3D.bin'
    manifest_path  = sparse_dir / 'pose_manifest.json'

    write_cameras_bin (cameras_path,  [camera_real, camera_synth])
    write_images_bin  (images_path,   images)
    write_points3d_bin(points3d_path, points3d)

    # Human-readable manifest for debugging and the thesis appendix
    manifest = {
        'camera_models': {
            'real_photos'     : camera_real,
            'synthetic_views' : camera_synth,
        },
        'zero123_offsets': [
            {'view_idx': i, 'rel_azimuth_deg': az, 'elevation_deg': el}
            for i, (az, el) in enumerate(ZERO123_OFFSETS)
        ],
        'images': [
            {
                'image_id'    : img['image_id'],
                'name'        : img['name'],
                'camera_id'   : img['camera_id'],
                'type'        : 'real' if img['camera_id'] == 1 else 'synthetic',
                'qvec'        : img['qvec'],
                'tvec'        : img['tvec'],
            }
            for img in images
        ],
        'num_init_points' : len(points3d),
        'init_cloud_type' : 'fibonacci_sphere_r0.5',
    }
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    # ── 7. SUMMARY ────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  COLMAP Sparse Reconstruction Written")
    print(f"{'='*70}")
    print(f"  cameras.bin   → {2} camera models  (real PINHOLE + synth PINHOLE)")
    print(f"  images.bin    → {len(images)} registered images  ({n_real} real + {n_synth} synthetic)")
    print(f"  points3D.bin  → {len(points3d)} init points  (Fibonacci sphere, r=0.5)")
    print(f"  pose_manifest.json → human-readable debug summary")
    print(f"\n  Output: {sparse_dir}")
    print(f"\n  Next step:")
    print(f"    python train.py -s \"{args.source_path}\" -m <output_path>")
    print(f"    (Remove --eval flag. Add --opacity_reset_interval 9000)")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
