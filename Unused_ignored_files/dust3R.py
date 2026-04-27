import os
import argparse
import torch
import numpy as np
from PIL import Image

from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

# 👇 THE MISSING IMPORT: This is required to turn file paths into Neural Tensors
from dust3r.utils.image import load_images


# --- THE MATH ENGINE ---
def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]
    ]) / 3.0
    vals, vecs = np.linalg.eigh(K)
    q = vecs[:, np.argmax(vals)]
    if q[3] < 0:
        q *= -1
    return np.array([q[3], q[0], q[1], q[2]])


# --- THE MANUAL BRIDGE ---
def export_colmap_manual(scene, out_dir, image_paths):
    os.makedirs(out_dir, exist_ok=True)

    print("      -> Extracting poses and focal lengths...")
    focals = scene.get_focals().detach().cpu().numpy()
    poses = scene.get_im_poses().detach().cpu().numpy()
    pts3d = scene.get_pts3d()
    masks = scene.get_masks()

    print("      -> Writing cameras.txt...")
    with open(os.path.join(out_dir, "cameras.txt"), "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for i, focal in enumerate(focals):
            img = Image.open(image_paths[i])
            w, h = img.size
            f_val = focal.item() if focal.size == 1 else focal[0]
            focal_pixel = f_val * max(w, h)
            f.write(f"{i + 1} PINHOLE {w} {h} {focal_pixel} {focal_pixel} {w / 2} {h / 2}\n")

    print("      -> Writing images.txt...")
    with open(os.path.join(out_dir, "images.txt"), "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for i, pose in enumerate(poses):
            w2c = np.linalg.inv(pose)
            R = w2c[:3, :3]
            t = w2c[:3, 3]
            qvec = rotmat2qvec(R)
            img_name = os.path.basename(image_paths[i])
            f.write(f"{i + 1} {qvec[0]} {qvec[1]} {qvec[2]} {qvec[3]} {t[0]} {t[1]} {t[2]} {i + 1} {img_name}\n")
            f.write("\n")

    print("      -> Writing points3D.txt (Sparsified)...")
    with open(os.path.join(out_dir, "points3D.txt"), "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")

        point_id = 1
        for i in range(len(image_paths)):
            valid_pts = masks[i].detach().cpu().numpy() > 0.5
            p3d = pts3d[i].detach().cpu().numpy()[valid_pts]

            H, W = masks[i].shape
            img = Image.open(image_paths[i]).convert('RGB').resize((W, H))
            colors = np.array(img)[valid_pts]

            subsample = 100
            p3d = p3d[::subsample]
            colors = colors[::subsample]

            for pt, c in zip(p3d, colors):
                f.write(f"{point_id} {pt[0]} {pt[1]} {pt[2]} {c[0]} {c[1]} {c[2]} 0.0\n")
                point_id += 1


def main(args):
    print("🚀 [1/4] Booting DUSt3R Neural Engine...")
    model = AsymmetricCroCo3DStereo.from_pretrained("naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt").to("cuda")
    model.eval()

    print(f"📂 [2/4] Loading Hybrid Data from: {args.input_dir}")
    image_paths = [os.path.join(args.input_dir, f) for f in sorted(os.listdir(args.input_dir)) if
                   f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # 👇 THE BUG FIX: Load images into Tensors before pairing
    print(f"      -> Converting {len(image_paths)} images to Neural Tensors...")
    images = load_images(image_paths, size=512)

    # 👇 THE 14-HOUR TRAP FIX:
    # If you have ~260 images, use a "swin-3" (sliding window) graph so it only compares
    # sequential frames, dropping the pairs from 69,000 down to roughly 1,500.
    print("      -> Creating optimized scene graph pairs...")

    # Check if we have a massive dataset or just a small FYP test
    if len(image_paths) > 50:
        graph_type = "swin-3"  # Fast mode for heavy datasets
        print("      -> ⚠️ Massive dataset detected! Using Sliding Window graph to save hours of compute.")
    else:
        graph_type = "complete"  # High accuracy mode for < 50 images

    pairs = make_pairs(images, scene_graph=graph_type, prefilter=None, symmetrize=True)

    print("🧠 [3/4] Running Neural 3D Alignment (Bypassing Physics Engine)...")
    with torch.no_grad():
        output = inference(pairs, model, device="cuda", batch_size=2)
        scene = global_aligner(output, device="cuda", mode=GlobalAlignerMode.PointCloudOptimizer)
        scene.compute_global_alignment(init="mst", niter=300, schedule="linear", lr=0.01)

    print(f"💾 [4/4] Manually generating COLMAP Database at: {args.out_dir}")
    export_colmap_manual(scene, args.out_dir, image_paths)

    print("✅✅ BRIDGE COMPLETE! Ready for Gaussian Splatting.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Folder with original + SV3D images")
    parser.add_argument("--out_dir", required=True, help="Where to save the fake COLMAP database")
    args = parser.parse_args()
    main(args)