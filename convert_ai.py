import os
import argparse
from pathlib import Path
import shutil
import sys
import subprocess
import pycolmap

# --- 1. SETUP DEPENDENCIES ---
if not os.path.exists("SuperGluePretrainedNetwork"):
    print("Cloning missing dependency: SuperGluePretrainedNetwork...")
    subprocess.run(["git", "clone", "https://github.com/magicleap/SuperGluePretrainedNetwork.git"], check=True)

sys.path.append(os.path.abspath("SuperGluePretrainedNetwork"))

try:
    import hloc
except ImportError:
    print("Installing AI matching libraries...")
    subprocess.run(["pip", "install", "-q", "git+https://github.com/cvg/Hierarchical-Localization/"], check=True)

from hloc import extract_features, match_features, reconstruction, pairs_from_exhaustive

# --- 2. ARGUMENTS ---
parser = argparse.ArgumentParser()
parser.add_argument('--source_path', type=Path, required=True)
parser.add_argument('--images', type=str, default="input")
args = parser.parse_args()

# Paths
dataset_path = args.source_path
images_dir = dataset_path / args.images
outputs = dataset_path / 'distorted'
sparse_dir = outputs / 'sparse'

# --- 3. PRE-CLEANUP ---
# Clean up OLD run files to avoid conflicts, but keep 'stereo' if it exists.
print("🧹 Cleaning up old run files...")
for folder in ['distorted', 'sparse', 'images']:
    folder_path = dataset_path / folder
    
    if folder_path.is_symlink():
        print(f"   - Unlinking: {folder}")
        folder_path.unlink()
    elif folder_path.exists():
        print(f"   - Removing folder: {folder}")
        shutil.rmtree(folder_path)

outputs.mkdir(parents=True, exist_ok=True)

# File paths for hloc
sfm_pairs = outputs / 'pairs-sfm.txt'
features = outputs / 'features.h5'
matches = outputs / 'matches.h5'

# --- 4. AI PIPELINE ---
print(f"🚀 (1/4) Starting AI Feature Extraction on {images_dir}...")
feature_conf = extract_features.confs['superpoint_aachen']
extract_features.main(feature_conf, images_dir, feature_path=features)

print(f"🚀 (2/4) Starting AI Matching (LightGlue)...")
try:
    pairs_from_exhaustive.main(sfm_pairs, image_list=extract_features.list_h5_names(features))
    match_conf = match_features.confs['superpoint+lightglue']
    match_features.main(match_conf, sfm_pairs, features=features, matches=matches)
except Exception as e:
    print(f"Error in matching: {e}")
    sys.exit(1)

print(f"🚀 (3/4) Running Reconstruction...")
# 🟢 ARCHITECTURE PATCH: Force COLMAP to treat real photos and SV3D frames as having independent camera lenses!
model = reconstruction.main(
    sparse_dir, 
    images_dir, 
    sfm_pairs, 
    features, 
    matches, 
    image_list=extract_features.list_h5_names(features),
    camera_mode=pycolmap.CameraMode.PER_IMAGE # 👈 THIS SAVES THE PIPELINE
)
# Ensure files are in sparse/0 for the Undistorter input
sparse_0_input = sparse_dir / "0"
sparse_0_input.mkdir(parents=True, exist_ok=True)

for filename in ['cameras.bin', 'images.bin', 'points3D.bin']:
    src = sparse_dir / filename
    dst = sparse_0_input / filename
    if src.exists():
        shutil.move(str(src), str(dst))

print(f"🚀 (4/4) Running Undistortion...")
cmd = [
    "xvfb-run", "-a", "colmap", "image_undistorter",
    "--image_path", str(images_dir),
    "--input_path", str(sparse_0_input),
    "--output_path", str(dataset_path),
    "--output_type", "COLMAP",
    "--max_image_size", "1600"
]
subprocess.run(cmd, check=True)

# --- 5. FINAL FIX FOR TRAIN.PY ---
# The Undistorter dumps files in 'sparse'. Train.py needs them in 'sparse/0'.
final_sparse = dataset_path / "sparse"
final_sparse_0 = final_sparse / "0"

print("🔧 Moving final model files to sparse/0 for training...")
final_sparse_0.mkdir(parents=True, exist_ok=True)

files_moved = 0
for filename in ['cameras.bin', 'images.bin', 'points3D.bin']:
    src = final_sparse / filename
    dst = final_sparse_0 / filename
    if src.exists():
        shutil.move(str(src), str(dst))
        files_moved += 1

if files_moved == 3:
    print(f"✅ Successfully moved {files_moved} files to {final_sparse_0}")
    print("✅ AI Pipeline Complete! Ready for 'python train.py'")
else:
    print(f"⚠️ Warning: Only moved {files_moved}/3 files. Check {final_sparse} if training fails.")