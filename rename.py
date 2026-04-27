import os
import shutil
import argparse
from tqdm import tqdm


def organize_dataset(source_dir, output_base_dir, model_category_name):
    """
    source_dir: The folder with your mixed processed files (e.g. final_dataset_v7...)
    output_base_dir: Where you want the clean folder (e.g. /content/drive/MyDrive/.../GS_input)
    model_category_name: The name of the object (e.g. "chair" or "lego")
    """

    # 1. Setup the specific folder for this object
    target_dir = os.path.join(output_base_dir, model_category_name, "input")

    # Clean old run if exists
    if os.path.exists(target_dir):
        print(f"🧹 Cleaning old folder: {target_dir}")
        shutil.rmtree(target_dir)
    os.makedirs(target_dir, exist_ok=True)

    print(f"📂 Source: {source_dir}")
    print(f"🎯 Target: {target_dir}")

    # 2. Get all valid image files
    valid_exts = ('.jpg', '.jpeg', '.png')
    if not os.path.exists(source_dir):
        print(f"❌ ERROR: Source directory not found: {source_dir}")
        return

    all_files = [f for f in os.listdir(source_dir) if f.lower().endswith(valid_exts)]

    if not all_files:
        print("❌ No files found in source directory!")
        return

    # 3. Sort them to ensure deterministic order
    all_files.sort()

    print(f"🚀 Processing {len(all_files)} images...")

    count = 1

    for filename in tqdm(all_files):
        # Construct the new sequential name (COLMAP preferred format)
        # e.g., 00001.jpg, 00002.jpg
        new_name = f"{count:05d}.jpg"

        src_path = os.path.join(source_dir, filename)
        dst_path = os.path.join(target_dir, new_name)

        # Copy and Rename
        shutil.copy2(src_path, dst_path)
        count += 1

    print("-" * 30)
    print(f"✅ Done! Organized {count - 1} images.")
    print(f"📂 Ready for COLMAP at: {target_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Organize dataset for Gaussian Splatting / COLMAP")

    # Define arguments
    parser.add_argument("--source", type=str, required=True, help="Path to the v7_final output folder")
    parser.add_argument("--output", type=str, required=True, help="Root folder for GS Input (e.g., input_database)")
    parser.add_argument("--name", type=str, required=True, help="Name of the model (e.g., chair, lego)")

    args = parser.parse_args()

    organize_dataset(args.source, args.output, args.name)