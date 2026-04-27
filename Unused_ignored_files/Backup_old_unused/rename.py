"""
Step 4: rename_sequence.py - Renames all images to 0001.jpg, 0002.jpg...
"""
import os
import argparse
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Directory containing images to rename")
    parser.add_argument("--prefix", default="", help="Optional prefix (e.g. 'img_')")
    args = parser.parse_args()

    # Allowed extensions
    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}

    # Get all files
    files = [f for f in os.listdir(args.input_dir) if os.path.splitext(f)[1].lower() in valid_exts]

    # Sort them to ensure order (Novel views usually come after originals alphabetically)
    files.sort()

    print(f"Found {len(files)} images. Renaming...")

    for i, filename in enumerate(tqdm(files)):
        old_path = os.path.join(args.input_dir, filename)

        # Get extension (keep original extension or force jpg?)
        # Let's force .jpg if you want uniformity, otherwise keep original
        ext = os.path.splitext(filename)[1].lower()
        if ext == ".jpeg": ext = ".jpg"

        # New Name: 0001.jpg
        new_name = f"{args.prefix}{i + 1:04d}{ext}"
        new_path = os.path.join(args.input_dir, new_name)

        os.rename(old_path, new_path)

    print(f"✅ Renamed {len(files)} images in {args.input_dir}")


if __name__ == "__main__":
    main()