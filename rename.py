#!/usr/bin/env python3
"""
Step 6 (CLEANUP): rename_sequence.py

LOGIC:
1. Separates 'Real' images (r_0.png) from 'Synthetic' images (synth_...).
2. Sorts 'Real' images numerically (0, 1, 2, 10...) not alphabetically (0, 10, 2...).
3. Renames everything to 0001.jpg, 0002.jpg... putting Real images FIRST.

This helps COLMAP initialize the robust real trajectory before adding synthetic data.
"""

import os
import argparse
import re
from tqdm import tqdm

def extract_number(filename):
    """Finds the first number in a filename for sorting."""
    match = re.search(r'(\d+)', filename)
    return int(match.group(1)) if match else 999999

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Directory containing images to rename")
    args = parser.parse_args()

    # Allowed extensions
    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}

    # Get all files
    all_files = [f for f in os.listdir(args.input_dir) if os.path.splitext(f)[1].lower() in valid_exts]

    if not all_files:
        print("❌ No images found in directory.")
        return

    # SPLIT into Real vs Synthetic
    # Assumption: Synthetic files start with "synth_" (from Step 4 script)
    synthetics = [f for f in all_files if f.startswith("synth_")]
    originals = [f for f in all_files if not f.startswith("synth_")]

    # SORT properly (Numerical Sort)
    # This ensures r_2.png comes before r_10.png
    originals.sort(key=extract_number)
    synthetics.sort(key=extract_number)

    # COMBINE: Originals First, Synthetics Last
    sorted_files = originals + synthetics

    print(f"Found {len(sorted_files)} images.")
    print(f"   - {len(originals)} Real Images (will be 0001 - {len(originals):04d})")
    print(f"   - {len(synthetics)} Synthetic Images (will follow after)")

    # RENAME
    # We rename to a temporary name first to avoid overwriting conflicts (e.g. renaming 1.jpg to 2.jpg while 2.jpg exists)
    # Strategy: Rename all to "temp_XXXX.jpg", then to "0001.jpg"

    print("Renaming...")

    # Pass 1: Rename to temp safe names
    temp_map = []
    for i, filename in enumerate(tqdm(sorted_files, desc="Processing")):
        old_path = os.path.join(args.input_dir, filename)

        # Force .jpg for consistency
        new_name = f"temp_{i+1:04d}.jpg"
        new_path = os.path.join(args.input_dir, new_name)

        os.rename(old_path, new_path)
        temp_map.append(new_name)

    # Pass 2: Rename temp to final
    for i, filename in enumerate(temp_map):
        old_path = os.path.join(args.input_dir, filename)
        final_name = f"{i+1:04d}.jpg"
        final_path = os.path.join(args.input_dir, final_name)

        os.rename(old_path, final_path)

    print(f"✅ Successfully renamed {len(sorted_files)} images.")
    print(f"   First: 0001.jpg (Real)")
    print(f"   Last:  {len(sorted_files):04d}.jpg (Synthetic)")

if __name__ == "__main__":
    main()