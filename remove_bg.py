import os
import argparse
from rembg import remove
from PIL import Image

def main():
    # 1. Setup Arguments for Terminal Execution
    parser = argparse.ArgumentParser(description="AI Background Removal: Swaps grey/white backgrounds for pure black.")
    parser.add_argument("--input_dir", required=True, help="Folder containing the raw Zero123++ images")
    parser.add_argument("--output_dir", required=True, help="Folder where the clean black-background images will be saved")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    # 2. Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Grab all valid image files
    valid_exts = ('.png', '.jpg', '.jpeg')
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_exts)]
    
    if not files:
        print(f"⚠️ No valid images found in {input_dir}")
        return

    print(f"🧹 Found {len(files)} images. Scrubbing backgrounds and applying black canvas...")

    # 3. Process the images
    processed_count = 0
    for filename in files:
        input_path = os.path.join(input_dir, filename)
        
        # Force .png output to preserve quality, even if input was .jpg
        base_name = os.path.splitext(filename)[0]
        output_path = os.path.join(output_dir, f"{base_name}.png")

        try:
            # Open image and use AI to extract the subject
            img = Image.open(input_path).convert("RGBA")
            subject_only = remove(img) # Background becomes transparent (Alpha=0)

            # Create a brand new pitch-black canvas
            black_canvas = Image.new("RGBA", subject_only.size, (0, 0, 0, 255))
            
            # Paste the subject perfectly onto the black canvas using its own alpha mask
            black_canvas.paste(subject_only, (0, 0), subject_only)
            
            # Convert back to standard RGB and save
            black_canvas.convert("RGB").save(output_path)
            
            processed_count += 1
            print(f"✅ Cleaned: {filename}")
            
        except Exception as e:
            print(f"❌ Failed to process {filename}: {e}")

    print(f"\n🚀 Successfully moved {processed_count} pristine images to:")
    print(f"📁 {output_dir}")

if __name__ == "__main__":
    main()