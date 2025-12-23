import os
import json
import numpy as np
from PIL import Image
import cv2


def prepare_mask_for_inpainting(mask_pil):
    """
    Prepare mask for SD inpainting model.

    SD Inpainting expects:
    - mask = 1 (white/255) -> area to INPAINT
    - mask = 0 (black/0) -> area to KEEP

    Returns binary mask with proper convention.
    """
    mask = np.array(mask_pil).astype("float32") / 255.0  # Normalize to [0, 1]

    # Binarize: threshold at 0.5
    mask_binary = (mask > 0.5).astype("float32")

    # Convert back to PIL (values: 0 or 255)
    mask_pil_binary = Image.fromarray((mask_binary * 255).astype("uint8"))

    return mask_pil_binary


def create_masked_image(image_pil, mask_pil, fill_mode="gray"):
    """
    Create masked image for SD inpainting conditioning.

    Args:
        image_pil: Original RGB image
        mask_pil: Binary mask (white=inpaint, black=keep)
        fill_mode: How to fill masked regions
            - "gray": Fill with gray (127, 127, 127) - RECOMMENDED
            - "black": Fill with black (0, 0, 0)
            - "white": Fill with white (255, 255, 255)
            - "noise": Fill with random noise

    Returns:
        PIL Image with masked regions filled
    """
    image = np.array(image_pil).astype("float32") / 255.0  # [H, W, 3]
    mask = np.array(mask_pil).astype("float32") / 255.0  # [H, W]

    # Ensure binary mask
    mask_binary = (mask > 0.5).astype("float32")
    mask_3c = np.repeat(mask_binary[:, :, None], 3, axis=2)  # [H, W, 3]

    # Choose fill method
    if fill_mode == "gray":
        fill_value = 0.5  # Gray
    elif fill_mode == "black":
        fill_value = 0.0
    elif fill_mode == "white":
        fill_value = 1.0
    elif fill_mode == "noise":
        fill_value = np.random.rand(*image.shape)
    else:
        fill_value = 0.5

    # Apply masking: keep original where mask=0, fill where mask=1
    if fill_mode == "noise":
        masked_image = image * (1.0 - mask_3c) + fill_value * mask_3c
    else:
        masked_image = image * (1.0 - mask_3c) + fill_value * mask_3c

    # Clip to valid range
    masked_image = np.clip(masked_image, 0.0, 1.0)

    return Image.fromarray((masked_image * 255).astype("uint8"))


def visualize_mask_overlay(image_pil, mask_pil, alpha=0.6):
    """
    Create visualization with red overlay on masked regions.
    Useful for verifying mask correctness.
    """
    image = np.array(image_pil).astype("float32") / 255.0
    mask = np.array(mask_pil).astype("float32") / 255.0
    mask_binary = (mask > 0.5).astype("float32")

    # Create red overlay
    overlay = image.copy()
    overlay[:, :, 0] = np.clip(overlay[:, :, 0] + mask_binary * 0.7, 0, 1)  # Add red

    # Blend
    result = image * (1 - alpha) + overlay * alpha
    result = np.clip(result, 0, 1)

    return Image.fromarray((result * 255).astype("uint8"))


def resize_to_multiple_of_8(pil_img):
    """
    Resize image to dimensions divisible by 8 (required for VAE).
    """
    w, h = pil_img.size
    new_w = w - (w % 8)
    new_h = h - (h % 8)
    if new_w == w and new_h == h:
        return pil_img
    return pil_img.resize((new_w, new_h), Image.BICUBIC)


def process_dataset(
        image_dir,
        mask_dir,
        output_dir,
        manifest_path,
        fill_mode="gray",
        create_visualization=True
):
    """
    Process dataset for SD inpainting using existing masks.

    Input structure:
        image_dir/
            image1.jpg
            image2.png
            ...
        mask_dir/
            image1.png  (or .jpg)
            image2.png
            ...

    Output structure:
        output_dir/
            masked_images/   - Images with filled regions
            masks/           - Processed binary masks
            visualizations/  - Debug overlays (red = inpaint area)
        manifest.json        - Metadata

    Args:
        image_dir: Directory with original images
        mask_dir: Directory with mask images (white=inpaint, black=keep)
        output_dir: Output directory
        manifest_path: Path to save manifest JSON
        fill_mode: How to fill masked regions ("gray", "black", "white", "noise")
        create_visualization: Whether to create debug visualizations
    """
    # Create output directories
    masked_image_dir = os.path.join(output_dir, "masked_images")
    processed_mask_dir = os.path.join(output_dir, "masks")
    vis_dir = os.path.join(output_dir, "visualizations")

    os.makedirs(masked_image_dir, exist_ok=True)
    os.makedirs(processed_mask_dir, exist_ok=True)
    if create_visualization:
        os.makedirs(vis_dir, exist_ok=True)

    manifest = []
    processed_count = 0
    skipped_count = 0

    print("Processing dataset for SD inpainting...")
    print(f"Image dir: {image_dir}")
    print(f"Mask dir: {mask_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Fill mode: {fill_mode}")
    print("-" * 60)

    # Get all image files
    valid_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
    image_files = [f for f in os.listdir(image_dir)
                   if os.path.splitext(f)[1].lower() in valid_extensions]

    if not image_files:
        print(f"❌ No images found in {image_dir}")
        return

    for fname in sorted(image_files):
        image_path = os.path.join(image_dir, fname)
        base_name = os.path.splitext(fname)[0]

        # Try to find matching mask with different extensions
        mask_path = None
        for ext in valid_extensions:
            potential_mask = os.path.join(mask_dir, base_name + ext)
            if os.path.exists(potential_mask):
                mask_path = potential_mask
                break

        if mask_path is None:
            print(f"⚠️  Skipping {fname}: no matching mask found")
            skipped_count += 1
            continue

        try:
            # Load images
            image = Image.open(image_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")  # Grayscale

            # Store original size
            original_size = image.size

            # Resize to multiples of 8 (required for VAE)
            image = resize_to_multiple_of_8(image)
            mask = resize_to_multiple_of_8(mask)

            # Ensure mask matches image size
            if image.size != mask.size:
                print(f"  Resizing mask to match image size: {image.size}")
                mask = mask.resize(image.size, Image.NEAREST)

            # Process mask to binary
            mask_processed = prepare_mask_for_inpainting(mask)

            # Check mask coverage
            mask_array = np.array(mask_processed)
            coverage = np.sum(mask_array > 127) / mask_array.size

            if coverage == 0:
                print(f"⚠️  Skipping {fname}: mask is completely black (nothing to inpaint)")
                skipped_count += 1
                continue

            if coverage == 1.0:
                print(f"⚠️  WARNING {fname}: mask is completely white (entire image will be inpainted!)")

            # Create masked image
            masked_image = create_masked_image(image, mask_processed, fill_mode=fill_mode)

            # Save outputs
            masked_image_path = os.path.join(masked_image_dir, f"{base_name}.png")
            processed_mask_path = os.path.join(processed_mask_dir, f"{base_name}.png")

            masked_image.save(masked_image_path)
            mask_processed.save(processed_mask_path)

            # Create visualization
            if create_visualization:
                vis = visualize_mask_overlay(image, mask_processed, alpha=0.6)
                vis_path = os.path.join(vis_dir, f"{base_name}_overlay.png")
                vis.save(vis_path)

            # Add to manifest
            manifest.append({
                "filename": fname,
                "original_image": image_path,
                "masked_image": masked_image_path,
                "mask": processed_mask_path,
                "original_size": list(original_size),
                "processed_size": list(image.size),
                "mask_coverage": float(coverage)
            })

            processed_count += 1
            status = "✓" if coverage < 0.5 else "⚠️"
            print(f"{status} {fname}: {coverage * 100:.1f}% will be inpainted")

        except Exception as e:
            print(f"❌ Error processing {fname}: {str(e)}")
            import traceback
            traceback.print_exc()
            skipped_count += 1
            continue

    # Save manifest
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print("-" * 60)
    print(f"✓ Processing complete!")
    print(f"  Processed: {processed_count} images")
    print(f"  Skipped: {skipped_count} images")
    print(f"  Manifest: {manifest_path}")
    print(f"\nOutputs:")
    print(f"  - Masked images: {masked_image_dir}")
    print(f"  - Masks: {processed_mask_dir}")
    if create_visualization:
        print(f"  - Visualizations: {vis_dir}")

    if processed_count > 0:
        avg_coverage = np.mean([m['mask_coverage'] for m in manifest])
        print(f"\nAverage mask coverage: {avg_coverage * 100:.1f}%")


def test_single_pair(image_path, mask_path, output_dir="test_output"):
    """
    Test processing on a single image-mask pair.
    Useful for checking if your masks are correct before batch processing.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\nTesting single image-mask pair...")
    print(f"Image: {image_path}")
    print(f"Mask: {mask_path}")

    # Load
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")

    # Resize
    image = resize_to_multiple_of_8(image)
    mask = resize_to_multiple_of_8(mask)

    if image.size != mask.size:
        mask = mask.resize(image.size, Image.NEAREST)

    # Process mask
    mask_processed = prepare_mask_for_inpainting(mask)

    # Check coverage
    coverage = np.sum(np.array(mask_processed) > 127) / np.array(mask_processed).size
    print(f"Mask coverage: {coverage * 100:.1f}%")

    # Create outputs with different fill modes
    for mode in ["gray", "black", "white", "noise"]:
        masked = create_masked_image(image, mask_processed, fill_mode=mode)
        masked.save(os.path.join(output_dir, f"masked_{mode}.png"))

    # Save processed mask
    mask_processed.save(os.path.join(output_dir, "mask_processed.png"))

    # Save visualization
    vis = visualize_mask_overlay(image, mask_processed, alpha=0.6)
    vis.save(os.path.join(output_dir, "visualization_overlay.png"))

    print(f"✓ Test outputs saved to: {output_dir}")
    print("\nCheck visualization_overlay.png:")
    print("  - Red areas = will be inpainted")
    print("  - Non-red areas = will be preserved")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Test mode: python script.py test <image_path> <mask_path>
        if len(sys.argv) < 4:
            print("Usage: python script.py test <image_path> <mask_path>")
            sys.exit(1)

        test_single_pair(sys.argv[2], sys.argv[3], output_dir="./test_output")

    else:
        # Process full dataset
        process_dataset(
            image_dir="./images",
            mask_dir="./masks",
            output_dir="./output",
            manifest_path="./manifest.json",
            fill_mode="gray",
            create_visualization=True
        )