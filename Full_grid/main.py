import os
from PIL import Image
import matplotlib.pyplot as plt

# ==========================================
# 🖍️ PASTE THE PATH TO ONE GRID IMAGE HERE
# ==========================================
FILE_TO_TEST = "Full_grid/FULL_GRID_r_3.png"

def test_slice_single_image(image_path):
    if not os.path.exists(image_path):
        print(f"❌ Error: File not found at {image_path}")
        print("Please check the path and try again.")
        return

    img = Image.open(image_path)
    w, h = img.size

    # --- 1. CORRECT TILE MATH (2 Cols, 3 Rows) ---
    tile_w = w // 2
    tile_h = h // 3

    print(f"📏 Original Grid: {w}x{h}")
    print(f"✂️  Cutting into:  {tile_w}x{tile_h} tiles (2 Cols x 3 Rows)")

    # --- 2. CORRECT SUBPLOT LAYOUT (3 Rows, 2 Cols) ---
    fig, axes = plt.subplots(3, 2, figsize=(8, 12))
    fig.suptitle(f"Slicing Test: {os.path.basename(image_path)}", fontsize=16)

    count = 0

    # --- 3. CORRECT LOOPS ---
    # We loop 3 rows down, 2 columns across
    for row in range(3):
        for col in range(2):
            # Calculate Cut Coordinates
            left = col * tile_w
            top = row * tile_h
            right = left + tile_w
            bottom = top + tile_h

            # Perform the Crop
            crop = img.crop((left, top, right, bottom))

            # Display in Notebook
            ax = axes[row, col]
            ax.imshow(crop)
            ax.set_title(f"View {count}\n(Row {row}, Col {col})")
            ax.axis('off')

            # Check for centering (Red Box Visualizer)
            rect = plt.Rectangle((0, 0), tile_w - 1, tile_h - 1, linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)

            count += 1

    plt.tight_layout()
    plt.show()

# Run the test
test_slice_single_image(FILE_TO_TEST)