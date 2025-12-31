import numpy as np
import cv2


def local_variance(gray, ksize=7):
    mean = cv2.blur(gray, (ksize, ksize))
    mean_sq = cv2.blur(gray ** 2, (ksize, ksize))
    return mean_sq - mean ** 2


def edge_energy(gray):
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    return np.abs(lap)


def detect_corruption(
    img_rgb,
    var_thresh=2.5,
    edge_thresh=2.0,
    bright_thresh=2.5,
    ksize=7,
):
    """
    img_rgb: H W 3 uint8
    returns: H W uint8 mask (255 = corrupted)
    """

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    # 1. Local variance
    var_map = local_variance(gray, ksize)

    # 2. Edge energy
    edge_map = edge_energy(gray)

    # 3. Brightness deviation
    mean_brightness = gray.mean()
    bright_map = np.abs(gray - mean_brightness)

    # Normalize maps
    def norm(x):
        return (x - x.mean()) / (x.std() + 1e-6)

    var_n = norm(var_map)
    edge_n = norm(edge_map)
    bright_n = norm(bright_map)

    # Thresholds
    var_mask = var_n > var_thresh
    edge_mask = edge_n < -edge_thresh   # blur → low edges
    bright_mask = bright_n > bright_thresh

    # Combine
    mask = var_mask | edge_mask | bright_mask
    mask = mask.astype(np.uint8) * 255

    # Morphological cleanup
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


import cv2
import matplotlib.pyplot as plt

img = cv2.imread("./test_image/train/images/00005.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

mask = detect_corruption(img)

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.title("Image")
plt.imshow(img)
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Mask")
plt.imshow(mask, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Overlay")
overlay = img.copy()
overlay[mask > 0] = [255, 0, 0]
plt.imshow(overlay)
plt.axis("off")

plt.show()