# 📸 Radiometric Corruption Generator & Robustness Suite

## 📌 Overview
This repository contains tools for **Dataset Augmentation** (`corrupt_dataset_v3.py`) and **Radiometric Integrity Screening** (`load_images_v23.py`).

The suite is designed to test the robustness of **Passive 3D Reconstruction** algorithms (NeRF, Gaussian Splatting) against real-world camera failures. Unlike traditional tests that use artificial geometric masks (black rectangles), this toolset focuses on **Sensor & Optical Failures**—errors that naturally occur in low-cost camera sensors during acquisition.

### **Key Features**
* **Sensor Saturation ("Deep Fried"):** Simulates color clipping common in "Vivid" modes.
* **Exposure Failure:** Simulates shutter errors ("Flash Bang" overexposure or "Crushed" underexposure).
* **Optical Defocus:** Simulates lens focus errors using Gaussian blur.
* **ISO Noise:** Simulates high-sensitivity grain common in low-light photography.
* **Sector Deletion:** Simulates "Missing Views" (e.g., restricted camera movement).

---

## 🛠️ Methodology: The "Sensor Failure" Hypothesis

This tool aligns with a **Passive Reconstruction** scope. Passive methods rely on ambient light and optical sensors; thus, the primary failure modes are **Radiometric** (Light/Color) and **Optical** (Lens), rather than Digital Transmission errors (data voids).

### **1. Saturation Boost (Sensor Clipping)**
* **Physics:** Simulates the loss of texture detail when a color channel hits its maximum value (255).
* **Effect on 3D:** NeRF/3DGS rely on subtle texture gradients to match features. Saturated regions become "flat" colors, causing matching failures.
* **Implementation:** Converts RGB to HSV and boosts the **Saturation (S)** channel by a factor of **2.0x to 4.0x**. Any value exceeding 255 is clipped, destroying texture data.

### **2. Exposure Failure (Dynamic Range Loss)**
* **Physics:** Simulates a shutter open too long (Overexposure) or too short (Underexposed).
* **Effect on 3D:**
    * *Overexposed:* White clipping removes all feature data (Histogram smashed to the right).
    * *Underexposed:* Signal-to-noise ratio drops to zero (Histogram smashed to the left).
* **Implementation:** Multiplies pixel values by factors **>2.5 (Flash)** or **<0.3 (Darkness)**.

### **3. Optical Defocus (Blur)**
* **Physics:** Simulates a lens focusing on the wrong plane (background vs. object).
* **Effect on 3D:** Softens edges, making depth estimation imprecise.
* **Implementation:** Applies a **Gaussian Blur kernel** (sizes ranging from 7x7 to 21x21) to mimic soft optics.

### **4. ISO Noise (Grain)**
* **Physics:** Simulates electronic thermal noise in the sensor.
* **Effect on 3D:** Algorithms interpret grain as "micro-geometry," creating floating artifacts (floaters) in the 3D scene.
* **Implementation:** Adds Gaussian noise ($\sigma = 30-80$) to the image tensor.

---

## 📊 Visual Verification & Debugging

When running the detection script (`load_images_v23.py`) with the `--debug` flag, the system generates analytical charts to verify these corruptions. Here is how to interpret them:

### **1. Interpreting the Saturation Heatmap**
The **Center Panel** of the debug output visualizes the Saturation channel.
* **Normal Image:** Shows a variety of colors (purple, orange, black).
* **"Deep Fried" Corruption:** The object appears **Glowing Yellow/Bright Orange**.
    * **Meaning:** Yellow indicates pixels closer to max saturation (255).
    * **Failure Criteria:** If the **"Neon Clip" ratio is > 5-10%**, the image is rejected as oversaturated.

### **2. Interpreting the Exposure Histogram**
The **Right Panel** shows the distribution of pixel brightness (0=Black, 255=White).
* **Normal Image:** The graph looks like a "Mountain" centered in the middle (e.g., 50–200 range).
* **Overexposed ("Flash Bang"):**
    * **Visual:** The mountain is smashed against the **Right Wall (255)**.
    * **Meaning:** Massive data loss in highlights.
* **Underexposed ("Crushed"):**
    * **Visual:** The mountain is smashed against the **Left Wall (0)**.
    * **Meaning:** Massive data loss in shadows.
* **Safe Zone:** Red dashed lines indicate the acceptable average brightness range (e.g., 30–220).

### **3. Interpreting Blur/Noise (MUSIQ Score)**
* **Visual:** These images may look "soft" or "grainy" to the naked eye.
* **Metric:** The MUSIQ score (0-100) quantifies this.
    * **Score < 40:** Rejected (Too Blurry/Noisy).
    * **Score > 70:** High Quality.

---

## 🚀 Usage

### **Prerequisites**
* Python 3.x
* OpenCV (`opencv-python`)
* NumPy
* Pillow
* tqdm
* Matplotlib (for debug graphs)

```bash
pip install opencv-python numpy pillow tqdm matplotlib

```

### **1. Generating Corruptions**

To corrupt a dataset with a 50% chance of sensor failure per image:

```bash
python corrupt_dataset_v3.py \
  --clean_dir ./nerf_synthetic/lego \
  --out_dir ./output_data \
  --corrupt_prob 0.5

```

### **2. Detecting & Screening**

To scan the dataset and generate debug graphs (Heatmaps/Histograms):

```bash
python load_images_v23.py \
  --input_dir ./output_data/lego \
  --mode synthetic \
  --debug

```

---

## ⚙️ Generator Arguments

| Argument | Description | Default |
| --- | --- | --- |
| `--clean_dir` | Path to the root of the input dataset. | **Required** |
| `--out_dir` | Path where the corrupted dataset will be saved. | **Required** |
| `--corrupt_prob` | Probability (0.0 - 1.0) of sensor failure. | `0.5` |
| `--use_geometry` | If set, reads `transforms.json` to delete the Negative-X sector. | `False` |
| `--delete_prob` | Randomly delete extra images (sparse data simulation). | `0.0` |

---

## 📂 Output Structure

The pipeline maintains compatibility with standard NeRF loaders:

```
output_data/
└── lego/
    ├── train/                <-- Mixed Clean & Corrupted images
    │   ├── r_0.png           (Clean)
    │   ├── r_1.png           (Defocus Blur)
    │   ├── r_2.png           (Saturation Fried)
    │   └── ...
    ├── processed_train/      <-- Clean copies saved by the loader script
    ├── debug_visuals/        <-- Histograms & Heatmaps generated by loader
    └── transforms_train.json <-- Camera poses

```