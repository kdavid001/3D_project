# 🛡️ Radiometric Screener & Novel View Router

## 📌 Overview
`load_images_v25.py` is an automated quality control and routing pipeline designed to screen datasets for **Sensor & Optical Failures** before they enter 3D reconstruction pipelines (NeRF, Gaussian Splatting).

Because passive reconstruction algorithms rely on consistent feature extraction, images with **Radiometric Anomalies** (Saturation Clipping, Exposure Failure) or **Optical Degradation** (Blur, Noise) frequently cause "floaters" or geometry collapse in the final model.

This script performs two critical tasks:
1.  **Screening:** Filters out defects using a **"Three-Pillar"** detection logic.
2.  **Routing:** Automatically tags healthy images for **Novel View Augmentation** (`NOVEL_VIEW`) and defective images for **Inpainting** (`REPAIR`).

---

## ⚙️ The "Three-Pillar" Logic

The script evaluates every image against three distinct failure modes. If **ANY** of these checks fail, the image is flagged for repair.

### **Pillar 1: Radiometric Saturation (Color Integrity)**
* **Hypothesis:** When a sensor's color channel clips (reaches 255), texture details are lost, leaving "flat" neon patches that cannot be feature-matched.
* **Method:**
    * Converts image to **HSV** colorspace.
    * Calculates the percentage of pixels with **Saturation > 250**.
* **Thresholds:**
    * **Natural:** Fails if > 5% of pixels are clipped.
    * **Synthetic:** Fails if > 8% of pixels are clipped.

### **Pillar 2: Exposure Integrity (Dynamic Range)**
* **Hypothesis:** "Flash Bangs" (Overexposure) or "Crushed Shadows" (Underexposure) result in zero data for the algorithm to use.
* **Method:**
    * Analyzes the **Value (V)** channel histogram of the object (ignoring background).
    * Checks if the average brightness falls outside the "Safe Zone."
* **Thresholds:**
    * **Overexposure:** Avg Brightness > 220 (out of 255).
    * **Underexposure:** Avg Brightness < 10 (synthetic) or < 30 (natural).

### **Pillar 3: Optical Quality (Blur & Noise)**
* **Hypothesis:** Defocus blur and high-ISO noise prevent precise edge detection.
* **Method:**
    * Uses **MUSIQ (Multi-scale Image Quality Transformer)**, a pre-trained AI metric.
    * This model aligns with human perception of sharpness and clarity.
* **Thresholds:**
    * **Fail:** MUSIQ Score < 40 (Natural) or < 65 (Synthetic).

---

## 🔀 Auto-Routing Logic

Unlike previous versions which simply "Accepted" or "Rejected" images, v25 automatically assigns a workflow decision to every image:

| Condition | Decision Tag | Downstream Action |
| :--- | :--- | :--- |
| **FAILED any Pillar** | `REPAIR` | Sent to **Inpainting Pipeline** to fix artifacts. |
| **PASSED all Pillars** | `NOVEL_VIEW` | Sent to **Img2Img Pipeline** to generate new camera angles (Data Augmentation). |

---

## 📊 Visual Verification & Debugging

When running with `--debug`, the script generates a **3-Panel Analysis Image** for every file. Here is how to interpret them:

### **Panel 1: Status & Reason**
* **Visual:** The original image with a colored title.
* **Red Title:** `FAIL` (e.g., `FAIL: Neon Clip (12.4%)`).
* **Green Title:** `PASS`.
* **Use Case:** Quick visual confirmation of *why* an image was rejected.

### **Panel 2: Saturation Heatmap**
* **Visual:** A heat map of the object.
* **Interpretation:**
    * **Black/Purple:** Normal saturation.
    * **Bright Yellow/Orange:** **"Deep Fried"** zones.
* **Failure Indicator:** If you see large patches of glowing yellow, the sensor was clipping color data.

### **Panel 3: Exposure Histogram**
* **Visual:** A grey mountain graph representing pixel brightness distribution (0=Black, 255=White).
* **Interpretation:**
    * **Centered Mountain:** Good exposure.
    * **Smashed to Right Wall:** Overexposed (Blown Highlights).
    * **Smashed to Left Wall:** Underexposed (Crushed Shadows).
* **Red Dashed Lines:** These mark the **Min/Max Safe Limits**. If the mountain's bulk is outside these lines, the image fails.

---

## 🚀 Usage

### **Prerequisites**
* Python 3.x
* PyTorch (CPU or CUDA/MPS)
* PyIQA (for MUSIQ)
* OpenCV, NumPy, Matplotlib

```bash
pip install torch pyiqa opencv-python numpy matplotlib tqdm

```

### **Running the Screener**

To scan a dataset (e.g., the output from the corruption generator):

```bash
python load_images_v25.py \
  --input_dir ./output_data/lego \
  --mode synthetic \
  --out_dir ./preprocessed \
  --debug

```

### **Arguments**

| Argument | Description | Default |
| --- | --- | --- |
| `--input_dir` | Root folder containing the dataset. | **Required** |
| `--mode` | `natural` (more lenient) or `synthetic` (stricter). | `synthetic` |
| `--out_dir` | Where to save the `manifest.json` and debug visuals. | `./output` |
| `--debug` | Enable generation of 3-panel diagnostic charts. | `False` |
| `--test_image` | Run analysis on a single file instead of a folder. | `None` |

---

## 📂 Output Artifacts

1. **`manifest.json`**: The master record used by downstream steps.
```json
[
  {
    "filename": "r_12.png",
    "score": 32.5,
    "decision": "REPAIR",
    "note": "Low Quality (32.5 < 65.0)"
  },
  {
    "filename": "r_13.png",
    "score": 75.2,
    "decision": "NOVEL_VIEW",
    "note": "High Quality - Selected for Novel View"
  }
]

```


2. **`processed_train/`**: A copy of the images (can be used as a clean set).
3. **`debug_visuals/`**: The 3-panel analysis images (only if `--debug` is ON).

---

## 🧠 Why This Matters for NeRF/3DGS?
NeRF algorithms assume that the color of a point stays consistent across views.

* **Saturation/Exposure Clipping** violates this by clamping values, making points look like flat sheets rather than textured surfaces.
* **Blur** smears features across pixels, causing the "Feature Matcher" to place points at the wrong depth.

By filtering these out **before** training, we prevent "floaters" (artifacts caused by noise) and "holes" (artifacts caused by clipping) in the final 3D model.
