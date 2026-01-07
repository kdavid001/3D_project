# 🛡️ Radiometric Screener & Router (Enhanced Visuals)

## 📌 Overview

`load_images_legend.py` is an advanced quality control pipeline designed to screen datasets for **3D Reconstruction (NeRF/Gaussian Splatting)**. This updated version features **Enhanced Diagnostic Graphs** with clear legends, axis labels, and a comprehensive "Inspection Report" to make debugging easier.

It automatically filters out images that would cause "floaters" or geometry collapse and routes them for repair, ensuring only high-quality data reaches the training stage.

---

## ⚙️ The "Five-Pillar" Logic

The script evaluates every image against **five** distinct failure modes using specific thresholds for `natural` (real-world) and `synthetic` (blender/render) data.

### **Pillar 1: Optical Quality (Blur & Noise)**

* **Method:** Uses **MUSIQ (Multi-scale Image Quality Transformer)** via `pyiqa` to score perceived sharpness.
* **Thresholds (Fail if Score < X):**
* **Natural:** < 40.0
* **Synthetic:** < 65.0



### **Pillar 2: Radiometric Saturation (Neon/Clipping)**

* **Method:** Analyzes the **Saturation (S)** channel in HSV space. It checks for "Neon" artifacts where color data is clipped.
* **Thresholds:**
* **Clipped Pixels:** > 5% (Natural) or > 2% (Synthetic).
* **Average Saturation:** > 180.0 (Natural) or > 160.0 (Synthetic).



### **Pillar 3: Exposure Integrity (Lighting)**

* **Method:** Analyzes the **Value (V)** channel to detect overexposure (blown highlights) or underexposure (crushed shadows).
* **Thresholds (Average Brightness 0-255):**
* **Min (Dark):** < 45.0
* **Max (Bright):** > 250.0 (Natural) or > 230.0 (Synthetic).



### **Pillar 4: Contrast (Flatness)**

* **Method:** Calculates the standard deviation of the V channel. Low variance means the image is "flat" and lacks feature definition.
* **Thresholds:**
* **Fail if:** < 10.0 (Natural) or < 25.0 (Synthetic).



### **Pillar 5: Color Cast (Tint)**

* **Method:** Converts to **LAB Colorspace** and measures the distance of the average pixel from neutral gray (128, 128).
* **Thresholds:**
* **Fail if:** Distance > 60.0 (Natural) or > 50.0 (Synthetic).



---

## 🔀 Auto-Routing Logic

Every image is automatically assigned a decision tag based on the Five Pillars:

| Condition | Decision Tag | Downstream Action |
| --- | --- | --- |
| **FAILED any Pillar** | `REPAIR` | Sent to **Inpainting Pipeline** to fix artifacts. |
| **PASSED all Pillars** | `NOVEL_VIEW` | Sent to **Img2Img Pipeline** to generate new camera angles. |

---

## 📊 Visual Verification (Enhanced)

When running with `--debug`, the script generates a **High-Fidelity 3-Panel Analysis Image** for every file:

### **Panel 1: Status & Quality**

* **Visual:** The original image.
* **Title:** Color-coded `PASS` (Green) or `REJECT` (Red).
* **Content:** Displays the specific **Reason** for failure (e.g., "Neon", "Blurry") and the raw MUSIQ score.

### **Panel 2: Color Balance (RGB Histogram)**

* **Visual:** A line graph showing the distribution of Red, Green, and Blue pixels.
* **Upgrades:**
* **Legend:** Clearly labels "Red Channel", "Green Channel", etc.
* **Axis Labels:** "Pixel Brightness" (X) and "Pixel Count" (Y).
* **Warning:** Displays a bold `❌ UNBALANCED` text overlay if a color cast is detected.



### **Panel 3: Inspection Report**

* **Visual:** A structured list of all 5 metrics vs. their limits.
* **Interpretation:**
* **Icons:** ✅ for Pass, ❌ for Fail.
* **Data:** Shows exact values (e.g., "Saturation: 12.4% (Limit 5.0%)").
* **Use Case:** Allows instant verification of *how close* an image was to failing.



---

## 🚀 Usage

### **Prerequisites**

* Python 3.x
* **PyIQA** (Critical for MUSIQ metric)
* OpenCV, NumPy, Matplotlib

```bash
pip install torch pyiqa opencv-python numpy matplotlib tqdm

```

### **Running the Screener**

```bash
python load_images_legend.py \
  --input_dir ./output_data/train \
  --mode natural \
  --out_dir ./preprocessed \
  --debug

```

### **Arguments**

| Argument | Description | Default |
| --- | --- | --- |
| `--input_dir` | Root folder containing the dataset. | **Required** |
| `--mode` | `natural` (lenient) or `synthetic` (strict). | `synthetic` |
| `--debug` | Generate the 3-panel charts with legends. | `False` |
| `--test_image` | Run analysis on a single file path. | `None` |

---

## 📂 Output Artifacts

1. **`manifest.json`**: Master log containing scores, decisions, and failure notes for every image.
2. **`processed_train/`**: Folder containing only the images that **Passed** (clean copy).
3. **`debug_visuals/`**: (If `--debug`) The detailed 3-panel report cards for manual inspection.

---

## 🧠 Why This Matters?

By strictly enforcing these 5 pillars, we ensure:

1. **Geometry:** No "flat" regions from clipping (Pillars 2 & 4).
2. **Texture:** Sharp features for matching (Pillar 1).
3. **Consistency:** Uniform lighting and white balance (Pillars 3 & 5), preventing the "chameleon effect" where an object changes color from different angles.

... [The Three Different Image Histograms](https://www.youtube.com/watch?v=bULVoGQNo74) ...
This video explains how to read RGB histograms, which corresponds directly to the new "Panel 2" visualization in the updated script.