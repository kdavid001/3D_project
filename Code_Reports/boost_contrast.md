# 🎨 CLAHE Contrast Booster For Dark/ Black Images

## 📌 Overview

`boost_contrast.py` is a specialized image enhancement utility designed to combat **"Feature Starvation"** in Photogrammetry and 3D Gaussian Splatting.

Reconstruction algorithms (like COLMAP) rely on detecting distinct "edges" and "corners" in an image. If your dataset was shot in flat lighting (overcast days, shadows) or has atmospheric haze, the images may look "smooth" to the computer, resulting in zero feature matches.

This script applies **CLAHE (Contrast Limited Adaptive Histogram Equalization)** to intelligently boost local contrast without destroying color fidelity.

---

## ⚙️ The Logic: Why LAB Color Space?

Most simple contrast tools work on RGB images directly, which often causes **"Color Shifting"** (e.g., boosting contrast makes a red apple turn orange or grey).

This script uses a smarter approach:

1. **Conversion:** Transforms the image from **BGR** to **LAB** Color Space.
* **L:** Lightness (Brightness only).
* **A:** Red/Green Axis.
* **B:** Blue/Yellow Axis.


2. **Isolation:** We apply the contrast boost **ONLY** to the **L channel**.
3. **Protection:** The A and B (Color) channels are left untouched.
4. **Recombination:** The boosted L is merged back with the original colors.

**Result:** Shadows get brighter, highlights get textured, but the *colors* remain mathematically identical to the original photo.

---

## 🚀 Key Features

* **CLAHE Algorithm:** Unlike global histogram equalization (which washes out images), this uses *Adaptive* equalization on small 8x8 grids. It fixes dark shadows without blowing out bright skies.
* **Smart Safety Check:** Detects if you are overwriting the input folder (`input == output`) and switches to "Safe Mode" to prevent accidental file deletion.
* **Haze Removal:** Effectively cuts through atmospheric haze or lens fog, making distant objects sharp enough for SfM (Structure-from-Motion).

---

## 🏃‍♂️ Usage

### **Prerequisites**

* Python 3.x
* OpenCV, NumPy, TQDM

```bash
pip install opencv-python numpy tqdm

```

### **Command**

```bash
python boost_contrast.py \
  --input_dir "/path/to/your/raw_images" \
  --output_dir "/path/to/your/enhanced_images"

```

### **Arguments**

| Argument | Description | Example |
| --- | --- | --- |
| `--input_dir` | Folder containing your source images (.jpg, .png). | `./data/raw` |
| `--output_dir` | Where to save the boosted images. | `./data/contrast` |

---

## 📊 Before & After Behavior

| Metric | Original Image | Boosted Image (Output) |
| --- | --- | --- |
| **Histogram** | Bunched in the middle (Grey/Flat). | Stretched across full range (0-255). |
| **Shadows** | Black void (No features). | Visible texture/grain. |
| **Colors** | Dull / Hazy. | Vibrant (due to higher local contrast). |
| **COLMAP Features** | Low (~200 points). | **High (~1500+ points).** |

---

## 🧠 Why This Matters for 3DGS?

3D Gaussian Splatting is particularly sensitive to "textureless regions."

* **Without Boost:** A white wall or a dark road looks like a single color. The AI cannot place splats there, leaving holes in the model.
* **With Boost:** The script reveals the micro-texture (grain) of the wall. The AI sees these distinct pixels and can successfully place 3D points, resulting in a fully dense reconstruction.

---

## ⚠️ Limitations

* **Noise Amplification:** Because this boosts local contrast, it will also make ISO noise (grain) more visible. If your images are extremely noisy, consider running a Denoising step *before* this script.
* **Clip Limit:** The script is hardcoded to `clipLimit=3.0`. This is an aggressive setting ideal for 3D reconstruction, but may look "over-processed" for artistic photography.