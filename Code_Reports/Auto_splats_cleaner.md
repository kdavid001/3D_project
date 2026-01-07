# 🧠 AI-Powered COLMAP Pipeline (hloc)

## 📌 Overview

`run_ai_pipeline.py` is a modern replacement for the standard COLMAP `convert.py` script. Instead of using "hand-crafted" feature detectors like SIFT (which fail on smooth walls or low texture), this pipeline uses **Deep Learning** to "see" and match features that traditional algorithms miss.

It automates the entire **Structure-from-Motion (SfM)** process using the **Hierarchical Localization (hloc)** library, taking you from raw images to a trained sparse point cloud ready for Gaussian Splatting.

---

## ⚙️ The "SuperPoint + LightGlue" Engine

This pipeline utilizes two specific AI models that are currently the gold standard in computer vision:

1. **Feature Extraction: SuperPoint**
* **Old Way (SIFT):** Looks for high-contrast "blobs" or corners. Fails on white walls or sky.
* **New Way (SuperPoint):** A neural network trained to find interest points even in textureless or repetitive regions.


2. **Feature Matching: LightGlue**
* **Old Way (Nearest Neighbor):** Matches points based on simple pixel similarity. Prone to outliers.
* **New Way (LightGlue):** A deep network that "reasons" about the geometry of the scene. It rejects bad matches by understanding the 3D context, resulting in cleaner clouds with fewer outliers.



---

## 🚀 Key Features

* **Zero-Setup Dependency:** Automatically clones and installs `SuperGlue` and `hloc` if they are missing.
* **Exhaustive Matching:** Forces the AI to compare every image against every other image, ensuring maximum connectivity for small datasets (<100 images).
* **Auto-Undistortion:** Includes a wrapper for the `colmap image_undistorter`. It reads the AI-generated camera model and corrects the images (straightening curved lines from lens distortion) so 3DGS can learn them.
* **"Train-Ready" Formatting:** Automatically moves the resulting `.bin` files into the specific `sparse/0` folder structure required by the official Gaussian Splatting `train.py`.

---

## 🏃‍♂️ Usage

### **Prerequisites**

* **System:** Linux (Colab/Ubuntu) with `colmap` installed (`sudo apt install colmap`).
* **Python:** 3.8+
* **GPU:** Required for SuperPoint/LightGlue inference.

### **Command**

```bash
python run_ai_pipeline.py \
  --source_path "/content/drive/MyDrive/.../gaussian_splatting/data/hotdog" \
  --images "input"

```

### **Arguments**

| Argument | Description | Default |
| --- | --- | --- |
| `--source_path` | The root folder of your project (containing the `input` folder). | **Required** |
| `--images` | The name of the subfolder containing your source images. | `input` |

---

## 🧠 Pipeline Logic (Step-by-Step)

1. **Cleanup:** Aggressively deletes old `distorted/` and `sparse/` folders to prevent mixing data from previous failed runs.
2. **Extraction (SuperPoint):** Analyzes every image in `input/` and saves keypoints to `features.h5`.
3. **Matching (LightGlue):** Compares keypoints across all image pairs and saves valid connections to `matches.h5`.
4. **Reconstruction (PyCOLMAP):** Solves the math to determine where the cameras were in 3D space.
* *Output:* A raw sparse model in `distorted/sparse`.


5. **Undistortion (COLMAP CLI):**
* Takes the raw model and images.
* Calculates the lens distortion (pinhole/radial).
* Saves **corrected** images to `images/` and a **new** model to `sparse/0`.


6. **Final Handshake:** Checks if `cameras.bin`, `images.bin`, and `points3D.bin` are in the correct `sparse/0` folder. If not, it moves them there manually.

---

## ⚠️ Common Issues

* **`Xvfb` Error:** If running on a headless server (like Colab), the script uses `xvfb-run` to fake a monitor for COLMAP. If this fails, ensure you installed `xvfb` (`sudo apt install xvfb`).
* **Memory OOM:** LightGlue is efficient, but "Exhaustive Matching" on >200 images grows exponentially (). If you crash, you may need to switch the matching script to `pairs_from_retrieval` (sequential) instead of exhaustive.

---

## 📂 Output Structure

After running, your dataset folder will be fully populated:

```text
/hotdog/
├── input/              # Your original images
├── distorted/          # Intermediate AI files (features.h5, matches.h5)
├── images/             # NEW: Undistorted, straight images (used for training)
├── sparse/
│   └── 0/              # The 3D Model
│       ├── cameras.bin
│       ├── images.bin
│       └── points3D.bin
└── run_ai_pipeline.py

```