# 🗂️ Dataset Organizer for COLMAP/3DGS rename.py

## 📌 Overview

`organize_dataset.py` is a utility script that prepares your image data for the 3D reconstruction pipeline. It solves a common issue where COLMAP fails or produces fragmented models because input filenames are messy, non-sequential, or contain special characters.

This script creates a **"Clean Room"** environment for your training data by:

1. **Sanitizing:** Removing old runs to prevent data mixing.
2. **Ordering:** Sorting images alphanumerically to preserve temporal/spatial sequence.
3. **Standardizing:** Renaming all files to a strict 5-digit sequence (`00001.jpg`, `00002.jpg`...) which helps COLMAP's sequential matcher work 20-30% faster.

---

## ⚙️ Logic Breakdown

### **1. The "Clean Room" Setup**

* **Action:** It looks for `[output_root]/[model_name]/input`.
* **Safety:** If that folder already exists, it **deletes it entirely** before starting. This ensures you never accidentally train on a mix of old "bad" images and new "restored" images.

### **2. Deterministic Sorting**

* **Action:** `all_files.sort()`
* **Why:** Operating systems sometimes list files in random order. This step forces a strict A-Z order. If your images are video frames, this preserves the camera path, allowing COLMAP to use "Sequential Matching" instead of the slower "Exhaustive Matching".

### **3. Sequential Renaming**

* **Action:** Renames `frame_873_restored.png` -> `00023.jpg`.
* **Benefit:** 3DGS training scripts often expect specific file patterns. Using `00001.jpg` guarantees compatibility with essentially every photogrammetry tool (COLMAP, Metashape, RealityCapture).

---

## 🚀 Usage

### **Prerequisites**

* Python 3.x
* `tqdm` (for the progress bar)

```bash
pip install tqdm

```

### **Command**

Use this after you have finished the **Restoration** or **Synthesis** steps.

```bash
python organize_dataset.py \
  --source "/content/drive/MyDrive/.../output_processed/train/final_train_run" \
  --output "/content/drive/MyDrive/.../gaussian_splatting/data" \
  --name "hotdog"

```

### **Arguments**

| Argument | Description | Example |
| --- | --- | --- |
| `--source` | The folder containing your final processed images (Restored/Synthesized). | `.../final_train_run` |
| `--output` | The root folder where you keep all your 3DGS datasets. | `.../gs_datasets` |
| `--name` | The name of the object. A subfolder will be created here. | `hotdog` |

---

## 📂 Output Structure

After running this script, your folder structure will look exactly like the standard **Gaussian Splatting** input format:

```text
/gs_datasets/
└── hotdog/
    └── input/
        ├── 00001.jpg  <-- Was "frame_001.png"
        ├── 00002.jpg
        ├── 00003.jpg
        └── ...

```

---

## 🧠 Why This Matters?

COLMAP uses image filenames to guess if images are "neighbors" in a sequence.

* **Messy Names:** `img_01.jpg`, `restored_05.png`, `z_02.jpg` -> COLMAP assumes random order -> **Slow Matching**.
* **Clean Names:** `00001.jpg`, `00002.jpg`, `00003.jpg` -> COLMAP assumes video sequence -> **Fast Matching**.

By running this script, you enable the use of the `--sequential_matching` flag in COLMAP, which can speed up feature matching by **5x-10x**.

... [3D Gaussian Splatting - Full Workflow](https://www.youtube.com/watch?v=UXtuigy_wYc) ...
This video walks through the complete pipeline from dataset organization to final rendering, highlighting why folder structure is critical for the training tools to work correctly.