# CoDiffusion: Generative 3D Reconstruction Pipeline

**CoDiffusion** is a fault-tolerant pipeline designed to rescue 3D reconstruction projects from low-quality, sparse, or corrupt input data (e.g., blurry video frames).

Instead of relying solely on traditional Structure-from-Motion (SfM), which fails on blurry inputs, this pipeline uses **Generative AI** to:

1. **Filter** usable keyframes from corrupt video.
2. **Synthesize** novel views (hallucinating missing angles) using **qZero123++**.
3. **Upscale** synthetic data to 4K resolution using **Real-ESRGAN**.
4. **Prepare** a robust dataset optimized for **Gaussian Splatting (3DGS)** or **COLMAP**.

---

## 🚀 Key Features

* **Graceful Degradation:** Turns "crash-prone" sparse datasets into dense, usable point clouds.
* **Multi-View Swarm Generation:** Generates 6 consistent novel views for every single good input image.
* **Auto-Cleaning:** Automatically removes backgrounds (`rembg`) and centers objects to prevent "double object" hallucinations.
* **Hybrid Resolution:** Blends original 4K photography with AI-upscaled synthetic details.
* **COLMAP-Optimized Sorting:** Prioritizes real geometry during initialization to ensure accurate scale and orientation.

---

## 🛠️ Installation

**Prerequisites:**

* NVIDIA GPU (A100/L4 recommended, T4 compatible).
* Python 3.10+
* CUDA 11.8+

**Install Dependencies:**

```bash
# 1. Core Generative Libraries
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install diffusers transformers accelerate rembg

# 2. Image Processing & Upscaling
pip install opencv-python pillow tqdm
pip install onnxruntime-gpu  # Important for Background Removal speed
pip install basicsr realesrgan

```

*Note: If you encounter `basicsr` errors regarding `torchvision`, the scripts include an auto-patcher to fix version conflicts.*

---

## 🏃‍♂️ Usage

### Phase 1: The "All-in-One" Generation

Run the main pipeline. This script handles filtering, background removal, Zero123++ synthesis, and Real-ESRGAN upscaling in a single pass to manage GPU memory efficiently.

```bash
python run_full_generation_v2.py \
  --input_dir "/path/to/your/step1_output" \
  --out_dir "/path/to/save/results"

```

* **Input:** The folder containing your `manifest.json` and processed images (from the Step 1 Quality Filter).
* **Output:** A folder named `final_dataset_[name]` containing high-res synthetic and real images.

### Phase 2: Sequence Preparation (Critical for COLMAP)

Before running 3D reconstruction, you must rename the files so that **Real Images** come first (initializing the geometry) and **Synthetic Images** come last (filling the gaps).

```bash
python rename_sequence.py \
  --input_dir "/path/to/save/results/final_dataset_name"

```

* **Result:** Images will be renamed to `0001.jpg`, `0002.jpg`...
* **Order:** Real Images (0001–00XX) -> Synthetic Images (00XX–01XX).

---

## 🧠 How It Works (The Pipeline)

1. **Ingestion & Filtering:** The system scans the `manifest.json` to find the highest-quality "Anchor Images" (high Laplacian variance, low blur).
2. **Preprocessing (RemBG):** Anchor images are stripped of their background and centered on a 512x512 canvas. This prevents the "floating artifacts" common in Zero123.
3. **View Synthesis (Zero123++):** The diffusion model generates 6 new camera angles (Front-Left, Back-Right, Top-Down, etc.) for each anchor.
4. **Super-Resolution (Real-ESRGAN):** Since diffusion models output low-res (512px) images, the upscaler expands them 4x (to 2048px) to match the original camera fidelity.
5. **Sequential Locking:** The renaming script ensures that when you run COLMAP, the SfM engine locks onto real features first, preventing the reconstruction from "drifting" into AI hallucinations.

---

## 📂 Project Structure

```text
CoDiffusion/
├── run_full_generation_v2.py   # Main Generation Pipeline (Synthesis + Upscale)
├── rename_sequence.py          # Sorting utility for COLMAP optimization
├── weights/                    # Stores Real-ESRGAN models (auto-downloaded)
└── README.md                   # This file

```

## ⚖️ Limitations

* **Texture Smoothing:** Synthetic views may look "cleaner" or smoother than real photos, potentially smoothing out very fine scratches or dirt.
* **Geometry Hallucination:** In extremely rare cases, Zero123++ may misinterpret complex hollow structures (like inside a tire) if the angle is completely blind.
* **VRAM Usage:** Requires ~16GB VRAM for stable execution (A100/L4). T4 users may need to reduce batch sizes.

## 🤝 Credits

* **Zero123++:** [Sudo-AI](https://www.google.com/search?q=https://github.com/Sudo-AI-3D/zero123plus) (View Synthesis)
* **Real-ESRGAN:** [Xinntao](https://github.com/xinntao/Real-ESRGAN) (Super-Resolution)
* **RemBG:** [Daniel Gatis](https://github.com/danielgatis/rembg) (Background Removal)