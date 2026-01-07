# CoDiffusion: Universal Generative 3D Pipeline

**CoDiffusion** is a fault-tolerant pipeline designed to rescue 3D reconstruction projects from low-quality, sparse, or corrupt input data.

Unlike traditional pipelines that fail on blurry or sparse inputs, this system uses a **Universal Diffusion Architecture** with two distinct modes:

1.  **Synthesis Mode (Zero123++):** Hallucinates *novel* camera angles to fill gaps in sparse datasets.
2.  **Restoration Mode (ControlNet):** Repairs *existing* noisy or blurry natural images without altering their geometry, ensuring COLMAP can track features.

---

## 🚀 Key Features

* **Dual-Mode Engine:** Switch between *generating new views* (Synthesis) or *fixing bad views* (Restoration) with a single flag.
* **Graceful Degradation:** Turns "crash-prone" datasets into dense, usable point clouds.
* **Natural Image Restoration:** Uses **ControlNet Tile** to denoise and sharpen real-world photos while preserving background context for SfM.
* **Multi-View Swarm Generation:** Generates 6 consistent novel views for every single anchor image (Synthesis Mode).
* **Hybrid Resolution:** Blends original 4K photography with AI-upscaled synthetic details (Real-ESRGAN).

---

## 🛠️ Installation

**Prerequisites:**

* NVIDIA GPU (A100/L4 recommended, T4 compatible).
* Python 3.10+
* CUDA 11.8+

**Install Dependencies:**

```bash
# 1. Core Generative Libraries & Accelerators
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
pip install diffusers transformers accelerate safetensors protobuf

# 2. ControlNet & Image Processing
pip install controlnet_aux rembg onnxruntime-gpu
pip install opencv-python pillow tqdm

# 3. Upscaling (Real-ESRGAN)
pip install basicsr realesrgan
```

*Note: If you encounter `basicsr` errors regarding `torchvision`, run the included auto-patcher snippet before execution.*

---

## 🏃‍♂️ Usage

The pipeline now operates in two modes using `diffusion_script.py`.

### Mode A: Synthesis (Default)

**Best for:** Sparse datasets, object scans, or when you have < 20 images.

* **Action:** Removes background -> Centers Object -> Generates 6 new angles per image -> Upscales.

```bash
python diffusion_script.py \
  --input_dir "/path/to/your/input_images" \
  --out_dir "/path/to/save/results" \
  --mode synthesis

```

### Mode B: Restoration (New)

**Best for:** Corrupt/blurry natural images (trains, landscapes, scenes).

* **Action:** Keeps background (No masking) -> Denoises & Sharpens using ControlNet -> Upscales.
* **Note:** This mode preserves the original camera perspective so COLMAP can still calculate the pose.

```bash
python diffusion_script.py \
  --input_dir "/path/to/your/corrupt_images" \
  --out_dir "/path/to/save/results" \
  --mode restoration \
  --prompt "high quality photo, detailed, sharp focus, 8k"

```

### Phase 2: Sequence Preparation (Critical for COLMAP)

After generating/restoring images, run the renaming utility to ensure COLMAP prioritizes the best images first.

```bash
python rename_sequence.py \
  --input_dir "/path/to/save/results/final_dataset_run"

```

* **Result:** Images are renamed to `0001.jpg`, `0002.jpg`...
* **Logic:** Real/Restored images are placed first (0001–00XX) to lock geometry; Synthetic images are placed last (00XX+).

---

## 🧠 How It Works (The Pipeline)

The system automatically routes logic based on your `--mode`:

### 1. Ingestion & Filtering

* **Synthesis:** Scans `manifest.json` for **"NOVEL_VIEW"** candidates (anchors).
* **Restoration:** Scans for **"REPAIR"** tags, or defaults to processing **ALL** images in the folder if no manifest exists.

### 2. The Diffusion Pass

* **Path A (Synthesis):** Images are masked (black BG) and centered. **Zero123++** generates 6 novel views.
* **Path B (Restoration):** Images are kept raw (with background). **ControlNet Tile** + **Stable Diffusion 1.5** hallucinates high-frequency details while locking onto the original geometry.

### 3. Super-Resolution

Both paths feed into **Real-ESRGAN**, which takes the 512px AI output and upscales it 4x (to 2048px+) to match modern camera fidelity.

---

## 📂 Project Structure

```text
CoDiffusion/
├── diffusion_script.py         # Universal Pipeline (Synthesis + Restoration)
├── rename_sequence.py          # Sorting utility for COLMAP optimization
├── weights/                    # Stores Real-ESRGAN/ControlNet models (auto-downloaded)
└── README.md                   # This file

```

## ⚖️ Limitations
* **Synthesis Hallucinations:** Zero123++ may misinterpret complex hollow structures (e.g., inside a tire) from blind angles.
* **Restoration "Dreaming":** If the `--prompt` is too specific (e.g., "hotdog") on a generic object, ControlNet might force-texture the object incorrectly. Use generic prompts ("high quality photo") for safety.
* **VRAM:** Restoration mode is lighter on VRAM (~8GB) than Synthesis mode (~16GB).

## 🤝 Credits
* **Zero123++:** [Sudo-AI](https://www.google.com/search?q=https://github.com/Sudo-AI-3D/zero123plus) (Novel Views)
* **ControlNet:** [lllyasviel](https://github.com/lllyasviel/ControlNet) (Restoration/Tile)
* **Real-ESRGAN:** [Xinntao](https://github.com/xinntao/Real-ESRGAN) (Super-Resolution)

```

```