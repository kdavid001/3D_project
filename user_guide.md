# Preprocessing Pipeline User Guide

This guide provides instructions for environment setup, installing dependencies, 
and using the data preprocessing scripts (`corrupt_data.py`, 
`process_file.py`, `mask2.py`, `ML_mask_gen.py`, etc.) for the dataset processing workflow.

---

## 1. Environment Setup

Before running any scripts, activate the correct conda or virtual environment.

```bash
conda deactivate
source nsenv/bin/activate
```mm;kz

---

## 2. Install Python Dependencies

Install all required packages using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

---

## 3. Data Corruption with `corrupt_data.py`

You can corrupt your dataset for data augmentation or robustness testing.

### For Natural Data

```bash
python3 corrupt_data.py \
  --clean_dir ./others/tandt_db/tandt/train \
  --out_dir ./output_train/ \
  --corrupt_prob 0.8 \
  --delete_prob 0.65 \
  --seed 42
```

### For NeRF Synthetic Data

```bash
python3 corrupt_data.py \
  --clean_dir ./nerf_synthetic/lego \
  --out_dir ./output_train \
  --use_geometry \
  --corrupt_prob 0.8 \
  --delete_prob 0.65 \
  --seed 42
```

---

## 4. Processing Files with `process_file.py`

### Single Image Test

```bash
python3 process_file.py \
  --mode natural \
  --test_image output_train/train/images/00025.jpg
```

### Batch Processing

```bash
python3 process_file.py \
  --mode natural \
  --input_dir ./output_train/train \
  --out_dir ./output_train/train
```

#### For NeRF Synthetic Data (Single Image)

```bash
python process_file.py --test_image ./output_processed/lego/train/r_82.png
```

#### NeRF Synthetic Batch

```bash
python process_file.py \
  --mode synthetic \
  --input_dir ./output_train/lego \
  --out_dir ./output_processed/lego \
  --debug
```

---

## 5. Mask Generation with `mask2.py`

### Single Image Test

```bash
python3 mask2.py \
  --manifest ./output_train/train/manifest.json \
  --test \
  --test_image 00025.jpg
```

### Batch Mask Generation

```bash
python3 mask2.py \
  --manifest ./output_train/train/manifest.json \
  --output ./inpainting_ready \
  --blur_thresh 100 \
  --texture_thresh 0.02
```

You can further tune parameters:

```bash
python mask2.py \
  --manifest ./output_train/train/manifest.json \
  --output ./inpainting_ready \
  --blur_thresh 100 \
  --texture_thresh 0.02 \
  --min_area 500 \
  --expand 5
```

---

## 6. Tunable Parameters

| Parameter         | Default | Lower =                | Higher =             |
|-------------------|---------|------------------------|----------------------|
| `--blur_thresh`   | 100     | More sensitive         | Less sensitive       |
| `--texture_thresh`| 0.02    | More sensitive         | Less sensitive       |
| `--min_area`      | 500     | Keep smaller regions   | Remove more noise    |
| `--expand`        | 5       | Less expansion         | More expansion       |
| `--feather`       | 0       | Hard edges             | Soft edges (3-10)    |

---

## 7. ML-based Mask Generation

```bash
python ML_mask_gen.py \
  --manifest ./output_processed/lego/manifest.json \
  --output ./inpainting_ready
```

---

## 8. Diffusion and Other Postprocess

```bash
python diffusion_script.py --step1_dir ./output_train/train --step2_dir ./inpainting_ready
```

---

## 9. File Renaming

```bash
python rename.py --input_dir ./preprocessed/final_results
```

---

## Notes

- Make sure to adjust file paths and parameter values as needed for your specific datasets and requirements.
- For more details on each script’s functionality and options, use the `--help` flag (e.g., `python mask2.py --help`).

---

**Maintainer:** kdavid001
````