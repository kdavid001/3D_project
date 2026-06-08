# Algorithm Summary: Radiometric Corruption Generator
# Source: corrupt_dataset_v3.py | Full algorithm: algorithm_for_codes/algo_corrupt_dataset.md

---

**Algorithm 1 (Summary): CORRUPT_DATASET**

**Input:**
- clean_dir     : path to clean image dataset
- corrupt_prob  : probability of applying corruption per image
- delete_prob   : probability of deleting an image (sparse simulation)
- use_geometry  : flag to enable camera-sector based deletion

**Output:**
- Corrupted dataset written to out_dir/

---

```
BEGIN CORRUPT_DATASET(clean_dir, out_dir, corrupt_prob, delete_prob, use_geometry)

  FOR each image I in clean_dir DO

    // Stage 1: Sparse deletion
    IF random() < delete_prob THEN
      SKIP I
      CONTINUE

    // Stage 2: Radiometric corruption
    IF random() < corrupt_prob THEN
      type ← RANDOM CHOICE {saturation, overexpose, underexpose, blur, noise}
      I ← APPLY degradation(type) to I

    // Stage 3: Geometric sector deletion (optional)
    IF use_geometry AND camera(I) in negative-X sector THEN
      I ← BLACK IMAGE

    SAVE I to out_dir

  END FOR

END CORRUPT_DATASET
```

---

**Degradation Types:**

| Type | Simulation | Parameter Range |
|---|---|---|
| Saturation | Sensor clipping — S channel boosted | factor ∈ [2.0, 4.0] |
| Overexpose | Highlight dynamic range loss | factor ∈ [2.5, 4.0] |
| Underexpose | Crushed shadow detail | factor ∈ [0.1, 0.3] |
| Blur | Optical defocus — Gaussian kernel | kernel ∈ {7, 11, 15, 21} |
| Noise | ISO grain — additive Gaussian noise | σ ∈ [30, 80] |

---
