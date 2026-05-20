# Algorithm: Radiometric Corruption Generator
# File: corrupt_dataset_v3.py

---

**Algorithm 1: CORRUPT_DATASET**

**Input:**
- clean_dir       : path to clean input image dataset
- out_dir         : path to save corrupted output
- corrupt_prob    : probability of applying corruption per image (0.0–1.0)
- use_geometry    : flag to delete sector based on camera transforms
- delete_prob     : probability of randomly deleting an image (sparse simulation)

**Output:**
- Corrupted dataset saved to out_dir/, mixing clean and corrupted images

---

```
BEGIN CORRUPT_DATASET(clean_dir, out_dir, corrupt_prob, use_geometry, delete_prob)

  images ← LIST all image files in clean_dir
  CREATE out_dir if it does not exist

  FOR each image I in images DO

    // Step 1: Sparse deletion (missing view simulation)
    IF random() < delete_prob THEN
      SKIP image I (do not copy to out_dir)
      CONTINUE

    // Step 2: Decide whether to apply corruption
    IF random() < corrupt_prob THEN
      corruption_type ← RANDOM CHOICE from {
        "saturation",   // sensor clipping
        "overexpose",   // shutter too long
        "underexpose",  // shutter too short
        "blur",         // optical defocus
        "noise"         // ISO grain
      }

      CASE corruption_type OF

        "saturation":
          // Sensor Saturation — "Deep Fried" effect
          hsv ← CONVERT I from RGB to HSV
          factor ← RANDOM FLOAT in [2.0, 4.0]
          hsv.S_channel ← CLIP(hsv.S_channel × factor, 0, 255)
          I_out ← CONVERT hsv back to RGB

        "overexpose":
          // Flash Bang — dynamic range loss in highlights
          factor ← RANDOM FLOAT in [2.5, 4.0]
          I_out ← CLIP(I × factor, 0, 255)

        "underexpose":
          // Crushed shadows — signal-to-noise collapse
          factor ← RANDOM FLOAT in [0.1, 0.3]
          I_out ← CLIP(I × factor, 0, 255)

        "blur":
          // Optical defocus — Gaussian blur kernel
          kernel_size ← RANDOM ODD INTEGER in {7, 11, 15, 21}
          I_out ← APPLY Gaussian blur with kernel (kernel_size × kernel_size) to I

        "noise":
          // ISO grain — thermal sensor noise
          sigma ← RANDOM FLOAT in [30, 80]
          noise_map ← SAMPLE Gaussian noise N(0, sigma) same shape as I
          I_out ← CLIP(I + noise_map, 0, 255)

      END CASE

    ELSE
      // No corruption — copy clean image
      I_out ← I

    END IF

    // Step 3: Geometric sector deletion (optional)
    IF use_geometry THEN
      Load camera transform for I from transforms.json
      IF camera is in the negative-X sector THEN
        I_out ← BLACK IMAGE (sector deleted)
      END IF
    END IF

    // Step 4: Save output
    SAVE I_out to out_dir with same filename as I

  END FOR

END CORRUPT_DATASET
```

---

**Notes:**
- Corruption types simulate real passive sensor failures, not artificial digital masks
- The output dataset is structurally identical to NeRF loader format (transforms.json preserved)
- Combine with load_images_v23.py (Algorithm 2) to screen the corrupted set downstream
