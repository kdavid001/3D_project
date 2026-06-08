# Algorithm Summary: Radiometric Integrity Screener
# Source: load_images_v23.py | Full algorithm: algorithm_for_codes/algo_load_images_screener.md

---

**Algorithm 2 (Summary): RADIOMETRIC_SCREEN**

**Input:**
- input_dir : path to dataset folder
- mode      : "natural" (lenient) or "synthetic" (strict)
- debug     : generate 3-panel diagnostic charts if True

**Output:**
- Screening log per image (pass/fail + reason)
- debug_visuals/ charts (if debug=True)

---

```
BEGIN RADIOMETRIC_SCREEN(input_dir, mode, debug)

  T ← LOAD thresholds for mode ("natural" or "synthetic")

  FOR each image I in input_dir DO

    failures ← EMPTY LIST

    // Pillar 1: Optical Quality
    IF MUSIQ(I) < T.musiq_min           → APPEND "Blurry/Noisy"

    // Pillar 2: Saturation
    S ← HSV(I).S_channel
    IF clipped_ratio(S) > T.clip_pct   → APPEND "Neon/Clipped"
    IF MEAN(S) > T.sat_avg             → APPEND "High Saturation"

    // Pillar 3: Exposure
    V ← HSV(I).V_channel
    IF MEAN(V) < T.bright_min          → APPEND "Underexposed"
    IF MEAN(V) > T.bright_max          → APPEND "Overexposed"

    // Pillar 4: Contrast
    IF STD_DEV(V) < T.contrast_min     → APPEND "Flat"

    // Pillar 5: Color Cast
    cast ← EUCLIDEAN_DIST(MEAN(LAB(I).A, LAB(I).B), (128,128))
    IF cast > T.color_cast             → APPEND "Color Cast"

    decision ← "PASS" if failures EMPTY else "FAIL"

    IF debug → SAVE 3-panel chart to debug_visuals/

  END FOR

  SAVE results to manifest.json

END RADIOMETRIC_SCREEN
```

---

**Threshold Comparison:**

| Pillar | Natural | Synthetic |
|---|---|---|
| MUSIQ min | 40.0 | 65.0 |
| Clip % max | 5% | 2% |
| Sat avg max | 180.0 | 160.0 |
| Brightness range | [45, 250] | [45, 230] |
| Contrast min | 10.0 | 25.0 |
| Color cast max | 60.0 | 50.0 |

---
