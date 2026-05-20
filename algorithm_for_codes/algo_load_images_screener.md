# Algorithm: Radiometric Integrity Screener
# File: load_images_v23.py

---

**Algorithm 2: RADIOMETRIC_SCREEN**

**Input:**
- input_dir  : path to dataset folder (may contain clean and corrupted images)
- mode       : "natural" or "synthetic" (controls threshold strictness)
- debug      : boolean — if True, generate visualisation charts

**Output:**
- Annotated screening log per image (pass/fail + reason)
- Histograms and heatmaps saved to debug_visuals/ (if debug=True)

---

```
BEGIN RADIOMETRIC_SCREEN(input_dir, mode, debug)

  // Thresholds depend on dataset type
  IF mode == "synthetic" THEN
    thresholds ← {
      musiq_min   : 65.0,
      clip_pct    : 0.02,   // 2%
      sat_avg     : 160.0,
      bright_min  : 45.0,
      bright_max  : 230.0,
      contrast_min: 25.0,
      color_cast  : 50.0
    }
  ELSE  // natural
    thresholds ← {
      musiq_min   : 40.0,
      clip_pct    : 0.05,   // 5%
      sat_avg     : 180.0,
      bright_min  : 45.0,
      bright_max  : 250.0,
      contrast_min: 10.0,
      color_cast  : 60.0
    }
  END IF

  results ← EMPTY LIST
  images  ← LIST all image files in input_dir

  FOR each image I in images DO

    failures ← EMPTY LIST

    // --- Pillar 1: Optical Quality (Blur / Noise) ---
    musiq_score ← COMPUTE MUSIQ no-reference quality score for I
    IF musiq_score < thresholds.musiq_min THEN
      APPEND "Blurry/Noisy" to failures
    END IF

    // --- Pillar 2: Radiometric Saturation ---
    hsv ← CONVERT I from BGR to HSV
    S   ← hsv.S_channel
    clipped_ratio ← COUNT(S > 240) / TOTAL_PIXELS(S)
    avg_saturation ← MEAN(S)

    IF clipped_ratio > thresholds.clip_pct THEN
      APPEND "Neon/Clipped" to failures
    END IF
    IF avg_saturation > thresholds.sat_avg THEN
      APPEND "High Saturation" to failures
    END IF

    // --- Pillar 3: Exposure Integrity ---
    V ← hsv.V_channel   // brightness channel
    avg_brightness ← MEAN(V)

    IF avg_brightness < thresholds.bright_min THEN
      APPEND "Underexposed" to failures
    END IF
    IF avg_brightness > thresholds.bright_max THEN
      APPEND "Overexposed" to failures
    END IF

    // --- Pillar 4: Contrast (Flatness) ---
    contrast ← STD_DEV(V)
    IF contrast < thresholds.contrast_min THEN
      APPEND "Flat/Low Contrast" to failures
    END IF

    // --- Pillar 5: Color Cast (Tint) ---
    lab ← CONVERT I from BGR to LAB
    a_channel ← lab.A_channel
    b_channel ← lab.B_channel
    avg_a  ← MEAN(a_channel)
    avg_b  ← MEAN(b_channel)
    // Neutral grey is at (128, 128) in LAB
    color_cast_dist ← SQRT((avg_a - 128)^2 + (avg_b - 128)^2)

    IF color_cast_dist > thresholds.color_cast THEN
      APPEND "Color Cast" to failures
    END IF

    // --- Decision ---
    IF failures is EMPTY THEN
      decision ← "PASS"
    ELSE
      decision ← "FAIL"
      reason   ← JOIN failures with ", "
    END IF

    APPEND {filename, musiq_score, decision, reason} to results

    // --- Debug Visualisation (optional) ---
    IF debug THEN
      Panel_1 ← DRAW original image with color-coded PASS/FAIL title
      Panel_2 ← PLOT RGB histogram with channel labels and "UNBALANCED" overlay if cast detected
      Panel_3 ← RENDER text inspection report listing all 5 pillar values vs thresholds
      SAVE 3-panel image to debug_visuals/
    END IF

  END FOR

  SAVE results to manifest.json

END RADIOMETRIC_SCREEN
```

---

**Notes:**
- MUSIQ is a no-reference neural quality metric (range 0–100). Does not require a reference image.
- Pillar 2 uses HSV saturation, not RGB channels — more robust to lighting variation
- LAB colour cast distance from (128, 128) measures chromatic neutrality
- Output manifest.json is consumed by diffusion_script_v0.py for routing decisions
