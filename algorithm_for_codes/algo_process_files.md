# Algorithm: Five-Pillar Quality Screener & Router
# File: process_files.py

---

**Algorithm 4: QUALITY_SCREEN_AND_ROUTE**

**Input:**
- input_dir : root folder containing the image dataset
- mode      : "natural" (lenient) or "synthetic" (strict) thresholds
- debug     : boolean — generate 3-panel diagnostic charts if True

**Output:**
- manifest.json : per-image decisions with quality scores and routing tags
- processed/    : copies of passing images
- debug_visuals/: 3-panel diagnostic report cards (if debug=True)

---

```
BEGIN QUALITY_SCREEN_AND_ROUTE(input_dir, mode, debug)

  // --- Define thresholds by dataset type ---
  IF mode == "synthetic" THEN
    T ← { musiq_min:65.0, clip_pct:0.02, sat_avg:160.0,
          bright_min:45.0, bright_max:230.0,
          contrast_min:25.0, color_cast:50.0 }
  ELSE  // natural
    T ← { musiq_min:40.0, clip_pct:0.05, sat_avg:180.0,
          bright_min:45.0, bright_max:250.0,
          contrast_min:10.0, color_cast:60.0 }
  END IF

  manifest ← EMPTY DICTIONARY
  images   ← LIST all images in input_dir

  FOR each image I in images DO

    fail_reasons ← EMPTY LIST

    // === PILLAR 1: Optical Quality ===
    q_score ← MUSIQ(I)            // neural no-reference IQA, range 0–100
    IF q_score < T.musiq_min THEN
      APPEND "blur" to fail_reasons
    END IF

    // === PILLAR 2: Radiometric Saturation ===
    hsv ← CONVERT I to HSV
    S   ← hsv.channel_S
    IF (COUNT pixels where S > 240) / TOTAL_PIXELS > T.clip_pct THEN
      APPEND "neon_clip" to fail_reasons
    END IF
    IF MEAN(S) > T.sat_avg THEN
      APPEND "high_saturation" to fail_reasons
    END IF

    // === PILLAR 3: Exposure Integrity ===
    V ← hsv.channel_V
    brightness ← MEAN(V)
    IF brightness < T.bright_min THEN
      APPEND "underexposed" to fail_reasons
    ELSE IF brightness > T.bright_max THEN
      APPEND "overexposed" to fail_reasons
    END IF

    // === PILLAR 4: Contrast / Flatness ===
    IF STD_DEV(V) < T.contrast_min THEN
      APPEND "flat" to fail_reasons
    END IF

    // === PILLAR 5: Color Cast ===
    lab    ← CONVERT I to LAB
    avg_a  ← MEAN(lab.channel_A)
    avg_b  ← MEAN(lab.channel_B)
    cast_d ← EUCLIDEAN_DISTANCE((avg_a, avg_b), (128, 128))
    IF cast_d > T.color_cast THEN
      APPEND "color_cast" to fail_reasons
    END IF

    // === ROUTING DECISION ===
    IF fail_reasons is EMPTY THEN
      tag ← "NOVEL_VIEW"     // good image → synthesis engine (Zero123++)
      COPY I to processed/
    ELSE
      tag ← "REPAIR"         // degraded image → restoration engine (ControlNet)
    END IF

    manifest[filename(I)] ← {
      score    : q_score,
      decision : tag,
      reasons  : fail_reasons
    }

    // === DEBUG VISUALISATION (optional) ===
    IF debug THEN
      panel_1 ← DRAW I with colour-coded PASS/FAIL title + MUSIQ score
      panel_2 ← PLOT RGB histogram with "UNBALANCED" text if cast_d > threshold
      panel_3 ← RENDER table: each pillar value vs limit, with PASS/FAIL icon
      SAVE 3-panel composite to debug_visuals/
    END IF

  END FOR

  WRITE manifest to manifest.json

END QUALITY_SCREEN_AND_ROUTE
```

---

**Routing Summary:**

| Condition              | Tag          | Downstream Component       |
|------------------------|--------------|----------------------------|
| All 5 pillars pass     | NOVEL_VIEW   | Zero123++ view synthesis   |
| Any pillar fails       | REPAIR       | ControlNet Tile restoration|

---

**Notes:**
- MUSIQ (Pillar 1) requires a pre-trained model download via pyiqa on first run
- manifest.json is the handshake file consumed by diffusion_script_v0.py (Algorithm 5)
- Pillar thresholds were calibrated empirically on the NeRF Synthetic dataset (hotdog, lego)
