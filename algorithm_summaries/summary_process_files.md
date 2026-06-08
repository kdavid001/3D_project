# Algorithm Summary: Five-Pillar Quality Screener & Router
# Source: process_files.py | Full algorithm: algorithm_for_codes/algo_process_files.md

---

**Algorithm 4 (Summary): QUALITY_SCREEN_AND_ROUTE**

**Input:**
- input_dir : root dataset folder
- mode      : "natural" (lenient) or "synthetic" (strict)
- debug     : generate 3-panel diagnostic charts if True

**Output:**
- manifest.json  : per-image decision (NOVEL_VIEW / REPAIR) with MUSIQ score
- processed/     : copies of passing images
- debug_visuals/ : 3-panel diagnostic charts (if debug=True)

---

```
BEGIN QUALITY_SCREEN_AND_ROUTE(input_dir, mode, debug)

  T ← LOAD thresholds for mode ("natural" or "synthetic")

  FOR each image I in input_dir DO

    fail_reasons ← EMPTY LIST

    // Pillar 1: Optical quality (neural IQA)
    IF MUSIQ(I) < T.musiq_min           → APPEND "blur"

    // Pillar 2: Saturation
    S ← HSV(I).S_channel
    IF clipped_ratio(S) > T.clip_pct   → APPEND "neon_clip"
    IF MEAN(S) > T.sat_avg             → APPEND "high_saturation"

    // Pillar 3: Exposure
    V ← HSV(I).V_channel
    IF MEAN(V) < T.bright_min          → APPEND "underexposed"
    IF MEAN(V) > T.bright_max          → APPEND "overexposed"

    // Pillar 4: Contrast
    IF STD_DEV(V) < T.contrast_min     → APPEND "flat"

    // Pillar 5: Colour cast
    cast ← EUCLIDEAN_DIST(MEAN(LAB.A, LAB.B), (128,128))
    IF cast > T.color_cast             → APPEND "color_cast"

    // Routing decision
    IF fail_reasons EMPTY THEN
      tag ← "NOVEL_VIEW"   // → Zero123++ synthesis
      COPY I to processed/
    ELSE
      tag ← "REPAIR"       // → ControlNet restoration
    END IF

    manifest[I] ← { score, decision: tag, reasons: fail_reasons }

    IF debug → SAVE 3-panel diagnostic chart to debug_visuals/

  END FOR

  WRITE manifest to manifest.json

END QUALITY_SCREEN_AND_ROUTE
```

---

**Routing Summary:**

| Outcome            | Tag        | Downstream             |
|--------------------|------------|------------------------|
| All 5 pillars pass | NOVEL_VIEW | Zero123++ (Algorithm 5)|
| Any pillar fails   | REPAIR     | ControlNet (Algorithm 5)|

---
