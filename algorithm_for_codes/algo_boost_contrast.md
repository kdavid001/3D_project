# Algorithm: CLAHE Contrast Booster
# File: boost_contrast.py

---

**Algorithm 3: BOOST_CONTRAST**

**Input:**
- input_dir  : folder containing source images (.jpg, .png)
- output_dir : folder to save contrast-enhanced images

**Output:**
- Contrast-enhanced images saved to output_dir/, preserving original colour fidelity

---

```
BEGIN BOOST_CONTRAST(input_dir, output_dir)

  images ← LIST all .jpg and .png files in input_dir

  // Safety check: prevent overwriting source files
  IF input_dir == output_dir THEN
    ENABLE safe_mode  // write to temporary folder first, then move
  END IF

  CREATE output_dir if it does not exist

  // Initialise CLAHE operator (Contrast Limited Adaptive Histogram Equalization)
  clahe ← CREATE CLAHE with {
    clipLimit    : 3.0,   // max contrast amplification per tile
    tileGridSize : (8, 8) // local neighbourhood tiles
  }

  FOR each image I in images DO

    // Step 1: Load image in BGR colour space
    img_bgr ← READ I as BGR array

    // Step 2: Convert from BGR to LAB colour space
    //   L channel: lightness (luminance only)
    //   A channel: red-green axis  (colour — DO NOT TOUCH)
    //   B channel: blue-yellow axis (colour — DO NOT TOUCH)
    img_lab ← CONVERT img_bgr from BGR to LAB

    // Step 3: Isolate the L (lightness) channel
    L, A, B ← SPLIT img_lab into three channels

    // Step 4: Apply CLAHE ONLY to the L channel
    //   This boosts local contrast adaptively across 8×8 grid tiles
    //   without blowing out globally bright regions
    L_enhanced ← APPLY clahe to L

    // Step 5: Recombine channels
    img_lab_enhanced ← MERGE(L_enhanced, A, B)

    // Step 6: Convert back to BGR for saving
    img_bgr_enhanced ← CONVERT img_lab_enhanced from LAB to BGR

    // Step 7: Save output
    output_path ← output_dir / filename(I)
    SAVE img_bgr_enhanced to output_path

  END FOR

END BOOST_CONTRAST
```

---

**Notes:**
- Operating in LAB colour space ensures that only luminance is modified; A and B (chrominance) channels are untouched, preserving the original colour hue
- CLAHE uses localised histogram equalisation per tile rather than global equalisation, preventing highlight blow-out while recovering shadow texture
- clipLimit=3.0 is aggressive — suitable for feature-starved 3DGS inputs. For artistic use, reduce to 1.5–2.0
- Output images are structurally identical to inputs and compatible with any downstream COLMAP or 3DGS pipeline
- If input_dir == output_dir, safe_mode writes to a temporary location first to prevent mid-batch corruption
