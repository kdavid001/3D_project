# Algorithm Summary: CLAHE Contrast Booster
# Source: boost_contrast.py | Full algorithm: algorithm_for_codes/algo_boost_contrast.md

---

**Algorithm 3 (Summary): BOOST_CONTRAST**

**Input:**
- input_dir  : folder containing source images
- output_dir : folder to write enhanced images

**Output:**
- Contrast-enhanced images (same filenames) in output_dir/

---

```
BEGIN BOOST_CONTRAST(input_dir, output_dir)

  clahe ← CREATE CLAHE { clipLimit: 3.0, tileGridSize: (8,8) }

  FOR each image I in input_dir DO

    img_bgr ← READ I

    // Work in LAB so only luminance is touched
    img_lab     ← CONVERT img_bgr BGR→LAB
    L, A, B     ← SPLIT img_lab
    L_enhanced  ← APPLY clahe to L     // A and B unchanged (colour preserved)

    img_out ← CONVERT MERGE(L_enhanced, A, B) LAB→BGR
    SAVE img_out to output_dir / filename(I)

  END FOR

END BOOST_CONTRAST
```

---

**Key Parameters:**

| Parameter    | Value  | Effect                                      |
|--------------|--------|---------------------------------------------|
| clipLimit    | 3.0    | Max per-tile contrast amplification         |
| tileGridSize | (8, 8) | Local neighbourhood tile count              |
| Colour space | LAB    | A/B channels untouched — hue is preserved   |

---
