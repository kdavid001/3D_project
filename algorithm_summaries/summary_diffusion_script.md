# Algorithm Summary: Dual-Mode Generative Augmentation Engine
# Source: diffusion_script_v0.py | Full algorithm: algorithm_for_codes/algo_diffusion_script.md

---

**Algorithm 5 (Summary): GENERATIVE_AUGMENTATION**

**Input:**
- input_dir : dataset folder (contains manifest.json + source images)
- out_dir   : destination for augmented output
- mode      : "synthesis" (object-centric) or "restoration" (degraded images)
- prompt    : text prompt for ControlNet (restoration mode only)

**Output:**
- final_{name}_run/
    - anchor_*.png       : real images on black 512×512 canvas
    - synth_*_v[0-5].png : Zero123++ novel views (rembg applied)
    - restored_*.png     : ControlNet-repaired images
- All outputs upscaled to 1024×1024 via RealESRGAN x4

---

```
BEGIN GENERATIVE_AUGMENTATION(input_dir, out_dir, mode, prompt)

  manifest ← LOAD manifest.json from input_dir

  // ============================================================
  // BRANCH A — SYNTHESIS (Zero123++)
  // ============================================================
  IF mode == "synthesis" THEN

    candidates ← FILTER manifest WHERE decision IN {NOVEL_VIEW, NONE}
    candidates ← SORT by score DESC, TAKE top 20

    LOAD Zero123++ v1.2 model

    FOR each anchor A in candidates DO

      // Prepare: black 512×512 canvas, object centred at 85%
      canvas ← CREATE 512×512 black image
      PASTE RESIZE(A, 85%) centred on canvas
      SAVE canvas as anchor_{stem}.png in temp_dir

      // Generate 6 novel views (2-column × 3-row output grid)
      grid_2x3 ← Zero123pp_infer(canvas, steps=75)

      FOR v in {0..5} DO
        tile ← CROP grid_2x3 at grid cell (col=v%2, row=v//2)
        tile ← rembg(tile)               // remove grey Zero123++ background
        PASTE tile on new 512×512 black canvas
        SAVE as synth_{stem}_v{v}.png in temp_dir
      END FOR

    END FOR

  // ============================================================
  // BRANCH B — RESTORATION (ControlNet Tile)
  // ============================================================
  ELSE IF mode == "restoration" THEN

    LOAD ControlNet Tile + Stable Diffusion 1.5

    FOR each entry E in manifest DO
      IF E.decision IN {REPAIR, BAD, DISCARD, blur} THEN
        I_restored ← pipe(
          image         = I_degraded,
          control_image = I_degraded,   // tile conditioning = input itself
          prompt        = prompt,
          strength      = 0.35,
          guidance      = 7.0,
          steps         = 30
        )
        SAVE as restored_{stem}.png in temp_dir
      ELSE
        COPY I as anchor_{stem}.png in temp_dir  // pass-through
      END IF
    END FOR

  END IF

  // ============================================================
  // SHARED: Upscale to 1024×1024 (both modes)
  // ============================================================
  LOAD RealESRGAN x4plus model

  FOR each file F in temp_dir DO
    IF "FULL_GRID_" in filename(F) THEN SKIP
    img ← upscaler.enhance(F, outscale=4)
    SAVE RESIZE(img, 1024×1024) to final_{name}_run/
  END FOR

END GENERATIVE_AUGMENTATION
```

---

**Output Naming Convention:**

| Prefix       | Source                        |
|--------------|-------------------------------|
| anchor_*     | Real photo, black BG          |
| synth_*_v[n] | Zero123++ tile, rembg applied |
| restored_*   | ControlNet output             |
| FULL_GRID_*  | Debug grid — skipped          |

---
