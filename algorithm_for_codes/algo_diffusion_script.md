# Algorithm: Dual-Mode Generative Augmentation Engine
# File: diffusion_script_v0.py

---

**Algorithm 5: GENERATIVE_AUGMENTATION**

**Input:**
- input_dir  : dataset folder containing manifest.json and source images
- out_dir    : destination for augmented output
- mode       : "synthesis" (object-centric) or "restoration" (degraded images)
- prompt     : text prompt for ControlNet (restoration mode only)

**Output:**
- final_{name}_run/ : augmented image set
    - anchor_*.png      : real/anchor images (black background, centred)
    - synth_*_v[0-5].png: Zero123++ novel views with background removed
    - restored_*.png    : ControlNet-repaired degraded images
- All outputs upscaled to 1024×1024 via RealESRGAN x4

---

```
BEGIN GENERATIVE_AUGMENTATION(input_dir, out_dir, mode, prompt)

  manifest ← LOAD manifest.json from input_dir
  temp_dir ← CREATE temporary working directory

  // ============================================================
  // BRANCH A — SYNTHESIS MODE (Zero123++ novel view generation)
  // ============================================================
  IF mode == "synthesis" THEN

    // Select top-quality anchor images from manifest
    candidates ← FILTER manifest WHERE decision IN {"NOVEL_VIEW", "NONE"}
    candidates ← SORT by score DESCENDING, TAKE top 20

    LOAD Zero123++ v1.2 model (stabilityai/zero123plus-v1.2)

    FOR each anchor A in candidates DO

      // Step 1: Prepare anchor image
      canvas  ← CREATE 512×512 black image
      A_scaled ← RESIZE A to fit within 85% of canvas
      A_centred ← PASTE A_scaled centered on canvas

      // Step 2: Save anchor (no background removal — already black BG)
      SAVE A_centred as anchor_{stem(A)}.png in temp_dir

      // Step 3: Generate 6 novel views via Zero123++
      grid_2x3 ← Zero123pp_infer(
        input_image = A_centred,
        steps       = 75
      )
      // grid_2x3 is a 1024×672 image containing 2 rows × 3 columns of 512×336 tiles

      // Step 4: Save debug grid (not used downstream)
      SAVE grid_2x3 as FULL_GRID_{stem(A)}.png in temp_dir

      // Step 5: Crop 6 individual view tiles
      // Grid layout: 2 columns × 3 rows
      FOR v in {0, 1, 2, 3, 4, 5} DO
        row ← v // 2          // integer division (3 rows)
        col ← v mod 2         // remainder (2 columns)
        tile_w ← grid_2x3.width  // 2
        tile_h ← grid_2x3.height // 3
        tile ← CROP grid_2x3 at (col×tile_w, row×tile_h, tile_w, tile_h)

        // Step 6: Remove Zero123++ grey background via rembg
        tile_nobg ← rembg(tile)       // alpha-matting to extract foreground
        black_canvas ← CREATE 512×512 black image
        PASTE tile_nobg (foreground only) centered on black_canvas

        // Step 7: Save synthetic view with naming convention
        SAVE result as synth_{stem(A)}_v{v}.png in temp_dir

      END FOR

    END FOR

  // ============================================================
  // BRANCH B — RESTORATION MODE (ControlNet Tile repair)
  // ============================================================
  ELSE IF mode == "restoration" THEN

    // Load ControlNet Tile pipeline (Stable Diffusion 1.5 backbone)
    controlnet ← LOAD ControlNetModel("lllyasviel/control_v11f1e_sd15_tile")
    pipe        ← LOAD StableDiffusionControlNetImg2ImgPipeline with controlnet

    FOR each entry E in manifest DO

      IF E.decision IN {"REPAIR", "BAD", "DISCARD", "blur"} THEN

        // Restore degraded image
        I_degraded ← LOAD image for E
        I_restored ← pipe(
          image           = I_degraded,
          control_image   = I_degraded,   // tile conditioning = input itself
          prompt          = prompt,
          controlnet_conditioning_scale = 0.35,
          guidance_scale  = 7.0,
          num_inf_steps   = 30
        )
        SAVE I_restored as restored_{stem(E)}.png in temp_dir

      ELSE
        // Pass-through: copy good images without modification
        I ← LOAD image for E
        SAVE I as anchor_{stem(E)}.png in temp_dir

      END IF

    END FOR

  END IF

  // ============================================================
  // SHARED PHASE — Upscaling (both modes)
  // ============================================================
  upscaler ← LOAD RealESRGAN model (RealESRGAN_x4plus)
  final_dir ← CREATE out_dir / "final_{name}_run/"

  FOR each file F in temp_dir DO

    IF filename(F) STARTS WITH "FULL_GRID_" THEN
      SKIP  // debug grids not upscaled
    END IF

    img ← LOAD F
    img_upscaled ← upscaler.enhance(img, outscale=4)
    img_1024     ← RESIZE img_upscaled to 1024×1024
    SAVE img_1024 to final_dir with same filename

  END FOR

  DELETE temp_dir

END GENERATIVE_AUGMENTATION
```

---

**Output File Naming Convention:**

| Prefix       | Origin                        | Recognised by downstream |
|--------------|-------------------------------|--------------------------|
| anchor_*     | Real photo, black BG          | convert_ai.py (pose est.)|
| synth_*_v[n] | Zero123++ tile, rembg applied | convert_ai.py (pose est.)|
| restored_*   | ControlNet output             | convert_ai.py (as anchor)|
| FULL_GRID_*  | Raw Zero123++ 2×3 debug grid  | Skipped — debug only     |

---

**Notes:**
- rembg background removal is applied to synth tiles only. Anchors already have black backgrounds from the canvas preparation step.
- ControlNet cannot recover information destroyed by severe degradation — it hallucinates plausible but geometrically incorrect texture.
- The basicsr functional_tensor compatibility patch must be applied before this script runs (required for RealESRGAN on newer PyTorch versions).
