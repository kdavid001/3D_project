# Algorithm: Sequential Dataset Organiser
# File: rename.py
# Status: NOT REQUIRED — skip in current pipeline (destroys anchor_/synth_ prefixes)

---

> **Note on Status:** rename.py is not used in the current pipeline.
> Running it strips the anchor_ / synth_ filename prefixes that convert_ai.py
> depends on for image classification. Document is retained for reference only.

---

**Algorithm 9: SEQUENTIAL_RENAME**

**Input:**
- source : path to folder containing augmented images (final_{name}_run/)
- output : root path for 3DGS dataset storage
- name   : object/scene identifier (creates a named subfolder)

**Output:**
- output/{name}/input/ : sequentially renamed images (00001.jpg, 00002.jpg, ...)
- Old input folder deleted to prevent data mixing

---

```
BEGIN SEQUENTIAL_RENAME(source, output, name)

  dest_root  ← output / name
  input_dir  ← dest_root / "input"

  // Step 1: Cleanup — delete old input folder to prevent stale image mixing
  IF input_dir EXISTS THEN
    DELETE input_dir recursively
  END IF
  CREATE input_dir

  // Step 2: Collect and sort source images
  images ← LIST all image files (.jpg, .jpeg, .png) in source
  images ← SORT images alphanumerically
  // Alphanumeric sort preserves any implicit temporal or spatial ordering

  // Step 3: Rename to strict 5-digit sequential format
  counter ← 1

  FOR each image I in images DO
    extension ← FILE_EXTENSION(I)                     // e.g., ".jpg"
    new_name  ← ZERO_PAD(counter, width=5) + extension // e.g., "00001.jpg"
    COPY I to input_dir / new_name
    counter ← counter + 1
  END FOR

END SEQUENTIAL_RENAME
```

---

**Output Structure:**

```
output/
└── {name}/
    └── input/
        ├── 00001.jpg
        ├── 00002.jpg
        ├── 00003.jpg
        └── ...
```

---

**When This Script Is Valid:**
- Only required when running standard COLMAP sequential matching (not used in current pipeline)
- If executed, must run AFTER diffusion_script_v0.py and BEFORE convert_ai.py
- The `anchor_` / `synth_` prefixes are destroyed by renaming, so any downstream component that relies on those prefixes will fail

**Current Pipeline Decision:**
Skip this script entirely. The anchor_/synth_ prefixes produced by diffusion_script_v0.py are preserved through to convert_ai.py, which does not depend on sequential filename ordering.
