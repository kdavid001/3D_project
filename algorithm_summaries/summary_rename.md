# Algorithm Summary: Sequential Dataset Organiser
# Source: rename.py | Full algorithm: algorithm_for_codes/algo_rename.md
# Status: NOT REQUIRED — destroys anchor_/synth_ prefixes needed by convert_ai.py

---

**Algorithm 9 (Summary): SEQUENTIAL_RENAME**

**Input:**
- source : augmented image folder (final_{name}_run/)
- output : root 3DGS dataset path
- name   : scene identifier (creates output/{name}/input/)

**Output:**
- output/{name}/input/ : images renamed to 00001.jpg, 00002.jpg, ...

---

```
BEGIN SEQUENTIAL_RENAME(source, output, name)

  input_dir ← output / name / "input"

  // Clean up old data to prevent mixing
  IF input_dir EXISTS → DELETE recursively
  CREATE input_dir

  // Collect and sort images
  images ← LIST all .jpg/.jpeg/.png in source
  images ← SORT alphanumerically

  // Copy with 5-digit sequential names
  counter ← 1
  FOR each image I in images DO
    new_name ← ZERO_PAD(counter, 5) + FILE_EXT(I)   // e.g. "00001.jpg"
    COPY I to input_dir / new_name
    counter ← counter + 1
  END FOR

END SEQUENTIAL_RENAME
```

---

**Why Not Used:**
- Strips anchor_* / synth_* filename prefixes that convert_ai.py depends on
- Only valid for standard COLMAP sequential matching, which is not used in this pipeline
- Skip this script entirely; prefixes are preserved through to pose estimation

---
