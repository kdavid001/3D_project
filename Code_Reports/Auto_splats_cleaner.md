# Auto Splats Cleaner — `run_ai_pipeline.py`

> **STATUS: ❌ NOT IN USE** — Earlier version of `convert_ai.py`. Superseded first by `convert_ai.py` and then by `inject_poses.py`. Kept for reference only.

---

## Overview

An earlier iteration of the AI-powered COLMAP pipeline using SuperPoint + LightGlue feature matching. Functionally identical to `convert_ai.py` — same SfM collapse problem on black-background images applies.

See `convert_ai.md` for the full breakdown of why this approach was replaced.

---

## Replacement Chain

```
run_ai_pipeline.py  (Auto_splats_cleaner)
        ↓  replaced by
convert_ai.py  (cleaner implementation, same approach)
        ↓  replaced by
inject_poses.py  (bypasses SfM entirely — current pipeline)
```

---

## When It Would Be Valid

Same conditions as `convert_ai.py`: natural scene images with original backgrounds, no Zero123++ synthetic views, sufficient texture for feature matching.

---

## Pipeline Logic

1. Cleanup old `distorted/` and `sparse/` folders
2. SuperPoint feature extraction → `features.h5`
3. LightGlue exhaustive matching → `matches.h5`
4. PyCOLMAP reconstruction → `distorted/sparse`
5. COLMAP image undistortion → `images/` + `sparse/0/`
6. Final handshake — moves `.bin` files to `sparse/0/`

---

## Usage

```bash
python run_ai_pipeline.py \
  --source_path "/content/drive/MyDrive/.../gaussian_splatting/data/hotdog" \
  --images "input"
```
