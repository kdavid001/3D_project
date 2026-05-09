# Dataset Organiser — `rename.py`

> **STATUS: ⚠️ NOT REQUIRED (SKIP IN CURRENT PIPELINE)** — Skipping this script is now preferred. Running it destroys the `anchor_` / `synth_` prefixes that `inject_poses.py` depends on for image classification.

---

## Why It Should Be Skipped

`rename.py` renames all output files to sequential names (`00001.jpg`, `00002.jpg`, …). This was originally needed for COLMAP sequential matching.

`inject_poses.py` classifies images by their `anchor_` and `synth_` prefix — not by filename order. If `rename.py` runs first:
- All `anchor_` prefixes are destroyed
- `inject_poses.py` cannot distinguish real photos from synthetic views
- You must manually pass `--num_real N` to recover the classification

**Current pipeline skips this step entirely.** If you do need sequential names for a different reason, pass `--num_real N` to `inject_poses.py` afterward.

---

## What It Does

Creates a clean room for COLMAP by:
1. Deleting old input folders to prevent data mixing
2. Sorting files alphanumerically (preserves temporal/spatial sequence)
3. Renaming all files to strict 5-digit sequence (`00001.jpg`, `00002.jpg`…)

---

## When It Is Still Valid

Only if you are running standard COLMAP (not `inject_poses.py`) and need sequential matching enabled. In that case run it after the diffusion step but before `convert_ai.py`.

---

## Usage

```bash
python rename.py \
  --source "/path/to/final_hotdog_run" \
  --output "/path/to/gaussian-splatting/data" \
  --name "hotdog"
```

| Argument | Description |
|---|---|
| `--source` | Folder with final processed images |
| `--output` | Root folder for 3DGS datasets |
| `--name` | Object name — creates subfolder here |

---

## Output Structure

```
/gs_datasets/
└── hotdog/
    └── input/
        ├── 00001.jpg
        ├── 00002.jpg
        └── ...
```
