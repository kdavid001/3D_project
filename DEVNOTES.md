# Development Notes

Internal notes on script evolution during the GA-3DGS project.
Not part of the public-facing documentation.

---

## corrupt_data.py — Version History

Originally `corrupt_dataset_v3.py`. Key changes leading to current version:
- **Removed:** Black Holes (Geometry Artifacts)
- **Added:** Radiometric Corruptions (Saturation, Exposure Extremes)
- **Kept:** Optical Corruptions (Defocus Blur, Noise)
- **Kept:** Sector Deletion (Missing Views logic)
- **Added:** Single Image Auto-Detect mode (generates isolated failure modes for thesis collages)

---

## diffusion_script_v0.py — Version History

Originally "Combined Pipeline V2.0 (Zero123++ Edition)".
This was the early-stage diffusion augmentation script before ViewCrafter
was adopted as the primary augmentation model. Zero123++ was evaluated but
produced hallucinated, object-centric outputs that were too geometrically
inconsistent for wide-baseline real-world scenes.
ViewCrafter replaced it in the final pipeline due to better photometric
consistency on real outdoor/indoor captures.

---

## process_file.py — Version History

Originally `load_images_legend.py`. Evolved from a basic MUSIQ-only screener
to the five-pillar system (MUSIQ + saturation + exposure + contrast + colour cast)
after MUSIQ alone was found to be domain-miscalibrated under artificial indoor lighting.
Key additions:
- RGB histogram panel with legend
- Per-pillar inspection report panel
- Hard/soft threshold gating logic
- Outdoor/indoor/synthetic threshold profiles
