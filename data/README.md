# Dataset Organization & Layout

This project uses the standard single-class YOLO format for human perception in mobile robotics.

### Expected Directory Structure
```text
data/
├── train/
│   ├── images/     # Training frames (*.jpg, *.png)
│   └── labels/     # YOLO bounding boxes (*.txt)
├── valid/
│   ├── images/     # Validation frames
│   └── labels/
├── test/
│   ├── images/     # Benchmark test frames
│   └── labels/
└── negatives/      # Background / empty industrial scenes for Hard Negative Mining (no labels)
```

### Bounding Box Annotation Format
Each line in `labels/*.txt` corresponds to:
```text
<class_id> <x_center> <y_center> <width> <height>
```
All coordinates are normalized between `0.0` and `1.0`. Class ID is `0` for `person`.

### Hard Negative Backgrounds
Images placed in `data/negatives/` represent scenes with high clutter (corridors, metal racks, chair legs, bright lighting reflections) but **zero humans**.
Running `scripts/hard_negative_mining.py` extracts false positives from this folder and injects them as negative examples into `data/train/`.
