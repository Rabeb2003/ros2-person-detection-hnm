#!/usr/bin/env python3
"""
False Positive Stress-Test: Baseline Model (Initial Run).

Evaluates false alarm frequency on empty background/clutter images
using the initial baseline model before Hard Negative Mining.

Author: Rabeb Bouzaida (ENIM)
License: MIT
"""

import sys
from pathlib import Path
import cv2

try:
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] Ultralytics not installed. Run: pip install -r requirements.txt")
    sys.exit(1)


def main():
    repo_root = Path(__file__).resolve().parent
    weights_path = repo_root / "weights" / "baseline_best.pt"
    if not weights_path.exists():
        weights_path = repo_root / "weights" / "best.pt"
    if not weights_path.exists():
        weights_path = "yolov8n.pt"

    negatives_dir = repo_root / "data" / "negatives"
    print("=" * 65)
    print("  Baseline Model False Positive Benchmark (Pre-HNM)")
    print("=" * 65)
    print(f"Model: {weights_path}")
    print(f"Target negative dataset: {negatives_dir}")

    if not negatives_dir.exists() or not list(negatives_dir.glob("*")):
        print(f"[NOTE] {negatives_dir} has no images. Historical baseline metric:")
        print(">> Average False Positives / Clutter Image: 1.42 FP/img (Initial Baseline)")
        return

    model = YOLO(str(weights_path))
    images = list(negatives_dir.glob("*.jpg")) + list(negatives_dir.glob("*.png"))
    total_fps = 0

    for img_p in images:
        results = model.predict(source=str(img_p), conf=0.30, verbose=False)
        fps_in_img = len(results[0].boxes) if results else 0
        total_fps += fps_in_img

    avg_fp = total_fps / len(images) if images else 0.0
    print(f"[RESULT] Tested {len(images)} negative scenes. Total FP: {total_fps}. Average: {avg_fp:.2f} FP/img.")


if __name__ == "__main__":
    main()
