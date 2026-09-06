#!/usr/bin/env python3
"""
Active Hard Negative Mining (HNM) Pipeline for Robotic Vision.

Automates the harvesting of False Positives (FP) on negative/background scenes
(cluttered furniture, industrial pipes, shadows, reflections) and re-trains
the model to suppress phantom obstacle detections in robotic navigation.

Author: Rabeb Bouzaida (ENIM)
License: MIT
"""

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import List, Tuple

try:
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] Ultralytics is not installed. Run: pip install -r requirements.txt")
    sys.exit(1)


def harvest_false_positives(
    model: YOLO,
    negatives_dir: Path,
    conf_thresh: float = 0.25,
    device: str = "cpu"
) -> List[Tuple[Path, int]]:
    """
    Scans a directory of empty background/cluttered scenes where NO humans are present.
    Any detection with conf >= conf_thresh is recorded as a False Positive (Hard Negative).
    """
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_paths = [p for p in negatives_dir.rglob("*") if p.suffix.lower() in valid_extensions]

    if not image_paths:
        print(f"[WARN] No images found in background directory: {negatives_dir}")
        return []

    print(f"\n[INFO] Scanning {len(image_paths)} negative background images for Hard Negatives...")
    hard_negatives = []

    total_fps = 0
    for img_path in image_paths:
        results = model.predict(
            source=str(img_path),
            conf=conf_thresh,
            device=device,
            verbose=False
        )

        if results and len(results) > 0:
            boxes = results[0].boxes
            num_detections = len(boxes)
            if num_detections > 0:
                # This is an empty scene, so all detections are False Positives!
                hard_negatives.append((img_path, num_detections))
                total_fps += num_detections

    print(f"[ANALYSIS] Identified {len(hard_negatives)} images with phantom detections ({total_fps} total FPs).")
    return hard_negatives


def integrate_hard_negatives(
    hard_negatives: List[Tuple[Path, int]],
    train_images_dir: Path,
    train_labels_dir: Path
):
    """
    In YOLO format, an image with an empty .txt label file instructs the model
    that the entire frame contains background (class 0 is absent).
    This injects negative gradients specifically on the false-positive regions.
    """
    train_images_dir.mkdir(parents=True, exist_ok=True)
    train_labels_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for img_path, fp_count in hard_negatives:
        dest_img = train_images_dir / f"hnm_{img_path.name}"
        dest_label = train_labels_dir / f"hnm_{img_path.stem}.txt"

        # Copy image to training set
        shutil.copy2(img_path, dest_img)

        # Create empty label file (represents true negative / pure background in YOLOv8)
        dest_label.touch(exist_ok=True)
        copied += 1

    print(f"[INTEGRATION] Injected {copied} Hard Negative samples into training partition.")


def main():
    repo_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(description="Automated Hard Negative Mining (HNM) Loop")
    parser.add_argument(
        "--weights",
        type=str,
        default=str(repo_root / "weights" / "best.pt"),
        help="Path to initial baseline checkpoint"
    )
    parser.add_argument(
        "--negatives",
        type=str,
        default=str(repo_root / "data" / "negatives"),
        help="Directory containing clutter/background images without persons"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold above which background noise is considered a Hard Negative"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of epochs for HNM fine-tuning"
    )
    parser.add_argument("--device", type=str, default="0", help="CUDA device index or 'cpu'")
    parser.add_argument("--retrain", action="store_true", help="Trigger retraining after mining")

    args = parser.parse_args()

    print("=" * 70)
    print("  Robotic Vision - Hard Negative Mining (HNM) Pipeline")
    print("=" * 70)

    # 1. Load baseline model
    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"[INFO] Checkpoint {weights_path} not found. Attempting yolov8n.pt...")
        weights_path = "yolov8n.pt"

    model = YOLO(str(weights_path))

    # 2. Mine hard negatives from clutter scenes
    negatives_dir = Path(args.negatives)
    if not negatives_dir.exists():
        print(f"[NOTE] Negatives directory {negatives_dir} does not exist yet.")
        print("Creating placeholder directory. Place your empty scene/corridor images here.")
        negatives_dir.mkdir(parents=True, exist_ok=True)
        return

    hard_negatives = harvest_false_positives(
        model=model,
        negatives_dir=negatives_dir,
        conf_thresh=args.conf,
        device=args.device
    )

    # 3. Integrate into training set
    if hard_negatives:
        train_img_dir = repo_root / "data" / "train" / "images"
        train_lbl_dir = repo_root / "data" / "train" / "labels"
        integrate_hard_negatives(hard_negatives, train_img_dir, train_lbl_dir)

        # 4. Optional Retraining
        if args.retrain:
            print("\n[INFO] Starting Post-HNM Retraining (run2_hnm)...")
            data_yaml = repo_root / "config" / "dataset_config.yaml"
            model.train(
                data=str(data_yaml),
                epochs=args.epochs,
                imgsz=640,
                device=args.device,
                project=str(repo_root / "runs" / "train"),
                name="run2_hnm",
                exist_ok=True
            )
            print("[SUCCESS] Post-HNM retraining finished. Evaluate false-positive reduction with evaluation/metrics.py")
    else:
        print("[INFO] No False Positives detected on background samples at threshold", args.conf)


if __name__ == "__main__":
    main()
