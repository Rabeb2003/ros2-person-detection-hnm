#!/usr/bin/env python3
"""
YOLOv8 Fine-Tuning Script for Robotic Person Detection.
Uses relative paths for portable cross-platform execution.

Author: Rabeb Bouzaida (ENIM)
License: MIT
"""

import argparse
import os
import sys
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] Ultralytics is not installed. Run: pip install -r requirements.txt")
    sys.exit(1)


def main():
    repo_root = Path(__file__).resolve().parent.parent
    default_data_config = repo_root / "config" / "dataset_config.yaml"
    default_weights = repo_root / "weights" / "best.pt"

    parser = argparse.ArgumentParser(description="Train YOLOv8 on Person Dataset")
    parser.add_argument(
        "--data",
        type=str,
        default=str(default_data_config),
        help="Path to dataset YAML configuration file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Base pretrained architecture (e.g. yolov8n.pt, yolov8s.pt)"
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--device", type=str, default="0", help="CUDA device index or 'cpu'")
    parser.add_argument("--name", type=str, default="person_detect_run", help="Experiment name")
    parser.add_argument("--project", type=str, default=str(repo_root / "runs" / "train"), help="Output project directory")

    args = parser.parse_args()

    print("=" * 70)
    print("  YOLOv8 Person Detection Training Pipeline")
    print("=" * 70)
    print(f"Base Model:       {args.model}")
    print(f"Dataset Config:   {args.data}")
    print(f"Epochs:           {args.epochs}")
    print(f"Batch Size:       {args.batch}")
    print(f"Image Resolution: {args.imgsz}")
    print(f"Compute Device:   {args.device}")
    print(f"Output Project:   {args.project}")
    print("=" * 70)

    # Initialize model
    model = YOLO(args.model)

    # Execute training
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
        exist_ok=True,
        save=True,
        plots=True,
        verbose=True
    )

    # Automatically copy best.pt to weights directory
    trained_best = Path(args.project) / args.name / "weights" / "best.pt"
    if trained_best.exists():
        os.makedirs(repo_root / "weights", exist_ok=True)
        import shutil
        shutil.copy2(trained_best, default_weights)
        print(f"\n[INFO] Best checkpoint saved directly to: {default_weights}")

    print("\n[SUCCESS] Training session completed successfully.")


if __name__ == "__main__":
    main()
