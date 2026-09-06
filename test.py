#!/usr/bin/env python3
"""
Quick inference sanity check script.

Author: Rabeb Bouzaida (ENIM)
License: MIT
"""

import sys
import numpy as np
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] Ultralytics not installed. Run: pip install -r requirements.txt")
    sys.exit(1)


def main():
    repo_root = Path(__file__).resolve().parent
    weights_path = repo_root / "weights" / "best.pt"

    if not weights_path.exists():
        print(f"[NOTE] Checkpoint {weights_path} not found. Testing with base yolov8n.pt...")
        weights_path = "yolov8n.pt"

    print("=" * 60)
    print("  Sanity Test: YOLOv8 Person Detection Inference")
    print("=" * 60)

    try:
        model = YOLO(str(weights_path))
        # Create a synthetic dummy frame (640x640) to verify model forward pass
        dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
        results = model.predict(source=dummy_frame, conf=0.30, device="cpu", verbose=False)

        print("[SUCCESS] Model initialized and executed forward pass without errors.")
        print(f"[INFO] Weights: {weights_path}")
        print(f"[INFO] Target Classes: {model.names[0] if hasattr(model, 'names') else 'person'}")
    except Exception as e:
        print(f"[FAIL] Inference test failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
