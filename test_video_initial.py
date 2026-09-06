#!/usr/bin/env python3
"""
Single Video Verification Script (Baseline Check).

Author: Rabeb Bouzaida (ENIM)
License: MIT
"""

import sys
from pathlib import Path
from inference.inference import PersonDetectorEngine

try:
    import cv2
except ImportError:
    print("[ERROR] OpenCV is not installed. Run: pip install -r requirements.txt")
    sys.exit(1)


def main():
    repo_root = Path(__file__).resolve().parent
    weights_path = repo_root / "weights" / "baseline_best.pt"
    if not weights_path.exists():
        weights_path = repo_root / "weights" / "best.pt"

    engine = PersonDetectorEngine(
        weights_path=str(weights_path),
        conf_threshold=0.30,
        device="cpu"
    )

    sample_videos = list((repo_root / "videos").glob("*.mp4"))
    if not sample_videos:
        print("[NOTE] No videos found in videos/ directory. Place a test video to run.")
        return

    test_vid = sample_videos[0]
    print(f"[INFO] Testing video sequence: {test_vid}")
    cap = cv2.VideoCapture(str(test_vid))
    count = 0
    while cap.isOpened() and count < 100:
        ret, frame = cap.read()
        if not ret:
            break
        dets = engine.detect(frame)
        count += 1

    cap.release()
    print(f"[SUCCESS] Processed {count} frames successfully.")


if __name__ == "__main__":
    main()
