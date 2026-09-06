#!/usr/bin/env python3
"""
Batch Video Evaluation Script.

Processes all video sequences in the videos/ directory and calculates
average inference latency and frame-by-frame detections.

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
    video_dir = repo_root / "videos"
    results_dir = repo_root / "video_results"
    results_dir.mkdir(parents=True, exist_ok=True)

    weights_path = repo_root / "weights" / "best.pt"
    engine = PersonDetectorEngine(
        weights_path=str(weights_path),
        conf_threshold=0.30,
        device="cpu"
    )

    video_files = list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.avi"))
    if not video_files:
        print(f"[NOTE] No video files found in {video_dir}. Place MP4/AVI sequences to run batch verification.")
        return

    print(f"[INFO] Found {len(video_files)} video sequence(s) to process.")
    for vid in video_files:
        print(f">> Processing: {vid.name}")
        cap = cv2.VideoCapture(str(vid))
        frames = 0
        total_detections = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            dets = engine.detect(frame)
            total_detections += len(dets)
            frames += 1
        cap.release()
        print(f"   Completed {frames} frames. Total person detections: {total_detections}.")


if __name__ == "__main__":
    main()
