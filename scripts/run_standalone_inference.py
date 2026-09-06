#!/usr/bin/env python3
"""
Standalone YOLOv8 Inference CLI (No ROS 2 required).

Supports single images, video files, directory batches, or live webcam streams.
Displays visual detection bounding boxes, confidence, FPS, and latency telemetry.

Author: Rabeb Bouzaida (ENIM)
License: MIT
"""

import argparse
import sys
import time
from pathlib import Path
import cv2

try:
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] Ultralytics is not installed. Run: pip install -r requirements.txt")
    sys.exit(1)


def process_stream(source, model, conf, iou, imgsz, device, save_output, output_dir):
    # Check if numeric (webcam index)
    if source.isdigit():
        source = int(source)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video source: {source}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0

    writer = None
    if save_output:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"annotated_{Path(str(source)).stem}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_file), fourcc, fps_in, (width, height))
        print(f"[INFO] Saving processed video to: {out_file}")

    prev_time = time.time()
    fps_smooth = 0.0

    print("[INFO] Processing stream. Press 'q' to stop.")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        start = time.time()
        results = model.predict(
            source=frame,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=device,
            classes=[0],
            verbose=False
        )
        latency = (time.time() - start) * 1000.0

        # Draw detections
        person_count = 0
        if results and len(results) > 0:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                score = float(box.conf[0].item())
                person_count += 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame, f"person {score:.2f}", (x1, max(y1 - 6, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2
                )

        now = time.time()
        fps_smooth = 0.9 * fps_smooth + 0.1 * (1.0 / (now - prev_time + 1e-6))
        prev_time = now

        # HUD overlay
        hud = f"FPS: {fps_smooth:.1f} | Latency: {latency:.1f}ms | Persons: {person_count}"
        cv2.putText(frame, hud, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if writer:
            writer.write(frame)

        # Show if in graphical environment
        try:
            cv2.imshow("Robotic Person Detection - YOLOv8", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except cv2.error:
            # Headless environment, skip imshow
            pass

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("[SUCCESS] Stream processing finished.")


def main():
    repo_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(description="Standalone YOLOv8 Inference Benchmark")
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Input source: image path, video path, or webcam index (default: 0)"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=str(repo_root / "weights" / "best.pt"),
        help="Path to model weights (.pt)"
    )
    parser.add_argument("--conf", type=float, default=0.30, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image resolution")
    parser.add_argument("--device", type=str, default="cpu", help="Compute device ('cpu' or '0')")
    parser.add_argument("--save", action="store_true", help="Save annotated images/video")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(repo_root / "test_results"),
        help="Directory to save visual output"
    )

    args = parser.parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"[WARN] Specified weights not found at {weights_path}. Using yolov8n.pt")
        weights_path = "yolov8n.pt"

    model = YOLO(str(weights_path))

    source_path = Path(args.source)
    if source_path.exists() and source_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        # Single image processing
        img = cv2.imread(str(source_path))
        start = time.time()
        results = model.predict(source=img, conf=args.conf, iou=args.iou, imgsz=args.imgsz, device=args.device, classes=[0])
        latency = (time.time() - start) * 1000.0

        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"person {float(box.conf[0]):.2f}", (x1, max(y1 - 6, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        print(f"[INFO] Inferred in {latency:.2f} ms. Detected {len(results[0].boxes)} persons.")
        if args.save:
            out_dir = Path(args.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"annotated_{source_path.name}"
            cv2.imwrite(str(out_path), img)
            print(f"[SAVED] Saved annotated image to: {out_path}")
    else:
        # Video or Webcam stream
        process_stream(
            source=args.source,
            model=model,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            save_output=args.save,
            output_dir=args.output_dir
        )


if __name__ == "__main__":
    main()
