"""
High-Level Inference Engine for Robotic Person Detection.

Provides reusable OOP wrappers around YOLOv8 inference, bounding box extraction,
and filtering for downstream robotic navigation stacks.

Author: Rabeb Bouzaida (ENIM)
License: MIT
"""

import os
import time
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


class PersonDetectorEngine:
    """
    Encapsulates YOLOv8 model inference with safety thresholds,
    bounding box extraction, and drawing utilities.
    """

    def __init__(
        self,
        weights_path: str = "weights/best.pt",
        conf_threshold: float = 0.30,
        iou_threshold: float = 0.45,
        img_size: int = 640,
        device: str = "cpu"
    ):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.img_size = img_size
        self.device = device
        self.weights_path = weights_path

        if YOLO is None:
            raise RuntimeError("Ultralytics package is missing. Run: pip install -r requirements.txt")

        if not os.path.exists(self.weights_path):
            # Fallback to base pretrained yolov8n
            self.weights_path = "yolov8n.pt"

        self.model = YOLO(self.weights_path)

    def detect(self, image: np.ndarray) -> List[Dict[str, Union[float, List[int]]]]:
        """
        Runs single-frame inference and returns formatted bounding boxes.

        Returns:
            List of dicts: [{'class': 'person', 'confidence': 0.88, 'bbox_xyxy': [x1, y1, x2, y2], 'center': (cx, cy)}]
        """
        results = self.model.predict(
            source=image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.img_size,
            device=self.device,
            classes=[0],
            verbose=False
        )

        detections = []
        if results and len(results) > 0:
            for box in results[0].boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()
                conf = float(box.conf[0].item())
                cx = (xyxy[0] + xyxy[2]) / 2.0
                cy = (xyxy[1] + xyxy[3]) / 2.0
                detections.append({
                    "class": "person",
                    "confidence": conf,
                    "bbox_xyxy": xyxy,
                    "center": (cx, cy),
                    "width": xyxy[2] - xyxy[0],
                    "height": xyxy[3] - xyxy[1]
                })

        return detections

    def annotate(
        self,
        image: np.ndarray,
        detections: List[Dict[str, Union[float, List[int]]]],
        latency_ms: Optional[float] = None
    ) -> np.ndarray:
        """Draws bounding boxes and telemetry overlay on image."""
        vis = image.copy()
        for det in detections:
            x1, y1, x2, y2 = det["bbox_xyxy"]
            score = det["confidence"]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"person {score:.2f}"
            cv2.putText(
                vis, label, (x1, max(y1 - 6, 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
            )

        if latency_ms is not None:
            text = f"Inference: {latency_ms:.1f}ms | Detections: {len(detections)}"
            cv2.putText(vis, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

        return vis
