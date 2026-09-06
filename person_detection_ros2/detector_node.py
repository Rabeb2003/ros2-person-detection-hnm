#!/usr/bin/env python3
"""
ROS 2 Humble Real-Time Person Detection Node using YOLOv8 with HNM.

Author: Rabeb Bouzaida (ENIM)
License: MIT
"""

import os
import sys
import time
from typing import Optional

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose

try:
    from cv_bridge import CvBridge
except ImportError:
    CvBridge = None

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


class PersonDetectorNode(Node):
    """
    ROS 2 Node subscribing to raw camera stream, executing YOLOv8 person detection,
    and publishing standardized 2D bounding boxes and annotated debug images.
    """

    def __init__(self):
        super().__init__('person_detector_node')

        # 1. Declare and retrieve parameters
        self.declare_parameter('camera_topic', '/camera/image_raw')
        self.declare_parameter('annotated_topic', '/person_detector/annotated_image')
        self.declare_parameter('detections_topic', '/person_detector/detections')
        self.declare_parameter('weights_path', 'weights/best.pt')
        self.declare_parameter('img_size', 640)
        self.declare_parameter('conf_threshold', 0.30)
        self.declare_parameter('iou_threshold', 0.45)
        self.declare_parameter('device', 'cpu')
        self.declare_parameter('publish_annotated', True)

        self.camera_topic = self.get_parameter('camera_topic').get_parameter_value().string_value
        self.annotated_topic = self.get_parameter('annotated_topic').get_parameter_value().string_value
        self.detections_topic = self.get_parameter('detections_topic').get_parameter_value().string_value
        self.weights_path = self.get_parameter('weights_path').get_parameter_value().string_value
        self.img_size = self.get_parameter('img_size').get_parameter_value().integer_value
        self.conf_threshold = self.get_parameter('conf_threshold').get_parameter_value().double_value
        self.iou_threshold = self.get_parameter('iou_threshold').get_parameter_value().double_value
        self.device = self.get_parameter('device').get_parameter_value().string_value
        self.publish_annotated = self.get_parameter('publish_annotated').get_parameter_value().bool_value

        self.get_logger().info('Initializing ROS 2 Person Detector Node...')
        self.get_logger().info(f'Camera Topic: {self.camera_topic}')
        self.get_logger().info(f'Confidence Threshold: {self.conf_threshold:.2f}')
        self.get_logger().info(f'Device: {self.device}')

        # 2. Initialize CV Bridge
        if CvBridge is not None:
            self.bridge = CvBridge()
        else:
            self.get_logger().warn('cv_bridge not found. Falling back to manual OpenCV conversions.')
            self.bridge = None

        # 3. Load YOLOv8 Model
        self.model = self._load_model(self.weights_path)

        # 4. Latency and FPS tracking
        self.prev_time = time.time()
        self.fps = 0.0

        # 5. QoS Profile for real-time video stream (Best Effort, non-blocking)
        qos_profile = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT
        )

        # 6. Subscribers & Publishers
        self.sub_image = self.create_subscription(
            Image,
            self.camera_topic,
            self.image_callback,
            qos_profile
        )

        self.pub_detections = self.create_publisher(
            Detection2DArray,
            self.detections_topic,
            10
        )

        if self.publish_annotated:
            self.pub_annotated = self.create_publisher(
                Image,
                self.annotated_topic,
                10
            )

        self.get_logger().info('Person Detector Node ready. Waiting for incoming camera frames...')

    def _load_model(self, path: str):
        """Loads the YOLOv8 model weights, with automatic fallback handling."""
        if YOLO is None:
            self.get_logger().error(
                'Ultralytics is not installed. Please run: pip install -r requirements.txt'
            )
            return None

        # Resolve relative paths
        if not os.path.isabs(path):
            candidates = [
                path,
                os.path.join(os.getcwd(), path),
                os.path.join(os.path.dirname(__file__), '..', path)
            ]
            for c in candidates:
                if os.path.exists(c):
                    path = os.path.abspath(c)
                    break

        if not os.path.exists(path):
            self.get_logger().warn(
                f'Trained weights not found at: {path}. Using pretrained "yolov8n.pt" for bootstrap.'
            )
            path = 'yolov8n.pt'

        try:
            self.get_logger().info(f'Loading YOLOv8 weights from: {path}')
            model = YOLO(path)
            self.get_logger().info('YOLOv8 model successfully loaded into memory.')
            return model
        except Exception as e:
            self.get_logger().error(f'Failed to load YOLOv8 model: {str(e)}')
            return None

    def image_callback(self, msg: Image):
        """Processes incoming camera image frames, runs inference, and publishes detections."""
        start_time = time.time()

        # 1. Convert ROS Image to OpenCV BGR
        try:
            if self.bridge:
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            else:
                cv_image = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, -1))
                if msg.encoding == 'rgb8':
                    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
        except Exception as e:
            self.get_logger().error(f'Image conversion failed: {str(e)}')
            return

        # 2. Run Inference
        detections_msg = Detection2DArray()
        detections_msg.header = msg.header

        annotated_frame = cv_image.copy() if self.publish_annotated else None
        person_count = 0

        if self.model is not None:
            try:
                results = self.model.predict(
                    source=cv_image,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    imgsz=self.img_size,
                    device=self.device,
                    classes=[0],  # 0 corresponds to class 'person'
                    verbose=False
                )

                if results and len(results) > 0:
                    boxes = results[0].boxes
                    for box in boxes:
                        cls_id = int(box.cls[0].item())
                        conf = float(box.conf[0].item())
                        xywh = box.xywh[0].cpu().numpy()
                        xyxy = box.xyxy[0].cpu().numpy().astype(int)

                        # Standard vision_msgs Detection2D structure
                        det = Detection2D()
                        det.header = msg.header
                        det.bbox.center.position.x = float(xywh[0])
                        det.bbox.center.position.y = float(xywh[1])
                        det.bbox.size_x = float(xywh[2])
                        det.bbox.size_y = float(xywh[3])

                        hyp = ObjectHypothesisWithPose()
                        hyp.hypothesis.class_id = 'person'
                        hyp.hypothesis.score = conf
                        det.results.append(hyp)

                        detections_msg.detections.append(det)
                        person_count += 1

                        # Draw bounding box for visualization
                        if annotated_frame is not None:
                            x1, y1, x2, y2 = xyxy
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            label = f'person {conf:.2f}'
                            cv2.putText(
                                annotated_frame, label, (x1, max(y1 - 8, 15)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                            )
            except Exception as e:
                self.get_logger().error(f'YOLOv8 inference error: {str(e)}')

        # 3. Publish standardized detections
        self.pub_detections.publish(detections_msg)

        # 4. Telemetry overlay & publish annotated image
        elapsed = time.time() - start_time
        latency_ms = elapsed * 1000.0

        current_time = time.time()
        time_diff = current_time - self.prev_time
        if time_diff > 0:
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / time_diff)
        self.prev_time = current_time

        if self.publish_annotated and annotated_frame is not None:
            # HUD overlay
            hud_text = f'FPS: {self.fps:.1f} | Latency: {latency_ms:.1f}ms | Persons: {person_count}'
            cv2.putText(
                annotated_frame, hud_text, (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2
            )

            try:
                if self.bridge:
                    out_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding='bgr8')
                else:
                    out_msg = Image()
                    out_msg.height, out_msg.width, _ = annotated_frame.shape
                    out_msg.encoding = 'bgr8'
                    out_msg.data = annotated_frame.tobytes()
                    out_msg.step = out_msg.width * 3
                out_msg.header = msg.header
                self.pub_annotated.publish(out_msg)
            except Exception as e:
                self.get_logger().error(f'Failed to publish annotated image: {str(e)}')


def main(args=None):
    rclpy.init(args=args)
    node = PersonDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('KeyboardInterrupt received. Shutting down detector node...')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
