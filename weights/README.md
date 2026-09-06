# Model Weights Directory

This directory hosts fine-tuned YOLOv8 weights.

### Default Checkpoint Placement
Drop your trained weights here:
```text
weights/
└── best.pt      # Fine-tuned YOLOv8n weights (Post-HNM)
```

### Automatic Bootstrap Fallback
If `weights/best.pt` is not provided, both the ROS 2 node (`detector_node`) and standalone scripts will automatically download and utilize standard `yolov8n.pt` as a baseline bootstrap.

### Training or Updating Weights
To generate a new fine-tuned checkpoint:
```bash
python3 scripts/train.py --epochs 50 --imgsz 640
```
Trained weights will automatically be updated in this folder.
