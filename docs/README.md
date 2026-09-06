# Documentation Assets & Demonstration

This folder contains graphical documentation artifacts referenced in the main `README.md`.

### Adding a Demonstration GIF or Video
To add an animated demo of your robot or Gazebo simulation:
1. Record a 5-10 second clip of the robot detecting persons.
2. Convert to GIF (`demo.gif`) or optimized MP4.
3. Place it in this directory:
   ```text
   docs/
   └── demo.gif
   ```
4. Link it in `README.md` right below the main header:
   ```markdown
   <p align="center">
     <img src="docs/demo.gif" width="80%" alt="ROS 2 Person Detection Demonstration" />
   </p>
   ```

### Current Artifacts
- `pr_curve.png`: Precision-Recall curve comparing initial baseline vs post-HNM.
- `confusion_matrix.png`: Normalized confusion matrix showing high true-positive classification and reduced background false alarms.
- `training_curves.png`: Loss and mAP progression curves across training epochs.
