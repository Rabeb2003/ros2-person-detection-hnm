#!/bin/bash
# ============================================================
# launch_simulation.sh
# Simulation complète : Gazebo (diff_robot + caméra) + Person Detector
# Utilise tmux pour tout lancer dans des terminaux séparés.
#
# Usage:
#   chmod +x launch/launch_simulation.sh
#   bash launch/launch_simulation.sh
# ============================================================

set -e

DETECTOR_WS="/home/rabeb/ros2-person-detection-hnm"
DIFF_DRIVE_WS="/home/rabeb/ros2_diff_drive_robot"
ROS_SETUP="/opt/ros/humble/setup.bash"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}  ROS 2 Simulation: Diff Robot + YOLOv8 Person Detection${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""

# Check tmux
if ! command -v tmux &> /dev/null; then
    echo -e "${RED}[ERROR] tmux is not installed. Run: sudo apt install tmux${NC}"
    exit 1
fi

# Kill any existing simulation session
tmux kill-session -t person_detection_sim 2>/dev/null || true

echo -e "${GREEN}[1/4] Starting Gazebo simulation (diff_robot with camera)...${NC}"
sleep 1

# Create tmux session
tmux new-session -d -s person_detection_sim -x 220 -y 50

# Window 0: Gazebo Simulation
tmux rename-window -t person_detection_sim:0 "Gazebo"
tmux send-keys -t person_detection_sim:0 \
  "source ${ROS_SETUP} && source ${DIFF_DRIVE_WS}/install/setup.bash && cd ${DIFF_DRIVE_WS} && echo '[Gazebo] Launching diff_robot simulation...' && ros2 launch diff_robot robot.launch.py rviz:=false" \
  Enter

echo -e "${GREEN}[2/4] Waiting 15s for Gazebo + robot to fully spawn...${NC}"
sleep 15

# Window 1: Person Detector Node
echo -e "${GREEN}[3/4] Starting YOLOv8 Person Detector Node...${NC}"
tmux new-window -t person_detection_sim -n "Detector"
tmux send-keys -t person_detection_sim:1 \
  "source ${ROS_SETUP} && cd ${DETECTOR_WS} && echo '[Detector] Starting person_detector_node...' && python3 -c \"
import sys
sys.path.insert(0, '${DETECTOR_WS}')
import rclpy
from person_detection_ros2.detector_node import main
main()
\"" \
  Enter

# Window 2: rqt Image View
echo -e "${GREEN}[4/4] Starting rqt image view for annotated detections...${NC}"
sleep 5
tmux new-window -t person_detection_sim -n "Visualizer"
tmux send-keys -t person_detection_sim:2 \
  "source ${ROS_SETUP} && echo '[Viz] Launching rqt_image_view...' && ros2 run rqt_image_view rqt_image_view /person_detector/annotated_image" \
  Enter

# Window 3: Status Monitor
tmux new-window -t person_detection_sim -n "Monitor"
tmux send-keys -t person_detection_sim:3 \
  "source ${ROS_SETUP} && source ${DIFF_DRIVE_WS}/install/setup.bash && sleep 20 && echo '' && echo '=== Active Topics ===' && ros2 topic list && echo '' && echo '=== Camera FPS ===' && ros2 topic hz /camera/image_raw --once && echo '' && echo '=== Detections ===' && ros2 topic echo /person_detector/detections --once" \
  Enter

# Attach to tmux session
echo ""
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}  Simulation started! Attaching to tmux session...${NC}"
echo -e "${GREEN}  Navigation: Ctrl+B then [0,1,2,3] to switch windows${NC}"
echo -e "${GREEN}  Quit:       Ctrl+B then & (kill current window)${NC}"
echo -e "${GREEN}  Exit all:   tmux kill-session -t person_detection_sim${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""

tmux attach-session -t person_detection_sim
