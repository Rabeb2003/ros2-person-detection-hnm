"""
person_detection_demo.launch.py
Launch file for person detection simulation demo.

Launches Gazebo with the person_detection_world (5 animated actors + cluttered environment),
spawns the diff_robot with its front RGB camera, and starts the YOLOv8 detector node.

Usage:
    ros2 launch diff_robot person_detection_demo.launch.py

Author: Rabeb Bouzaida (ENIM)
"""

import os
import sys
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    SetEnvironmentVariable,
    TimerAction,
    LogInfo
)
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    package_share = get_package_share_directory('diff_robot')

    # ── File paths ──────────────────────────────────────────────────────────
    urdf_file_path  = os.path.join(package_share, 'urdf',   'diff_robot.urdf')
    rviz_config     = os.path.join(package_share, 'urdf',   'rviz.rviz')
    world_file_path = os.path.join(package_share, 'world',  'person_detection_world.world')
    gazebo_models_path = os.path.join(package_share, 'world', 'models')

    # Add ~/.gazebo/models so person_walking mesh resolves correctly
    gazebo_model_dirs = os.path.expanduser('~/.gazebo/models') + ':' + gazebo_models_path

    with open(urdf_file_path, 'r') as f:
        robot_desc = f.read()

    # ── Detector workspace ────────────────────────────────────────────────
    detector_ws = os.path.expanduser('~/ros2-person-detection-hnm')
    weights_path = os.path.join(detector_ws, 'weights', 'best.pt')
    if not os.path.exists(weights_path):
        weights_path = 'yolov8n.pt'   # auto-download fallback

    # ── Launch arguments ─────────────────────────────────────────────────
    rviz_arg = DeclareLaunchArgument(
        'rviz', default_value='true',
        description='Launch RViz2 for visualization'
    )
    detector_arg = DeclareLaunchArgument(
        'detector', default_value='true',
        description='Launch YOLOv8 person detector node'
    )
    conf_arg = DeclareLaunchArgument(
        'conf_threshold', default_value='0.30',
        description='YOLOv8 confidence threshold (0.0 - 1.0)'
    )
    device_arg = DeclareLaunchArgument(
        'device', default_value='cpu',
        description='Inference device: cpu or 0 (CUDA)'
    )

    # ── 1. Gazebo Classic ─────────────────────────────────────────────────
    gazebo = ExecuteProcess(
        cmd=[
            'gazebo', '--verbose',
            '-s', 'libgazebo_ros_init.so',
            '-s', 'libgazebo_ros_factory.so',
            world_file_path
        ],
        output='screen'
    )

    # ── 2. Robot State Publisher ─────────────────────────────────────────
    robot_state_pub = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[
            {'robot_description': robot_desc},
            {'use_sim_time': True},
        ],
        remappings=[('/diff_drive_controller/cmd_vel_unstamped', '/cmd_vel')]
    )

    # ── 3. Spawn robot at origin (camera facing the actors) ──────────────
    spawn_robot = TimerAction(
        period=8.0,
        actions=[
            Node(
                package='gazebo_ros',
                executable='spawn_entity.py',
                arguments=[
                    '-file', urdf_file_path,
                    '-entity', 'diff_robot',
                    '-x', '-2.0',    # Behind origin — actors are at +3m to +8m
                    '-y', '0.0',
                    '-z', '0.15',
                    '-Y', '0.0'      # Robot faces +X direction (toward actors)
                ],
                output='screen'
            ),
            LogInfo(msg='[SIM] Robot spawned at (-2, 0). Camera faces +X toward actors.')
        ]
    )

    # ── 4. RViz2 (after spawn) ────────────────────────────────────────────
    rviz2 = TimerAction(
        period=12.0,
        actions=[
            Node(
                package='rviz2',
                executable='rviz2',
                name='rviz2',
                output='screen',
                arguments=['-d', rviz_config],
                condition=IfCondition(LaunchConfiguration('rviz'))
            )
        ]
    )

    # ── 5. YOLOv8 Person Detector Node ───────────────────────────────────
    # Launched 15s after start to ensure camera topic is publishing
    detector_node = TimerAction(
        period=15.0,
        actions=[
            Node(
                package='person_detection_ros2',
                executable='detector_node',
                name='person_detector_node',
                output='screen',
                parameters=[{
                    'camera_topic':    '/camera/image_raw',
                    'weights_path':    weights_path,
                    'conf_threshold':  LaunchConfiguration('conf_threshold'),
                    'device':          LaunchConfiguration('device'),
                    'publish_annotated': True,
                }],
                condition=IfCondition(LaunchConfiguration('detector'))
            ),
            LogInfo(msg='[DETECTOR] YOLOv8 Person Detector Node started on /camera/image_raw')
        ]
    )

    # ── 6. rqt_image_view for live annotated detections ──────────────────
    rqt_view = TimerAction(
        period=18.0,
        actions=[
            Node(
                package='rqt_image_view',
                executable='rqt_image_view',
                name='detection_viewer',
                output='screen',
                arguments=['/person_detector/annotated_image'],
                condition=IfCondition(LaunchConfiguration('detector'))
            ),
            LogInfo(msg='[VIZ] rqt_image_view showing /person_detector/annotated_image')
        ]
    )

    return LaunchDescription([
        # Arguments
        rviz_arg,
        detector_arg,
        conf_arg,
        device_arg,
        # Sequenced launch
        SetEnvironmentVariable('GAZEBO_MODEL_PATH', gazebo_model_dirs),
        LogInfo(msg='[SIM] Launching person_detection_world — 5 animated actors'),
        gazebo,
        robot_state_pub,
        spawn_robot,   # t+8s
        rviz2,         # t+12s
        detector_node, # t+15s
        rqt_view,      # t+18s
    ])
