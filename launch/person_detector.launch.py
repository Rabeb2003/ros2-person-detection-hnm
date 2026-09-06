import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Default paths
    default_config_path = ""
    try:
        pkg_share = get_package_share_directory('person_detection_ros2')
        default_config_path = os.path.join(pkg_share, 'config', 'detector_params.yaml')
    except Exception:
        # Fallback when running from source directory without install
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        default_config_path = os.path.join(current_dir, 'config', 'detector_params.yaml')

    # Launch configuration variables
    camera_topic_arg = DeclareLaunchArgument(
        'camera_topic',
        default_value='/camera/image_raw',
        description='Input camera image topic name (sensor_msgs/msg/Image)'
    )

    conf_threshold_arg = DeclareLaunchArgument(
        'conf_threshold',
        default_value='0.30',
        description='Confidence threshold for YOLOv8 person detection (0.0 to 1.0)'
    )

    weights_path_arg = DeclareLaunchArgument(
        'weights_path',
        default_value='weights/best.pt',
        description='Path to trained YOLOv8 model weights (.pt)'
    )

    device_arg = DeclareLaunchArgument(
        'device',
        default_value='cpu',
        description='Inference compute device ("cpu" or "0" for CUDA)'
    )

    params_file_arg = DeclareLaunchArgument(
        'params_file',
        default_value=default_config_path,
        description='Path to ROS 2 parameters YAML configuration file'
    )

    detector_node = Node(
        package='person_detection_ros2',
        executable='detector_node',
        name='person_detector_node',
        output='screen',
        parameters=[
            LaunchConfiguration('params_file'),
            {
                'camera_topic': LaunchConfiguration('camera_topic'),
                'conf_threshold': LaunchConfiguration('conf_threshold'),
                'weights_path': LaunchConfiguration('weights_path'),
                'device': LaunchConfiguration('device'),
            }
        ]
    )

    return LaunchDescription([
        camera_topic_arg,
        conf_threshold_arg,
        weights_path_arg,
        device_arg,
        params_file_arg,
        detector_node
    ])
