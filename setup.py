import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'person_detection_ros2'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Rabeb Bouzaida',
    maintainer_email='rabeb.bouzaida@enim.u-monastir.tn',
    description='ROS 2 Humble real-time YOLOv8 person detector with Hard Negative Mining (HNM)',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'detector_node = person_detection_ros2.detector_node:main',
        ],
    },
)
