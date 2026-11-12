import os
from glob import glob

from setuptools import setup

package_name = 'peg_detection'

setup(
    name=package_name,
    version='0.2.0',
    packages=[package_name],
    data_files=[
        # Lets ROS find this package by name.
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Install launch and config files so `ros2 launch` can find them.
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='kshitij',
    maintainer_email='kshitij@todo.todo',
    description='Colour-based 3D peg + socket detection in the robot frame (RealSense D435i, ROS 2 Jazzy)',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            # ros2 run peg_detection peg_detector
            'peg_detector = peg_detection.peg_detector_node:main',
            # ros2 run peg_detection generate_marker
            'generate_marker = peg_detection.generate_marker:main',
            # ros2 run peg_detection frame_calibration   (drawn axes -> robot link1)
            'frame_calibration = peg_detection.frame_calibration:main',
        ],
    },
)
