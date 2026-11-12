"""Start the RealSense driver (aligned depth, synced) and the peg detector.

    ros2 launch peg_detection detection.launch.py
    ros2 launch peg_detection detection.launch.py launch_camera:=false   # driver already running
    ros2 launch peg_detection detection.launch.py pointcloud:=true       # cloud for the RViz check

Parameters = config/detection_params.yaml, then ~/omx_calibration/peg_detector_overrides.yaml
(written by the calibration wizard) on top, if it exists.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _detector(context):
    params = [LaunchConfiguration('params_file').perform(context)]
    over = os.path.expanduser(LaunchConfiguration('calibration_file').perform(context))
    if over and os.path.isfile(over):
        params.append(over)
        print(f'[detection.launch] using calibration overrides: {over}')
    else:
        print(f'[detection.launch] no calibration overrides at {over} (not calibrated yet)')
    return [Node(package='peg_detection', executable='peg_detector', name='peg_detector',
                 output='screen', emulate_tty=True, parameters=params)]


def generate_launch_description():
    pkg_share = get_package_share_directory('peg_detection')
    default_params = os.path.join(pkg_share, 'config', 'detection_params.yaml')
    rs_launch = os.path.join(get_package_share_directory('realsense2_camera'),
                             'launch', 'rs_launch.py')
    camera = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(rs_launch),
        condition=IfCondition(LaunchConfiguration('launch_camera')),
        launch_arguments={
            'align_depth.enable': 'true',
            'enable_sync': 'true',
            'rgb_camera.color_profile': '640x480x30',
            'depth_module.depth_profile': '640x480x30',
            'initial_reset': 'true',
            'diagnostics_period': '1.0',
            'pointcloud.enable': LaunchConfiguration('pointcloud'),
        }.items(),
    )
    return LaunchDescription([
        DeclareLaunchArgument('launch_camera', default_value='true',
                              description='Also start the RealSense driver'),
        DeclareLaunchArgument('pointcloud', default_value='false',
                              description='Publish the RealSense point cloud (RViz alignment check)'),
        DeclareLaunchArgument('params_file', default_value=default_params,
                              description='Detector parameter YAML'),
        DeclareLaunchArgument('calibration_file',
                              default_value='~/omx_calibration/peg_detector_overrides.yaml',
                              description='Overrides written by the calibration wizard'),
        camera,
        OpaqueFunction(function=_detector),
    ])
