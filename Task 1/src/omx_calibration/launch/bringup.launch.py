"""Everything the calibration (and later the pick-and-place) needs, on one laptop:

    ros2 launch omx_calibration bringup.launch.py
      arm:=true            ROBOTIS OpenMANIPULATOR-X bringup (ros2_control, controllers, TF)
      init_position:=false default: base is multi-turn, use `ros2 run omx_calibration go_home`
      port_name:=/dev/ttyUSB0
      vision:=true         RealSense + peg detector (+ calibration overrides)
      viewer:=true         rqt_image_view on /peg_detector/annotated_image
      pointcloud:=false
Then, in a second terminal:  ros2 run omx_calibration wizard
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    C = LaunchConfiguration
    arm = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution(
            [FindPackageShare('open_manipulator_bringup'), 'launch', 'open_manipulator_x.launch.py'])),
        condition=IfCondition(C('arm')),
        launch_arguments={'init_position': C('init_position'), 'port_name': C('port_name')}.items())
    vision = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution(
            [FindPackageShare('peg_detection'), 'launch', 'detection.launch.py'])),
        condition=IfCondition(C('vision')),
        launch_arguments={'pointcloud': C('pointcloud')}.items())
    viewer = Node(package='rqt_image_view', executable='rqt_image_view',
                  arguments=['/peg_detector/annotated_image'], condition=IfCondition(C('viewer')))
    return LaunchDescription([
        DeclareLaunchArgument('arm', default_value='true'),
        DeclareLaunchArgument('init_position', default_value='false'),
        DeclareLaunchArgument('port_name', default_value='/dev/ttyUSB0'),
        DeclareLaunchArgument('vision', default_value='true'),
        DeclareLaunchArgument('viewer', default_value='true'),
        DeclareLaunchArgument('pointcloud', default_value='false'),
        arm,
        TimerAction(period=3.0, actions=[vision]),     # let the arm/USB settle first
        TimerAction(period=6.0, actions=[viewer]),
    ])
