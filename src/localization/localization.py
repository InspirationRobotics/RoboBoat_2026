import os
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, LaunchConfiguration

from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
import launch_ros

def generate_launch_description():
    
    # IMU-GPS Fusion node
    # start_ekf_local = Node(
    #     package='robot_localization',
    #     executable='ekf_node',
    #     name='ekf_localization_node',
    #     output='screen',
    #     parameters=['/workspace/src/localization/ekf_imu.yaml']
    #     )
    
    # LiDAR-IMU Fusion node
    start_ekf_local = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_localization_node',
        output='screen',
        parameters=['/workspace/src/localization/ekf_local.yaml'],
        # remappings=[('odometry/filtered', 'odometry/local')]  ,
        # arguments=['--ros-args', '--log-level', 'debug'],
        )
    
    # start_ekf_global = Node(
    #     package='robot_localization',
    #     executable='ekf_node',
    #     name='ekf_filter_node_map',
    #     output='screen',
    #     # parameters=[os.path.join(pkg_share, 'config/dual_ekf_navsat.yaml')],
    #     parameters=['/workspace/src/localization/ekf_global.yaml'],
    #     remappings=[('odometry/filtered', 'odometry/global')]
    #     )
    
    # navsat_transform= Node(
    #     package='robot_localization',
    #     executable='navsat_transform_node',
    #     name='navsat_transform',
    #     output='screen',
    #     parameters=['/workspace/src/localization/navsat.yaml'],
    #     # remappings=[('/livox/imu', 'imu/data')]
    #     )

    return LaunchDescription([
        start_ekf_local,
        # start_ekf_global,
        # navsat_transform
    ])