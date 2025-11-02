#!/bin/bash
timestamp=$(date +"%m%d_%H%M%S")
bag_name="rosbag2_${timestamp}_mapping"

tmux new-session -d -s "percept" "bash -c 'ros2 run perception dai_node; exec bash'"
tmux split-window -h "bash -c 'ros2 launch livox_ros_driver2 msg_MID360_launch.py xfer_format:=1; exec bash'"
tmux split-window -h "bash -c 'ros2 run perception lidar_camera_projection; exec bash'"

#Fast-LIO
tmux new-window -t percept:1 "bash -c 'cd /livox_ws/livox_odm && . install/setup.bash && ros2 launch fast_lio mapping.launch.py config_file:=mid360.yaml; exec bash'"
# tmux split-window -h "bash -c 'cd src/sensors/gps && python3 um982_driver.py; exec bash'"

tmux new-window -t percept:2 "bash -c 'python3 src/mapping/mapper.py; exec bash'"
tmux split-window -h "bash -c 'python3 src/navigation/map_navigation.py; exec bash'"
# tmux split-window -h "bash -c 'ros2 bag record -a'"
# tmux split-window -h "bash -c 'ros2 bag record -o $bag_name /dai_node/annotated_image /centroids /steering_command /oak/rgb/projected; exec bash'"

tmux attach-session -t "percept"