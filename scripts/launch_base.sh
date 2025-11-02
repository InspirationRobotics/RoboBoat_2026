#!/bin/bash
timestamp=$(date +"%m%d_%H%M%S")
bag_name="rosbag2_${timestamp}_base"

tmux new-session -d -s "nav" "bash -c 'python3 src/navigation/buoy_navigation.py; exec bash'"
tmux split-window -h "bash -c 'ros2 bag record -o $bag_name /annotated_image /steering_command'"

tmux attach-session -t "nav"