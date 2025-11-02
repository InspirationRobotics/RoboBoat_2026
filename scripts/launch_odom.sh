#!/bin/bash

tmux new-session -d -s "percept" "bash -c 'ros2 launch depthai_ros_driver camera.launch.py; exec bash'"
tmux split-window -h "bash -c 'cd src/sensors/launch && ros2 launch msg_MID360_launch.py; exec bash'"
tmux split-window -h "bash -c 'cd src/sensors/launch && ros2 launch msg_MID360_launch1.py; exec bash'"
tmux new-window -t percept:1 "bash -c 'cd /livox_ws/livox_odm && . install/setup.bash && ros2 launch fast_lio mapping.launch.py config_file:=mid360.yaml; exec bash'"
tmux split-window -h "bash -c 'cd src/sensors/gps && python3 um982_driver.py; exec bash'"
tmux new-window -t percept:2 "bash -c 'cd src/localization && python3 frame_remapper.py; exec bash'"
tmux split-window -h "bash -c 'ros2 bag record -a'"

tmux attach-session -t "percept"