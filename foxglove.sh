tmux new-session -d -s "foxglove" "bash -c 'ros2 launch foxglove_bridge foxglove_bridge_launch.xml send_buffer_limit:=100000000; exec bash'"
