# Start from Ubuntu
FROM humble-gpu-ubuntu22.04:latest

RUN sudo apt-get install ros-humble-foxglove-bridge

RUN mkdir /livox_ws && \
    cd /livox_ws && \
    git clone https://github.com/Livox-SDK/Livox-SDK2.git && \
    cd ./Livox-SDK2/ && \
    mkdir build && \
    cd build && \
    cmake .. && make -j2 && \
    sudo make install

RUN mkdir -p /livox_ws/livox_ros2_ws/src && \
    cd /livox_ws/livox_ros2_ws/src && \
    git clone https://github.com/Livox-SDK/livox_ros_driver2.git

COPY config/MID360_config.json livox_ws/livox_ros2_ws/src/livox_ros_driver2/config/MID360_config.json

SHELL ["/bin/bash", "-c"]

RUN source /opt/ros/humble/setup.bash && \
    cd /livox_ws/livox_ros2_ws/src/livox_ros_driver2 && \
    ./build.sh humble && \
    source /livox_ws/livox_ros2_ws/install/setup.bash

# Remove bash commands from the base image 
RUN sed -i '/source \/opt\/ros\/iron\/setup.bash/d' /etc/bash.bashrc
RUN sed -i '/source \/workspace\/oak_ws\/install\/setup.bash/d' ~/.bashrc

RUN echo "source /livox_ws/livox_ros2_ws/install/setup.bash" >> ~/.bashrc && \
    echo "source /workspace/install/setup.bash" >> ~/.bashrc 

# Set up a working directory
WORKDIR /workspace

# Default command
CMD ["bash"]
