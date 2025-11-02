docker run \
    --runtime nvidia \
    --name container_name \
    -e TZ=America/Los_Angeles \
    -it \
    --privileged \
    --net=host \
    --gpus all \
    --shm-size=2g \
    -e DISPLAY=$DISPLAY \
    -e QT_X11_NO_MITSHM=1 \
    -e XDG_RUNTIME_DIR=/tmp/runtime-root \
    -v /dev/bus/usb:/dev/bus/usb \
    --device-cgroup-rule='c 189:* rmw' \
    --device /dev/video0 \
    --volume='/dev/input:/dev/input' \
    --volume='/home/jetson/.Xauthority:/root/.Xauthority:rw' \
    --volume='/tmp/.X11-unix/:/tmp/.X11-unix' \
    --volume='./:/workspace' \
    roboboat-final:latest 