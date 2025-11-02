#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image

import torch
import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO
from cv_bridge import CvBridge
import depthai as dai
from src.read_yaml import extract_configuration

class YoloBuoyNavigator(Node):
    def __init__(self):
        super().__init__('yolo_buoy_navigator')
        # Load YOLO model

        config_file = extract_configuration()
        if config_file is None:
            self.get_logger().error("Failed to extract configuration file.")
            return
        model_path = config_file['yolo']['cuda_path']

        self.setup_depthai_pipeline()
        print("Loading YOLOv8n model...")
        self.model = YOLO(model_path)
        self.model.to('cuda') # need jetson-compatible pytorch
        print("... Done.")
        
        # ROS2 publisher
        self.publisher = self.create_publisher(String, 'steering_command', 10)
        self.image_pub = self.create_publisher(Image, 'annotated_image', 10)
        self.bridge = CvBridge()

        # History of buoy positions
        self.red_history = deque(maxlen=20)
        self.green_history = deque(maxlen=20)

        # Video source (replace with DepthAI if needed)
        self.timer = self.create_timer(0.03, self.process_frame)

    def setup_depthai_pipeline(self):
        pipeline = dai.Pipeline()

        cam_rgb = pipeline.createColorCamera()
        cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam_rgb.setInterleaved(False)
        cam_rgb.setFps(30)

        xout = pipeline.create(dai.node.XLinkOut)
        xout.setStreamName("video")
        cam_rgb.video.link(xout.input)

        self.device = dai.Device(pipeline)
        self.video_queue = self.device.getOutputQueue(name="video", maxSize=1, blocking=False)

    def process_frame(self):
        in_frame = self.video_queue.get()
        frame = in_frame.getCvFrame()
        results = self.model.predict(frame, verbose=False, device="cuda")
        detections = results[0]

        red_positions, green_positions = [], []
        self.displayFrame("rgb", frame, detections)

        for box in detections.boxes:
            x1, y1, x2, y2 = map(float, box.xyxy[0].tolist()) 
            conf = float(box.conf[0])
            label = float(box.cls[0])  

            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            if label == 16: # red buoy
                red_positions.append((x_center, y_center))
            elif label == 11: # green buoy
                green_positions.append((x_center, y_center))

        self.update_history(self.red_history, red_positions)
        self.update_history(self.green_history, green_positions)

        self.predict_and_draw_path(frame, self.red_history, self.green_history)

        steering = self.calculate_steering_command(frame, self.red_history, self.green_history)

        msg = String()
        msg.data = steering
        self.publisher.publish(msg)

        cv2.putText(frame, f"Steering: {steering}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 2)
        # cv2.imshow("RoboBoat Navigation", frame)
        # cv2.waitKey(1)
        
        # Publish annotated image
        ros_img = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        self.image_pub.publish(ros_img)

    def update_history(self, history, positions):
        for pos in positions:
            history.append(pos)

    def predict_and_draw_path(self, frame, red_history, green_history):
        if len(red_history) > 1 and len(green_history) > 1:
            red_points = np.array([[int(x), int(y)] for x, y in red_history], np.int32)
            green_points = np.array([[int(x), int(y)] for x, y in green_history], np.int32)

            red_points = red_points[np.argsort(red_points[:, 1])]
            green_points = green_points[np.argsort(green_points[:, 1])]

            if len(red_points) > 1 and len(green_points) > 1:
                path = np.vstack((red_points, green_points[::-1]))
                overlay = frame.copy()
                cv2.fillPoly(overlay, [path], (0, 255, 255), lineType=cv2.LINE_AA)
                cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

    def calculate_steering_command(self, frame, red_history, green_history):
        if red_history and green_history:
            red = red_history[-1]
            green = green_history[-1]

            midpoint_x = (red[0] + green[0]) / 2
            center_x = frame.shape[1] / 2

            if midpoint_x < center_x - 80:
                return "Left"
            elif midpoint_x > center_x + 80:
                return "Right"
            else:
                return "Straight"
        return "No Command"

    def displayFrame(self, name, frame, detections):
        color = (255, 0, 0)

        for box in detections.boxes:
            conf = float(box.conf[0])
            label = float(box.cls[0])  
            label_tag = 'Red Buoy' if label == 16 else 'Green Buoy' if label == 11 else 'Unknown'

            bbox = list(map(int, box.xyxy[0].tolist()))
            cv2.putText(frame, label_tag, (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(frame, f"{int(conf * 100)}%", (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        # Show the frame
        # cv2.imshow(name, frame)



def main(args=None):
    rclpy.init(args=args)
    node = YoloBuoyNavigator()  # Replace with actual path
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
