#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String

import cv2
import numpy as np
from collections import deque
import sys
sys.path.insert(1, '../perception/')

from src.read_yaml import extract_configuration


class FuseBuoyNavigator(Node):
    def __init__(self):
        super().__init__('fuse_buoy_navigator')

        config = extract_configuration()

        self.visualize = config['navigation']['visualize']
        self.frame_width = config['camera']['image_size']['width']
        self.frame_height = config['camera']['image_size']['height']

        self.red_history = deque(maxlen=20)    # Only x values
        self.green_history = deque(maxlen=20)  # Only x values

        self.subscription = self.create_subscription(
            Float32MultiArray,
            '/centroids',
            self.centroid_callback,
            10
        )

        self.publisher = self.create_publisher(String, 'steering_command', 10)
        self.timer = self.create_timer(0.1, self.visualize_and_publish)
        self.count = 0
        self.thresh = 3
        print('Initialization Success.')

    def centroid_callback(self, msg):
        data = msg.data
        if len(data) < 6:
            self.get_logger().warn("Invalid centroid message length")
            return
        
        if data[0] != -1:
            self.red_history.append(data[0])   # red_x
        
        if data[3] != -1:
            self.green_history.append(data[3]) # green_x

    def visualize_and_publish(self):
        frame = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)

        if self.red_history and self.green_history:
            red_x = self.red_history[-1]
            green_x = self.green_history[-1]
            midpoint_x = int((red_x + green_x) / 2)
            center_x = self.frame_width // 2

            if midpoint_x < center_x - 100:
                steering = "Left"
            elif midpoint_x > center_x + 100:
                steering = "Right"
            else:
                steering = "Straight"

            # if center_x < red_x +  self.frame_width * 0.2 :
            #     steering = "Right"
            # elif center_x > green_x -  self.frame_width * 0.2:
            #     steering = "Left"
            # else:
            #     steering = "Straight"

            if self.visualize:
                for x in self.red_history:
                    cv2.circle(frame, (int(x), self.frame_height // 3), 5, (0, 0, 255), -1)
                for x in self.green_history:
                    cv2.circle(frame, (int(x), 2 * self.frame_height // 3), 5, (0, 255, 0), -1)

                cv2.circle(frame, (midpoint_x, self.frame_height // 2), 8, (255, 255, 0), -1)
                cv2.line(frame, (center_x, 0), (center_x, self.frame_height), (255, 255, 255), 1)
                cv2.putText(frame, f"Steering: {steering}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                cv2.imshow("RoboBoat Navigation", frame)
                cv2.waitKey(1)

        else:
            steering = "No Command"
            
        # if self.count != 0:
        #     print(f"Force silence ({self.count}/{self.thresh})")
        #     steering = "No Command"

        self.publisher.publish(String(data=steering))
        
        self.count += 1
        self.count %= self.thresh


def main(args=None):
    rclpy.init(args=args)
    node = FuseBuoyNavigator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
