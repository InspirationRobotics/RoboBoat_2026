"""Importing float32, float64, string, and int16 data types to initialize variables with
Importing ROS, time, numpy, math
Importing IMU"""

#!/usr/bin/env python
import rclpy
from rclpy.node import Node
import time
import numpy as np
from std_msgs.msg import Float32
from std_msgs.msg import Float64
from std_msgs.msg import String
from std_msgs.msg import Int16
from sensor_msgs.msg import NavSatFix
from sensor_msgs.msg import Imu
import timer
# from geometry_msgs.msg import Quaternion
# from geometry_msgs.msg import Pose
# from geometry_msgs.msg import Point
# from geographic_msgs.msg import WayPoint
# from geographic_msgs.msg import GeoPoint
# from geographic_msgs.msg import KeyValue
# from geographic_msgs.msg import GeoPath
# from geographic_msgs.msg import GeoPoseStamped


import math

"""Initialize variables/constants:
Current latitude/longitude (deg)
Target angle/lat/longitude (deg)
Booleans for arrival status, waypoint start and finish
Target distance (m)
Minimum distance constant (threshold for meeting waypoint)
Angle threshold of 10 degrees
"""


class auto_nav(Node):

    def __init__(self):
        super().__init__('auto_nav')
        # self.current_lat = 0
        # self.current_lon = 0
        self.target_angle = 0.0
        # self.target_lon = 0
        # self.target_lat = 0
        # self.arrived = True
        # self.waypoint_done = False
        # self.waypoint_started = False
        # self.station_started = False
        self.target_distance = 0
        # TODO: tune threshold
        self.MIN_DIST = 0.00003 #in lat/lon
        # TODO: finetune
        self.ANGLE_THR = 10 # angle in degrees
        self.arrived_pub = self.create_publisher(String, '/wamv/navigation/arrived', 10)
        self.mc_torqeedo = self.create_publisher(String, '/wamv/torqeedo/motor_cmd', 10)
        self.navigation_input=self.create_publisher(String,'navigation_input',10)
        # subscriber
        self.subscription = self.create_subscription(
            String,
            'steering_command',
            self.steering_callback,
            10
        )
        self.waypoint_list = []
        self.waypoint_index = 0
        self.msg = String()
        print("Navigator Init done")

    def steering_callback(self, msg):
        command = msg.data
        print(command)
        if command == 'Left':
            self.turn_left()
        elif command == 'Right':
            self.turn_right()
        elif command == 'Straight':
            self.move_forward()
        else:
            self.stop()

    def turn_left(self):
        dir_to_move = "a"  # left
        self.msg.data=dir_to_move
        self.navigation_input.publish(self.msg)

    def turn_right(self):
        dir_to_move = "d"  # right
        self.msg.data=dir_to_move
        self.navigation_input.publish(self.msg)

    def move_forward(self):
        dir_to_move = "w"  # forward
        self.msg.data=dir_to_move
        self.navigation_input.publish(self.msg)

    def stop(self):
        dir_to_move = "s"
        self.msg.data=dir_to_move
        self.navigation_input.publish(self.msg)


    def test_move(self):
        print("TESTING THRUSTERS")
        while True:
            self.move_forward()
            print("go straight")
            time.sleep(5)
            self.turn_left()
            print("turn left")
            time.sleep(5)
            self.turn_right()
            print("turn right")
            time.sleep(5)
            # Add more movements as needed



def main(args=None):
    rclpy.init()
    navigator = auto_nav()
    #navigator.test_move()
    rclpy.spin(navigator)

if __name__ == '__main__':
    main()
