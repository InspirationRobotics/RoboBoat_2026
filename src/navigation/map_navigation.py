#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from nav_msgs.msg import Odometry
import numpy as np
import ast
import math


class MapBasedNavigator(Node):
    def __init__(self):
        super().__init__('map_based_navigator')

        self.subscription_map = self.create_subscription(
            String,
            '/landmark_map',
            self.map_callback,
            10
        )
        self.subscription_odom = self.create_subscription(
            Odometry,
            '/Odometry',
            self.odom_callback,
            10
        )

        self.publisher = self.create_publisher(String, 'steering_command', 10)
        self.timer = self.create_timer(0.2, self.compute_navigation)

        self.latest_map = {}
        self.boat_pos = np.array([0.0, 0.0])
        self.boat_yaw = 0.0

    def map_callback(self, msg):
        try:
            self.latest_map = ast.literal_eval(msg.data)
        except Exception as e:
            self.get_logger().error(f"Failed to parse map: {e}")

    def odom_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self.boat_pos = np.array([x, y])

        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.boat_yaw = math.atan2(siny_cosp, cosy_cosp)  # heading angle

    def compute_navigation(self):
        red_buoys = np.array(self.latest_map.get(16, []))
        green_buoys = np.array(self.latest_map.get(11, []))

        if len(red_buoys) == 0 or len(green_buoys) == 0:
            self.publisher.publish(String(data="No Command"))
            return

        # Pair buoys based on y-axis (forward direction), forming virtual gates
        gate_midpoints = []
        for r in red_buoys:
            for g in green_buoys:
                mid = (np.array(r) + np.array(g)) / 2.0
                gate_midpoints.append(mid)

        gate_midpoints = np.array(gate_midpoints)

        # Filter gates that are ahead of the boat (positive projection on forward vector)
        forward_vec = np.array([math.cos(self.boat_yaw), math.sin(self.boat_yaw)])
        relative_vecs = gate_midpoints - self.boat_pos
        projections = np.dot(relative_vecs, forward_vec)

        forward_gates = gate_midpoints[projections > 0]
        if len(forward_gates) == 0:
            self.publisher.publish(String(data="No Forward Gate"))
            return

        # Pick the nearest forward gate
        dists = np.linalg.norm(forward_gates - self.boat_pos, axis=1)
        target_midpoint = forward_gates[np.argmin(dists)]

        # Compute lateral error (dot with perpendicular vector)
        perp_vec = np.array([-forward_vec[1], forward_vec[0]])
        lateral_error = np.dot(target_midpoint - self.boat_pos, perp_vec)

        # Thresholds
        LATERAL_THRESHOLD = 0.3  # meters

        if lateral_error > LATERAL_THRESHOLD:
            command = "Right"
        elif lateral_error < -LATERAL_THRESHOLD:
            command = "Left"
        else:
            command = "Straight"

        self.get_logger().info(f"Steering: {command}, Lateral Error: {lateral_error:.2f} m")
        self.publisher.publish(String(data=command))


def main(args=None):
    rclpy.init(args=args)
    node = MapBasedNavigator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()