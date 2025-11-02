#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu  # Change to Odometry, etc., if needed

class FrameIDRemapper(Node):
    def __init__(self):
        super().__init__('frame_id_remapper')

        # Parameters you can override via CLI or launch
        self.declare_parameter('input_topic', '/livox/imu')
        self.declare_parameter('output_topic', '/imu')
        self.declare_parameter('new_frame_id', 'body')

        input_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        output_topic = self.get_parameter('output_topic').get_parameter_value().string_value
        self.new_frame_id = self.get_parameter('new_frame_id').get_parameter_value().string_value

        self.sub = self.create_subscription(Imu, input_topic, self.callback, 10)
        self.pub = self.create_publisher(Imu, output_topic, 10)

        # self.get_logger().info(f"Remapping frame_id of {input_topic} to {output_topic} with frame_id='{self.new_frame_id}'")

    def callback(self, msg: Imu):
        msg.header.frame_id = self.new_frame_id
        self.pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = FrameIDRemapper()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
