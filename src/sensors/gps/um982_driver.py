import sys
import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix
from nav_msgs.msg import Odometry
from tf_transformations import quaternion_from_euler, euler_from_quaternion


from um982 import UM982Serial


class UM982DriverROS2(Node):

    def _ros_log_debug(self, log_data):
        self.get_logger().debug(str(log_data))

    def _ros_log_info(self, log_data):
        self.get_logger().info(str(log_data))

    def _ros_log_warn(self, log_data):
        self.get_logger().warn(str(log_data))

    def _ros_log_error(self, log_data):
        self.get_logger().error(str(log_data))


    def __init__(self) -> None:
        super().__init__('um982_serial_driver')
        self.declare_parameter('port', '/dev/ttyUSB0')
        self.declare_parameter('baud', 115200)

        port = self.get_parameter('port').get_parameter_value().string_value
        baud = self.get_parameter('baud').get_parameter_value().integer_value

        try:
            self.um982serial = UM982Serial(port, baud)
            self._ros_log_info(f'serial {port} open successfully!')
        except Exception as e:
            self._ros_log_error(f'serial {port} do not open!')
            self._ros_log_error(e) 
            sys.exit(0)

        self.um982serial.start()

        self.fix_pub        = self.create_publisher(NavSatFix, '/gps/fix',     10)
        self.pub_timer      = self.create_timer(1/20, self.pub_task)

    def pub_task(self):
        serial_data = self.um982serial.data

        if not serial_data.is_valid():
            return
        
        lat, lon, heading = serial_data
        
        this_time = self.get_clock().now().to_msg()

        # Step 1: Publish GPS Fix Data
        fix_msg = NavSatFix()
        fix_msg.header.stamp = this_time
        fix_msg.header.frame_id = 'gps'
        fix_msg.latitude = lat
        fix_msg.longitude = lon

        fix_msg.position_covariance[0] = 0
        fix_msg.position_covariance[4] = 0
        fix_msg.position_covariance[8] = 0

        fix_msg.position_covariance_type = NavSatFix.COVARIANCE_TYPE_UNKNOWN
        self.fix_pub.publish(fix_msg)

    def run(self):
        if rclpy.ok():
            rclpy.spin(self)

    def stop(self):
        self.um982serial.stop()
        self.pub_timer.cancel()


import time
import signal

def signal_handler(sig, frame):
    um982_driver.stop()
    time.sleep(0.1)
    if rclpy.ok():
        rclpy.shutdown()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)
rclpy.init()
um982_driver = UM982DriverROS2()


def main():
    um982_driver.run()

if __name__ == "__main__":
    main()
