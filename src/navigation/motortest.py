import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from t200 import T200  # Assuming T200 class is in a file called t200.py in the same folder

class NavigationServer(Node):
    def __init__(self):
        super().__init__('navigation_server')

        self.declare_parameters(
            namespace='',
            parameters=[
                ('navigation_input', 'navigation_input'),
                ('move_forward_cmd', 'w'),
                ('move_backward_cmd', 'x'),
                ('turn_left_cmd', 'a'),
                ('turn_right_cmd', 'd'),
                ('stop_cmd', 's'),
                ('serial_port', '/dev/ttyACM2')
            ]
        )

        # Load parameters
        self.servo_topic_name = self.get_param_str('navigation_input')
        self.move_forward_cmd = self.get_param_str('move_forward_cmd')
        self.move_backward_cmd = self.get_param_str('move_backward_cmd')
        self.turn_left_cmd = self.get_param_str('turn_left_cmd')
        self.turn_right_cmd = self.get_param_str('turn_right_cmd')
        self.stop_cmd = self.get_param_str('stop_cmd')
        self.serial_port = self.get_param_str('serial_port')

        # Initialize T200 thruster controller
        self.thrusters = T200(port=self.serial_port)

        # Create subscriber
        self.subscriber = self.create_subscription(
            String,
            self.servo_topic_name,
            self.serial_listener_callback,
            10
        )

        self.get_logger().info('Navigation server is running.')

    def get_param_str(self, name):
        return self.get_parameter(name).get_parameter_value().string_value

    def serial_listener_callback(self, msg):
        command = msg.data
        self.get_logger().info(f"Received command: {command}")

        if command == self.move_forward_cmd:
            self.thrusters.set_thrusters(-0.3, -0.3, 0.3, -0.3)
        elif command == self.move_backward_cmd:
            self.thrusters.set_thrusters(0.3, 0.3, -0.3, 0.3)
        elif command == self.turn_left_cmd:
            self.thrusters.set_thrusters(-0.15, -0.15, -0.15, -0.15)
        elif command == self.turn_right_cmd:
            self.thrusters.set_thrusters(-0.15, -0.15, 0.15, 0.15)
        elif command == self.stop_cmd:
            self.thrusters.set_thrusters(0.0, 0.0, 0.0, 0.0)
        else:
            self.get_logger().warn(f"Unknown command: {command}")

        # Motors stop after 100ms of any command 
        # time.sleep(0.1)
        # self.thrusters.set_thrusters(0.0, 0.0, 0.0, 0.0)

        # timer = threading.Timer(0.1, self.thrusters.set_thrusters(0.0, 0.0, 0.0, 0.0))
        # timer.start()

def main(args=None):
    rclpy.init(args=args)
    navigation_server = NavigationServer()
    rclpy.spin(navigation_server)
    navigation_server.thrusters.stop_thrusters()
    navigation_server.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
