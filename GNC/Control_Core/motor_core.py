"""
This script only do certain motion, magnitude between 0 and 1
"""
from API.Motors import t200

class MotorCore:
    def __init__(self, motor_port : str = "/dev/ttyACM2",debug=False):
        self.t200 = t200.T200(motor_port,debug=debug)

    def surge(self, magnitude):
        """Configures for forward (positive magnitude) or backward (negative magnitude) movement"""
        self.t200.set_thrusters(-magnitude, -magnitude, magnitude, -magnitude)

    def stay(self):
        """Sets all motors to no power."""
        self.t200.set_thrusters(0,0,0,0)

    def stop(self):
        """Stop motors."""
        self.t200.stop_thrusters()

    def slide(self, magnitude):
        """Sliding (strafing) in horizontal direction without rotating, positive is left, negative is right."""
        # NOTE: We want positive to be right, negative to be right.
        self.t200.set_thrusters(magnitude,-magnitude,magnitude,magnitude)

    def veer(self, magnitudeF, magnitudeB):
        """ 
        Move forward and yaw at the same time.
        Arguments:
            magnitudeF : magnitude for front motors
            magnitudeB : magnitude for back motors

        Usage:
            Positive magnitudeF is forward
            Positive magnitudeB is clockwise
        """
        self.t200.set_thrusters(-magnitudeF,-magnitudeF,magnitudeB,magnitudeB)

    def rotate(self, magnitude):
        """Rotate, all motors help rotation Positive magnitude is clockwise, negative is counterclockwise."""
        self.t200.set_thrusters(-magnitude,magnitude,magnitude,magnitude)
