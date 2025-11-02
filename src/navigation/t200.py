import time
import serial
from threading import Thread, Lock

class Arduino:
    """
    Class to set up serial connection with the Arduino.

    Args:
        port (str): Port to initiate serial connection with. Defaults to "/dev/tty/ACM0".
        baudrate (int): Baudrate to set serial connection to. Defaults to 115200.
    """
    def __init__(self, port="/dev/ttyACM2", baudrate=9600):
        self.arduino = serial.Serial(port=port, baudrate=baudrate, timeout = 0.1)

        self.send_PWM([1500] * 4)
        time.sleep(1)
 
    def send_PWM(self, command):
        """
        Writes the PWMs to the arduino.

        Args:
            command (list): List of PWMs to send.
        """
        for index, value in enumerate(command):
            command[index] = str(value)
        parsed_pwms = ",".join(command)
        print(parsed_pwms)
        # Example parsed_pwms : 1500,1600,1500,1750
        self.arduino.write(parsed_pwms.encode())

class T200(Arduino):
    """
    Class to set the PWM values for each motor. Inherits from the Arduino class.

    Args:
        port (str): Port to initiate serial connection with. Defaults to "/dev/ttyACM0"
        debug (bool): Mode of whether or not to execute functions or not (debug = no execution). Defaults to false.
    """

    def __init__(self, port = "/dev/ttyACM2", debug = False):
        self.debug = debug

        if not debug:
            super().__init__(port)

            # These are the variables for motor speed between -1 and 1. Converted to PWM values later.
            self.forward_port_speed = 0
            self.forward_starboard_speed = 0
            self.aft_port_speed = 0
            self.aft_starboard_speed = 0

            self.motor_PWM_list = [1500] * 4
            
            self.send_thread = Thread(target = self.set_speed_thread, daemon = True)
            self.lock = Lock()
            self.active = True
            
            self.send_thread.start()

    def debug_decorator(func):
        """
        Decorator so that when in debug mode, the function passed in just returns, it does not execute.
        """
        def wrapper(self, *args, **kwargs):
            if self.debug:
                return
            
            return func(self, *args, **kwargs)
        
        return wrapper

    @debug_decorator
    def set_thrusters(self, forward_port, forward_starboard, aft_port, aft_starboard):
        """
        Gets incoming calculated values for each motor (between -1 and 1), and stores them to be calculated into PWM values.
        Forces the values to be in between -1, 1, if not already.
        
        Args:
            forward_port (float) : Value between -1 and 1  
            forward_starboard (float) : Value between -1 and 1 
            aft_port (float) : Value between -1 and 1 
            aft_starboard (float) : Value between -1 and 1 
        """
        with self.lock:
            self.forward_port_speed = min(0.85, max(forward_port, -0.85))
            self.forward_starboard_speed = min(0.85, max(forward_starboard, -0.85))
            self.aft_port_speed = min(0.85, max(aft_port, -0.85))
            self.aft_starboard_speed = min(0.85, max(aft_starboard, -0.85))

    def set_speed_thread(self):
        """
        Setting thruster PWMs based on expected value between -1 and 1 for each motor.
        Scales between 1200-1800 PWM where 0 is 1500.
        PWMs between 1460 and 1540 will provide no thrust (kg f), so we have to take that into account when setting motor speed.
        For more information see "Performance Charts" under technical specification in the following link: 
        https://bluerobotics.com/store/thrusters/t100-t200-thrusters/t200-thruster-r2-rp/
        """
        while self.active:
            with self.lock:
                motor_speeds = [self.forward_port_speed, self.forward_starboard_speed, self.aft_port_speed, self.aft_starboard_speed]
                
                for index, value in enumerate(motor_speeds):
                    # First calculate motor speeds
                    list_value = value
                    PWM_value = int(1500 + list_value * 300)

                    # Handling cases where PWM value is in the set [1460, 1540] but is not 1500
                    if PWM_value != 1500 and abs(PWM_value - 1500) <= 40:
                        if PWM_value - 1500 < 0:
                            PWM_value = 1460
                        
                        if PWM_value - 1500 > 0:
                              PWM_value = 1540

                    # Change the value at the right index
                    self.motor_PWM_list[index] = PWM_value

            self.send_PWM(self.motor_PWM_list)
            time.sleep(1.5)
    
    @debug_decorator
    def stop_thrusters(self):
        """
        Kills thrusters, closes serial connection.
        """
        self.set_thrusters(0, 0, 0, 0)
        time.sleep(5)
        self.active = False
        self.send_thread.join(2)
        self.arduino.close()
        print("Thrusters stopped")

    # NOTE: This might be dangerous.
    def __del__(self):
        self.stop_thrusters()

