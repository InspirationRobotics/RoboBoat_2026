import queue
import time
import threading
import csv  # Fix: Import csv module
from typing import List, Tuple
from GNC.Control_Core import motor_core
from API.Util import gis_funcs

# Initialize motors
motor_port = "/dev/ttyACM2"
motors = motor_core.MotorCore(motor_port)

lock = threading.Lock
initHeading = motors.update_position()
"""
self.position_data = {
            'current_position' : self.sensor_fuse.get_position(),
            'current_heading' : self.sensor_fuse.get_heading(),
            'current_velocity' : self.sensor_fuse.get_velocity()
        }
"""

gpsON = True
def gps_info():
    while(gpsON):
        pass