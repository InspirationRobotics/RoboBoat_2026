"""
This code has never been tested (as of 03/08/2025) because the IMU was never mounted on Barco Polo. As such, this module has not been tested.
"""

import time
# Need to download Adafruit-blinka locally before this file will work
import board
import busio

from adafruit_bno08x import (
    BNO_REPORT_ACCELEROMETER,
    BNO_REPORT_LINEAR_ACCELERATION,
    BNO_REPORT_GYROSCOPE,
    BNO_REPORT_MAGNETOMETER,
    BNO_REPORT_ROTATION_VECTOR,
    BNO_REPORT_GRAVITY
)
from adafruit_bno08x.i2c import BNO08X_I2C

import numpy as np
from pathlib import Path
from typing import Tuple, Any
from threading import Thread, Lock
from math import atan2, sqrt, pi

class IMUData:
    """
    Class to be used as a data type to store the data from the IMU.

    Args:
        # NOTE: Not exactly sure what the arguments are; will need to test to find out, and will need to update documentation accordingly, specifically order
        of axes in the tuples

        All arguments default to None.
        accel (tuple): Three axes of acceleration (gravity + linear motion) in m/s^2
        gyro (tuple): Three axes of 'rotation speed' in rad/s
        mag (tuple): Three axes of magnetic field sensing in micro Tesla (uT)
        quat (tuple): Four point quaternion output
        euler (tuple): Euler representation of quaternion output.

    Attributes include all arguments, but also include timestamp, which is the current time.
    """
    def __init__(self, accel : tuple = None, gyro : tuple = None, mag : tuple = None, quat : tuple = None, euler : tuple = None):
        self.accel = accel
        self.gyro = gyro
        self.mag = mag
        self.quat = quat
        self.euler = euler
        self.timestamp = time.time()
        
    @staticmethod
    def _quat_to_euler(quat : Tuple[float, float, float, float]) -> Tuple[float, float, float]:
        """
        Quaterinion to euler conversion.
        """
        # This function should convert a quaternion to its euler representation
        w, x, y, z = quat
        # Roll
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = atan2(sinr_cosp, cosr_cosp)
        # Pitch
        sinp = sqrt(1 + 2 * (w * y - x * z))
        cosp = sqrt(1 - 2 * (w * y - x * z))
        pitch = 2 * atan2(sinp, cosp) - pi / 2
        # Yaw
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = atan2(siny_cosp, cosy_cosp)
        return roll, pitch, yaw
        
    def __setattr__(self, name: str, value: Any) -> None:
        """
        Sets the attribute of a key-value pair in a dictionary. Also sets the timestamp (key) of the dictionary to the current time (value).

        Args:
            name (str): Key of the dictionary position.
            value: Value of the dictionary position with the given key. Can be any type.
        """
        self.__dict__[name] = value
        self.__dict__["timestamp"] = time.time()

    def __str__(self) -> str:
        """
        Returns f-string of all attributes of the IMUData object.
        """
        return f"""Time: {self.timestamp}, Acceleration: {self.accel}, Gyroscopic rotation: {self.gyro}, 
            Magnetic field strength: {self.mag}, Quaternion: {self.quat}, Euler: {self.euler}"""
        
    def __repr__(self) -> str:
        return self.__str__()


class IMU:
    """
    Class to handle all low-level IMU functionality. Upon instantiation automatically initiates connection with the IMU.

    Args:
        callback (func): Callback function to perform some action with the IMU data. Defaults to None.
        threaded (bool): Whether to start the IMU thread to get updated data. Defaults to True.
    """
    """
    Specs:
    https://www.adafruit.com/product/4754
    """

    def __init__(self, callback = None, threaded : bool = True):
        self.threaded = threaded
        self.callback = callback

        self.bno = self.imu_init()
        self.data : IMUData = IMUData(None, None, None, None)
        
        self.lock = Lock()
        self.active = True
        self.imu_thread = Thread(target=self.__imu_thread, daemon=True)

        if threaded:
            self.imu_thread.start()

    def __del__(self):
        """
        Delete function for the class; occurs when there are no longer any references to the class.
        Stops the IMU thread.
        """
        if self.threaded:
            self.active = False
            self.imu_thread.join(2)

    def imu_init(self):
        """
        Initializes the IMU connection. Starts the I2C connection, and configures the IMU object, which 
        handles all functionality, like the reading of data.

        Returns:
            bno (busion.I2C): Configured IMU object.
        """
        # Frequency is 400 kHz
        i2c = busio.I2C(board.SCL, board.SDA, frequency=400000)
        bno = BNO08X_I2C(i2c)
        bno.enable_feature(BNO_REPORT_LINEAR_ACCELERATION)
        bno.enable_feature(BNO_REPORT_GYROSCOPE)
        bno.enable_feature(BNO_REPORT_MAGNETOMETER)
        bno.enable_feature(BNO_REPORT_ROTATION_VECTOR)
        bno.enable_feature(BNO_REPORT_GRAVITY)
        return bno

    def _get_single_data(self) -> IMUData:
        """
        Gets data from a single "moment" from the IMU. Also auto-handles quaternion to euler conversion.

        Returns:
            data (IMUData): IMUData object with all attributes filled (see IMUData's declaration for more information).
        """
        data = IMUData()
        accel = list(self.bno.linear_acceleration)
        for i in range(3):
            accel[i] = accel[i] if abs(accel[i]) > 0.005 else 0
        data.accel = tuple(accel) # - self.accel_calib if self.calibrated else accel
        data.gyro = self.bno.gyro
        data.mag = self.bno.magnetic
        data.quat = self.bno.quaternion
        data.euler = IMUData._quat_to_euler(data.quat)
        return data

    def __imu_thread(self):
        """
        Callback function to run continuously on the IMU thread.
        Simply reads the data from the IMU, stores it in an IMUData object, then saves that object as an attribute of the IMU class (self.data).
        """
        while self.active:
            with self.lock:
                data = self._get_single_data()
                self.data = data
            if self.callback:
                self.callback(data)
            time.sleep(0.01)

    def get_data(self):
        """
        Gets data from the IMU, either through manual reading and storage of the IMUData object
        (if not configured to thread), or just the attribute of the IMU class.
        """
        if not self.threaded:
            return self._get_single_data()
        with self.lock:
            return self.data
