"""
File that defines the class SensorFuse, which handles the creation of a Kalman filter, with inputs integrated between both the GPS and IMU.
NOTE: This Kalman filter does not work that well because it is linear, meaning that it does not work for 2nd derivative (acceleration) changes.
"""

import time
import numpy as np
from scipy.spatial.transform import Rotation as R
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from typing import Union, Tuple

from API.GPS.gps_api import GPS, GPSData
# from API.IMU.imu_api import IMU, IMUData

# TODO: Understand calculations better, and update documentation as understanding improves.
# Reference file: https://github.com/InspirationRobotics/RX24-OCB/blob/main/motor_control/sensor_fuse.py#L140 

class SensorFuse:
    def __init__(self, *, enable_filter = True, gps_port : str = "/dev/ttyUSB0", gps_baudrate : int = 115200, use_imu : bool = False, heading_offset : float = 0):

        if enable_filter:
            self.kf = self.create_filter()
            self.imu_dt = time.time() # Variable to compare previous calculation time to current calculation time -- see _update_IMUfilter()
        
        self.raw_data = GPSData(None, None, None)
        self.filter = enable_filter

        self.connected = False
        self.gps = GPS(gps_port, gps_baudrate, callback = self._gps_callback, offset=heading_offset)
        if use_imu and enable_filter:
            pass
            # self.imu = IMU(callback= self._imu_callback)

    def _gps_callback(self, data : GPSData):
        """
        Callback for GPS, usage for Kalman filter. If filter is enabled, checks to make sure GPS data is valid, then updates the kalman filter with the data.
        Otherwise just stores the data from the GPS as an attribute of the class.
        
        Args:
            data (GPSData): Data from the GPS -- contains lat, lon, heading info.
        """
        if data.is_valid():
            self.connected = True
            if self.filter:
                z = np.array([data.lat, data.lon, data.heading])
                self.kf.update(z)
            else:
                self.raw_data = data

    def _imu_callback(self, data):
        """
        Callback for the IMU, usage for Kalman filter. Once data is passed in, callback modifies the data and stores it in the appropriate 
        places in the Kalman filter.
        """
        self._update_IMUfilter(data)

    def get_position(self) -> Union[Tuple[float, float], None]:
        """
        Returns either the lat, lon data stored in the Kalman filter from the GPS, or the actual raw (lat, lon) data from the GPS.
        """
        if not self.connected:
            return None
        if self.filter:
            # Get lat, lon from Kalman filter
            return tuple(self.kf.x[:2])
        else:
            return (self.raw_data.lat, self.raw_data.lon)

    def get_velocity(self) -> Union[Tuple[float, float], None]:
        """
        Returns the latitude change in meters per second and longitude change in meters per second
        based on lat and lon velocity from the Kalman filter.
        """
        if not self.connected or not self.filter:
            return None
        global_velocity = tuple(self.kf.x[2:4])
        lat_vel_mps = global_velocity[0] * 111320
        lon_vel_mps = global_velocity[1] * 111320 * np.cos(np.radians(self.kf.x[0]))
        return lat_vel_mps, lon_vel_mps

    def get_relative_velocity(self) -> Union[Tuple[float, float], None]:
        """
        Get the relative velocity from the GPS.
        NOTE: This function is not complete (or even started).
        """
        if not self.connected or not self.filter:
            return None
        # global_velocity = self.get_velocity()
        # return global_velocity

    def get_heading(self) -> Union[float, None]:
        """
        Returns the heading of the ASV from the Kalman filter.
        """
        if not self.connected:
            return None
        if self.filter:
            return float(self.kf.x[4])
        else:
            return self.raw_data.heading
        

    # ----------------- Kalman Filter -----------------
    # See documentation: https://filterpy.readthedocs.io/en/latest/kalman/KalmanFilter.html

    def create_filter(self) -> KalmanFilter:
        """
        Create Kalman filter with five states (lat, lon, vel_lat, vel_lon, heading),
        and three measurements (lat, lon, heading).

        Returns:
            KalmanFilter: Fully configured Kalman filter. Initalizes with high initial uncertainty.
        """

        # Five state variables, three inputs
        kf = KalmanFilter(dim_x=5, dim_z=3)

        kf.x = np.array([0., 0., 0., 0., 0.])

        # State transition matrix
        kf.F = np.array([   [1., 0., 1., 0., 0.], # lat
                            [0., 1., 0., 1., 0.], # lon
                            [0., 0., 1., 0., 0.], # vel_lat
                            [0., 0., 0., 1., 0.], # vel_lon
                            [0., 0., 0., 0., 1.]]) # heading
       
        # Measurement matrix
        kf.H = np.array([   [1., 0., 0., 0., 0.], # lat
                            [0., 1., 0., 0., 0.], # lon
                            [0., 0., 0., 0., 1.]])  # Heading from GPS
        
        # Covariance matrix (P) is the initial uncertainty at the state -- set it to high
        kf.P = np.eye(5) * 1000

        # Process noise matrix
        kf.Q = np.eye(5)

        # Measurement noise matrix
        kf.R = np.array([   [0.1, 0., 0.], # lat
                            [0., 0.1, 0.], # lon
                            [0., 0., 0.1]]) # heading
        
        return kf
    
    def _update_IMUfilter(self, data):
        """
        Updates the Kalman filter with the IMU data. Specifically calculates the latitude velocity, longitude velocity, 
        latitude/longitude velocity change in degrees per second, and the yaw of the ASV, all from the IMU data or data already stored in the Kalman filter.
        All of these calculated variables then go in the Kalman filter in the appropriate spot. Finally, runs the predict method on the Kalman filter.

        Args:
            data (IMUData): Data from the IMU.
        """
        # Run only after there is GPS data.
        if not self.connected:
            return
        
        accel = np.array(data.accel)
        quat = np.array(data.quat)

        # TODO: Look up what these calculations are supposed to do.
        norm_quat = np.linalg.norm(quat)
        if norm_quat == 0:
            return
        
        quat /= norm_quat

        # Rotate accelerometer data into global coordinates.
        r_quat = R.from_quat(quat)
        euler_angles = r_quat.as_euler('xyz', degrees=True) # Get Roll (X), Pitch (Y), Yaw (Z)
        roll = euler_angles[0]
        pitch = euler_angles[1]
        yaw = self.get_heading()
        r_matrix = R.from_euler('xyz', [roll, pitch, yaw], degrees=True).as_dcm()
        global_accel = np.dot(r_matrix, accel)

        # Calculating delta time for IMU.
        dt = time.time() - self.imu_dt
        self.imu_dt = time.time()

        # To velocity from acceleration.
        accel_vel_change = global_accel[:2] * dt

        # Convert units from lat, lon
        current_lat = self.kf.x[0]
        lat_vel_deg = accel_vel_change[0] / 111320 # Convert to degrees latitude per second
        lon_vel_deg = accel_vel_change[1] / (111320 * np.cos(np.radians(current_lat))) # Convert to degrees longitude per second

        # Now the kalman filter can be updated.
        self.kf.F[0, 2] = dt # lat_vel
        self.kf.F[1, 3] = dt # lon_vel
        self.kf.x[2] += lat_vel_deg
        self.kf.x[3] += lon_vel_deg
        self.kf.x[4] = yaw
        self.kf.predict()
