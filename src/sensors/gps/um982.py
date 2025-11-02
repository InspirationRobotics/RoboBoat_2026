"""
File to handle all low-level GPS functionality.
Includes the class definitions of GPSData -- object to store GPS data, and GPS -- class to handle all low-level parsing of NMEA GPS data.
"""

import os
import time
import threading
from typing import List, Tuple
from pathlib import Path
from serial import Serial
from threading import Thread, Lock


# Need to import pynmeagps locally
from pynmeagps import NMEAReader, NMEAMessage

"""
GPS Specifications (Beitan GPS module):

https://www.qso.com.ar/datasheets/Receptores%20GNSS-GPS/NMEA_Format_v0.1.pdf
Specs : GNGGA
"""


class GPSData:
    """
    Class to handle the parsing of GPS data in a much more convenient way.

    Args:
        self.lat (float) : GPS-based latitude
        self.lon (float) : GPS-based longitude
        self.headt (float) : GPS-based absolute heading
    """

    def __init__(self, lat : float, lon : float, headt : float):
        self.lat = lat
        self.lon = lon
        self.heading = headt
        self.timestamp = time.time()

    def is_valid(self):
        """
        Checks to make sure data is valid by making sure there is a lat, lon, heading in the data
        """
        return self.lat and self.lon # and self.heading

    def __setattr__(self, name: str, value) -> None:
        """
        Puts GPS data into a dictionary and attaches the time stamp of when the data was listed
        """
        self.__dict__[name] = value
        self.__dict__["timestamp"] = time.time()

    def __str__(self) -> str:
        """
        Returns f-string of latitude, longitude, heading
        """
        return f"Lat: {self.lat}, Lon: {self.lon}, Heading: {self.heading}"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def __iter__(self):
        return iter((self.lat, self.lon, self.heading))
    
class UM982Serial(threading.Thread):
    """
    Class to handle all GPS functionalitiy.

    Args:
        serialport (str): Port between GPS and Jetson
        baudrate (int): Message rate on serial connection (defaults to 115200)
        callback (func): Function to run on the GPS data when data is collected
        threaded (bool): Whether or not to create a GPS thread (defaults to True)
        offset (float): Keyword argument, whether or not to take into account any sort of offset for the GPS
    """

    def __init__(self, serialport : str = "/dev/ttyUSB0", baudrate : int = 115200, offset : float = None):
        super().__init__()
        self.ser = Serial(serialport, baudrate, timeout=3)
        self.nmr = NMEAReader(self.ser)
        self.isRUN          = True
        self.offset = 125.5-270
        self.data : GPSData = GPSData(None, None, None)

    def stop(self):
        self.isRUN = False
        time.sleep(0.1)
        self.ser.close()
        
    def read_frame(self):
        parsed_data : NMEAMessage
        raw_data, parsed_data = self.nmr.read() 
        try:
            if parsed_data.msgID == 'GGA':
                    self.data.lat = parsed_data.lat
                    self.data.lon = parsed_data.lon
            elif parsed_data.msgID == 'THS':
                    self.data.heading = (parsed_data.headt + self.offset) % 360
        except Exception as e:
            pass
        print(self.data)
        return self.data

    def run(self):
        while self.isRUN:
            self.read_frame()
    
    

if __name__ == "__main__":
    um982 = UM982Serial("/dev/ttyUSB0", 115200, offset=225)
    um982.start()