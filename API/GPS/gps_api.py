"""
File to handle all low-level GPS functionality.
Includes the class definitions of GPSData -- object to store GPS data, and GPS -- class to handle all low-level parsing of NMEA GPS data.
"""

import os
import time
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
        return self.lat and self.lon and self.heading

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
    
class GPS:
    """
    Class to handle all GPS functionalitiy.

    Args:
        serialport (str): Port between GPS and Jetson
        baudrate (int): Message rate on serial connection (defaults to 115200)
        callback (func): Function to run on the GPS data when data is collected
        threaded (bool): Whether or not to create a GPS thread (defaults to True)
        offset (float): Keyword argument, whether or not to take into account any sort of offset for the GPS
    """

    def __init__(self, serialport : str = "/dev/ttyUSB0", baudrate : int = 115200, callback = None, threaded : bool = True, *, offset : float = None):
        stream = Serial(serialport, baudrate, timeout=3)
        self.nmr = NMEAReader(stream)
        self.threaded = threaded
        self.callback = callback
        self.offset = offset if offset is not None else self.load_heading_offset()

        self.data : GPSData = GPSData(None, None, None)
        self.lock = Lock()

        self.active = True
        self.gps_thread = Thread(target=self.__gps_thread, daemon=True)
        if threaded:
            self.gps_thread.start()

    def __del__(self):
        """
        Function to delete the thread for the GPS.
        """
        
        if self.threaded:
            self.active = False
            self.gps_thread.join(2)

    def __update_data(self, parsed_data):
        """
        Updates GPSData object by putting latest data from NMEA reader into each respective attribute of GPSData.

        Args:
            parsed_data (class): GPS data that was parsed, with data being organized into specific attributes (.lat, .lon, etc).
        """
        try:
            if parsed_data.msgID == 'GGA':
                    self.data.lat = parsed_data.lat
                    self.data.lon = parsed_data.lon
            elif parsed_data.msgID == 'THS':
                    self.data.heading = (parsed_data.headt + self.offset) % 360
        except Exception as e:
            # print("Error grabbing data")
            # print(e)
            pass

    """
    Example GPS data before parse (i.e. what the GPS spits out in terminal when plugged into a host computer):

    $GNGGA,034927.50,3255.43416509,N,11702.31616403,W,1,14,0.8,219.4295,M,-33.3221,M,,*44
    $GNTHS,331.8678,A*19
    $GNGGA,034928.00,3255.43422338,N,11702.31617660,W,1,15,0.8,219.3912,M,-33.3221,M,,*49
    $GNTHS,331.8955,A*19
    $GNGGA,034928.50,3255.43425758,N,11702.31621373,W,1,14,0.9,219.3470,M,-33.3221,M,,*42
    $GNTHS,331.4784,A*17
    $GNGGA,034929.00,3255.43427297,N,11702.31624825,W,1,15,0.8,219.2937,M,-33.3221,M,,*40
    $GNTHS,331.5406,A*1F
    """

    def __gps_thread(self):
        """
        Function that continually runs while the GPS thread remains in existence.
        Parses the data via NMEA reader, and updates the GPSData object with the latest data.
        The function then does executes the callback function for the class, passing in the most updated GPSData object. 
        The callback function passed into the class is unique to each individual file.
        """
        parsed_data : NMEAMessage
        while self.active:
            raw_data, parsed_data = self.nmr.read() # Blocking
            with self.lock:
                self.__update_data(parsed_data)
            if self.callback and self.data.is_valid():
                self.callback(self.data)

    def __get_single_data(self) -> GPSData:
        """
        Get a single line of data from the GPS, by reading the data via NMEA reade, and adding it to the GPSData object.

        Returns:
            self.data (GPSData): GPSData object (contains : [self.lat, self.lon, self.heading]) with most updated data.
        """
        parsed_data : NMEAMessage
        raw_data, parsed_data = self.nmr.read() # Blocking
        self.__update_data(parsed_data)
        return self.data

    def get_data(self) -> GPSData:
        """
        Obtains data from the GPSData class, either through single lines (not threaded), or just returning the object (threaded).

        Returns:
            GPSData: Most updated data from the GPS, organized via NMEA reader.
        """
        if self.threaded:
            with self.lock:
                return self.data
        else:
            return self.__get_single_data()
    
    def load_heading_offset(self):
        """
        Load the calculated GPS heading offset from the specified place (API/GPS/Config/gps_offset.txt).

        Returns:
            offset (float): The GPS heading offset.
        """
        curr_path = Path("API/GPS")
        config_path = curr_path / "Config"
        if not config_path.exists():
            os.mkdir(config_path)
        if not (config_path / "gps_offset.txt").exists():
            return 0 # Default offset
        with open(config_path / "gps_offset.txt", "r") as f:
            offset = float(f.read())
        return offset

    def calibrate_heading_offset(self, calib_time : int = 5):
        """
        Averages out the heading data and subtracts it from the most recent heading to generate GPS heading offset.
        Puts the calcuated offset in the specified place (API/GPS/Config/gps_offset.txt).

        Args:
            calib_time (int): Time to receive data from GPS before calculating offset. Defaults to 5 seconds.
        """
        heading_data = []
        input("Press Enter to start calibration")
        print("Calibrating GPS heading offset...")
        start_time = time.time()
        while time.time() - start_time < calib_time:
            data = self.get_data()
            if data.is_valid():
                heading_data.append(data.heading)
            time.sleep(0.3)
        avg_heading = sum(heading_data) / len(heading_data)
        print(f"Average heading: {avg_heading}")
        current_heading = input("Enter current heading: ")
        offset = float(current_heading) - avg_heading
        print(f"Offset: {offset}")
        data = input("Save offset? (y/n): ")
        if data.lower() == "y":
            curr_path = Path("API/GPS")
            config_path = curr_path / "Config"
            if not config_path.exists():
                os.mkdir(config_path)
            with open(config_path / "gps_offset.txt", "w") as f:
                f.write(str(offset))
                print("Offset saved.")
        print("Calibration complete.")
        
    def save_waypoints(self):
        """
        Method to save current waypoints in a log (.txt) file. Will open a file in the Test_Scripts/GPS_Tests directory.
        The data written to the log file will be in the form: 
            "{rounded_time} % {data.lat} % {data.lon} % {data.heading}\n"

        Pressing Ctrl + C or "q" will break the loop and close the log file.
        """

        if not self.gps_thread.is_alive():
            print("GPS thread is required to save waypoints")
            return
        if self.callback is not None:
            print("GPS cannot have a callback set to save waypoints")
            return
        name = input("Enter the name for the file: ")
        auto = True if input("Auto log waypoints at 2hz? (y/n)").lower() == "y" else False
        input("Press any key to begin logging waypoints")
        curr_path = Path("Test_Scripts/API_Tests/GPS_Tests")
        missions_path = curr_path / "Missions"
        if not missions_path.exists():
            os.mkdir(missions_path)
        log = open(f'{missions_path}/{name}.txt', "w")

        # Creating callback function to simply write the data to the log file, since there should be no callback function passed in.
        def auto_callback(data : GPSData):
            log.write(f"{data.lat} % {data.lon} % {data.heading}\n")

        with self.lock:
            self.callback = auto_callback if auto else None
        cnt = 1
        while True:
            try:
                if not auto:
                    key = input("Press any key to save a waypoint (or q to quit)")
                    if key.lower() == "q":
                        break
                    data = self.get_data()
                    rounded_time = int(time.time())
                    log.write(f"{rounded_time} % {data.lat} % {data.lon} % {data.heading}\n")
                    print(f"Waypoint {cnt} saved")
                    cnt+=1
                else:
                    time.sleep(1)
            except KeyboardInterrupt:
                break
        print(f"Waypoints saved to {name}.txt")
        log.close()

    @staticmethod
    def load_waypoints(filename : str) -> List[Tuple[float, float]]: 
        """
        Returns a list of waypoints in the form of a list of tuples (lat, lon). This is after a log file has already been created.

        Args:
            filename (str): File name of the .txt file with the coordinates to load.

        Returns:
            waypoints (list): List of tupled (lat, lon) waypoints. Ignores time and heading.
        """

        if filename is None:
            return None
        curr_path = Path("Test_Scripts/API_Tests/GPS_Tests")
        missions_path = curr_path / "Missions"
        if not missions_path.exists():
            os.mkdir(missions_path)
        file_path = missions_path / filename
        if not file_path.exists():
            print(f"File {filename} does not exist")
            return None
        waypoints = []
        with open(file_path, "r") as f:
            for line in f:
                try:
                    # NOTE: This code has to be retested in order to ensure that the new time command still works. However, it should be fine for now.
                    time, lat, lon, heading = line.split(" % ")
                    waypoints.append((float(lat), float(lon))) #ignore heading for now
                except:
                    pass
        return waypoints
    
if __name__ == "__main__":
    gps = GPS("/dev/ttyUSB0", 115200, None, True)
    print("Waiting...")
    time.sleep(3)
    gps.calibrate_heading_offset(calib_time=5)