"""
This is a manager class
This script meant to collect information from gps and perception, and start background threads
We will also do some simple calculation here
Normalized relative bearing angle
object location base on perception and heading.
    convert relative angle to abs -> find lat lon
"""

from API.GPS.gps_api import GPS, GPSData
from Perception.Perception_Core.perception_core import CameraCore
from threading import Thread, Lock
from queue import Queue
import math
class infoCore:
    def __init__(self,modelPath:str ,labelMap:list):
        # Stop event to control the manager core and background threads
        self.manager_stop_event = None
        
        # Initialize GPS and Camera
        self.Camera = CameraCore(model_path=modelPath,labelMap=labelMap)

    def start_collecting(self):
        # A Thread is started when you initialize the GPS object
        self.GPS = GPS(serialport = "/dev/ttyUSB0", baudrate= 115200, callback = None, threaded= True, offset = -146) 
        self.Camera.start()   # Start Perception Thread
        pass

    def stop_collecting(self):
        self.GPS.__del__()
        self.Camera.stop()

    def getInfo(self): # return object information and GPS data
        detections = self.Camera.get_object_depth()  
        gpsData = self.GPS.get_data()
        boat_heading    = gpsData.heading
        boat_lat        = gpsData.lat
        boat_lon        = gpsData.lon
        for object in detections:
            R = 6371000  # Earth radius in meters
            heading = math.radians((boat_heading + object["angle"]) % 360)     # abs heading [0,2pi]
            distance = object["depth"]
            # Compute change in latitude and longitude
            delta_lat = (distance / R) * math.cos(heading)
            delta_lon = (distance / R) * math.sin(heading) / math.cos(math.radians(boat_lat)) # compute in radians

            # Convert back to degrees
            lat_obj = boat_lat + math.degrees(delta_lat)
            lon_obj = boat_lon + math.degrees(delta_lon)

            # add location to dictionary
            object["location"] = {"lat":lat_obj,"lon":lon_obj}

        return gpsData, detections

    def getFrame(self):
        return self.Camera.visualize()  

    def getFrameRaw(self):
        return self.Camera.getFrameRaw()    

    def getGPSData(self) ->GPSData:
        """DEPRECATED it's now integrated in getInfo"""
        return self.GPS.get_data()
    
    def switchModel(self,modelPath:str,labelMap:str):
        self.Camera.switchModel(modelPath=modelPath,labelMap=labelMap)
