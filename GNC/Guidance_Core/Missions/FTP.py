from API.Util import gis_funcs
from GNC.Control_Core  import motor_core
from GNC.info_core import infoCore
from GNC.Guidance_Core.mission_helper import MissionHelper
import API.Util.gis_funcs as gpsfunc
import threading
import math
import time
from API.Servos.mini_maestro import MiniMaestro
import cv2
import numpy as np
class FTP:
    def __init__(self, *, infoCore, motors):
        self.info = infoCore
        self.motors = motors

        # Threshold to look for objects.
        self.threshold = 0.7

        self.cur_ang = None
        self.cur_dis = None

    def updateDelta(self,lat,lon):
        gpsdata = self.info.getGPSData()
        print(f"waypoints| lat: {lat} | lon: {lon}")
        self.cur_ang =  gis_funcs.relative_bearing(lat1=gpsdata.lat,lon1=gpsdata.lon,lat2=lat,lon2=lon,current_heading=gpsdata.heading)
        self.cur_dis =  gis_funcs.haversine(lat1=gpsdata.lat,lon1=gpsdata.lon,lat2=lat,lon2=lon) # this return angle to the range (-180,180)
        print(f"abs head: {gpsdata.heading} | lat: {gpsdata.lat} | lon: {gpsdata.lon}")
        print(f"delta ang: {self.cur_ang} | delta dis: {self.cur_dis}")
        # normalize angle to value between 0 and 1
        self.cur_ang /= 180
        return self.cur_ang, self.cur_dis
    
    def run(self,endpoint,tolerance=1.5):
        # Find the two closest objects that are lowest on the screen.
        # Find the midpoint of both objects(pixel values).
        # If the difference between one midpoint is greater than the other, yaw to force the values to be within equal within a certain 
        # tolerance (while always moving forward)
        # Always move forward while yawing
        gpsData, detections = self.info.getInfo()
        
        self.cur_ang,self.cur_dis = self.updateDelta(gpsData.lat,gpsData.lon)
        while self.cur_dis<tolerance:
            gpsData, detections = self.info.getInfo()
            # Find the lowest two detections.
            # Going down the screen will actually increase the value, so lowest will be 1.
            # Going across the screen increases the value (xmin -> xmax).

            # Find the lowest of each type.
            # Types are "green", "red", "yellow", "cross", "triangle"
            # "bbox": (detection.xmin, detection.ymin, detection.xmax, detection.ymax)

            red_object_list = []
            green_object_list = []

            red_detected = False
            green_detected = False
            threshold = 0.9
            if len(detections)>=1 and detections is not None:
                # collect objects
                for detection in detections:
                    if detection["label"] == "green_buoy" or detection["label"] == "green_pole_buoy":
                        green_object_list.append(detection)

                    if detection["label"] == "red_buoy" or detection["label"] == "red_pole_buoy":
                        red_object_list.append(detection)

                # find min
                red_min = -1
                red_min_detection = None
                green_min = -1
                green_min_detection = None

                for object in red_object_list:
                    if(object["bbox"][3]>red_min   and object["bbox"][3]<threshold):
                        red_min_detection = object
                        red_detected = True

                for object in green_object_list:
                    if(object["bbox"][3]>green_min and object["bbox"][3]<threshold):
                        green_min_detection = object
                        green_detected = True

                # find midpoint
                red_center      = (red_min_detection["bbox"][0] + red_min_detection["bbox"][2])/2 if red_detected else 0
                green_center    = (green_min_detection["bbox"][0] + green_min_detection["bbox"][2])/2 if green_detected else 1
                path_center     = (red_center + green_center)/2
                print(f"Red : {red_center}, Green : {green_center}")
            else:
                path_center = 0.5

            # control the motor
            midpoint = 0.5
            screen_tolerance = 0.15
            delta_center = path_center - midpoint
            if(delta_center > screen_tolerance):
                """turn left"""
                print("\nturn left")
                self.motors.veer(0.8,-0.5)
            elif(delta_center < -screen_tolerance):
                """turn right"""
                print("\nturn right")
                self.motors.veer(0.8, 0.5)
            else:
                print("\nsurge")
                self.motors.surge(1)

            # update del dis
            self.cur_ang,self.cur_dis = self.updateDelta(gpsData.lat,gpsData.lon)

            time.sleep(0.05)

        # NOTE: Need to write an actual executable file.
    def stop(self):
        self.info.stop_collecting()
        print("Background Threads stopped")
        self.motors.stop()
        print("Motors stopped")
if __name__ == "__main__":
    config     = MissionHelper()
    print("loading configs")
    config     = config.load_json(path="GNC/Guidance_Core/Config/barco_polo.json")
    info       = infoCore(modelPath=config["competition_model_path"],labelMap=config["competition_label_map"])
    print("start background threads")
    info.start_collecting()
    motor      = motor_core.MotorCore("/dev/ttyACM2") # load with default port "/dev/ttyACM2"
    mission    = FTP(infoCore=info, motors=motor)

    try:
        mission.run(endpoint=config["FTP_endpoint"],tolerance=1.5)
        mission.stop()
    except KeyboardInterrupt:
        mission.stop()


            



