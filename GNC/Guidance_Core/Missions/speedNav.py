"""Keep the target at x=0.25  part of the screen, when heading is 180 + original, we go back to initial position"""
from GNC.info_core import infoCore
from GNC.Guidance_Core.waypointNav import waypointNav
from GNC.Control_Core.motor_core import MotorCore
from GNC.Guidance_Core.mission_helper import MissionHelper
import time
import math

class SpeedChallenge(MissionHelper):
    def __init__(self):
        self.config     = MissionHelper().load_json(path="GNC/Guidance_Core/Config/barco_polo.json")
        self.info       = infoCore(modelPath=self.config["test_model_path"],labelMap=self.config["test_label_map"])
        self.motor      = MotorCore()
        self.wayPNav    = waypointNav()

        self.initPos    = []    #(lat,lon,heading)
        
    def start(self):
        self.info.start_collecting()
        time.sleep(1)
        gpsData = self.info.getGPSData()
        self.initPos.append((gpsData.lat,gpsData.lon,gpsData.heading))

        self.wayPNav.loadWaypoints(self.initPos)

    def surround(self):
        """go around the target"""
        while (abs(self.info.getGPSData().heading-self.initPos[2])<=170):
            _, detections = self.info.getInfo()
            for object in detections:
                if(object["label"] == "TARGETNAME"):
                    # find location
                    state = (object["bbox"][0] + object["bbox"][2])/2 > 0.25
                    # motor control
                    if state:
                        self.motor.slide(-0.5)
                    else:
                        self.motor.yaw(0.2,0.2,-0.4,-0.4)
                elif(object["label"] == "OBSTACLENAME"):
                    state = (object["bbox"][0] + object["bbox"][2])/2 > 0.5
                    if state:
                        # if on right side
                        self.motor
                        pass
        
            time.sleep(0.05)
        
            # end loop while loop

        # ready to go back
        self.wayPNav.run()

        print("reached initial position")

    def stop(self):
        self.info.stop_collecting()
        print("Background Threads stopped")


"""Below is the function to determine whether the light is on or not"""
import cv2
import numpy as np

def detect_clusters(image):
    """ Detects and counts red and green clusters in an image. """
    
    # Convert image to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define color ranges for red and green
    red_lower1 = np.array([0, 120, 70])     # Red (lower boundary)
    red_upper1 = np.array([10, 255, 255])  
    red_lower2 = np.array([170, 120, 70])   # Red (upper boundary)
    red_upper2 = np.array([180, 255, 255])

    green_lower = np.array([35, 100, 100])  # Green boundary
    green_upper = np.array([85, 255, 255])

    # Create masks
    red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
    red_mask = red_mask1 + red_mask2

    green_mask = cv2.inRange(hsv, green_lower, green_upper)

    # Count clusters using contours
    num_red = len(cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0])
    num_green = len(cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0])

    return num_red, num_green

def determine_light(num_red, num_green):
    """ Determines the light color based on detected clusters. """
    return "Red" if num_red > num_green else "Green"

# Example usage
image = cv2.imread("buoys.jpg")  # Load image of buoys
num_red, num_green = detect_clusters(image)
light_color = determine_light(num_red, num_green)

print(f"Red clusters: {num_red}, Green clusters: {num_green}")
print(f"Light color: {light_color}")
