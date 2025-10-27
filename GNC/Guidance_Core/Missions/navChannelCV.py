from GNC.info_core import infoCore
from GNC.Guidance_Core.mission_helper import MissionHelper
from GNC.Control_Core import motor_core
from API.Util import gis_funcs 
import cv2
import numpy as np
import time

threshold = 100

def detect_buoy_red(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 120, 70]), np.array([10, 255, 255])) + \
           cv2.inRange(hsv, np.array([170, 120, 70]), np.array([180, 255, 255]))
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    h, w, _ = frame.shape
    center_x = w // 2
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > threshold:
            lowest_point = max(largest_contour, key=lambda point: point[0][1])  # Find point with max y-coordinate
            x, y = tuple(lowest_point[0])  # (x, y)
            
            # Draw contours and the lowest point
            cv2.drawContours(frame, [largest_contour], -1, (0, 0, 255), 2)  # Red contours
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)  # Red dot at lowest point
            
            return (x, y), frame

    return None, frame


def detect_buoy_green(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([40, 40, 40]), np.array([80, 255, 255]))
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    h, w, _ = frame.shape
    center_x = w // 2
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > threshold:
            lowest_point = max(largest_contour, key=lambda point: point[0][1])  # Find point with max y-coordinate
            x, y = tuple(lowest_point[0])  # (x, y)
            
            # Draw contours and the lowest point
            cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 2)  # Green contours
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  # Green dot at lowest point
            
            return (x, y), frame

    return None, frame


def navigate_boat(frame):
    """Processes the frame to detect buoys and determine navigation instructions."""
    red_buoy, frame = detect_buoy_red(frame)
    green_buoy, frame = detect_buoy_green(frame)
    
    print("Red Buoy:", red_buoy)
    print("Green Buoy:", green_buoy)

    h, w, _ = frame.shape
    center_x = w // 2
    
    if red_buoy and green_buoy:
        gate_center_x = (red_buoy[0] + green_buoy[0]) // 2
        
        if gate_center_x < center_x - 40:
            command = "Turn Left"
        elif gate_center_x > center_x + 40:
            command = "Turn Right"
        else:
            command = "Move Forward"
    elif green_buoy:
        command = "Turn Left"
    elif red_buoy:
        command = "Turn Right"
    else:
        command = "Searching for buoys"
    
    # Show the frame with contours
    cv2.imshow("Buoy Detection", frame)
    
    return command, frame


"""***"""
config     = MissionHelper()
print("loading configs")
config     = config.load_json(path="GNC/Guidance_Core/Config/barco_polo.json")
info       = infoCore(modelPath=config["sign_model_path"],labelMap=config["sign_label_map"])
print("start background threads")
info.start_collecting()
motor      = motor_core.MotorCore("/dev/ttyACM2",debug=False) # load with default port "/dev/ttyACM2"
time.sleep(2)
print("rest 2 seconds")
GPS, _ = info.getInfo()
calc_lat, calc_lon = gis_funcs.destination_point(GPS.lat, GPS.lon, GPS.heading, 15)

try:

    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        gps, detections = info.getInfo()
        
        command, processed_frame = navigate_boat(info.getFrame())
        if command=="Turn Left":
            print("turn left")
            motor.veer(0.6,-0.15)
        elif command == "Turn Right":
            print("turn right")
            motor.veer(0.6, 0.15)
        elif command == "Move Forward":
            print("Forward")
            motor.surge(0.8)
        elif command == "Searching for buoys":
            print("Lost")
            motor.rotate(0.1)
        time.sleep(0.1)

        # TODO add stop statement
        if (gis_funcs.haversine(gps.lat,gps.lon,calc_lat,calc_lon)<=3): # 3 meter tolernace
            print("mission finished")
            info.stop_collecting()
            motor.stay()
            motor.stop()
            break

except KeyboardInterrupt:
    print("mission finished")
    info.stop_collecting()
    motor.stay()
    motor.stop()

info.stop_collecting()
motor.stay()
motor.stop()
    


