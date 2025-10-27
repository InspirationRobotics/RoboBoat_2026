from GNC.Control_Core  import motor_core
from GNC.info_core import infoCore
from GNC.Guidance_Core.mission_helper import MissionHelper
from GNC.Guidance_Core import waypointNav
from GNC.Guidance_Core.Missions import navChannel
import API.Util.gis_funcs as gpsfunc
from API.Util import gps_waypoint_sequence_parse as waypoint_parse
import threading
import math
import time
from API.Servos.mini_maestro import MiniMaestro

config     = MissionHelper()
print("loading configs")
config     = config.load_json(path="GNC/Guidance_Core/Config/barco_polo.json")
info       = infoCore(modelPath=config["competition_model_path"],labelMap=config["competition_label_map"])
print("start background threads")
info.start_collecting()
motor      = motor_core.MotorCore("/dev/ttyACM2") # load with default port "/dev/ttyACM2"
NNAV    = waypointNav.waypointNav(infoCore=info, motors=motor)
Servo = MiniMaestro(port="/dev/ttyACM0")
# load waypoints
nav = navChannel.navChannel(infoCore=info, motors=motor)
lat, lon = nav.run()
nav_lat, nav_lon = gpsfunc.destination_point(lat, lon, 313, 20)
tolerance = 1.5 # Meters

waypoints  = waypoint_parse.process_waypoints(waypoint_path=config["waypoint_json"], sequence_path=config["waypoint_sequence"])
# waypoints.insert(0,{"lat" : nav_lat, "lon" : nav_lon})

try:
    for index, p in enumerate(waypoints):
        # if (index == 5):
        #     motor.stay()
        #     time.sleep(2)
        #     Servo.set_pwm(1,1500)
        #     Servo.set_pwm(1,1800)
        #     time.sleep(10)
        #     Servo.set_pwm(1,1500)
            
        if(index==8):
            motor.stay()
            time.sleep(1)
            # Verified three yellow buoys on course
            motor.rotate(0.5)
            time.sleep(20)
            motor.stay()
            time.sleep(2)
            
        nav_thread = threading.Thread(target=NNAV.run, args=(p, 1.5), daemon=True)
        nav_thread.start()
        nav_thread.join()  # ✅ WAIT for thread to finish before stopping motors

    NNAV.stop()  # ✅ Stop everything AFTER all waypoints are reached
except KeyboardInterrupt:
    print("\n[!] KeyboardInterrupt detected! Stopping mission...")
    NNAV.stopThread()  # ✅ Use stop event to signal stop
    nav_thread.join()
    NNAV.stop()  # Stop motors and background threads
    print("[✔] Mission stopped cleanly.")

print("program finished")