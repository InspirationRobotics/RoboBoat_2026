from GNC.Control_Core  import motor_core
from GNC.info_core import infoCore
from GNC.Guidance_Core.mission_helper import MissionHelper
from GNC.Guidance_Core import waypointNav
from GNC.Guidance_Core.Missions import navChannel
import API.Util.gis_funcs as gpsfunc
import threading
import math
import time
from API.Servos.mini_maestro import MiniMaestro
from GNC.Guidance_Core.Missions.FTP_Cv import cvCore

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
nav_lat, nav_lon = gpsfunc.destination_point(lat, lon, 313, 25)
tolerance = 1.5 # Meters

# def start_waypoint(point, tolerance : float = 1.0):
#     nav_thread = threading.Thread(target=NNAV.run, args=(point, tolerance), daemon=True) # arguemnets: waypoint(dict), tolerance(float)->in meters
#     nav_thread.start()
#     nav_thread.join()
#     print("[Mission] Waypoint reached.")

waypoints  = NNAV._readLatLon(file_path = config["waypoint_file"])
waypoints.insert(0,{"lat" : nav_lat, "lon" : nav_lon})

# Follow the path thread start
FTP = cvCore()
FTP_Thread = threading.Thread(target=FTP.control_loop,args=(motor,False),daemon=True)
FTP_Thread.start()
time.sleep(120) # this is our time out
FTP_Thread.join()
try:
    for index, p in enumerate(waypoints):
        if(index==int):
            pass
            Servo.set_pwm(1,1500)
            Servo.set_pwm(1,1800)
            time.sleep(5)
            Servo.set_pwm(1,1500)
            
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

# # shooting water
# maestro = MiniMaestro(port="/dev/ttyACM0")

# print("water gun")

# maestro.set_pwm(1, 1500)  # Move servo on channel 1
# time.sleep(1)
# maestro.set_pwm(1, 1800)  # Move servo on channel 1   
# time.sleep(10)
# maestro.set_pwm(1, 1500)  # Move servo on channel 1
# time.sleep(1)

print("program finished")