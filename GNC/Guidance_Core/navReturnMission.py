from GNC.Control_Core  import motor_core
from GNC.info_core import infoCore
import API.Util.gis_funcs as gpsfunc
from GNC.Guidance_Core.waypointNav import waypointNav
from GNC.Guidance_Core.mission_helper import MissionHelper
from GNC.Guidance_Core.Missions import navChannel, FTP
import threading, queue
import math
import time

config     = MissionHelper()
data     = config.load_json(path="GNC/Guidance_Core/Config/barco_polo.json")
config.parse_config_data(data)
print("[MISSION] Loaded configuration.")

info       = infoCore(config.model_path, config.model_label_map)
print("[MISSION] Starting background threads.")
info.start_collecting()
motors     = motor_core.MotorCore(config.motor_port) 

# NAV Block
mission = navChannel.navChannel(infoCore=info, motors=motors)
lat, lon = mission.run()
nav_lat, nav_lon = gpsfunc.destination_point(lat, lon, 310, 20)
return_lat, return_lon = gpsfunc.destination_point(lat, lon, 40, 20)

# Initial point
first_point = {"lat" : lat, "lon" : lon}
# Point on other end of NAV
nav_point = {"lat" : nav_lat, "lon" : nav_lon}
# Point near the black buoy gates
return_point = {"lat" : return_lat, "lon" : return_lon}

NNAV = waypointNav(infoCore=info, motors=motors)
tolerance = 1.5

def start_waypoint(point, tolerance : float = 1.0):
    nav_thread = threading.Thread(target=NNAV.run, args=(point, tolerance), daemon=True) # arguemnets: waypoint(dict), tolerance(float)->in meters
    nav_thread.start()
    nav_thread.join()
    print("[Mission] Waypoint reached.")

start_waypoint(nav_point)
start_waypoint(return_point)
start_waypoint(first_point)

NNAV.stop()

# # # FTP Block
# # folow_the_path_queue = queue.Queue()
# # follow_the_path = FTP.FTP(infoCore=info, motors=motors)
# # FTP_thread = threading.Thread(target=follow_the_path.run, args=(folow_the_path_queue,), daemon=True)
# # FTP_thread.start()
# # FTP_thread.join()
# # FTP_point = folow_the_path_queue.get()
# # FTP_point = {"lat" : FTP_point[0], "lon": FTP_point[1]}

# start_waypoint(FTP_point)

