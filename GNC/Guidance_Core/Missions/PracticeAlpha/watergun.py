from GNC.Control_Core  import motor_core
from GNC.info_core import infoCore
from GNC.Guidance_Core.mission_helper import MissionHelper
from GNC.Guidance_Core.waypointNav import waypointNav
from GNC.Guidance_Core.Missions import navChannel
import API.Util.gis_funcs as gpsfunc
import threading
import math
import time
from API.Servos.mini_maestro import MiniMaestro
from GNC.Guidance_Core.Missions.FTP_Cv import cvCore
# Log file
LOG_FILE = "cpu_usage_log.csv"


# Load config
config = MissionHelper().load_json(path="GNC/Guidance_Core/Config/barco_polo.json")

# Define paths to models
MODEL_1 = config["test_model_path"]
MODEL_2 = config["competition_model_path"]

# Label Map (Ensure it matches your detection classes)
LABELMAP_1 = config["test_label_map"]
LABELMAP_2 = config["competition_label_map"]

# Initialize info Core
infocore = infoCore(MODEL_2,LABELMAP_2)
infocore.start_collecting()  # Starts background threads
motor      = motor_core.MotorCore("/dev/ttyACM2") # load with default port "/dev/ttyACM2"
NNAV    = waypointNav(infoCore=infocore, motors=motor)
servo = MiniMaestro(port="/dev/ttyACM0")
try:
    start_time = time.time()
    shoot = False
    while((time.time()-start_time)<30 and not shoot):
        gps, detections = infocore.getInfo()

        for object in detections:
            if(object["label"] == "black_triangle"):
                print("detected")
                points = object["location"]
                print("running waypoint")
                print(points)
                NNAV.run(points=points,tolerance=0.5)
                print("shooting water")
                servo.set_pwm(1,1800)
                time.sleep(20)
                servo.set_pwm(1,1500)
                shoot = True
                break
except KeyboardInterrupt:
    print("program finished")
    servo.set_pwm(1,1500)
    NNAV.stop()
    infocore.stop_collecting()
    motor.stop()
            




