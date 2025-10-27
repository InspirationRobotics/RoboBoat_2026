"""For rescue delivery"""
from GNC.info_core import infoCore
from GNC.Guidance_Core.waypointNav import waypointNav
from GNC.Control_Core.motor_core import MotorCore
from GNC.Guidance_Core.mission_helper import MissionHelper
from API.Servos import mini_maestro
import time
import math

class Rescue(MissionHelper):
    def __init__(self):
        # Init config
        self.config = MissionHelper()
        print("loading configs")
        self.config = self.config.load_json(path="GNC/Guidance_Core/Config/barco_polo.json")

        self.info       = infoCore(modelPath=self.config["sign_model_path"],labelMap=self.config["sign_label_map"])
        self.motor      = MotorCore("/dev/ttyACM2") # load with default port "/dev/ttyACM2"
        self.wayPNav    = waypointNav(infoCore=self.info, motors=self.motor)
        self.servo      = mini_maestro.MiniMaestro(self.config["mini_maestro_port"])

        self.cross      = False
        self.triangle   = False

        self.crossObject    = []
        self.triangleObject = []

        self.objectDetected = False
        self.duration = 20
        self.startTime = time.time()

        # # Maybe? declare mini maestro channel for water gun and racketball launcher also the PWM value
        self.racquetball_launcher_channel   = 0
        self.water_cannon_channel           = 0
        self.launchPWM                      = 1600
        self.nominalPWM                     = 1800

    def start(self):
        print("Background thread started")
        self.info.start_collecting()

    def run(self):
        on = True
        while(on):
            gps, detections = self.info.getInfo()
            self.objectDetected = False
            for object in detections:
                print(object)
                label = object["label"]
                self.objectDetected = True
                if(label=="Cross" or label=="Black boat"):
                    self.wayPNav.loadWaypoints([object["location"]])
                    self.wayPNav.run(tolerance=2.5)
                    
                    # code for water gun and racketball
                    self.servo.set_pwm(self.racquetball_launcher_channel, self.launchPWM)
                    time.sleep(2)
                    self.servo.set_pwm(self.racquetball_launcher_channel, self.nominalPWM)
                    # self.cross = True

                elif(label=="Triangle"or label== "Orange boat"):
                    self.wayPNav.loadWaypoints([object["location"]])
                    self.wayPNav.run(tolerance=2.5)

                    # code for water gun and racketball
                    self.servo.set_pwm(self.water_cannon_channel, self.launchPWM)
                    time.sleep(2)
                    self.servo.set_pwm(self.water_cannon_channel, self.nominalPWM)

                    # self.triangle = True

            if(self.objectDetected == False and self.duration > (time.time() - self.startTime)):
                self.motor.rotate(0.2) # rotate to find target
            else:
                self.motor.stay()
            # else: 
            #     on = False # stop the while loop

            
            time.sleep(0.05) # sampling rate

        print("[RESCUE DELIVERIES] Mission finished.")

    def stop(self):
        self.wayPNav.stop()
        


if __name__ == "__main__":
    mission = Rescue()
    try:
        mission.start()
        mission.run()
        mission.stop()
    except KeyboardInterrupt:
        mission.stop()