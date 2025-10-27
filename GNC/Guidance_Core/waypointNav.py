"""
This is the script for waypoint navigation
"""
from GNC.Control_Core  import motor_core
from GNC.info_core import infoCore
from GNC.Guidance_Core.mission_helper import MissionHelper
import API.Util.gis_funcs as gpsfunc
import threading
import math
import time
from API.Servos.mini_maestro import MiniMaestro

class waypointNav:
    def __init__(self , infoCore, motors):  
        self.info               = infoCore
        self.motor              = motors

        self.waypoints :list    = None
        
        self.cur_ang            = None
        self.cur_dis            = None

        self.stop_event = threading.Event()  # STOP event


    def _loadConfig(self,file_path:str = "GNC/Guidance_Core/Config/barco_polo.json"):
        self.config = MissionHelper()
        self.config = self.config.load_json(path=file_path)

    def _loadWaypoints(self):
        self._loadConfig()
        print(f"path: {self.config['waypoint_file']}")
        self.waypoints = self._readLatLon(self.config['waypoint_file'])
        print("\nWaypoints: ")
        for points in self.waypoints:
            print(points)

    def loadWaypoints(self,points):
        """This is used when waypoint nav is not readed from txt"""
        self.waypoints = list(points)
        pass

    def _readLatLon(self,file_path:str)->list:
        lat_lon_list = []
    
        with open(file_path, 'r') as file:
            for line in file:
                lat, lon = map(float, line.strip().split(','))
                lat_lon_list.append({'lat': lat, 'lon': lon})
        
        return lat_lon_list  # dict in list

    def start(self):
        # load waypoints
        print("Loading waypoints...")
        self._loadWaypoints()



    def stop(self):
        self.info.stop_collecting()
        print("Background Threads stopped")
        self.motor.stop()
        print("Motors stoped")

    def stopThread(self):
        self.stop_event.set()  # Signal thread to stop
        print("Stop event set.")

    def run(self,points : dict = None, tolerance:int = 1.5):
        """Main logic of waypoint navigation"""
        distanceTolerance = tolerance       # 3 meters tolerance
        latin = points["lat"]  #lat
        lonin = points["lon"]  #lon
    
        # update bearing angle and distance
        self.updateDelta(lat=latin, lon=lonin)
        
        # store current distance
        initDis = self.cur_dis

        while(self.cur_dis>distanceTolerance):
            if self.stop_event.is_set():  # Check if stop was requested
                print("Stopping navigation thread.")
                return  # Exit thread
            # set max motor power pwm
            MAXFRONT    = 1
            MAXBACK     = 0.7

            # TODO test different graph and its impact on the performance, 
            # Try ^2.5 for turning power
            # Try ^0.4 for thruster power
            # Equation: x^3 why? Higher turning power at a greater angle, decreases as angle decreases, also can be + or - depend on angle
            turningPower = MAXBACK * self.cur_ang
            
            # Equation: 1-|x^0.2| why? concave up and decreasing as angle increase
            # TODO I think we need to add another varaible to slow down when distance is smaller
            thrusterPower = MAXFRONT # * (0.5* (self.cur_dis / (initDis-distanceTolerance)+0.5)) if(self.cur_dis<distanceTolerance*3) else MAXFRONT
	    
	    # thrusterPower = MAXFRONT
            # Veer based  on angle and distance
            # apply expoential relationship for turning power and angle
            self.motor.veer(thrusterPower,turningPower)
            # 0.1 s interval
            time.sleep(0.01)

            # update information
            self.updateDelta(lat=latin, lon=lonin)
        
        print("waypoint reached")
        
    def updateDelta(self,lat,lon):
        gpsdata = self.info.getGPSData()
        print(f"waypoints| lat: {lat} | lon: {lon}")
        self.cur_ang =  gpsfunc.relative_bearing(lat1=gpsdata.lat,lon1=gpsdata.lon,lat2=lat,lon2=lon,current_heading=gpsdata.heading)
        self.cur_dis =  gpsfunc.haversine(lat1=gpsdata.lat,lon1=gpsdata.lon,lat2=lat,lon2=lon) # this return angle to the range (-180,180)
        print(f"abs head: {gpsdata.heading} | lat: {gpsdata.lat} | lon: {gpsdata.lon}")
        print(f"delta ang: {self.cur_ang} | delta dis: {self.cur_dis}")
        # normalize angle to value between 0 and 1
        self.cur_ang /= 180
        return self.cur_ang, self.cur_dis



if __name__ == "__main__":
    config     = MissionHelper()
    print("loading configs")
    config     = config.load_json(path="GNC/Guidance_Core/Config/barco_polo.json")
    info       = infoCore(modelPath=config["competition_model_path"],labelMap=config["competition_label_map"])
    print("start background threads")
    info.start_collecting()
    motor      = motor_core.MotorCore("/dev/ttyACM2") # load with default port "/dev/ttyACM2"
    mission    = waypointNav(infoCore=info, motors=motor)

    # load waypoints
    waypoints  = mission._readLatLon(file_path = config["waypoint_file"])
    
    try:
        for p in waypoints:
            nav_thread = threading.Thread(target=mission.run, args=(p, 1.0), daemon=True)
            nav_thread.start()
            nav_thread.join()  # ✅ WAIT for thread to finish before stopping motors
        mission.stop()  # ✅ Stop everything AFTER all waypoints are reached
    except KeyboardInterrupt:
        print("\n[!] KeyboardInterrupt detected! Stopping mission...")
        mission.stopThread()  # ✅ Use stop event to signal stop
        nav_thread.join()
        mission.stop()  # Stop motors and background threads
        print("[✔] Mission stopped cleanly.")

    # shooting water
    maestro = MiniMaestro(port="/dev/ttyACM0")

    print("water gun")

    maestro.set_pwm(1, 1500)  # Move servo on channel 1
    time.sleep(1)
    maestro.set_pwm(1, 1800)  # Move servo on channel 1   
    time.sleep(2)
    maestro.set_pwm(1, 1500)  # Move servo on channel 1
    time.sleep(1)

    print("program finished")


