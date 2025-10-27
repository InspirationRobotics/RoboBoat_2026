"""
Test script to test GPS functionality. This includes connectivity and data parsing into a readable format.
Either puts data into a log or prints parsed data directly on the screen.

This test is considered successful when the GPS is connected either to a personal user's computer or to the Jetson on the ASV, and either:
- Prints the data from the GPS quickly (> 3 Hz) on terminal.
- Creates a log (.txt) file in the directory RoboBoat_2025/Test_Scripts/API_Tests/GPS_Tests/GPSLogs.
The data should be in the form of: ["Lat: {latitude}, Lon: {longitude}, Heading: {heading}"]
"""

from API.GPS.gps_api import GPS, GPSData

import time

def log_gps():
    # log = open(f'Test_Scripts/API_Tests/GPS_Tests/GPSLogs/GPSlog_{int(time.time())}.txt', "w")
    log = open(f'GNC/Guidance_Core/Config/waypoints.txt', "w")

    def callback(data : GPSData):
        log.write(str(data) + '\n')

    gps = GPS('/dev/ttyUSB0', 115200, callback=callback)

    rate = 2 # Period
    print(f"Beginning logging process @ one waypoint every {rate} seconds")
    while True:
        try:
            time.sleep(rate)
        except KeyboardInterrupt:
            break

    del gps
    log.close()

def print_gps():

    def callback(data : GPSData):
        print(data)

    gps = GPS('/dev/ttyUSB0', 115200, callback=callback, offset=270.96788823529414)
    while True:
        try:
            time.sleep(1)
        except KeyboardInterrupt:
            break

    del gps

if __name__ == "__main__":
    # To test print ability
    # print_gps()

    
    # To test log ability
    log_gps()