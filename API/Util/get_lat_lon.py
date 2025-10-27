import re
import time
from  API.GPS.gps_api import GPS, GPSData  # Ensure you import the correct GPS module

def log_gps():
    # Open the log file for writing waypoints
    log = open(f'GNC/Guidance_Core/Config/waypoints.txt', "w")

    def parse_coordinates(gps_string):
        """Extract latitude and longitude from a GPS data string."""
        match = re.search(r"Lat: ([\d\.\-]+), Lon: ([\d\.\-]+)", gps_string)
        if match:
            lat, lon = match.groups()
            return float(lat), float(lon)
        return None

    def callback(data: GPSData):
        """Callback function to process incoming GPS data."""
        gps_string = f"Lat: {data.lat}, Lon: {data.lon}, Heading: {data.heading}"
        # log.write(gps_string + '\n')  # Log raw GPS data

        parsed_lat_lon = parse_coordinates(gps_string)
        if parsed_lat_lon:
            lat, lon = parsed_lat_lon
            log.write(f"{lat}, {lon}\n")  # Save lat, lon as waypoints

    gps = GPS('/dev/ttyUSB0', 115200, callback=callback)

    rate = 2  # Period in seconds
    print(f"Beginning logging process @ one waypoint every {rate} seconds")
    
    try:
        while True:
            time.sleep(rate)
    except KeyboardInterrupt:
        print("Logging interrupted. Closing file.")
    finally:
        del gps
        log.close()

if __name__ == "__main__":
    log_gps()