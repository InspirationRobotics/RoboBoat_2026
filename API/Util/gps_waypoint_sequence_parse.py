import math
import json

def apply_offset(lat, lon, dx, dy):
    """
    Apply a Cartesian offset (dx, dy) in meters to a latitude/longitude point.
    Args:
        dx (int, float) : Positive moves the point east, negative moves the point west.
        dy (int, float) : Positive moves the point north, negative moves the point south.
    """
    R = 6378137  # Earth's radius in meters
    dlat = (dy / R) * (180 / math.pi)
    dlon = (dx / (R * math.cos(math.radians(lat)))) * (180 / math.pi)
    return lat + dlat, lon + dlon

def process_waypoints(waypoint_path, sequence_path):
    """Read waypoints from a file, apply an offset, and overwrite the file."""
    
    # Load main data file (position and offset information)
    with open(waypoint_path, "r") as file:
        waypoints = json.load(file)

    # Load sequence file (list of keys in order)
    with open(sequence_path ,"r") as file:
        sequence = json.load(file)

    # initialize new waypoint list
    new_waypoints = []
    # Process data in the given sequence
    for key in sequence:
        if key in waypoints:  # Ensure key exists in data.json
            position = waypoints[key]["position"]
            offset =   waypoints[key]["offset"]
            newPosition = apply_offset(position[0],position[1],offset[0], offset[1])
            new_waypoints.append({"lat": newPosition[0], "lon": newPosition[1]})
        else:
            print(f"Warning: {key} not found in data.json")

    return new_waypoints

if __name__ =="__main__":
    # Example usage
    waypoint_path = "GNC/Guidance_Core/Config/waypoints.json"
    sequence_path = "GNC/Guidance_Core/Config/waypoint_sequence.json"

    wayP = process_waypoints(waypoint_path,sequence_path)
    # print(wayP)
    for w in wayP:
        print(f"[{w['lat']},{w['lon']}],")