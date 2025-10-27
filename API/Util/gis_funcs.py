"""
Utility GPS functions. To be used in map and motor core code.
Assumes spherical earth.

In general, this script contains functions that take in GPS data
and use those data to caculate useful informaiton
"""

import math
from typing import Tuple, Union

R = 6371000 # Earth's radius in meters.

"""
---------------------------- TESTED FUNCTIONS --------------------------------------
"""


def haversine(lat1, lon1, lat2, lon2) -> float:
    """
    Calculate the great-circle distance between two points
    on the Earth (specified in decimal degrees).
    Returns distance in meters.
    """
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(a ** 0.5)  # Faster than math.sqrt(a)
    r = 6371000  # Radius of Earth in meters (use 3956 for miles)
    
    return c * r  # Distance in meters

def bearing(lat1, lon1, lat2, lon2) -> float:
    """
    Calculate the bearing between two points
    on the earth (specified in decimal degrees).
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])

    # calculate the bearing
    y = math.sin(lon2-lon1) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(lon2-lon1)
    solved_bearing = math.degrees(math.atan2(y, x))
    solved_bearing = (solved_bearing + 360) % 360
    return solved_bearing

def relative_bearing(lat1, lon1, lat2, lon2, current_heading) -> float:
    """
    Calculate the relative bearing between two points on Earth (in decimal degrees)
    relative to the current heading. The result is normalized to (-180, 180] degrees.
    """
    abs_bearing = bearing(lat1, lon1, lat2, lon2)
    relative_bearing = (abs_bearing - current_heading + 180) % 360 - 180  
    return 180 if relative_bearing == -180 else relative_bearing  # Ensures -180 → 180



def vector_to_target(pos1 : Tuple, pos2 : Tuple, current_heading) -> Tuple[float, float, float]:
    """
    Calculate the vector to the target point (retuns vx, vy, total distance).
    """
    lat1, lon1 = pos1
    lat2, lon2 = pos2
    dist = haversine(lat1, lon1, lat2, lon2)
    bearing = relative_bearing(lat1, lon1, lat2, lon2, current_heading)
    vx = math.cos(math.radians(bearing)) * dist
    vy = math.sin(math.radians(bearing)) * dist
    # normalize
    if abs(vx) > 1 or abs(vy) > 1:
        max_val = abs(vx) + abs(vy)
        vx/=max_val
        vy/=max_val
    # vx is forwards
    # vy is lateral
    return vx, vy, dist

def destination_point(lat, lon, bearing, distance) -> Tuple[float, float]:
    """
    Calculate the destination point (lat, lon) given a starting point, bearing (desired), and distance in meters
    """
    # convert decimal degrees to radians
    lon, lat, bearing = map(math.radians, [lon, lat, bearing])

    distance = distance/1000

    # calculate the destination point
    lat2 = math.asin(math.sin(lat) * math.cos(distance/6371) + math.cos(lat) * math.sin(distance/6371) * math.cos(bearing))
    lon2 = lon + math.atan2(math.sin(bearing) * math.sin(distance/6371) * math.cos(lat), math.cos(distance/6371) - math.sin(lat) * math.sin(lat2))
    lat2 = math.degrees(lat2)
    lon2 = math.degrees(lon2)
    return lat2, lon2

"""
------------------------------------------------------------------
"""

"""
----------------------------- FUNCTIONS THAT REQUIRE CAREFUL PROOF-READING (ARE PROBABLY INACCURATE) ---------------------------------
"""
# NOTE that these functions may or may not be inaccurate; I put R as the wrong constant when initally testing, have not changed it since (as of 2/11/2025).

def calculate_midpoint(point1 : Tuple[float, float], point2: Tuple[float , int]) -> Tuple[float, float]:
    """
    Return the midpoint of two points. Points will be in the form of (lat, lon).
    """
    lat_1, lon_1 = map(math.radians, point1)
    lat_2, lon_2 = map(math.radians, point2)

    Bx = math.cos(lat_2) * math.cos(lon_2 - lon_1)
    By = math.cos(lat_2) * math.sin(lon_2 - lon_1)

    lat_mid = math.atan2(
        math.sin(lat_1) + math.sin(lat_2),
        math.sqrt((math.cos(lat_1) + Bx)**2 + (By)**2)
    )
    lon_mid = lon_1 + math.atan2(By, math.cos(lat_1) + Bx)
    lat_mid, lon_mid = map(math.degrees, [lat_mid, lon_mid])

    lon_mid = (lon_mid + 180) % 360 - 180

    return (lat_mid, lon_mid)

def calculate_heading_to_waypoint(current_point : Tuple[float, float], waypoint : Tuple[float, float]) -> float:
    curr_lat = current_point[0]
    curr_lon = current_point[1]
    waypoint_lat = waypoint[0]
    waypoint_lon = waypoint[1]

    curr_lat, curr_lon, waypoint_lat, waypoint_lon = map(math.radians, [curr_lat, curr_lon, waypoint_lat, waypoint_lon])
    dlon = waypoint_lon - curr_lon

    x = math.cos(waypoint_lat) * math.sin(dlon)
    y = math.cos(curr_lat) * math.sin(waypoint_lat) - math.sin(curr_lat) * math.cos(waypoint_lat) * math.cos(dlon)
    bearing = math.atan2(x, y)

    # Convert back to degrees and normalize.
    bearing = (math.degrees(bearing) + 360) % 360
    return bearing

def calculate_waypoint_from_vector(current_point : Tuple[float, float], bearing : float, magnitude : float) -> Tuple[float, float]:
    curr_lat, curr_lon = map(math.radians, current_point)
    bearing = math.radians(bearing)
    angular_distance = magnitude/R

    waypoint_lat = math.asin(
        math.sin(curr_lat) * math.cos(angular_distance) + math.cos(curr_lat) * math.sin(angular_distance) * math.cos(bearing)
    )
    waypoint_lon = curr_lon + math.atan2(
        math.sin(bearing) * math.sin(angular_distance) * math.cos(curr_lat),
        math.cos(angular_distance) - (math.sin(curr_lat) * math.sin(waypoint_lat))
    )

    waypoint_lat, waypoint_lon = map(math.degrees, [waypoint_lat, waypoint_lon])
    return(waypoint_lat, waypoint_lon)

# NOTE: These functions commented are extremely inaccurate.
# def calculate_latitude_distance(lat_1 : float, lat_2 : float) -> float:
#     lat1, lat2 = map(math.radians, [lat_1, lat_2])
#     distance_latitude = R * abs(lat2 - lat1)
#     return distance_latitude

# def calculate_longitude_distance(lon_1 : float, lon_2 : float, average_latitude : float) -> float:
#     lon1, lon2, latitude = map(math.radians, [lon_1, lon_2, average_latitude])
#     distance_longitude = R * abs(lon2 - lon1) * math.cos(latitude)
#     return distance_longitude

def extend_vector(vector_tail : Tuple[float, float], curr_vector_head : Tuple[float, float], scale : float) -> Tuple[float, float]:
    curr_lat, curr_lon = map(math.radians, vector_tail)
    # curr_waypoint_lat, curr_waypoint_lon = map(math.radians, curr_vector_head)
    distance = haversine(vector_tail, curr_vector_head)
    scaled_distance = distance * scale
    angular_distance = scaled_distance/R

    bearing = calculate_heading_to_waypoint(vector_tail, curr_vector_head)
    bearing = math.radians(bearing)

    new_lat = math.asin(
        math.sin(curr_lat) * math.cos(angular_distance) + 
        math.cos(curr_lat) * math.sin(angular_distance) * math.cos(bearing)
    )

    new_lon = curr_lon + math.atan2(
        math.sin(bearing) * math.sin(angular_distance) * math.cos(curr_lat),
        math.cos(angular_distance) - math.sin(curr_lat) * math.sin(new_lat)
    )

    new_lat, new_lon = map(math.degrees, [new_lat, new_lon])

    return (new_lat, new_lon)

"""
------------------------------------------------------------------
"""

if __name__ == '__main__':

    import random

    def _generate_random_coords(distance_between):
        # Generates two random coordinates that are distance_between apart
        # Generate one random coordinate
        lat1 = random.uniform(-90, 90)
        lon1 = random.uniform(-180, 180)

        # Generate a second random coordinate that is distance_between apart
        bearing = random.uniform(0, 360)
        lat2, lon2 = destination_point(lat1, lon1, bearing, distance_between)
        return lat1, lon1, lat2, lon2

    # test the functions
    lat1, lon1 = 27.3743248995, -82.4529466212
    lat2, lon2 = 27.374976, -82.454223
    # print(haversine(lat1, lon1, lat2, lon2))
    # print(bearing(lat1, lon1, lat2, lon2))
    # print(relative_bearing(lat1, lon1, lat2, lon2, 90))
    print(vector_to_target([lat1, lon1], [lat2, lon2], 291.8644882352941))
    # print(destination_point(lat1, lon1, 90, 100))
    # for i in range(10):
    #     ran_dist = random.uniform(0, 4)
    #     lat1, lon1, lat2, lon2 = _generate_random_coords(ran_dist)
    #     vx, vy, dist = vector_to_target([lat1, lon1], [lat2, lon2], 90)
    #     scale = min(dist/3, 1)
    #     target_vector = [vx*scale, vy*scale]
    #     print(f"Dist: {dist}, Target Vector: {target_vector}")
