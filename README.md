# RoboBoat_2025 (Barco Polo Software Package v2.0)
Team Inspiration's Codebase for the 2025 RoboBoat Competition. We utilize a single ASV (autonomous surface vessel). See [Hardware Prerequisites](#hardware-prerequisites) for a more detailed specification of our current system.
Before developing ANY code, please read the [format](#format) section thoroughly.

## Structure
As of the time of writing (3/8/2025), the repository will utilize the following format in order to organize effectively.

```bash
RoboBoat_2025
|-- API
|    | -- Camera
|    |     | -- # Here is all of the low-level camera functionality.
|    | -- GPS
|    |     | -- # Here are all of the low-level GPS functionalities.
|    | -- IMU
|    |     | -- # Here is all of the low-level IMU functionality.
|    | -- Motors
|    |     | -- # Here is all of the low-level motor code.
|    | -- Actuators
|    |     | -- # Currently not developed, projected place to have the low-level control code for the racketball launcher and water cannon.
|    | -- Util
|    |     | -- # Here is all of the low-level functionality used across multiple sensors/devices.
|    
|-- GNC
|    | -- Control_Core
|    |     | -- # All of the code to control the ASV. This is mainly about allowing the ASV to move precisely and accurately.
|    | -- Guidance_Core
|    |     | -- # This is the highest level module of the whole codebase -- here are where the high-level mission/waypoint navigation algorithms are located. 
|
|-- Perception
|    | -- Models
|    |     | -- # Binary files that contain the weights of pre-trained ML models (YOLO -- .blob or .pt format)
|    | -- Perception_Core
|    |     | -- # Handles accessing camera streams, running ML models, and returning actionable results.
|   
|-- Test_Scripts
|   | -- API_Tests
|   |     | -- Camera_Tests
|   |     |     | -- Connectivity Tests
|   |     |     |     | -- # Tests to determine whether a camera can be accessed properly.
|   |     | -- GPS_Tests
|   |     |     | -- GPSLogs
|   |     |     |     | -- # GPS data logging files created by GPS Test files to test GPS API functionality.
|   |     |     | -- Missions
|   |     |     |     | -- # GPS data logging files created by GPS Test files to test GPS API functionality, specifically on mission usage.
|   |     |     | -- # All tests to confirm low-level GPS capabilities.
|   |     | -- IMU_Tests
|   |     |     | -- # All tests to confirm low-level IMU capabilities.
|   |     | -- Motor_Tests
|   |     |     | -- # All tests to confirm low-level motor capabilities.
|   |
|   | -- GNC_Tests
|   |     | -- Control_Tests
|   |     |    | -- # Tests to confirm control code. This will include station keeping, all basic movements, and moving to a given waypoint accurately.
|   |     | -- Nav_Tests
|   |     |     | -- # All tests to confirm navigation capabilities for the ASV. These rely on lower-level functionalities like the GPS and IMU..
|
|-- Setup
|   | -- # Shell scripts to install all necessary functionalities given the right hardware. 
```

## Usage
After cloning the repository, navigate to the RoboBoat_2025 repository:
```bash
cd RoboBoat_2025
```

### Running the system

Scripts should be run as modules from the root directory (i.e. inside RoboBoat_2025, but not in any folder within RoboBoat_2025). 
Here is an example to run the test_serial script inside Test_Scripts:
```bash
git pull
python3 -m Test_Scripts.API_Tests.Motor_Tests.test_serial
```

### Miscellaneous Files
- "__pycache__" files should NOT be pushed or pulled from the cloud -- there is a .gitignore file in the root directory that ignores changes to these 
files. Please make sure you do not override this and push __pycache__ files to the cloud.
- There should be "__init__" files located in each folder in our repository. This is because our (python) scripts are run as modules.
- As stated in [Usage](#usage), the API, GNC, and Perception folders are set up as python packages. As such, each of these have a "setup.py" file.

## Dependencies

### Hardware
Barco Polo sensor payloads entail a Beitan GPS module, BNO085 Adafruit IMU, Oak-D LR, USB Arducam.
Barco Polo actuators include a four T200 thrusters in a holonomic configuration, a racquetball launcher, and a water cannon. The motors are connected to an Arduino Uno, while the launcher and cannon are connected to a Polulu Mini Maestro.
Our current electrical diagram can be found here: https://drive.google.com/file/d/1p00fLm1HOUzSkRoLzEd6WDd1H0qSmS08/view?usp=drive_link
Hardware list can be found here: https://docs.google.com/spreadsheets/d/1ug_iB4X5CtvmAm4iIsihrIVFvDHcIRy0a3RY6WEW07w/edit?gid=0#gid=0

### Software
This repository is designed to run on Barco Polo. Most dependenceis can be installed using bash scripts located in the Setup folder. Any version of Ubuntu will most likely work, 22.04 Jammy Jellyfish is the version on Barco Polo (03/09/2025)

## Format
### Best Practices/Standards for Development
Please conform to the following guidelines when developing code:
1. All code should be tested and verified in all of its functionality before being pushed to the cloud. If this is not feasible, such as when having to push code remotely to enable testing on the ASV, please make note of it, using the "NOTE" keyword in a comment at the top of the file or function.
2. Besides noting the status of functionality if warranted (see [1.](#1.)), it is important that all code is well-documented and readable. Please see [Documentation Guidelines](#documentation-guidelines) for more information.
3. When importing files and classes, make sure the imports are absolute, not relative. This decreases problems to fix when restructuring.

### Documentation Guidelines
The goal is to keep our code well-documented and well-organized. To ensure this, please:
- Add a header at the top of each file detailing what its intended purpose is.
- Add docstrings to each class and function, explaining the purpose of the class/function.
- For functions that take arguments/return specific items, please include in the docstring the arguments expected, along with the type, as well as the type of object returned and what its significance is.
- Add "#" comments throughout the code, especially for long/more convoluted files to explain low-level processes.

## Future Development
Current Version: 2.0.0

At competition, we were only able to use waypoints to complete actual missions. Oak-D models and CV can be run. Example usage can be found in the API_Tests/Camera_Tests and Test_Scripts/Perception_Tests directories. 

### Version Control/Naming
Post-competition Barco Polo will be considered version 2.0. Individual commits will not necessarily increase version number. Small confirmed method changes in one area will be a .1 change (i.e. from 2.3 to 2.4), while major changes in one or more core sections will be considered a 1. change (i.e. 3.3 to 4.0). Each of the "core areas" will be defined below.

### Control
Control is defined as actuation of servos and thrusters (interfaces between the high-level mission algorithms (i.e. where to move to at a high level) at the actuators). Improvements include:
1. Better waypoint navigation to include strafe and backwards surge maneuvers. Right now waypoint navigation is a simple P controller that only yaws and surges in the forward direction.
2. Creating a PID controller instead of a simple P controller to more accurately move to a waypoint.
3. Create more indivdualized functions that allow to navigate to a certain cartesian vector, as well as rotate to a given heading (relative and absolute).
4. Create extended kalman filer (EKF) between differential GPS and IMU to enable more accurate state position.

### Perception
Perception is defined as anything that increases the system's knowledge of the environment, aside from its movement/position. 
1. Create both computer vision (CV) algorithms and YOLO models that enable actual accurate detection of objects of interest:
    - CV includes SIFT, Contours, HSV masks, CLAHE, etc.
    - While YOLO has its already built workflow, it is important that team members are able to understand how to train models (i.e. how to label datasets to optimize performance.)
2. Use the DepthAI libraries to implement preprocessing, occupancy grid, diff map, etc.
3. Integrate solid state LIDAR

### Mapping
Mapping integrates perception and control modules to determine the state of the ASV relative to its environment. This includes pre-run surveying and autonomous mapping.
1. Pre-Course Surveying:
    - Implement a more accurate GPS waypoint worflow. Possibilities include obtaining GPS module, more accurate rangefinder, creating lines of intersection between reference points to ensure that the point is in the correct position.
    - Create a visualizer/app to plot waypoints on a course. Plotting waypoints and movement inside this visualizer, and then running the program will automatically be the mission path for the system.
2. Autonomous Mapping:
    - 

### Mission Planning
TBD

