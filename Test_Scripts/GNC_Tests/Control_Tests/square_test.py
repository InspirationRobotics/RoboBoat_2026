"""
To test ability to move in square (surge sway).

The test is considered successful if:
- The ASV moves forward for 4 seconds.
- The ASV moves backwards for 4 seconds.
- The ASV strafes (moves laterally) right for 4 seconds.
- The ASV strafes left for 4 seconds.
"""

import time
from GNC.Control_Core import motor_core

motors = motor_core.MotorCore("/dev/ttyACM2")


while(True):
    user_input = input("command>> ")
    # To run the square maneuver, just input "square" (see last elif)
    if(str(user_input) == "q"):
        print("quiting the code")
        start = time.perf_counter_ns()
        motors.stop()
        end = time.perf_counter_ns()
        print(f"Takes {end-start} ns to stop")
        break
    elif(str(user_input) == "w"):
        start = time.perf_counter_ns()
        motors.surge(0.5)
        end = time.perf_counter_ns()
        print(f"elapsed time: {end-start} ns")
    elif (str(user_input) == "s"):
        start = time.perf_counter_ns()
        motors.surge(-0.5)
        end = time.perf_counter_ns()
        print(f"elapsed time: {end - start} ns")
    elif(str(user_input) =="a"):
        start = time.perf_counter_ns()
        motors.slide(0.5)
        end = time.perf_counter_ns()
        print(f"elapsed time: {end-start} ns")
    elif(str(user_input) =="d"):
        start = time.perf_counter_ns()
        motors.slide(-0.5)
        end = time.perf_counter_ns()
        print(f"elapsed time: {end - start} ns")
    elif(str(user_input) =="0"):
        start = time.perf_counter_ns()
        motors.stay()
        end = time.perf_counter_ns()
        print(f"elapsed time: {end - start} ns")
    elif(str(user_input) =="square"):
        def wait():
            print("Stay for 2 s")
            motors.stay()
            time.sleep(2)

        start = time.perf_counter_ns()
        motors.surge(0.3)
        time.sleep(4)
        end = time.perf_counter_ns()
        print(f"Takes {time.perf_counter_ns()} ns to go forward")

        wait()

        start = time.perf_counter_ns()
        motors.slide(-0.3)
        time.sleep(4)
        end = time.perf_counter_ns()
        print(f"Takes {time.perf_counter_ns()} ns to slide right")

        wait()

        start = time.perf_counter_ns()
        motors.surge(-0.3)
        time.sleep(4)
        end = time.perf_counter_ns()
        print(f"Takes {time.perf_counter_ns()} ns to go backward")

        wait()

        start = time.perf_counter_ns()
        motors.slide(0.3)
        time.sleep(4)
        end = time.perf_counter_ns()
        print(f"Takes {time.perf_counter_ns()} ns to slide left")

        wait()

        print("finished the square")
