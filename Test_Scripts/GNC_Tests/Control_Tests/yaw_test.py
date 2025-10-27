from GNC.Control_Core import motor_core
import time

# Make motors
motors = motor_core.MotorCore("/dev/ttyACM2")

# Set speed to 1 for 20 seconds. Take out a stopwatch and learn
# how much time three rotations take, then change this argument

# It took about 5-6 seconds to complete each revolution, three
# were completed in 17 seconds with some movement, the rotation
# was sufficient
try:
    motors.rotate(0.5)
    time.sleep(20)
    motors.stay()
    time.sleep(2)
except KeyboardInterrupt:
    motors.stay()
    time.sleep(1)
    motors.stop()
    time.sleep(2)
    
