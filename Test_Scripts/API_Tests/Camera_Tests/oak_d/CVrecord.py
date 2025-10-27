import cv2
import depthai as dai

# Create DepthAI pipeline
pipeline = dai.Pipeline()

# Create a node for the left color camera
CAM_A = pipeline.create(dai.node.ColorCamera)

# Create an output node
xout_A = pipeline.create(dai.node.XLinkOut)
xout_A.setStreamName("CAM_A")

# Configure the color camera (Left)
CAM_A.setBoardSocket(dai.CameraBoardSocket.CAM_A)  # Use CAM_A for the left camera
CAM_A.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
CAM_A.setIspScale(2, 3)  # Reduce resolution if needed
CAM_A.setFps(30)

# Link the camera output
CAM_A.isp.link(xout_A.input)

# Connect to the device and start the pipeline
with dai.Device(pipeline) as device:
    CAM_A_q = device.getOutputQueue(name="CAM_A", maxSize=4, blocking=False)

    print("Capturing images...")

    # Wait for the first frame to determine resolution
    first_msg = CAM_A_q.get()
    if first_msg is None:
        print("Error: No frames received from CAM_A!")
        exit(1)
    
    first_frame = first_msg.getCvFrame()
    height, width = first_frame.shape[:2]
    frame_size = (width, height)
    print(frame_size)

    # Define video output parameters
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
    fps = CAM_A.getFps()
    print(fps)
    output_file = "dockingData.mp4"


    # Initialize VideoWriter
    out = cv2.VideoWriter(output_file, fourcc, fps, frame_size)
    try:
        while True:
            # Get frames safely
            msg_A = CAM_A_q.get()
            if msg_A is None:
                print("Warning: No frame from CAM_A, skipping...")
                continue
            
            camA_frame = msg_A.getCvFrame()

            # Display image
            out.write(camA_frame)

            if cv2.waitKey(1) == ord('q'):
                break
    
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        out.release()
